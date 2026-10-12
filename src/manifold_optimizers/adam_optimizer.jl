function OptimizerCache(::Adam, x::OptimizerSolution)
    AdamCache(_copy(x), _zero(x), _zero(x))
end
Hessian(::Adam, ::OptimizerProblem, ::OptimizerSolution{T}) where {T} = NoHessian{T}()

# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
# The invariant is enforced by the outer constructors below.
"""
    AdamCache <: OptimizerCache

Cache for the gradient optimizer.

# Fields
- `x::`[`Manifold`](@ref): the solution,
- `g`: the gradient (for the *manifold case* this is in [`AbstractLieAlgHorMatrix`](@ref) form),
- `δ`: the direction,
- `Δg`: difference in gradients, needed for [`OptimizerStatus`](@ref),
- `g̃`: scratch for [`latest_gradient`](@ref); see [`GradientCache`](@ref),
- `g̃_is_current`: whether `g̃` is the gradient at `x`; see [`store_gradient!`](@ref),
- `section`: the [`GlobalSection`](@ref).
"""
struct AdamCache{T, MT, VT, ST} <: OptimizerCache{T}
    x::MT
    g::VT
    δ::VT
    Δg::VT
    g̃::VT
    g̃_is_current::Base.RefValue{Bool}
    m₁::VT
    m₂::VT
    section::ST
end

first_moment(cache::AdamCache) = cache.m₁
second_moment(cache::AdamCache) = cache.m₂

# The `new` in a body of its own, reached with every member already computed and passed in, so that
# this frame infers from its *own signature* rather than through the tree that built them.
#
# This is not tidiness. On a large nested container, the inference of one body that composes every
# member and ends in a ten-field `new` with four large type parameters costs most of the compile time
# of `OptimizerCache(Adam, ps)`, although no piece of it is expensive alone.
# `scripts/adam_cache_attribution.jl` measures it. A frame that infers from its own signature removes
# the cost, where a `@noinline` on the composing body does not.
@noinline function _adam_cache(x::OptimizerSolution{T}, g::VT, δ::VT, Δg::VT, g̃::VT,
        m₁::VT, m₂::VT, sec::ST) where {T, VT, ST}
    AdamCache{T, typeof(x), VT, ST}(x, g, δ, Δg, g̃, Ref(false), m₁, m₂, sec)
end

function AdamCache(x::OptimizerSolution{T}, g::AT, δ::AT, Δg::AT) where {
        T, AT <: GradientStorage{T}}
    sec = GlobalSection(_copy(x))
    g̃ = _similar(g)
    _fill!(g̃, T(NaN))
    _adam_cache(x, g, δ, Δg, g̃, _similar(g), _similar(g), sec)
end

function AdamCache(x::OptimizerSolution{T}, g::AT, δ::AT) where {
        T, AT <: GradientStorage{T}}
    Δg = _similar(g)
    _fill!(Δg, T(NaN))
    AdamCache(x, g, δ, Δg)
end

function AdamCache(x::OptimizerSolution{T}, g::GradientStorage{T}) where {T}
    δ = _zero(g)
    AdamCache(x, g, δ)
end

function AdamCache(x::OptimizerSolution{T}) where {T}
    g = _zero(x)
    _fill!(g, T(NaN))
    AdamCache(x, g)
end

direction(cache::AdamCache) = cache.δ
rhs(cache::AdamCache) = direction(cache)
# As for `MomentumCache`: `rhs` aliases the direction, which is `-m₁/(√m₂ + δ)` and not `-∇f`.
steepest_descent!(cache::AdamCache) = _steepest_descent_from_gradient!(cache)

# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
# The invariant is enforced by the outer constructors below.
"""
    AdamState <: OptimizerState

State for the gradient optimizer.
"""
mutable struct AdamState{T, OT, GS, VT} <: OptimizerState{T}
    section::GS
    iterations::Int

    x::OT
    x̄::OT
    g::VT
    ḡ::VT
    m₁::VT
    m₂::VT
    f::T
    f̄::T
end

first_moment(state::AdamState) = state.m₁
second_moment(state::AdamState) = state.m₂

function AdamState(x::OST, g::GradientStorage{T}) where {T, OST <: OptimizerSolution{T}}
    _x = _copy(x)
    _g = _copy(g)
    gs = GlobalSection(_x)
    # note that the moments have to be initialized with zeros (and not with `_similar`):
    # they are read in the first call to `update!(::AdamCache, ...)` before they are
    # written to for the first time, so uninitialized memory would be used there.
    AdamState{T, typeof(_x), typeof(gs), typeof(_g)}(
        gs, 0, _x, _similar(_x), _g, _similar(_g),
        _zero(_g), _zero(_g), T(NaN), T(NaN))
end

AdamState(x::OptimizerSolution) = AdamState(x, _zero(x))

OptimizerState(::Adam, x...) = AdamState(x...)

# The first step weighs the stored moments by `(β - β¹)/(1 - β¹) = 0`, which does not clear a moment
# that an earlier solve left non-finite, since `0 ⋅ Inf` is `NaN`.
function initialize_state!(state::AdamState{T}) where {T}
    _fill!(first_moment(state), zero(T))
    _fill!(second_moment(state), zero(T))
    state
end

# `AdamWithEuclideanDecay` shares the cache and the state, and its moments are `Adam`'s.
function advance_state!(
        state::AdamState, cache::AdamCache, ::Union{Adam, AdamWithEuclideanDecay})
    _copyto!(section(state), section(cache))
    _copyto!(first_moment(state), first_moment(cache))
    _copyto!(second_moment(state), second_moment(cache))
    state
end

function update!(cache::AdamCache{T}, state::AdamState{T}, gradient::Gradient{T},
        β₁::T, β₂::T, δ::T, t::Integer, x::OptimizerSolution{T}) where {T}
    # first, and before the two `_copyto!`s below; see `store_gradient!`
    store_gradient!(cache, state, gradient, x)
    _copyto!(section(cache), section(state))
    _copyto!(solution(cache), x)
    # `t` is the iteration number and is counted from one: `solve!` calls
    # `increase_iteration_number!` *before* `solver_step!`, so `t = state.iterations` is already
    # the number of the step that is being taken here, and the first direction is the `sign(∇L)`
    # that the bias correction is supposed to produce. Note that the correction is
    # undefined for `t = 0` (`1 - β^0 = 0`), so a call sequence that forgets to increment fails
    # here rather than quietly producing `NaN`s.
    @assert t ≥ 1 "the bias-corrected Adam moments are undefined before the first iteration (t = $(t)); `increase_iteration_number!` has to be called before the step"
    # the moments are stored in bias-corrected form, hence the `- β^t` in the numerators of
    # `fac₁₁` and `fac₂₁` (see the docstring of [`Adam`](@ref)).
    fac₁₁ = (β₁ - β₁^t) / (1 - β₁^t)
    fac₁₂ = (1 - β₁) / (1 - β₁^t)
    fac₂₁ = (β₂ - β₂^t) / (1 - β₂^t)
    fac₂₂ = (1 - β₂) / (1 - β₂^t)
    # One broadcast per leaf for both moments and the direction, into the cache, so that the step
    # allocates nothing. The direction is `-m₁/(√m₂ + δ)`, which is *not* scaled by a learning rate:
    # that is the line search's `α` (see [`default_linesearch`](@ref)).
    function adam_leaf!(d, m₁, m₂, m̄₁, m̄₂, g)
        m₁ .= fac₁₁ .* m̄₁ .+ fac₁₂ .* g
        m₂ .= fac₂₁ .* m̄₂ .+ fac₂₂ .* g .^ 2
        d .= .-(m₁ ./ (sqrt.(m₂) .+ δ))
    end
    _storagewise!(adam_leaf!, direction(cache), first_moment(cache), second_moment(cache),
        first_moment(state), second_moment(state), gradient_array(cache))

    cache
end

function update!(cache::AdamCache{T}, state::AdamState{T}, gradient::Gradient{T},
        method::Adam{T}, x::OptimizerSolution{T}) where {T}
    update!(cache, state, gradient, method.β₁, method.β₂, method.δ, state.iterations, x)
end
