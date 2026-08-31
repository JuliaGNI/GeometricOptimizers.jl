function OptimizerCache(::Adam{T}, x::OptimizerSolution{T}) where {T}
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
    m̃₂::VT
    section::ST
end

first_moment(cache::AdamCache) = cache.m₁
second_moment(cache::AdamCache) = cache.m₂
_second_moment(cache::AdamCache) = cache.m̃₂

# The `new` in a body of its own, reached with every member already computed and passed in, so that
# this frame infers from its *own signature* rather than through the tree that built them.
#
# This is not tidiness. Building an `OptimizerCache(Adam, ps)` for a 16 × 24 nested container cost
# about 88 s on Julia 1.11.9, and `scripts/adam_cache_attribution.jl` put **81.49 s of it in the
# four-argument constructor's body** — while every piece that body calls is under a second and a half
# (`_zero` 1.21, `_fill!` 0.38, `_similar` 0.89, `_copy` 0.54, `GlobalSection(_copy(x))` 0.87) and
# `parameterlayout` and `flatten` measured after all of them are 2.33 s of the 81, under 3 %. Nothing in
# it is expensive; composing them in one inferred body was, and a ten-field `new` with four large type
# parameters at the end of that body is the composition. On 1.13 the same body is 3.54 s — 23× cheaper —
# with every individual piece *dearer*, which is what says it is inference and not work.
#
# Splitting rather than annotating, and that choice is evidence-led: the one adjacent thing already
# tried is open issue D1's `@noinline`, which did nothing there, whereas D1's other control — flattening
# the nesting so each frame infers from its signature — took 940 s to 6.53 s. This is that control
# applied here. See the changelog for what it measured.
@noinline function _adam_cache(x::OptimizerSolution{T}, g::VT, δ::VT, Δg::VT, g̃::VT,
        m₁::VT, m₂::VT, m̃₂::VT, sec::ST) where {T, VT, ST}
    AdamCache{T, typeof(x), VT, ST}(x, g, δ, Δg, g̃, Ref(false), m₁, m₂, m̃₂, sec)
end

function AdamCache(x::OptimizerSolution{T}, g::AT, δ::AT, Δg::AT) where {
        T, AT <: GradientStorage{T}}
    sec = GlobalSection(_copy(x))
    g̃ = _similar(g)
    _fill!(g̃, T(NaN))
    _adam_cache(x, g, δ, Δg, g̃, _similar(g), _similar(g), _similar(g), sec)
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

solution(cache::AdamCache) = cache.x
gradient(cache::AdamCache) = cache.g
gradient_array(cache::AdamCache) = gradient(cache)
latest_gradient(cache::AdamCache) = cache.g̃
function refresh_latest_gradient!(cache::AdamCache, g::Gradient)
    _refresh_latest_gradient!(cache, g)
end
function latest_gradient_is_current(cache::AdamCache, state::OptimizerState, x::OptimizerSolution)
    _latest_gradient_is_current(cache, state, x)
end
invalidate_latest_gradient!(cache::AdamCache) = _invalidate_latest_gradient!(cache)
function gradient_difference!(cache::AdamCache, ::OptimizerState)
    _latest_gradient_difference!(cache)
end
direction(cache::AdamCache) = cache.δ
rhs(cache::AdamCache) = direction(cache)
# As for `MomentumCache`: `rhs` aliases the direction, which is `-m₁/(√m₂ + δ)` and not `-∇f`.
steepest_descent!(cache::AdamCache) = _steepest_descent_from_gradient!(cache)
section(cache::AdamCache) = cache.section

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
    m̃₂::VT
    f::T
    f̄::T
end

solution(state::AdamState) = state.x
previous_solution(state::AdamState) = state.x̄
gradient(state::AdamState) = state.g
previous_gradient(state::AdamState) = state.ḡ
value(state::AdamState) = state.f
previous_value(state::AdamState) = state.f̄
first_moment(state::AdamState) = state.m₁
second_moment(state::AdamState) = state.m₂
_second_moment(state::AdamState) = state.m̃₂

section(state::AdamState) = state.section

function AdamState(x::OST, g::GradientStorage{T}) where {T, OST <: OptimizerSolution{T}}
    _x = _copy(x)
    _g = _copy(g)
    gs = GlobalSection(_x)
    # note that the moments have to be initialized with zeros (and not with `_similar`):
    # they are read in the first call to `update!(::AdamCache, ...)` before they are
    # written to for the first time, so uninitialized memory would be used there.
    AdamState{T, typeof(_x), typeof(gs), typeof(_g)}(
        gs, 0, _x, _similar(_x), _g, _similar(_g),
        _zero(_g), _zero(_g), _zero(_g), T(NaN), T(NaN))
end

AdamState(x::OptimizerSolution) = AdamState(x, _zero(x))

OptimizerState(::Adam, x...) = AdamState(x...)

function update!(state::AdamState{T}, gradient_array::GradientStorage{T},
        direction::GradientStorage{T}, _first_moment::GradientStorage{T},
        _second_moment::GradientStorage{T},
        x::OptimizerSolution{T}, f::Callable, retraction) where {T}
    _copyto!(previous_solution(state), solution(state))
    _copyto!(previous_gradient(state), gradient(state))
    state.f̄ = value(state)
    _copyto!(solution(state), x)
    _copyto!(gradient(state), gradient_array)
    _copyto!(first_moment(state), _first_moment)
    _copyto!(second_moment(state), _second_moment)
    state.f = f(x)

    update_section!(section(state), direction, retraction)

    state
end

function update!(state::AdamState, opt::Optimizer, x::OptimizerSolution)
    update!(
        state, gradient_array(cache(opt)), direction(cache(opt)), first_moment(opt.cache),
        second_moment(opt.cache), x, problem(opt).F, opt.retraction)
end

function update!(cache::AdamCache{T}, state::AdamState{T}, gradient::Gradient{T},
        β₁::T, β₂::T, δ::T, t::Integer, x::OptimizerSolution{T}) where {T}
    # first, and before the two `_copyto!`s below; see `store_gradient!`
    store_gradient!(cache, state, gradient, x)
    _copyto!(section(cache), section(state))
    _copyto!(solution(cache), x)
    # `t` is the iteration number and is counted from one: `solve!` calls
    # `increase_iteration_number!` *before* `solver_step!`, so `t = state.iterations` is already
    # the number of the step that is being taken here. It used to be incremented a second time
    # (`_t = t + 1`), which made the very first direction `0.7425⋅sign(∇L)` instead of the
    # `sign(∇L)` that the bias correction is supposed to produce. Note that the correction is
    # undefined for `t = 0` (`1 - β^0 = 0`), so a call sequence that forgets to increment fails
    # here rather than quietly producing `NaN`s.
    @assert t ≥ 1 "the bias-corrected Adam moments are undefined before the first iteration (t = $(t)); `increase_iteration_number!` has to be called before the step"
    # the moments are stored in bias-corrected form, hence the `- β^t` in the numerators of
    # `fac₁₁` and `fac₂₁` (see the docstring of [`Adam`](@ref)).
    fac₁₁ = (β₁ - β₁^t) / (1 - β₁^t)
    fac₁₂ = (1 - β₁) / (1 - β₁^t)
    fac₂₁ = (β₂ - β₂^t) / (1 - β₂^t)
    fac₂₂ = (1 - β₂) / (1 - β₂^t)
    _copyto!(first_moment(cache), _mul(fac₁₁, first_moment(state)))
    _add!(first_moment(cache), _mul(fac₁₂, gradient_array(cache)))
    _copyto!(second_moment(cache), _mul(fac₂₁, second_moment(state)))
    _add!(second_moment(cache), _mul(fac₂₂, _square(gradient_array(cache))))
    # `m̃₂ = √m₂ + δ`; note that the square root must not be applied to `m₂` in place, as
    # `m₂` is stored in the state in `update!(::AdamState, ...)` afterwards.
    _copyto!(_second_moment(cache), second_moment(cache))
    _rac!(_second_moment(cache))
    _add!(_second_moment(cache), δ)
    # the direction is `-m₁/(√m₂ + δ)`, which is *not* scaled by a learning rate: that is the
    # line search's `α` (see [`default_linesearch`](@ref)).
    _copyto!(direction(cache), first_moment(cache))
    _div!(direction(cache), _second_moment(cache))
    _rmul!(direction(cache), -1)

    cache
end

function update!(cache::AdamCache{T}, state::AdamState{T}, gradient::Gradient{T},
        method::Adam{T}, x::OptimizerSolution{T}) where {T}
    update!(cache, state, gradient, method.β₁, method.β₂, method.δ, state.iterations, x)
end
