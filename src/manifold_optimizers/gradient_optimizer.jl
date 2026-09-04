function OptimizerCache(::GradientMethod, x::OptimizerSolution)
    GradientCache(_copy(x), _zero(x), _zero(x))
end
function Hessian(::GradientMethod, ::OptimizerProblem, ::OptimizerSolution{T}) where {T}
    NoHessian{T}()
end

struct NoHessian{T} <: Hessian{T} end

# The other half of the owned `Hessian` functor error; see the note on
# `(::IterativeHessian)(::AbstractMatrix, ::OptimizerSolution)` in
# `optimizers/iterative_hessians/bfgs/hessian_bfgs.jl` for why the two are split.
function (hes::NoHessian)(::AbstractMatrix, ::OptimizerSolution)
    error("This has to be called together with a cache.")
end

# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
# The invariant is enforced by the outer constructors below.
"""
    GradientCache <: OptimizerCache

Cache for the gradient optimizer.

# Fields
- `x::`[`Manifold`](@ref): the solution,
- `g`: the gradient (for the *manifold case* this is in [`AbstractLieAlgHorMatrix`](@ref) form),
- `δ`: the direction,
- `Δg`: difference in gradients,
- `g̃`: scratch for [`latest_gradient`](@ref),
- `g̃_is_current`: whether `g̃` is the gradient at `x`; see [`store_gradient!`](@ref),
- `section`: the [`GlobalSection`](@ref).

# Implementation

!!! info "Why `g̃` is a field and not an alias for `g`"
    The line search evaluates ``\\varphi'(\\alpha)`` *into* an array of the cache — that is what makes
    [`trial_slope`](@ref) allocation-free — and [`solver_step!`](@ref) refreshes the same array at the
    accepted iterate. Neither may land in `g`, because `g` is ``\\nabla{}f(x_k)`` at the iterate the
    direction was built from and the state updates read it afterwards: `update!(::MomentumState, ...)`
    accumulates it into the momentum, and with `g` shared that recursion was measurably wrong (see
    the CHANGELOG entry for issue A2). [`MomentumCache`](@ref) and [`AdamCache`](@ref) carry the same
    field for the same reason.
"""
struct GradientCache{T, MT, VT, ST} <: OptimizerCache{T}
    x::MT
    g::VT
    δ::VT
    Δg::VT
    g̃::VT
    g̃_is_current::Base.RefValue{Bool}
    section::ST
end

function GradientCache(x::OptimizerSolution{T}, g::AT, δ::AT, Δg::AT) where {
        T, AT <: GradientStorage{T}}
    sec = GlobalSection(_copy(x))
    g̃ = _similar(g)
    _fill!(g̃, T(NaN))
    GradientCache{T, typeof(x), typeof(g), typeof(sec)}(x, g, δ, Δg, g̃, Ref(false), sec)
end

function GradientCache(x::OptimizerSolution{T}, g::AT, δ::AT) where {
        T, AT <: GradientStorage{T}}
    Δg = _similar(g)
    _fill!(Δg, T(NaN))
    GradientCache(x, g, δ, Δg)
end

function GradientCache(x::OptimizerSolution{T}, g::GradientStorage{T}) where {T}
    δ = _similar(g)
    _fill!(δ, T(NaN))
    GradientCache(x, g, δ)
end

function GradientCache(x::OptimizerSolution{T}) where {T}
    g = _zero(x)
    _fill!(g, T(NaN))
    GradientCache(x, g)
end

solution(cache::GradientCache) = cache.x
# `gradient` and `gradient_array` are the same array here, as they are on `NewtonOptimizerCache`.
# Only `gradient` was missing, and `trial_slope`'s `AbstractVector` branch calls it, so the three
# first-order methods used to throw a `MethodError` on any line search that evaluates `φ'`.
gradient(cache::GradientCache) = cache.g
gradient_array(cache::GradientCache) = gradient(cache)
latest_gradient(cache::GradientCache) = cache.g̃
function refresh_latest_gradient!(cache::GradientCache, g::Gradient)
    _refresh_latest_gradient!(cache, g)
end
function latest_gradient_is_current(cache::GradientCache, state::OptimizerState, x::OptimizerSolution)
    _latest_gradient_is_current(cache, state, x)
end
invalidate_latest_gradient!(cache::GradientCache) = _invalidate_latest_gradient!(cache)
# `∇f(x_{k+1}) - ∇f(x_k)`, the successive difference `OptimizerStatus` prints as `|g(x) - g(x')|`,
# from the two gradients the cache holds rather than from a `state.ḡ` that is two iterates behind
# here. See `gradient_difference!`.
function gradient_difference!(cache::GradientCache, ::OptimizerState)
    _latest_gradient_difference!(cache)
end
direction(cache::GradientCache) = cache.δ
rhs(cache::GradientCache) = direction(cache)
section(cache::GradientCache) = cache.section

# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
# The invariant is enforced by the outer constructors below.
"""
    GradientState <: OptimizerState

State for the gradient optimizer.
"""
mutable struct GradientState{T, OT, GS, VT} <: OptimizerState{T}
    section::GS
    iterations::Int

    x::OT
    x̄::OT
    g::VT
    ḡ::VT
    f::T
    f̄::T
end

solution(state::GradientState) = state.x
previous_solution(state::GradientState) = state.x̄
gradient(state::GradientState) = state.g
previous_gradient(state::GradientState) = state.ḡ
value(state::GradientState) = state.f
previous_value(state::GradientState) = state.f̄

section(state::GradientState) = state.section

function GradientState(x::OST, g::GradientStorage{T}) where {T, OST <: OptimizerSolution{T}}
    _x = _copy(x)
    _g = _copy(g)
    gs = GlobalSection(_x)
    GradientState{T, typeof(_x), typeof(gs), typeof(_g)}(
        gs, 0, _x, _similar(_x), _g, _similar(_g), T(NaN), T(NaN))
end

GradientState(x::OptimizerSolution) = GradientState(x, _zero(x))

OptimizerState(::GradientMethod, x...) = GradientState(x...)

function update!(state::GradientState{T}, gradient_array::GradientStorage{T},
        direction::GradientStorage{T}, x::OptimizerSolution{T},
        f::Callable, retraction, observer = NoStepObserver()) where {T}
    _copyto!(previous_solution(state), solution(state))
    _copyto!(previous_gradient(state), gradient(state))
    state.f̄ = value(state)
    _copyto!(solution(state), x)
    _copyto!(gradient(state), gradient_array)
    state.f = observe_optimizer_phase(observer, :objective) do
        f(x)
    end

    observe_optimizer_phase(observer, :retraction_application) do
        update_section!(section(state), direction, retraction)
    end

    state
end

function update!(state::GradientState, opt::Optimizer, x::OptimizerSolution)
    update!(state, gradient_array(cache(opt)), direction(cache(opt)),
        x, problem(opt).F, opt.retraction, step_observer(opt))
end

# function compute_direction!(opt::Optimizer{T,OM}, ::GradientState) where {T,OM<:GradientMethod}
#     direction(opt) .= rhs(opt)
# end

function update!(cache::GradientCache{T}, state::GradientState{T},
        gradient::Gradient{T}, ::Hessian{T}, x::OptimizerSolution{T}) where {T}
    # first, and before the two `_copyto!`s below: it compares `solution(cache)` against `x` and
    # `section(cache)` against `section(state)`, which those overwrite
    store_gradient!(cache, state, gradient, x)
    _copyto!(section(cache), section(state))
    _copyto!(solution(cache), x)
    _copyto!(direction(cache), gradient_array(cache))
    _rmul!(direction(cache), -1)

    cache
end

# this should be moved to a different file
function update!(state::BFGSState{T}, opt::Optimizer{T}, x::OptimizerSolution{T}) where {T}
    observer = step_observer(opt)
    f = observe_optimizer_phase(observer, :objective) do
        problem(opt).F(x)
    end
    update!(
        state, direction(cache(opt)), gradient(opt), x, f, opt.retraction, observer)
end
