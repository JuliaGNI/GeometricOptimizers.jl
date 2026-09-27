# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
# The invariant is enforced by the inner constructor's signature, which is dispatch and therefore
# costs nothing.
"""
    BFGSState <: OptimizerState

The [`OptimizerState`](@ref) corresponding to the [`BFGS`](@ref) method.

# Keys
- `x`: the current iterate,
- `x̄`: the previous iterate,
- `s`: stores the previous direction. This needs to be stored in addition to the *previous solution* because of the manifold case.
- `ḡ`: the gradient the last step was built from,
- `f`: the objective at `x`,
- `f̄`: the objective at `x̄`,
- `Q`

The unbarred fields are the current iterate's and the barred ones the previous iterate's, as
[`update!`](@ref) maintains them, and the accessors are the [`OptimizerState`](@ref) defaults:
`value` reads `f` and is exported, `GeometricOptimizers.solution`,
`GeometricOptimizers.previous_solution` and `GeometricOptimizers.previous_value` are internal.
"""
mutable struct BFGSState{T, AT, GT, MT, GS} <: OptimizerState{T}
    x::AT
    x̄::AT
    s::GT
    ḡ::GT
    f::T
    f̄::T
    Q::MT
    iterations::Int

    section::GS

    function BFGSState(x̄::AT,
            ḡ::GT,
            f̄::T,
            Q::MT) where {T, AT <: OptimizerSolution{T},
            GT <: GradientStorage{T}, MT <: AbstractMatrix{T}}
        section = GlobalSection(x̄)
        state = new{T, AT, GT, MT, typeof(section)}(
            _copy(x̄), x̄, _similar(ḡ), ḡ, f̄, f̄, Q, 0, section)
        _poison!(state)
    end
end

function BFGSState(x̄::OptimizerSolution{T}, ḡ::GradientStorage{T}, f̄::T) where {T}
    BFGSState(_copy(x̄), _copy(ḡ), f̄, _alloc_q(x̄))
end
function BFGSState(x̄::OptimizerSolution{T}, ḡ::GradientStorage{T}) where {T}
    BFGSState(_copy(x̄), _copy(ḡ), zero(T))
end
BFGSState(x̄::OptimizerSolution) = BFGSState(_copy(x̄), _zero(x̄))

# `Q` is sized by the *intrinsic* dimension of the parameters, i.e. by the length of the flattening of
# the space the direction lives in. SimpleSolvers' fallback is `x * x'`, which for anything but a
# plain vector is the wrong shape: for a bare `StiefelManifold` of size `(3, 1)` it gives `3 × 3`
# where the horizontal lift has only 2 free parameters, and the cache and the state then disagree
# about how big `Q` is.
#
# `_alloc_q` sizes `Q` for a manifold and for a parameter set alike. It is not `SimpleSolvers`'
# `alloc_h(::NetworkParameters)`, which sizes by `mapparameters(zero, ·)`, and `zero` of a manifold
# point is the point's own shape and not its lift (issue #21). On a plain vector the parameters are
# the flat coordinates, and `SimpleSolvers`' method is right, and it keeps the vector's array type,
# so a device vector gets a device `Q`.
_alloc_q(x::AbstractVector) = alloc_h(x)

function _alloc_q(x::OptimizerSolution{T}) where {T}
    # `_zero(x)` for the reason `QuasiNewtonCache` gives: the lift's dimension, not the dense one
    n = flatlength(_zero(x))
    fill(T(NaN), n, n)
end

# `Manifold` is this package's type, so this method is this package's to own (issue #16).
alloc_h(x::Manifold) = _alloc_q(x)

OptimizerState(::BFGS, x_args...) = BFGSState(x_args...)

inverse_hessian(state::BFGSState) = state.Q

@doc raw"""
    restart!(state::BFGSState)

Reset the inverse-Hessian approximation to the identity, so that the next direction is the
steepest-descent one.

This is the `BFGSState` method of [`restart!`](@ref); `DFPState` is an alias for `BFGSState`, so it
covers `DFP` as well. [`solver_step!`](@ref) calls it when a line search reports that it could not
decrease the merit — see [`linesearch_rejected`](@ref).

!!! info "Why the direction alone is not enough"
    [`ensure_descent!`](@ref) already substitutes the steepest-descent direction for one step when
    ``Q`` has stopped being positive definite, but it leaves ``Q`` itself alone, so every subsequent
    direction comes from the same damaged approximation. Measured on the SVD problem of
    `test/optimizer_convergence/svd_optim.jl`, ``\lambda_\mathrm{min}(Q)`` reached `-398` and stayed
    negative for the rest of the solve. Discarding ``Q`` is what actually recovers.
"""
function restart!(state::BFGSState)
    inverse_hessian(state) .= one(inverse_hessian(state))
    state
end

# The one initialisation, which the constructor and `initialize_state!` share: the tangents and the
# objectives read `NaN` until a step writes them, and `Q` starts at the identity. The points are not
# filled, because a manifold point cannot hold `NaN` (issue #22). The first secant pair is then `NaN`
# and `curvature_is_usable` rejects it, so the first step is steepest descent.
function _poison!(state::BFGSState{T}) where {T}
    _fill!(state.s, T(NaN))
    _fill!(state.ḡ, T(NaN))
    state.f = T(NaN)
    state.f̄ = T(NaN)
    inverse_hessian(state) .= one(inverse_hessian(state))

    state
end

# `update!(state::BFGSState, ::Gradient, x, retraction)` was deleted in 0.6.0. It had no caller: for a
# `BFGSState` the live path is `update!(state, opt, x, f)` in `gradient_optimizer.jl`, which reaches
# the method below and hands it the objective value `solve!` evaluated at `x`. Nothing in `src/`,
# `test/`, `docs/` or `scripts/` called the deleted form, and neither does `GeometricMachineLearning`
# or `GMLDatasets`.
#
# It is named here because of what it did rather than because it is missed: it set `f̄` with
# `gradient.F(flatten(T, x)[1])`, and `gradient.F` is the closure `_x -> F(unflatten(layout, _x))` —
# so that line flattened `x` and unflattened it again to compute `F(x)`, allocating twice for a value
# the caller already had. The six-argument method takes `f` as an argument, which is the fix; a
# scratch buffer would only have made the round trip cheaper.

function _copyto!(sec::GlobalSection{T, AT, Nothing}, Y::AT) where {
        T, AT <: AbstractVector{T}}
    sec.Y .= Y
end

function update!(state::BFGSState{T}, direction::GradientStorage{T}, gradient::Gradient,
        x::XT, f::T, retraction,
        observer = NoStepObserver()) where {
        T, XT <: OptimizerSolution{T}}
    _copyto!(state.x̄, state.x)
    _copyto!(state.x, x)
    # `ḡ` is deliberately *not* refreshed here. This runs at the end of the iteration, at the same
    # iterate `x` that the next `Δg = ∇f(x) - ḡ` is formed at, so writing `∇f(x)` here made `Δg`
    # identically zero and the quasi-Newton `Q` update never fired on any iteration. The
    # quasi-Newton cache advances `ḡ` itself, right after it has used it.
    state.f̄ = state.f
    state.f = f

    _copyto!(state.s, direction)
    observe_optimizer_phase(observer, :retraction_application) do
        update_section!(section(state), state.s, retraction)
    end
    _copyto!(state.section, x)

    state
end
