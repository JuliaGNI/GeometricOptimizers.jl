"""
    OptimizerMethod <: SolverMethod

The `OptimizerMethod` is used in [`Optimizer`](@ref) and determines the algorithm that is used.
"""
abstract type OptimizerMethod <: SolverMethod end

"""
    QuasiNewtonOptimizerMethod <: OptimizerMethod

Includes [`_BFGS`](@ref) and [`_DFP`](@ref).
"""
abstract type QuasiNewtonOptimizerMethod <: OptimizerMethod end

@doc raw"""
    Newton

Newton's method: the direction solves ``\nabla^2f(x)\delta = -\nabla{}f(x)`` with the exact Hessian,
which [`SimpleSolvers.HessianAutodiff`](@extref) supplies.

Unlike [`_BFGS`](@ref) and [`_DFP`](@ref) this needs no approximation to build up, so it converges in
few iterations, but it also inherits the Hessian's indefiniteness: where ``\nabla^2f`` is not positive
definite the direction ascends, and [`ensure_descent!`](@ref) substitutes the steepest-descent
direction for it.
"""
struct Newton <: OptimizerMethod end

Hessian(::Newton, ForOBJ::Union{Callable,OptimizerProblem}, x::AbstractVector) = HessianAutodiff(ForOBJ, x)
HessianAutodiff(F::OptimizerProblem, x) = HessianAutodiff(F.F, x)

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct _DFP <: QuasiNewtonOptimizerMethod end

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct _BFGS <: QuasiNewtonOptimizerMethod end

"""
The gradient descent algorithm.
"""
struct GradientMethod <: OptimizerMethod end

@doc raw"""
    MomentumMethod(α)

The gradient descent algorithm with momentum, i.e. the *heavy ball* method.

Stores the *momentum coefficient* `α`. The momentum is accumulated as
```math
    p \gets \alpha{}p + \nabla{}L
```
and the direction is ``-p``.

!!! info "The step size is not stored here"
    Like every [`OptimizerMethod`](@ref), `MomentumMethod` only produces a direction; how far
    the optimizer goes along it is the line search's business. A fixed learning rate ``\eta``
    is therefore expressed as `linesearch = Static(η)`, which is also the default (see
    [`default_linesearch`](@ref)).
"""
struct MomentumMethod{T} <: OptimizerMethod
    α::T

    MomentumMethod(α::T=DEFAULT_MOMENTUM_α) where {T} = new{T}(α)
end

@doc raw"""
    Adam(T; β₁, β₂, δ)

The Adam optimizer, with the defaults suggested in [goodfellow2016deep; page 301](@cite).

The cache consists of a first and a second moment, stored in *bias-corrected* form, i.e.
updated in the ``t``-th iteration (counted from ``t = 1``) as
```math
m_1 \gets \frac{\beta_1 - \beta_1^t}{1 - \beta_1^t}m_1 + \frac{1 - \beta_1}{1 - \beta_1^t}\nabla{}L,
```
```math
m_2 \gets \frac{\beta_2 - \beta_2^t}{1 - \beta_2^t}m_2 + \frac{1 - \beta_2}{1 - \beta_2^t}\nabla{}L\odot\nabla{}L,
```
from which the direction is computed as ``-m_1/(\sqrt{m_2} + \delta)``.

`T` is the element type of the parameters that are to be optimized; unlike
[`MomentumMethod`](@ref), `Adam` is *not* converted by [`Optimizer`](@ref), so
`Adam(Float32)` is needed for `Float32` parameters.

!!! info "The learning rate is not stored here"
    `Adam` only produces a direction, of magnitude ``\approx{}1`` per component; the learning
    rate ``\eta`` is the line search's `α`, i.e. it is passed as `linesearch = Static(η)`,
    which is also the default (see [`default_linesearch`](@ref)). `Adam` used to carry an `η`
    field that was never applied to the direction, so `Adam(1e-3)` and `Adam(1e2)` gave
    identical results; it is gone, and because it used to be the *first positional* argument,
    `β₁`, `β₂` and `δ` are keyword arguments now so that an old `Adam(1e-3)` call fails
    instead of silently setting `β₁ = 1e-3`.
"""
struct Adam{T} <: OptimizerMethod
    β₁::T
    β₂::T
    δ::T

    # the defaults are written as `Float64` literals so that `T(9.0e-1)` is `0.9` and not
    # `Float64(9.0f-1) = 0.8999999761581421`; for `T = Float32` they are the same numbers
    function Adam(::Type{T}=Float64; β₁=9.0e-1, β₂=9.9e-1, δ=1.0e-8) where {T}
        new{T}(T(β₁), T(β₂), T(δ))
    end
end

const DEFAULT_MOMENTUM_α = 0.01

const DEFAULT_LEARNING_RATE = 1.0e-3

"""
    default_linesearch(T, method)

Return the line search that [`Optimizer`](@ref) uses for `method` if none is supplied.

[`GradientMethod`](@ref), [`MomentumMethod`](@ref) and [`Adam`](@ref) determine a direction
but no step size, so for them the `α` of a [`SimpleSolvers.Static`](@extref) line search *is*
the learning rate and the default is `Static(DEFAULT_LEARNING_RATE)`. The methods that build a
(quasi-)Newton direction come with a scale of their own and default to
[`SimpleSolvers.Backtracking`](@extref).
"""
default_linesearch(::Type{T}, ::OptimizerMethod) where {T} = Backtracking(T)
default_linesearch(::Type{T}, ::Union{GradientMethod,MomentumMethod,Adam}) where {T} = Static(T(DEFAULT_LEARNING_RATE))

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::_DFP) = print(io, "DFP")
Base.show(io::IO, alg::_BFGS) = print(io, "BFGS")
