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

@doc raw"""
    AdamWithEuclideanDecay(T; β₁, β₂, δ, λ)

[`Adam`](@ref) with the *decoupled* weight decay of [loshchilov2019decoupled](@cite), which is
the **Euclidean** ``\lambda{}x`` and therefore acts on the unconstrained weights only.

The moments are the ones of [`Adam`](@ref) — this method shares its cache and its state — and
the decay is applied to the *direction* instead of to the gradient, i.e. the direction is
```math
-\left(\frac{m_1}{\sqrt{m_2} + \delta} + \lambda{}x\right)
```
so that a step with learning rate ``\eta`` (see below) is
```math
x \gets x - \eta\frac{m_1}{\sqrt{m_2} + \delta} - \eta\lambda{}x.
```
That is what *decoupled* means: ``\lambda{}x`` never enters ``m_1`` or ``m_2``, so the amount
by which a weight is shrunk does not depend on the size of its gradient. Adding ``\lambda{}x``
to the gradient instead — ``L^2`` regularization, which is what plain [`Adam`](@ref) on a
penalized objective does — is *not* the same thing, because the second moment then rescales the
penalty away.

# Why the name says *Euclidean*

``\lambda{}x`` is the gradient of ``\frac{\lambda}{2}||x||^2``, and on the manifolds of this
package that function is constant: ``||Y||_F^2 = \mathrm{tr}(Y^TY) = n`` for every
``Y\in{}St(N, n)``, and likewise on the [`GrassmannManifold`](@ref). Its Riemannian gradient
is therefore zero,
```math
\mathtt{rgrad}(Y, \lambda{}Y) = \lambda{}Y - Y(\lambda{}Y)^TY = \lambda{}Y - \lambda{}Y = \mathbb{O},
```
so this decay does nothing at all to a manifold weight. The method decays the ordinary arrays
of a `NamedTuple` of parameters and leaves the [`Manifold`](@ref)s in it alone — which is the
case it exists for, a network that keeps Stiefel weights next to unconstrained ones — and on a
*bare* [`Manifold`](@ref) it *is* [`Adam`](@ref), for every ``\lambda``. Passing a nonzero `λ`
together with parameters that are entirely manifolds is therefore warned about rather than
silently ignored.

Whether a *Riemannian* weight decay should exist alongside this one, and what it would decay
towards, is [issue #28](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28); the name
`AdamW` is held in reserve for its answer, which is why this method is not called that.

# Arguments

As for [`Adam`](@ref), `T` is the element type of the parameters
(`AdamWithEuclideanDecay(Float32)` for `Float32` parameters) and the learning rate ``\eta`` is
*not* stored here: it is the `α` of the line search, i.e. `linesearch = Static(η)`, which is
also the default (see [`default_linesearch`](@ref)). `λ` is hence multiplied by ``\eta``,
exactly as in [loshchilov2019decoupled](@cite), where the two are decoupled from each other but
the decay is still scaled by the schedule.
"""
struct AdamWithEuclideanDecay{T} <: OptimizerMethod
    β₁::T
    β₂::T
    δ::T
    λ::T

    # see the remark on the `Float64` literals in `Adam`
    function AdamWithEuclideanDecay(::Type{T}=Float64; β₁=9.0e-1, β₂=9.9e-1, δ=1.0e-8, λ=DEFAULT_WEIGHT_DECAY) where {T}
        new{T}(T(β₁), T(β₂), T(δ), T(λ))
    end
end

"""
    AdamW(args...)

Deliberately undefined; use [`AdamWithEuclideanDecay`](@ref).

The decoupled weight decay of AdamW is the Euclidean ``λx``, whose Riemannian gradient vanishes
identically on every manifold of this package, so a method called `AdamW` here would be `Adam`
under a second name for anyone optimizing on a manifold — and would say so nowhere. Rather than
let that be discovered at run time, the name errors and points at the one that describes what it
does. It is kept free for a *Riemannian* weight decay, should
[issue #28](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28) conclude that one is
called for.
"""
function AdamW(args...; kwargs...)
    error("`AdamW` is deliberately not defined: its weight decay is the Euclidean `λx`, which " *
          "is identically zero on a `Manifold` weight (`rgrad(Y, λY) = 𝕆`), so a method by that " *
          "name would silently be `Adam` on a manifold. Use `AdamWithEuclideanDecay`, which " *
          "decays the unconstrained weights and says so, or `Adam` for no decay at all. See " *
          "https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28.")
end

"""
The methods that build their direction from parameters of their own (and have no Hessian),
as opposed to the (quasi-)Newton methods, which need the Hessian and have no parameters.
"""
const FirstOrderMethodWithState = Union{MomentumMethod,Adam,AdamWithEuclideanDecay}

const DEFAULT_MOMENTUM_α = 0.01

const DEFAULT_LEARNING_RATE = 1.0e-3

# the default of [loshchilov2019decoupled](@cite) and of `torch.optim.AdamW`
const DEFAULT_WEIGHT_DECAY = 1.0e-2

"""
    default_linesearch(T, method)

Return the line search that [`Optimizer`](@ref) uses for `method` if none is supplied.

[`GradientMethod`](@ref), [`MomentumMethod`](@ref), [`Adam`](@ref) and
[`AdamWithEuclideanDecay`](@ref)
determine a direction but no step size, so for them the `α` of a
[`SimpleSolvers.Static`](@extref) line search *is* the learning rate and the default is
`Static(DEFAULT_LEARNING_RATE)`. The methods that build a (quasi-)Newton direction come with a
scale of their own and default to [`SimpleSolvers.Backtracking`](@extref).
"""
default_linesearch(::Type{T}, ::OptimizerMethod) where {T} = Backtracking(T)
default_linesearch(::Type{T}, ::Union{GradientMethod,FirstOrderMethodWithState}) where {T} = Static(T(DEFAULT_LEARNING_RATE))

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::_DFP) = print(io, "DFP")
Base.show(io::IO, alg::_BFGS) = print(io, "BFGS")
