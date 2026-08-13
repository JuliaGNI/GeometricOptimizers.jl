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

@doc raw"""
    DEFAULT_DFP_c₂

The curvature constant [`default_linesearch`](@ref) gives [`_DFP`](@ref)'s
[`SimpleSolvers.StrongWolfe`](@extref) search.

`StrongWolfe`'s own default is `0.9`, the value [nocedal2006numerical](@cite) recommends for Newton and
quasi-Newton methods. That is too loose for `_DFP`: the strong Wolfe conditions are then already
satisfied at ``\alpha = 1`` on 99.4% of iterations, so the bracketing phase never grows the step and the
solve crawls exactly as it does under [`SimpleSolvers.Backtracking`](@extref). `0.1` — the value the
same reference recommends where a more accurate line search is needed — makes the expansion fire on
94.5% of iterations, with a median ``\alpha`` of 8.
"""
const DEFAULT_DFP_c₂ = 0.1

@doc raw"""
    default_linesearch(T, method)

Return the line search that [`Optimizer`](@ref) uses for `method` if none is supplied.

Everything except [`Adam`](@ref) defaults to [`SimpleSolvers.Backtracking`](@extref): the
(quasi-)Newton methods because they build a direction with a scale of its own, and
[`GradientMethod`](@ref) and [`MomentumMethod`](@ref) because a searching line search is what makes
them *converge* rather than merely descend. Both produce genuine descent directions, so a
backtracking search always has an `α` to find.

`Adam` is the exception and keeps a fixed `Static(DEFAULT_LEARNING_RATE)`. Its direction is
``-m_1/(\sqrt{m_2} + \delta)``, a moving average that is deliberately *not* required to descend on any
individual step, so a sufficient-decrease search has nothing to work with and would spend every such
step reporting that it found no descent direction. [`AdamWithEuclideanDecay`](@ref) shares the
exception for the same reason: it adds ``-\lambda{}x`` to that direction, which does not make it a
descent direction either.

!!! info "This changed in 0.2.0"
    `GradientMethod` and `MomentumMethod` used to default to `Static(DEFAULT_LEARNING_RATE)` as well.
    They could not do anything else: until the line search learned to take its trial step through the
    retraction (see [`trial_iterate!`](@ref)), `Static` was the only line search that worked on
    manifold parameters at all. Pass `linesearch = Static(η)` to get the old fixed learning rate back.

!!! tip "Why `Backtracking`, when the searching methods need far fewer iterations"
    Because iterations are the wrong unit. `Backtracking` returns the *first* `α` that decreases `f`
    enough, while `Bisection`, [`SimpleSolvers.Quadratic`](@extref) and
    [`SimpleSolvers.BierlaireQuadratic`](@extref) bracket and then refine a line *minimum*, which costs
    an order of magnitude more merit evaluations per iteration. Counting objective evaluations rather
    than iterations, on the SVD problem of `test/optimizer_convergence/svd_optim.jl` (`Geodesic`;
    `Static` needs ≈4 evaluations per iteration, so subtract that for the search's own cost):

    | search | evals/iteration | `_BFGS`: iters / evals | `_DFP`: iters / evals |
    |---|---|---|---|
    | `Backtracking` | **25** | 113 / **2 857** | 3 000+ / 75 012 (no convergence) |
    | `StrongWolfe` (`c₂ = 0.9`) | 36 | 159 / 5 708 | 3 000+ / 105 054 (no convergence) |
    | `StrongWolfe` (`c₂ = 0.1`) | 82 | 118 / 6 738 | 201 / **16 466** |
    | `BierlaireQuadratic` | 102 | 170 / 17 340 | 322 / 27 484 |
    | `Quadratic` | 129 | 173 / 22 267 | 189 / 18 313 |
    | `Bisection` | 583 | 143 / 83 353 | 134 / 78 698 |

    So for `_BFGS` the extra iterations are a bargain: `Backtracking` does the job in **6× less work**
    than the cheapest searching method, and its lower final accuracy (2.6e-11 against 9.3e-16) is far
    past the convergence gate anyway. The same holds on the sphere problem (44 evaluations against 81
    to 127) and for the first-order methods, where `Bisection` burns 1.8M evaluations to `Backtracking`'s
    79 500 for the same 3 000 iterations. Reach for a searching method when iteration count is what you
    are paying for — a very expensive objective, or an outer loop that is bounded in iterations.

!!! warning "[`_DFP`](@ref) is the exception, and defaults to `StrongWolfe(c₂ = 0.1)`"
    `Backtracking` starts its trial step at `α = 1` and only ever *shrinks*. That is fine for a method
    whose direction is already scaled like a Newton step — `_BFGS` accepts `α = 1` on 74% of its
    iterations and converges in 113 — but `_DFP` produces a systematically *under-scaled* direction,
    and a backtracking search has no mechanism to lengthen it. On the SVD problem it then accepts
    `α = 1` on **100%** of its iterations and crawls to the gradient gate in **49 679** of them.
    Raising only `Backtracking`'s initial trial step to 3 gives 229, which is what identifies the
    ceiling rather than the method as the cause.

    What `_DFP` needs is a search that can *grow* the step, and the cheapest of those is
    `StrongWolfe` — but only with a curvature constant tight enough to reject `α = 1`. At its own
    default `c₂ = 0.9` the strong Wolfe conditions are already satisfied at `α = 1` on 99.4% of
    iterations, so the bracketing phase never fires and it crawls just like `Backtracking`. At
    `c₂ = ` [`DEFAULT_DFP_c₂`](@ref) the expansion fires on 94.5% of iterations with a median `α` of 8,
    and it is then the cheapest converging option on *both* retractions:

    | search | `_DFP` on `Geodesic` | `_DFP` on `Cayley` |
    |---|---|---|
    | `StrongWolfe` (`c₂ = 0.1`) | 201 iters / **16 466** evals | 274 / **23 312** |
    | `Quadratic` | 189 / 18 313 | 555 / 54 230 |
    | `BierlaireQuadratic` | 322 / 27 484 | 990 / 80 787 |
    | `Bisection` | 134 / 78 698 | 96 / 55 493 |

    `Bisection` needs the fewest *iterations* but four to five times the work. `Quadratic` is close on
    `Geodesic` and falls apart on `Cayley`, which is the default retraction — probably because
    [`trial_slope`](@ref) is only first-order correct there and `Quadratic` uses ``\varphi'``
    *quantitatively* in its polynomial fit, where `Bisection` uses only its sign and `StrongWolfe` only
    compares it against ``\varphi'(0)``.

    None of this is a property of DFP: given a search that can exceed 1 it is competitive with `_BFGS`.
    See JuliaGNI/SimpleSolvers.jl#174 for the upstream half of the story.

!!! note "SimpleSolvers 0.11 gives `Backtracking` an expansion phase"
    `Backtracking(T; expand = true)` on SimpleSolvers `main` (0.11, unreleased at the time of writing,
    so the `SimpleSolvers = "0.10"` compat bound here still rules it out) lengthens the step when the
    *first* trial is accepted — the trigger identified in #174 — growing it by at most a factor `q = 10`
    per round for at most `nexpand = 3` rounds while the step both satisfies sufficient decrease and
    strictly improves the merit.

    Measured on the SVD problem, it is close to free and helps both quasi-Newton methods:

    | | `expand = false` | `expand = true` |
    |---|---|---|
    | `_BFGS`, `Geodesic` | 113 iters / 2 857 evals | **93 / 2 374** |
    | `_BFGS`, `Cayley` | 136 / 3 431 | **118 / 3 006** |
    | `_DFP`, `Geodesic` | no convergence / 75 012 | **830 / 21 540** |
    | `_DFP`, `Cayley` | no convergence / 75 011 | **1 237 / 31 995** |

    The cost is under 4% per iteration (25.0 to 26.0 evaluations), and on a well-scaled problem it is
    *exactly* nothing: on the sphere the evaluation counts are identical with and without it, because
    the extrapolation model declines to propose a longer step before any merit is evaluated. `_BFGS`
    then takes `α > 1` on 20% of its iterations and `_DFP` on 59% (median 2.55, hitting the
    ``q^{\mathrm{nexpand}} = 1000`` ceiling at the top).

    So `expand = true` is the right default for every `Backtracking` above once the compat bound moves,
    and it removes `_DFP`'s pathology outright — but not its exception: at 21 540 and 31 995 evaluations
    it is still 1.3 to 1.4 times the work of `StrongWolfe(c₂ = 0.1)`, so `_DFP` keeps that. The suite
    passes against 0.11.0 unchanged.
"""
default_linesearch(::Type{T}, ::OptimizerMethod) where {T} = Backtracking(T)
default_linesearch(::Type{T}, ::Union{Adam,AdamWithEuclideanDecay}) where {T} = Static(T(DEFAULT_LEARNING_RATE))
default_linesearch(::Type{T}, ::_DFP) where {T} = StrongWolfe(T; c₂=T(DEFAULT_DFP_c₂))

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::_DFP) = print(io, "DFP")
Base.show(io::IO, alg::_BFGS) = print(io, "BFGS")
