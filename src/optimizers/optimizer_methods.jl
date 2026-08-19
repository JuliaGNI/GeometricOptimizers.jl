"""
    OptimizerMethod <: SolverMethod

The `OptimizerMethod` is used in [`Optimizer`](@ref) and determines the algorithm that is used.
"""
abstract type OptimizerMethod <: SolverMethod end

"""
    QuasiNewtonOptimizerMethod <: OptimizerMethod

Includes [`BFGS`](@ref) and [`DFP`](@ref).
"""
abstract type QuasiNewtonOptimizerMethod <: OptimizerMethod end

# The defaults of the methods below, collected here so that each one is defined before the
# constructor that uses it as a keyword default. They are written as `Float64` literals so that
# `T(1.0e-3)` is `1.0f-3` for `T = Float32` rather than `Float64(1.0e-3)` rounded twice.
const DEFAULT_MOMENTUM_α = 0.01

const DEFAULT_LEARNING_RATE = 1.0e-3

# the default of [loshchilov2019decoupled](@cite) and of `torch.optim.AdamW`
const DEFAULT_WEIGHT_DECAY = 1.0e-2

@doc raw"""
    Newton

Newton's method: the direction solves ``\nabla^2f(x)\delta = -\nabla{}f(x)`` with the exact Hessian,
which [`SimpleSolvers.HessianAutodiff`](@extref) supplies.

Unlike [`BFGS`](@ref) and [`DFP`](@ref) this needs no approximation to build up, so it converges in
few iterations, but it also inherits the Hessian's indefiniteness: where ``\nabla^2f`` is not positive
definite the direction ascends, and [`ensure_descent!`](@ref) substitutes the steepest-descent
direction for it.
"""
struct Newton <: OptimizerMethod end

Hessian(::Newton, ForOBJ::Union{Callable,OptimizerProblem}, x::AbstractVector) = HessianAutodiff(ForOBJ, x)
HessianAutodiff(F::OptimizerProblem, x) = HessianAutodiff(F.F, x)

@doc raw"""
    DFP

The Davidon–Fletcher–Powell method, from [nocedal2006numerical](@cite): the direction is
``-Q\nabla{}f(x)`` for an approximate *inverse* Hessian ``Q``, updated from the secant pair
``\delta = x^{(k)} - x^{(k-1)}``, ``\gamma = \nabla{}f^{(k)} - \nabla{}f^{(k-1)}`` by
```math
Q \gets Q - \frac{Q\gamma\gamma^TQ}{\gamma^TQ\gamma} + \frac{\delta\delta^T}{\delta^T\gamma}
```
(equation 6.15 there). Its state is [`BFGSState`](@ref), under the alias [`DFPState`](@ref), and the
update itself lives in [`DFPCache`](@ref), which symmetrizes ``Q\gamma\gamma^TQ`` explicitly and
skips the update when the secant pair fails [`curvature_is_usable`](@ref).

Where [`BFGS`](@ref) is the better-conditioned choice and the default, `DFP` is here because it is
the other classical member of the family and because the two differ in exactly the way that shows
what the line search is doing: DFP's direction is systematically *under*-scaled, so it wants a
median ``\alpha`` well above 1 and is starved by a search that cannot exceed it. See
[`default_linesearch`](@ref), which measures this, converges `DFP` under the default expanding
`Backtracking`, and gives `StrongWolfe(T; c₂ = 0.1)` as the steadier explicit choice for a
DFP-heavy workload.

Takes no parameters: like every [`OptimizerMethod`](@ref) it only produces a direction, and how far
the optimizer travels along it is the line search's business.
"""
struct DFP <: QuasiNewtonOptimizerMethod end

@doc raw"""
    BFGS

The Broyden–Fletcher–Goldfarb–Shanno method, from [nocedal2006numerical](@cite), and the default
`algorithm` of [`Optimizer`](@ref). The direction is ``-Q\nabla{}f(x)`` for an approximate *inverse*
Hessian ``Q``, updated from the secant pair ``\delta = x^{(k)} - x^{(k-1)}``,
``\gamma = \nabla{}f^{(k)} - \nabla{}f^{(k-1)}`` by
```math
Q \gets Q - \frac{\delta\gamma^TQ + Q\gamma\delta^T
    - \left(1 + \frac{\gamma^TQ\gamma}{\delta^T\gamma}\right)\delta\delta^T}{\delta^T\gamma}
```
Its state is [`BFGSState`](@ref) and the update itself lives in [`BFGSCache`](@ref), which skips the
update when the secant pair fails [`curvature_is_usable`](@ref) — the condition that keeps ``Q``
positive definite. [`restart!(::BFGSState)`](@ref) discards ``Q`` when a line search reports that it
could not decrease the merit.

Unlike [`Newton`](@ref) this builds its curvature up over the iterations rather than evaluating a
Hessian, so it takes more of them; unlike [`DFP`](@ref) it produces a direction already scaled like
a Newton step, and accepts ``\alpha = 1`` on most of them. Both run on an `AbstractVector`, a
`NamedTuple` of parameters and a bare `Manifold` alike: ``Q`` is sized by the *intrinsic* dimension
and the secant pair is taken in the horizontal lift, so nothing in the update needs a vector-valued
point.

Takes no parameters: like every [`OptimizerMethod`](@ref) it only produces a direction, and how far
the optimizer travels along it is the line search's business.
"""
struct BFGS <: QuasiNewtonOptimizerMethod end

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

!!! info "A searching line search is not the way to make this converge"
    Because the direction has magnitude ``\approx{}1`` per component whatever the gradient is,
    a step that does not shrink leaves `Adam` circling the minimiser at that distance rather
    than settling on it. A *searching* line search does not fix that — it picks each step from
    the merit and has no reason to drive the sequence to zero — it only makes each step a
    better one: on the two-sphere problem of `test/manifold_linesearch_tests.jl` the searching
    options need 251–331 iterations where [`GradientMethod`](@ref) and [`MomentumMethod`](@ref)
    need 9–64. [`DecayingStatic`](@ref) is the setting under which `Adam` terminates on a
    criterion by construction, because its schedule drives the step to zero itself.

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
    NonGeometricAdam(T; β₁, β₂, δ)

*Cayley ADAM*, [li2020efficient; Algorithm 2](@cite), as an experimental Stiefel-only baseline for
[`Adam`](@ref).

The moments are stored in *bias-corrected* form, as [`Adam`](@ref)'s are, but the second one is a
**scalar**:
```math
m_1 \gets \frac{\beta_1 - \beta_1^t}{1 - \beta_1^t}m_1 + \frac{1 - \beta_1}{1 - \beta_1^t}\bar{G},
```
```math
m_2 \gets \frac{\beta_2 - \beta_2^t}{1 - \beta_2^t}m_2 + \frac{1 - \beta_2}{1 - \beta_2^t}\lVert\bar{G}\rVert^2,
```
where ``\bar{G}\in\mathfrak{g}^\mathrm{hor}`` is the gradient in the [global tangent space
representation](@ref "Global Tangent Spaces"), and the direction is
``-m_1/\sqrt{m_2 + \delta}``.

That one scalar is the entire difference from [`Adam`](@ref), and it is what *non-geometric* names
here: ``\lVert\bar{G}\rVert^2`` is a squared gradient *norm* where Adam's ``m_2`` is a squared
*gradient*, so the second moment carries no direction and the method assigns one adaptive learning
rate to the whole matrix instead of one per coordinate. The name describes that choice — the
source's own name for the algorithm is *Cayley ADAM*, and the source does not call it
non-geometric. What it *is* is a reproduction of a published algorithm, so it is a baseline and not
a lesser optimizer: on some objectives it will beat [`Adam`](@ref).

Only a single `StiefelManifold{T}` solution is supported; ordinary arrays, `NamedTuple`s, Grassmann
solutions and mixed parameter trees throw an `ArgumentError`. As for [`Adam`](@ref), `T` is the
element type of the parameters and is not converted by [`Optimizer`](@ref).

!!! info "The Cayley transform is the retraction here, and it is exact"
    The source's lines 12–14 approximate the Cayley transform with two fixed-point iterations, and
    its line 11 caps the step length to keep that iteration contractive. Neither is ported: this
    package retracts exactly — [`Cayley`](@ref), the default, evaluates the transform through the
    Sherman-Morrison-Woodbury formula — and it admits any other [`AbstractRetraction`](@ref),
    [`Geodesic`](@ref) included. The step is bounded, but by [`step_αmax`](@ref) and for a different
    reason; `step_ceiling = 1/2π` recovers the source's bound up to its use of the induced 1-norm.

!!! info "The learning rate is not stored here"
    As for [`Adam`](@ref): the direction has magnitude ``\approx{}1``, and the source's learning
    rate ``l`` is the line search's `α`, i.e. `linesearch = Static(η)`.

`ADAM_MATHS.md` maps every symbol of the source's Algorithm 2 onto the code, and records the three
places where this reproduction deliberately departs from it.
"""
struct NonGeometricAdam{T} <: OptimizerMethod
    β₁::T
    β₂::T
    δ::T

    function NonGeometricAdam(::Type{T}=Float64; β₁=9.0e-1, β₂=9.9e-1, δ=1.0e-8) where {T<:AbstractFloat}
        β₁T, β₂T, δT = T(β₁), T(β₂), T(δ)
        0 ≤ β₁T < 1 || throw(ArgumentError("β₁ must satisfy 0 ≤ β₁ < 1"))
        0 ≤ β₂T < 1 || throw(ArgumentError("β₂ must satisfy 0 ≤ β₂ < 1"))
        δT ≥ 0 || throw(ArgumentError("δ must be nonnegative"))
        new{T}(β₁T, β₂T, δT)
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

``\lambda{}x`` is the gradient of ``\frac{\lambda}{2}||x||^2``, which is *constant* on the
[`StiefelManifold`](@ref) and the [`GrassmannManifold`](@ref) — both are compact, with
``||Y||_F^2 = \mathrm{tr}(Y^TY) = n`` — so its Riemannian gradient
``\mathtt{rgrad}(Y, \lambda{}Y)`` vanishes identically and the decay does nothing at all to a
manifold weight. The method therefore decays the ordinary arrays of a `NamedTuple` of parameters
and leaves the [`Manifold`](@ref)s in it alone — which is the case it exists for, a network that
keeps Stiefel weights next to unconstrained ones — and on a *bare* [`Manifold`](@ref) it *is*
[`Adam`](@ref), for every ``\lambda``. Passing a nonzero `λ` together with parameters that are
entirely manifolds is therefore warned about rather than silently ignored.

The derivation, the Grassmann case and why the name `AdamW` is held in reserve for a
*Riemannian* decay instead (see
[issue #28](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28)) are in the
[Weight Decay on Manifolds](@ref) page.

# Arguments

As for [`Adam`](@ref), `T` is the element type of the parameters
(`AdamWithEuclideanDecay(Float32)` for `Float32` parameters) and the learning rate ``\eta`` is
*not* stored here: it is the `α` of the line search, i.e. `linesearch = Static(η)`, which is
also the default (see [`default_linesearch`](@ref)). `λ` is hence multiplied by ``\eta``,
exactly as in [loshchilov2019decoupled](@cite), where the two are decoupled from each other but
the decay is still scaled by the schedule.

!!! warning "Do not pass a searching line search"
    For [`Adam`](@ref) a sufficient-decrease search is merely wasted work. Here it is worse than
    that, because the merit the line search minimizes is the *bare* objective ``f`` and not
    ``f + \frac{\lambda}{2}||x||^2``: the penalty this method exists to apply is not part of the
    function whose decrease the search insists on, so the search picks its ``\alpha`` in order to
    undo the decay's contribution as far as it can. On top of that ``\alpha`` then varies from
    step to step, and since the decay per step is ``\alpha\lambda``, the regularization strength
    would be whatever the search happened to choose rather than what was asked for.

    A fixed `Static(η)` — the default — is the setting in which ``\lambda`` means what
    [loshchilov2019decoupled](@cite) says it means. [`DecayingStatic`](@ref) is also well defined
    and is the analogue of AdamW's schedule: it shrinks the decay along with the step, so
    ``\lambda`` keeps its interpretation relative to ``\eta`` while both go to zero.
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
[`Adam`](@ref) and the variants of it that differ only in how the direction is finished off, i.e.
that share [`AdamCache`](@ref), [`AdamState`](@ref) and the moment recursion. Currently
[`AdamWithEuclideanDecay`](@ref).
"""
const AdamFamily = Union{Adam,AdamWithEuclideanDecay,NonGeometricAdam}

"""
The methods whose `update!` needs the *method* rather than a
[`SimpleSolvers.Hessian`](@extref), because they carry state of their own out of which the
direction is built — a momentum or a pair of moments. The (quasi-)Newton methods are the
complement: their direction comes from the Hessian and the method object holds nothing.

[`GradientMethod`](@ref) is in neither group: it has no Hessian *and* no state, and takes the
Hessian branch because `NoHessian` is all that branch needs from it.

`solver_step!` also uses this to decide whether [`ensure_descent!`](@ref) applies. It must not:
a momentum and a moment average are deliberately allowed not to descend on an individual step,
and the decay of [`AdamWithEuclideanDecay`](@ref) tilts the direction further away from the
gradient still.

That exemption is about the direction *before* the search, and it used to be extended to the
restart *after* it as well. It no longer is — see [`linesearch_rejected`](@ref) and issue A7.
Declining to overrule the direction is not the same as taking the longest step along it, which
is what a rejected search hands back.
"""
const FirstOrderMethodWithState = Union{MomentumMethod,AdamFamily}

@doc raw"""
    default_linesearch(T, method)

Return the line search that [`Optimizer`](@ref) uses for `method` if none is supplied.

Everything except [`Adam`](@ref) defaults to
[`SimpleSolvers.Backtracking`](@extref)`(T; expand = true)`: the (quasi-)Newton methods because they
build a direction with a scale of its own, and [`GradientMethod`](@ref) and [`MomentumMethod`](@ref)
because a searching line search is what makes them *converge* rather than merely descend. All produce
genuine descent directions, so a backtracking search always has an `α` to find.

`expand = true` is what lets that search *lengthen* a step as well as shorten it, and it is not the
SimpleSolvers default — see the tip below for why it is the default here.

The [`AdamFamily`](@ref) methods are the exception and keep a fixed `Static(DEFAULT_LEARNING_RATE)`.
[`Adam`](@ref)'s direction is ``-m_1/(\sqrt{m_2} + \delta)``, a moving average that is deliberately
*not* required to descend on any individual step, so a sufficient-decrease search has nothing to work
with and would spend every such step reporting that it found no descent direction.

For [`AdamWithEuclideanDecay`](@ref) a fixed step is not merely the cheaper choice but the only one
under which ``\lambda`` means what it is documented to mean. Two reasons, and the first is the one
that matters:

 1. The merit the line search minimizes is the *bare* objective ``f``. It is not
    ``f + \frac{\lambda}{2}||x||^2``, because this package regularizes by adding ``-\lambda{}x`` to
    the direction rather than by penalizing the objective — that is what *decoupled* means. So the
    penalty is invisible to the search, which will spend its ``\alpha`` undoing the decay's
    contribution to ``f`` as far as it can. The search is not approximating a line minimum of the
    function the method is actually descending.
 2. The decay per step is ``\alpha\lambda``, so a varying ``\alpha`` makes the effective
    regularization strength whatever the search happened to pick on that step.

[`DecayingStatic`](@ref) is the exception to the exception: it varies ``\alpha`` too, but on a
*schedule* rather than in response to the merit, which is exactly AdamW's own learning-rate
schedule and leaves ``\lambda`` its meaning relative to ``\eta``.

!!! info "This changed in 0.2.0"
    `GradientMethod` and `MomentumMethod` used to default to `Static(DEFAULT_LEARNING_RATE)` as well.
    They could not do anything else: until the line search learned to take its trial step through the
    retraction (see [`trial_iterate!`](@ref)), `Static` was the only line search that worked on
    manifold parameters at all. Pass `linesearch = Static(η)` to get the old fixed learning rate back.

!!! tip "Why a backtracking search, and why `expand`"
    `Backtracking` returns the *first* `α` that decreases `f` enough, while `Bisection`,
    [`SimpleSolvers.Quadratic`](@extref) and [`SimpleSolvers.BierlaireQuadratic`](@extref) bracket and
    then refine a line *minimum*, which costs an order of magnitude more merit evaluations per
    iteration. Iterations are therefore the wrong unit to compare them in. Counting objective
    evaluations instead, on the SVD problem of `test/optimizer_convergence/svd_optim.jl` (`Geodesic`;
    `Static` needs ≈4 evaluations per iteration, so subtract that for the search's own cost):

    | search | evals/iteration | `BFGS`: iters / evals | `DFP`: iters / evals |
    |---|---|---|---|
    | `Backtracking(expand = true)` | **26** | **95** / **2 441** | 768 / 20 001 |
    | `Backtracking` (shrink only) | 25 | 136 / 3 457 | 48 322 / 1 208 157 |
    | `StrongWolfe` (`c₂ = 0.1`) | 58 | 135 / 7 893 | 218 / **18 127** |
    | `StrongWolfe` (`c₂ = 0.9`) | 36 | 159 / 5 687 | 12 717 / 445 497 |
    | `BierlaireQuadratic` | 106 | 130 / 13 781 | 121 / 13 491 |
    | `Quadratic` | 138 | 111 / 15 377 | 175 / 18 122 |
    | `Bisection` | 589 | 133 / 78 658 | 136 / 80 001 |

    (Regenerate with `scripts/retraction_accuracy.jl`. Every number here moved by a few percent in
    0.2.0 when the geodesic retraction stopped losing accuracy on a large lift — a more accurate
    exponential is a different trajectory. The ordering, which is what the table is for, did not.)

    Three cells are **not** regenerated by that script and are older measurements: the two
    `StrongWolfe (c₂ = 0.9)` cells and the `DFP` cells of `Backtracking (shrink only)` and
    `BierlaireQuadratic`, none of which is one of its `COMBINATIONS`. They predate SimpleSolvers 0.12
    and the step ceiling. Everything else in the table is current; see the note on open issue C9 about
    which figures in this package have a named harness behind them and which do not.

    **What the step ceiling moved here: nothing.** Every cell the script regenerates is reproduced to
    the digit with the ceiling on and off, on both retractions, at `DEFAULT_STEP_CEILING = 1`. The
    ceiling does not bind on this starting point, which is the whole design — what it buys is on the
    *other* starting points, where it is the difference between converging and ending off the
    manifold. See [`DEFAULT_STEP_CEILING`](@ref) and the seed spreads in `svd_optim.jl`.

    That was not true of the ceiling as first written, and the difference is issue A15. Deriving the
    bound from `2π` over the norm of the *whole* direction made each block of a `NamedTuple` pay for
    its neighbours, and on this problem — where both blocks are manifolds — combining them in
    quadrature tightened it by up to `√2`. That was enough to bind on three cells here (`BFGS`
    `Quadratic` 111 → 120 iterations, `BFGS` `BierlaireQuadratic` 130 → 113, `DFP` `Quadratic`
    175 → 308) and on nothing under `Cayley`. Deriving it per block instead removes all three. The
    lesson is worth keeping: those cells looked like the price of the ceiling and were the price of a
    sloppy norm.

    Every evaluation count in this table but one is ten higher than it was before `rg` became the
    residual at the iterate a solve returns (issue A8), and every iteration count but one is unchanged
    — and it is the same row both times: `StrongWolfe (c₂ = 0.1)` for `BFGS` went 136 → 135
    iterations and so 8 074 → 7 893 evaluations, which is the one cell that moved *down*. Ten is
    exactly one gradient evaluation on this problem — `GradientAutodiff` costs ten objective calls for
    its 60 parameters, and the count above includes those — and it is the refresh at the *last*
    iterate, the one no `update!` follows and so the one nothing reuses. Per solve, not per iteration:
    the reuse in `store_gradient!` is what makes the difference `10` rather than `10 × iterations`.

    A shrink-only backtracking search starts at `α = 1` and can never exceed it, which is right for a
    direction already scaled like a Newton step — `BFGS` accepts `α = 1` on 74% of its iterations —
    but wrong for one that is systematically *under*-scaled. `DFP` wants a median `α` of 8, so it
    accepts the ceiling on **100%** of its iterations and crawls to the gate in 48 322 of them.
    `expand = true` lets an accepted *first* trial step be lengthened while each longer trial still
    satisfies sufficient decrease and strictly improves the merit, at most `nexpand = 3` rounds of at
    most `q = 10` each.

    That fixes `DFP` outright and makes `BFGS` slightly better as well, at a cost of under 4% per
    iteration — and of exactly nothing on a well-scaled problem, since the extrapolation reuses
    ``\varphi(0)``, ``\varphi'(0)`` and ``\varphi(\alpha)``, all known once the trial step is accepted,
    so declining to expand costs no evaluation at all. On the sphere problem the evaluation counts are
    identical with and without it.

    Reach for one of the bracketing methods when iteration count rather than evaluation count is what
    you are paying for — a very expensive objective, or an outer loop bounded in iterations. For the
    first-order methods that trade is poor: `Bisection` burns 1.8M evaluations against
    `Backtracking`'s 79 500 for the same 3 000 iterations.

!!! note "[`DFP`](@ref) converges under the default, and [`SimpleSolvers.StrongWolfe`](@extref) is still the steadier explicit choice"
    DFP's direction stays under-scaled — the expansion phase makes that harmless rather than absent, so
    `DFP` needs 768 iterations on `Geodesic` and 1 366 on `Cayley` where `BFGS` needs 95 and 118, on
    the starting point the test suite uses. Over eight starting points on the same problem it ranges
    over 385–1 118 (`Geodesic`) and 466–1 177 (`Cayley`). (That upper bound read 1 366 — the *pinned*
    value rather than the spread's — which `svd_optim.jl` corrected and this docstring did not.)

    Those ranges used to be 512–77 890 and 465–3 834, and the difference is
    [`curvature_is_usable`](@ref). `Q` became badly conditioned (κ ≈ 1e9) because it was being built
    from secant pairs with ``\delta^T\gamma \leq 0``, which the guard on the update did not reject; how
    quickly the expansion phase dug it back out was close to arbitrary. Enforcing the curvature
    condition removes a factor of 92 from the spread on `Geodesic` and, on this problem, most of the
    reason `DFP` had a reputation for being unpredictable.

    `StrongWolfe(T; c₂ = 0.1)` remains the choice to pass explicitly on a DFP-heavy workload, now on
    cost rather than on reliability: 218 and 279 iterations on that starting point, 296–868 and 198–515
    across the eight, 18 127 and 23 828 evaluations against the default's 20 001 and 35 339, and 1.6×
    to 2.2× faster in wall clock (the two wall-clock figures are older measurements: 0.155 s against
    0.246 s on `Geodesic`, 0.205 s against 0.451 s on `Cayley`, neither of which `svd_tables`
    produces). `Bisection` is steadier still (99–141 / 102–124) at four to five times the work.

    `c₂ = 0.1` and not `StrongWolfe`'s own default of `0.9`: at `0.9` the strong Wolfe conditions are
    already satisfied at `α = 1` on 99.4% of iterations, so its bracketing phase never fires and it
    crawls just as a shrink-only `Backtracking` does — 12 717 iterations against 218. `0.1` is the
    value [nocedal2006numerical](@cite) recommends where a more accurate line search is needed, and it
    makes the expansion fire on 94.5% of iterations.

    `Quadratic` is competitive on `Geodesic` (308 iterations) and falls behind on `Cayley` (529) — and
    the explanation this entry used to give for that is now measured and wrong. It read "probably
    because [`trial_slope`](@ref) is only first-order correct there"; `trial_slope` is exact under
    `Cayley` as of the [`retraction_differential`](@ref), which moved this figure from 550 to 529 and
    left the gap. What the polynomial searches are actually sensitive to is the size of the step they
    occasionally ask for, which is what [`DEFAULT_STEP_CEILING`](@ref) bounds — that was issue A1b, and
    bounding it is what closed it. The remaining `Geodesic`/`Cayley` gap is not that: it is the same
    gap with the ceiling switched off (175 against 529) and with it on (308 against 529), so the
    ceiling narrows it rather than explaining it, and no measurement here accounts for the rest.

    None of this is a property of DFP as such: given a search that can exceed `α = 1` it is competitive
    with `BFGS`. The expansion phase exists because of this package — see
    JuliaGNI/SimpleSolvers.jl#174, which was filed from these measurements and released in
    SimpleSolvers 0.11.
"""
default_linesearch(::Type{T}, ::OptimizerMethod) where {T} = Backtracking(T; expand=true)
default_linesearch(::Type{T}, ::AdamFamily) where {T} = Static(T(DEFAULT_LEARNING_RATE))

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::DFP) = print(io, "DFP")
Base.show(io::IO, alg::BFGS) = print(io, "BFGS")
