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

Runs on an `AbstractVector` and on nothing else. The exact Hessian is what makes it Newton's method:
a bare [`Manifold`](@ref) has no Riemannian Hessian to build, and for a
[`NeuralNetworkParameters.NetworkParameters`](@extref) the Hessian is not built over the flattening.
Both are rejected at [`Optimizer`](@ref) and [`OptimizerState`](@ref) construction with a message
saying so. [`BFGS`](@ref) and [`DFP`](@ref) take all three.
"""
struct Newton <: OptimizerMethod end

function Hessian(::Newton, ForOBJ::Union{Callable, OptimizerProblem}, x::AbstractVector)
    HessianAutodiff(ForOBJ, x)
end
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
update itself lives in [`QuasiNewtonCache`](@ref), which symmetrizes ``Q\gamma\gamma^TQ`` explicitly and
skips the update when the secant pair fails [`curvature_is_usable`](@ref).

Where [`BFGS`](@ref) is the better-conditioned choice and the default, `DFP` is here because it is
the other classical member of the family and because the two differ in exactly the way that shows
what the line search is doing: DFP's direction is systematically *under*-scaled, so it wants a
median ``\alpha`` well above 1 and is starved by a search that cannot exceed it. See
[`default_linesearch`](@ref): `DFP` converges under the default expanding `Backtracking`, and
`StrongWolfe(T; c₂ = 0.1)` is the explicit choice for a DFP-heavy workload.

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
Its state is [`BFGSState`](@ref) and the update itself lives in [`QuasiNewtonCache`](@ref), which skips the
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
    MomentumMethod(; α)

The gradient descent algorithm with momentum, i.e. the *heavy ball* method.

Stores the *momentum coefficient* `α`. The momentum is accumulated as
```math
    p \gets \alpha{}p + \nabla{}L
```
and the direction is ``-p``.

`α` carries no element type of the parameters: [`Optimizer`](@ref) and [`TrainingOptimizer`](@ref)
convert it once, with `change_precision`, to the element type of the parameters they are given.

!!! info "The step size is not stored here"
    Like every [`OptimizerMethod`](@ref), `MomentumMethod` only produces a direction; how far
    the optimizer goes along it is the line search's business. A fixed learning rate ``\eta``
    is therefore expressed as `linesearch = η`, or `linesearch = Static(η)`.
"""
struct MomentumMethod{T} <: OptimizerMethod
    α::T

    MomentumMethod{T}(α) where {T} = new{T}(α)
end

MomentumMethod(; α = DEFAULT_MOMENTUM_α) = MomentumMethod{typeof(float(α))}(float(α))

@doc raw"""
    Adam(; β₁, β₂, δ)

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

The coefficients carry no element type of the parameters: [`Optimizer`](@ref) and
[`TrainingOptimizer`](@ref) convert them once, with `change_precision`, to the element type of the
parameters they are given. `Adam()` optimizes `Float32` and `Float64` parameters alike.

!!! info "A searching line search is not the way to make this converge"
    Because the direction has magnitude ``\approx{}1`` per component whatever the gradient is,
    a step that does not shrink leaves `Adam` circling the minimiser at that distance rather
    than settling on it. A *searching* line search does not fix that — it picks each step from
    the merit and has no reason to drive the sequence to zero — it only makes each step a
    better one: on the two-sphere problem of `test/integration/manifold_linesearch_tests.jl` the searching
    options need 251–331 iterations where [`GradientMethod`](@ref) and [`MomentumMethod`](@ref)
    need 9–64. [`DecayingStatic`](@ref) is the setting under which `Adam` terminates on a
    criterion by construction, because its schedule drives the step to zero itself.

!!! info "The learning rate is not stored here"
    `Adam` only produces a direction, of magnitude ``\approx{}1`` per component; the learning
    rate ``\eta`` is the line search's `α`, i.e. it is passed as `linesearch = η` or
    `linesearch = Static(η)`, and [`default_linesearch`](@ref) is the default. `β₁`, `β₂` and `δ`
    are keyword arguments, so that a positional `Adam(1e-3)` meant as a learning rate fails instead
    of setting `β₁`.
"""
struct Adam{T} <: OptimizerMethod
    β₁::T
    β₂::T
    δ::T

    Adam{T}(β₁, β₂, δ) where {T} = new{T}(β₁, β₂, δ)
end

# The defaults are `Float64` literals, so that `change_precision(Float32, Adam())` holds `0.9f0` and
# not a `Float32` rounded from a `Float32` literal.
function Adam(; β₁ = 9.0e-1, β₂ = 9.9e-1, δ = 1.0e-8)
    β₁, β₂, δ = promote(float(β₁), float(β₂), float(δ))
    Adam{typeof(δ)}(β₁, β₂, δ)
end

@doc raw"""
    ScalarMomentAdam(; β₁, β₂, δ, ambient_norm)

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

That one scalar is the entire difference from [`Adam`](@ref), and it is what the name says:
``\lVert\bar{G}\rVert^2`` is a squared gradient *norm* where Adam's ``m_2`` is a squared *gradient*,
so the second moment carries no direction and the method assigns one adaptive learning rate to the
whole matrix instead of one per coordinate. That is a reproduction of a published algorithm and not a
straw man — on some objectives it will beat [`Adam`](@ref).

Only a single `StiefelManifold{T}` solution is supported; ordinary arrays, `NamedTuple`s, Grassmann
solutions and mixed parameter trees throw a `MethodError`. As for [`Adam`](@ref), the
coefficients carry no element type of the parameters, and [`Optimizer`](@ref) converts them.

!!! info "The Cayley transform is the retraction here, and it is exact"
    The source's lines 12–14 approximate the Cayley transform with two fixed-point iterations, and
    its line 11 caps the step length to keep that iteration contractive. Neither is ported: this
    package retracts exactly — [`Cayley`](@ref), the default, evaluates the transform through the
    Sherman-Morrison-Woodbury formula — and it admits any other [`AbstractRetraction`](@ref),
    [`Geodesic`](@ref) included. The step is bounded, but by [`step_αmax`](@ref) and for a different
    reason; `step_ceiling = 1/2π` recovers the source's bound up to its use of the induced 1-norm.

!!! warning "The direction has magnitude ``\approx{}1`` *in total*, not per component"
    This is where the shared default learning rate stops meaning the same thing in both methods, and
    it matters for the comparison this one exists for. [`Adam`](@ref)'s direction is
    ``-m_1/(\sqrt{m_2}+\delta)`` **componentwise**, so every entry has magnitude ``\approx{}1`` and
    ``\lVert\delta\rVert \approx \sqrt{\dim}``. Here the divisor is one number chosen to normalize
    the *whole* lift, so ``\lVert\delta\rVert \approx 1`` and each component is ``\approx{}1/\sqrt{\dim}``.

    Both methods take [`default_linesearch`](@ref)'s `Static(DEFAULT_LEARNING_RATE)`, so the same
    ``\eta`` buys a step ``\sqrt{\dim}`` times shorter here than it does for [`Adam`](@ref). Scale
    ``\eta`` accordingly before reading anything into a side-by-side run. It also means
    [`step_αmax`](@ref)'s ``2\pi{}c/\lVert\delta\rVert`` is ``\approx 2\pi{}c`` and so essentially
    never binds on this method.

!!! info "The learning rate is not stored here"
    As for [`Adam`](@ref), the source's learning rate ``l`` is the line search's `α`, i.e. it is
    passed as `linesearch = Static(η)`, which is also the default.

# Arguments

`β₁`, `β₂` and `δ` are the source's ``\beta_1``, ``\beta_2`` and ``\varepsilon``, with
``0 \le \beta_1, \beta_2 < 1`` and ``\delta \ge 0`` validated. `ambient_norm` selects which
``\lVert\cdot\rVert^2`` the second moment accumulates and defaults to `false`; see
[`GeometricOptimizers._squared_gradient_norm`](@ref), which is also where the reason the two choices
are not interchangeable up to a constant is written down.

# Extended help

The source's Algorithm 2, a table mapping each of its symbols onto the code, the derivation that makes
its lines 8–10 *be* [`global_rep`](@ref), and the two departures above with their reasons are on the
[Optimizer Methods](@ref "Standard Neural Network Optimizers") page. The third departure is the
momentum's transport: the source re-projects ``M_k`` onto the tangent space at the new iterate (its
equation (6)) where [`update_section!`](@ref) carries the lift here, which agrees for a Lie group and
differs on a proper homogeneous space — the one departure that changes the iterates against the
authors' implementation on the same objective and seed.

What is recorded only here is three discrepancies *between* the source and its own implementation,
which are not departures of this port:

  - **``v_1``.** Its line 2 initializes it to ``1``, `stiefel_optimizer.py` to ``0``
    (`v_buffer = torch.zeros([1])`), and line 6's ``1 - \beta_2^k`` is ``0`` at the ``k = 0`` its loop
    starts from. ``0`` is followed, which is also [`Adam`](@ref)'s convention here and which makes the
    first direction a normalized gradient step.
  - **``\varepsilon`` inside or outside the root.** Its line 7 and its implementation
    (`vnew_hat.add(epsilon).sqrt()`) both put it inside; [`Adam`](@ref) here puts it outside,
    ``\sqrt{m_2}+\delta``. The source's placement is used, so this is the one convention
    `ScalarMomentAdam` does *not* share with [`Adam`](@ref).
  - **``s``, the number of fixed-point iterations.** Its line 1 gives ``s = 2``; `Cayley_loop` in
    `gutils.py` runs `for i in range(5)`. Neither number reaches this method, because the iteration
    they count is the approximation of the Cayley transform that the admonition above says is not
    ported — [`Cayley`](@ref) is evaluated exactly. It is recorded because a reader comparing this
    port against the authors' code will meet it.
"""
struct ScalarMomentAdam{T} <: OptimizerMethod
    β₁::T
    β₂::T
    δ::T
    ambient_norm::Bool

    function ScalarMomentAdam{T}(β₁, β₂, δ, ambient_norm::Bool) where {T <: AbstractFloat}
        0 ≤ β₁ < 1 || throw(ArgumentError("β₁ must satisfy 0 ≤ β₁ < 1"))
        0 ≤ β₂ < 1 || throw(ArgumentError("β₂ must satisfy 0 ≤ β₂ < 1"))
        δ ≥ 0 || throw(ArgumentError("δ must be nonnegative"))
        new{T}(β₁, β₂, δ, ambient_norm)
    end
end

function ScalarMomentAdam(; β₁ = 9.0e-1, β₂ = 9.9e-1, δ = 1.0e-8, ambient_norm::Bool = false)
    β₁, β₂, δ = promote(float(β₁), float(β₂), float(δ))
    ScalarMomentAdam{typeof(δ)}(β₁, β₂, δ, ambient_norm)
end

@doc raw"""
    AdamWithEuclideanDecay(; β₁, β₂, δ, λ)

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

The derivation and the Grassmann case are in the [Weight Decay on Manifolds](@ref) page. Whether
a *Riemannian* decay is called for is an open question.

# Arguments

As for [`Adam`](@ref), the coefficients carry no element type of the parameters, and the learning
rate ``\eta`` is *not* stored here: it is the `α` of the line search, i.e. `linesearch = η`, and
[`default_linesearch`](@ref) is the default. `λ` is hence multiplied by ``\eta``,
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

    AdamWithEuclideanDecay{T}(β₁, β₂, δ, λ) where {T} = new{T}(β₁, β₂, δ, λ)
end

# see the remark on the `Float64` literals in `Adam`
function AdamWithEuclideanDecay(;
        β₁ = 9.0e-1, β₂ = 9.9e-1, δ = 1.0e-8, λ = DEFAULT_WEIGHT_DECAY)
    β₁, β₂, δ, λ = promote(float(β₁), float(β₂), float(δ), float(λ))
    AdamWithEuclideanDecay{typeof(δ)}(β₁, β₂, δ, λ)
end

"""
[`Adam`](@ref) and the methods that build their direction from a pair of bias-corrected moment
averages as it does. What the union selects for is that shared *character* and not a shared
implementation: the direction is a moving average that is deliberately allowed not to descend on an
individual step, which is what [`default_linesearch`](@ref)'s fixed `Static` and the
[`ensure_descent!`](@ref) exemption in [`FirstOrderMethodWithState`](@ref) are about.

The implementation is shared by only some of them, so do not read the union as a promise about the
cache:

- [`AdamWithEuclideanDecay`](@ref) differs only in how the direction is finished off, so it reuses
  [`AdamCache`](@ref) and [`AdamState`](@ref) outright and `OptimizerState(AdamWithEuclideanDecay(), x)`
  returns an `AdamState`.
- [`ScalarMomentAdam`](@ref) has neither, because its second moment is a *number* rather than an
  element of `𝔤ʰᵒʳ` and its recursion is therefore not `Adam`'s. It carries
  `ScalarMomentAdamCache` and [`ScalarMomentAdamState`](@ref) of its own.
"""
const AdamFamily = Union{Adam, AdamWithEuclideanDecay, ScalarMomentAdam}

"""
The methods whose `update!` needs the *method* rather than a
[`SimpleSolvers.Hessian`](@extref), because they carry state of their own out of which the
direction is built — a momentum or a pair of moments. The (quasi-)Newton methods are the
complement: their direction comes from the Hessian and the method object holds nothing.

[`GradientMethod`](@ref) has no Hessian *and* no state; with these methods it forms the
[`FirstOrderMethod`](@ref)s, whose `update!` takes the method.

[`ensure_descent!`](@ref) has a no-op method for these. It must not apply to them:
a momentum and a moment average are deliberately allowed not to descend on an individual step,
and the decay of [`AdamWithEuclideanDecay`](@ref) tilts the direction further away from the
gradient still.

That exemption is about the direction *before* the search, and not about the restart *after* it —
see [`linesearch_rejected`](@ref). Declining to overrule the direction is not the same as taking the
longest step along it, which is what a rejected search hands back.
"""
const FirstOrderMethodWithState = Union{MomentumMethod, AdamFamily}

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
[`Adam`](@ref)'s direction is ``-m_1/(\sqrt{m_2} + \delta)`` and [`ScalarMomentAdam`](@ref)'s is
``-m_1/\sqrt{m_2 + \delta}``; either way it is a moving average that is deliberately *not* required to
descend on any individual step, so a sufficient-decrease search has nothing to work with and would
spend every such step reporting that it found no descent direction.

One caveat on sharing the *value* `DEFAULT_LEARNING_RATE` across the family: [`Adam`](@ref) normalizes
componentwise and [`ScalarMomentAdam`](@ref) normalizes the whole lift at once, so the same ``\eta``
is a step ``\sqrt{\dim}`` shorter for the latter. See the warning in its docstring.

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

Pass `linesearch = Static(η)` for a fixed learning rate with any method.

!!! tip "Why a backtracking search, and why `expand`"
    `Backtracking` returns the *first* `α` that decreases `f` enough, while `Bisection`,
    [`SimpleSolvers.Quadratic`](@extref) and [`SimpleSolvers.BierlaireQuadratic`](@extref) bracket and
    then refine a line *minimum*, which costs an order of magnitude more merit evaluations per
    iteration. So compare them by objective evaluations, not by iterations.

    A shrink-only backtracking search starts at `α = 1` and can never exceed it. That is right for a
    direction already scaled like a Newton step, and wrong for one that is systematically
    *under*-scaled, such as the direction of [`DFP`](@ref). `expand = true` lengthens an accepted
    *first* trial step while each longer trial satisfies sufficient decrease and improves the merit,
    for at most `nexpand = 3` rounds of at most a factor `q = 10` each. On a well-scaled problem it
    costs no evaluation, because the extrapolation reuses ``\varphi(0)``, ``\varphi'(0)`` and
    ``\varphi(\alpha)``, which are known once the trial step is accepted.

    Use a bracketing method when you pay for iterations rather than evaluations — a very expensive
    objective, or an outer loop bounded in iterations.

!!! note "[`DFP`](@ref) and [`SimpleSolvers.StrongWolfe`](@extref)"
    `DFP` converges under the default, but its direction stays under-scaled.
    `StrongWolfe(T; c₂ = 0.1)` is the choice to pass explicitly on a DFP-heavy workload: it takes fewer
    evaluations. Use `c₂ = 0.1` and not the default `0.9` of `StrongWolfe`: at `0.9` the strong Wolfe
    conditions hold at `α = 1` on nearly every iteration, so the bracketing phase does not start.
    `0.1` is the value that [nocedal2006numerical](@cite) recommends where a more accurate line search
    is necessary.

The manual page [Linesearches on Manifolds](@ref) gives the evaluation counts on the SVD problem.
"""
default_linesearch(::Type{T}, ::OptimizerMethod) where {T} = Backtracking(T; expand = true)
default_linesearch(::Type{T}, ::AdamFamily) where {T} = Static(T(DEFAULT_LEARNING_RATE))

"""
    FirstOrderMethod

The methods that build their direction from the gradient alone: [`GradientMethod`](@ref) and the
[`FirstOrderMethodWithState`](@ref). These are the methods a
[`TrainingOptimizer`](@ref) takes, alone or two of them in a [`CompositeMethod`](@ref), and the
methods whose `update!(cache, state, gradient, method, x)` takes the method rather than a
[`SimpleSolvers.Hessian`](@extref).
"""
const FirstOrderMethod = Union{GradientMethod, FirstOrderMethodWithState}

@doc raw"""
    CompositeMethod(; manifold, array)

Two [`FirstOrderMethod`](@ref)s for a parameter set whose leaves are not all of one kind: `manifold`
steps every [`Manifold`](@ref) leaf, and `array` every other leaf.

[`ScalarMomentAdam`](@ref) is the case that needs it: it steps a single [`StiefelManifold`](@ref),
because a *scalar* second moment pooled across a set means nothing. So a network with Stiefel
weights beside ordinary `Matrix` and `Vector` leaves has no single method that covers it:

```julia
CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam())
```

The split is by `Manifold` and not by `StiefelManifold`, so a `GrassmannManifold` leaf under
`ScalarMomentAdam` is a `MethodError` when its cache is built. Any other method, or a composite, in
either place is a `MethodError` at construction.

A composite is for [`TrainingOptimizer`](@ref) on a parameter set. Every leaf gets a cache and a
state of its own method, in the element type of the set, and moves exactly as it would under a
`TrainingOptimizer` of its own; nothing is pooled across leaves. The leaves share the step size of
each step. A composite is not an [`OptimizerMethod`](@ref), so [`Optimizer`](@ref) and
[`solve!`](@ref), whose line search finds one step length for the direction of the whole set, do
not take it.
"""
struct CompositeMethod{M <: FirstOrderMethod, A <: FirstOrderMethod}
    manifold::M
    array::A
end

CompositeMethod(; manifold, array) = CompositeMethod(manifold, array)

# The method that steps the leaf `x`. A parameter set is not a leaf, and is a `MethodError`.
leafmethod(method::CompositeMethod, ::Manifold) = method.manifold
leafmethod(method::CompositeMethod, ::AbstractArray) = method.array

"""
    default_step_size(method)

The fixed step size a [`TrainingOptimizer`](@ref) takes for `method` if its `linesearch` keyword is
not given: `1e-2` for [`GradientMethod`](@ref) and [`MomentumMethod`](@ref), and
`DEFAULT_LEARNING_RATE = 1e-3` for the [`AdamFamily`](@ref), whose direction has magnitude
``\\approx{}1`` per component whatever the gradient is.

A [`CompositeMethod`](@ref) steps all of its leaves at one step size, so its default is the one its
two methods agree on, and an `ArgumentError` asking for `linesearch` where they disagree.
"""
default_step_size(::Union{GradientMethod, MomentumMethod}) = 1.0e-2
default_step_size(::AdamFamily) = DEFAULT_LEARNING_RATE

function default_step_size(method::CompositeMethod)
    (; manifold, array) = method
    η, η′ = default_step_size(manifold), default_step_size(array)
    η == η′ || throw(ArgumentError(
        "the two methods of a CompositeMethod, $(nameof(typeof(manifold))) and " *
        "$(nameof(typeof(array))), have different default step sizes, $(η) and $(η′); " *
        "pass `linesearch`"))
    η
end

# The conversion `Optimizer` and `TrainingOptimizer` apply once, to the element type of the
# parameters. A method without coefficients is returned as it is.
change_precision(::Type, method::OptimizerMethod) = method
change_precision(::Type{T}, m::MomentumMethod) where {T} = MomentumMethod{T}(T(m.α))
change_precision(::Type{T}, m::Adam) where {T} = Adam{T}(T(m.β₁), T(m.β₂), T(m.δ))
function change_precision(::Type{T}, m::ScalarMomentAdam) where {T}
    ScalarMomentAdam{T}(T(m.β₁), T(m.β₂), T(m.δ), m.ambient_norm)
end
function change_precision(::Type{T}, m::AdamWithEuclideanDecay) where {T}
    AdamWithEuclideanDecay{T}(T(m.β₁), T(m.β₂), T(m.δ), T(m.λ))
end
function change_precision(::Type{T}, m::CompositeMethod) where {T}
    CompositeMethod(change_precision(T, m.manifold), change_precision(T, m.array))
end

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::DFP) = print(io, "DFP")
Base.show(io::IO, alg::BFGS) = print(io, "BFGS")
