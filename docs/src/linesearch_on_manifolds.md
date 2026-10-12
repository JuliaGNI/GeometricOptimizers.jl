```@meta
CurrentModule = GeometricOptimizers
```

# Linesearches on Manifolds

A line search on a manifold is not the vector construction with a different inner product. Three
things change: the trial point has to be formed with the retraction, the derivative of the merit has
to be paired in the intrinsic coordinates, and the accepted step is no longer content to stay below
``\alpha = 1``. This page collects those three points and their consequences for the quasi-Newton
methods.

## A line-search trial point must use the retraction

For an ordinary vector, a line search evaluates the one-dimensional merit function

```math
\varphi(\alpha) = f(x_k + \alpha{}p_k).
```

That construction is not available when ``x_k`` lies on a manifold: adding a tangent or
horizontal-lift direction to a point on ``\operatorname{St}(N,n)`` generally leaves the manifold.
[`linesearch_problem`](@ref) therefore has to retract rather than add, for every *searching* line
search on a set of [`StiefelManifold`](@ref) points. [`SimpleSolvers.Static`](@extref) is the one case
that would appear to work without it, and only because it never evaluates the merit at all.

The manifold trial point is formed by applying the selected retraction to the current point and
direction: [`trial_iterate!`](@ref) uses the same construction as [`solver_step!`](@ref) does once the
line search has settled on ``\alpha``. The `AbstractVector` implementation still takes the
allocation-free [`SimpleSolvers.compute_new_iterate!`](@extref) path.

The derivative has to use the same tangent-space representation as the direction, which is why
[`trial_slope`](@ref) brings the gradient and the direction together through [`global_rep`](@ref) — on
a manifold the two need not have matching shapes to begin with.

That turns a fixed-step workaround into a genuine search:

| Problem | Fixed step | Searching |
| --- | --- | --- |
| Sphere, [`GradientMethod`](@ref) | 28 iterations | 2 iterations, ``\|\nabla f\| = 2.7\times10^{-16}`` |
| Sphere, [`MomentumMethod`](@ref) | 23 iterations | 2 iterations, ``\|\nabla f\| = 1.7\times10^{-15}`` |
| SVD, best available | 1000 iterations, relative error ``10^{-2}`` | [`BFGS`](@ref) in 93 iterations, relative error below ``10^{-8}`` |

The fixed-step column is not merely slower. With `Static(0.01)` the SVD solve reaches only
``\|\nabla f\| = 8\times10^{-2}`` after 1000 iterations, and ``1.9\times10^{-3}``,
``2.1\times10^{-4}``, ``4.0\times10^{-5}`` at 5000, 20 000 and 60 000; reaching the convergence gate
that way would take of the order of a million iterations.

## Pair gradients and directions intrinsically

`LinearAlgebra.dot` on an [`AbstractLieAlgHorMatrix`](@ref) is the *ambient* Frobenius product, and it
counts each off-diagonal block of the horizontal lift twice — so it comes out at exactly twice the
product of the lift's free parameters. Those free parameters are the coordinates everything else in
this package is expressed in: they size ``Q``, they are the coordinates [`_flat_scratch`](@ref) holds
the flat form in before forming
its outer product, and they are what the line search's ``\alpha`` parameterizes. A central-difference
check showed the consequence: paired ambiently, the slope came out as ``2\varphi'(\alpha)`` where
``\varphi'(\alpha)`` was wanted.

[`_dot`](@ref) is the intrinsic pairing, and it is what four quantities need:

1. [`trial_slope`](@ref), so that the line-search derivative has the right scale.
2. The quasi-Newton denominator ``\delta^\mathsf{T}\gamma``, so that it agrees with the flattened
   ``T_1``, ``T_2`` and ``\gamma^\mathsf{T}Q\gamma`` it divides.
3. The predicted decrease ``\widetilde{\Delta f}``, so that it is comparable with the measured
   ``\Delta f``.
4. [`ensure_descent!`](@ref)'s descent test, for the same reason: the ambient product is twice the
   intrinsic one, and on a manifold that is a scalar-indexed, device-incompatible computation besides.

Getting that scale right improved [`BFGS`](@ref) on the SVD problem from 176 to 113 iterations on
[`Geodesic`](@ref) with a shrink-only [`SimpleSolvers.Backtracking`](@extref), and from 197 to 93 on
[`Cayley`](@ref) with [`SimpleSolvers.Bisection`](@extref).

## The generator of the trial curve turns with the step

``\varphi'(\alpha) = \langle\nabla f(x(\alpha)), B\rangle`` holds only where
``\alpha \mapsto \mathrm{retract}(\alpha B)`` is a one-parameter subgroup. [`Geodesic`](@ref) is one;
[`Cayley`](@ref) is not, and pairing against ``B`` there gave a slope that was exact at
``\alpha = 0`` and drifted from there — 8.9% out at ``\alpha = 0.5``, 36% at ``\alpha = 1`` and 143%
at ``\alpha = 2`` on a ``\operatorname{St}(6,3)`` problem, against a central difference of the merit
the search itself evaluates.

[`retraction_differential`](@ref) supplies the generator that turns with the step. With
``M = (\mathbb{I} - \frac{\alpha}{2}\bar{B})^{-1}``,

```math
\frac{d}{d\alpha}\mathrm{Cayley}(\alpha\bar{B}) = M\bar{B}M,
\qquad\text{so}\qquad
\frac{dx}{d\alpha} = W(\alpha)\,\big(M^\mathsf{T}\bar{B}M\big)\,E ,
```

with ``W(\alpha)`` the frame the trial point was built in. Both inverses go through
[`lift_factors`](@ref) and the Woodbury identity, exactly as [`cayley`](@ref) does, so no
``N \times N`` matrix is formed and the cost is ``O(Nn^2 + n^3)``. ``\alpha = 0`` returns ``\bar{B}``
untouched, which is why the `Backtracking` default — the one search that evaluates ``\varphi'`` at
``\alpha = 0`` only — pays nothing for this and is bit-identical across the change, as is every
`Geodesic` figure in this package.

The merit *value* was always exact for either retraction, so what this changes is the searches that
read ``\varphi'``: `Bisection`, the two polynomial fits, and
[`SimpleSolvers.StrongWolfe`](@extref). With the differential in place the two retractions no longer
differ in what a derivative-based search sees, and the choice between them is the cost one — see
[What they cost](@ref) on the [Exponential Algorithms](@ref) page for the benchmark.

## Preserve symmetry in the DFP inverse Hessian

The DFP update contains a term of the form

```math
Q\,\gamma\gamma^\mathsf{T}\,Q,
```

which is symmetric in exact arithmetic and not symmetric when its two matrix products are formed
independently in floating point. The error accumulates: ``\|Q - Q^\mathsf{T}\|/\|Q\|`` grows from
``8\times10^{-16}`` after five iterations to ``1.6\times10^{-11}`` after twenty thousand, at which
point `eigvals(Q)` starts returning complex numbers for a matrix that cannot have them. The update is
therefore symmetrized explicitly. The BFGS cache gets this for free, because it adds ``T_1 + T_2``
where ``T_2`` is built as the exact transpose of ``T_1``.

## A manifold step does not want ``\alpha \le 1``

A shrink-only backtracking search starts at ``\alpha = 1`` and can never exceed it. That ceiling is
right for a direction already scaled like a Newton step — [`BFGS`](@ref) accepts ``\alpha = 1`` on
74% of its iterations — and wrong for one that is systematically *under*-scaled. [`DFP`](@ref) wants
a median ``\alpha`` of 8, so under a shrink-only search it accepts the ceiling on 100% of its
iterations and crawls to the gradient gate in 47 115 of them. Raising only the initial trial step,
from 1 to 3, brought the same solve to 229 — which is what identifies the ceiling rather than DFP
itself as the cause.

The remedy is the expansion phase of SimpleSolvers: `Backtracking(T; expand = true)` lengthens an accepted *first* trial step for at most `nexpand = 3`
rounds of at most a factor `q = 10` each, while every longer trial still satisfies sufficient decrease
and strictly improves the merit. It is close to free — under 4% more objective evaluations per
iteration, and *exactly* nothing on a well-scaled problem, because the extrapolation reuses
``\varphi(0)``, ``\varphi'(0)`` and ``\varphi(\alpha)``, all of which are known once the trial step
has been accepted. That is why it is the default here for every method except [`Adam`](@ref); see
[`default_linesearch`](@ref) for the per-method choices.

The table gives the objective evaluations on the SVD problem of `test/verification/svd_optim.jl`,
with `Geodesic`, on the starting point that the test suite uses. `Static` needs about 4 evaluations
per iteration; subtract that for the cost of the search itself. The figures are from
GeometricOptimizers 0.2.0 with SimpleSolvers 0.11, and `scripts/retraction_accuracy.jl` makes them
again. That script does not make the two `StrongWolfe (c₂ = 0.9)` cells, and the `DFP` cells of
`Backtracking` (shrink only) and `BierlaireQuadratic`; these come from an earlier measurement.

| search | evals/iteration | `BFGS`: iters / evals | `DFP`: iters / evals |
|---|---|---|---|
| `Backtracking(expand = true)` | **26** | **95** / **2 441** | 768 / 20 001 |
| `Backtracking` (shrink only) | 25 | 136 / 3 457 | 48 322 / 1 208 157 |
| `StrongWolfe` (`c₂ = 0.1`) | 58 | 135 / 7 893 | 218 / **18 127** |
| `StrongWolfe` (`c₂ = 0.9`) | 36 | 159 / 5 687 | 12 717 / 445 497 |
| `BierlaireQuadratic` | 106 | 130 / 13 781 | 121 / 13 491 |
| `Quadratic` | 138 | 111 / 15 377 | 175 / 18 122 |
| `Bisection` | 589 | 133 / 78 658 | 136 / 80 001 |

The step ceiling of the next section does not bind on this starting point: every cell that the
script makes is the same with the ceiling on and off, on both retractions.

Two conclusions from those measurements are worth stating separately, because they are properties of
the manifold problem rather than of any one search:

- **Compare line searches by objective evaluations, not by iterations.**
  [`SimpleSolvers.Bisection`](@extref) needs the fewest iterations of any option here, and roughly 583
  objective evaluations per iteration to get them, against 26 for `Backtracking(expand = true)`.
  Bracketing methods refine a line *minimum*; a backtracking search returns the first ``\alpha`` that
  decreases ``f`` enough. Reach for a bracketing method when iterations rather than evaluations are
  what you pay for — a very expensive objective, or an outer loop bounded in iterations.
- **A search that uses ``\varphi'`` quantitatively is the one that notices a bad step, not the one
  that causes it.** [`SimpleSolvers.Quadratic`](@extref) is competitive on `Geodesic` (175 iterations
  for `DFP`) and falls behind on `Cayley` (529). A slope that is only first-order correct is not the
  cause: the slope is exact under both retractions, and the gap remains. The step ceiling below
  narrows it without accounting for the rest, and nothing measured here explains the remainder.

  A *divergence* on an over-long step is a different matter; see the next section.

## A bounded merit is not a bound on the step

This is the property of a compact manifold that no line search is written to expect.

``\varphi(\alpha) = f(\Lambda\,\mathrm{retract}(\alpha\bar{B})E)`` is **bounded** in ``\alpha``:
[`Cayley`](@ref) converges to a fixed rotation as ``\alpha \to \infty`` and [`Geodesic`](@ref) is
periodic in it. On ``\operatorname{St}(10,3)`` against a random target, ``\varphi`` at
``\alpha = 10^9`` is `5.4862` under `Cayley` and `4.7299` under `Geodesic`, against `4.8499` at
``\alpha = 0`` — so under `Geodesic` a step of ``10^9`` is a *genuine decrease*, and a search that
reports one is telling the truth.

In the Euclidean case the same ``\alpha`` gives ``f(x + \alpha{}p) = 3.4\times10^{18}`` against
`1.1e1`, and the search's own sufficient-decrease test throws it out. That is why a line search
carries no guard of its own here: **on a vector space the merit is the guard.** On a manifold it is
not, and the only thing wrong with the step is that retracting a lift of that norm loses the manifold.

Measured on the SVD problem, `Quadratic` returned ``\alpha = 4.3\times10^7`` on a direction of norm
`5.54`, a step of ``\|\alpha\delta\| = 2.4\times10^8``. Nothing about the direction was wrong there —
``\|\delta\|`` was in the same range as the two steps before it and ``Q`` was well conditioned
(``\lambda_\mathrm{max} = 3.86``) — and two iterations later, after [`restart!`](@ref) had put ``Q``
back to the identity so that the direction *was* ``-\nabla{}f``, the same search handed back
``8.6\times10^7`` on steepest descent. So a better direction was never the remedy; a bound on the step
was.

### The bound, and why it takes two packages

The bound is geometric. Retracting a lift is a rotation, so past ``\|\alpha\bar{B}\| \approx 2\pi``
a longer step reaches nothing new — and ``2\pi/\|\delta\|`` is not a property of ``\varphi``, changes
at every solver step, and cannot be measured by a line search. Hence the split:

- **SimpleSolvers** takes a per-call ceiling as `params.αmax`, honoured by all six of its line
  searches, and stops the bracketing *at* the ceiling rather than clamping afterwards — so the merit
  is measured at the step that is handed back, and the evaluations past it are saved rather than
  wasted. Its own default of `65536` is a backstop against unbounded extrapolation and, upstream is
  explicit about this, is nowhere near tight enough to bound the step: at
  ``\|\delta\| \approx 5.5`` it still permits ``\|\alpha\delta\| = 3.6\times10^5``.
- **This package** supplies the value, because choosing it is a statement about the geometry.
  [`solver_step!`](@ref) passes ``c\,2\pi/\|\delta\|`` through [`linesearch_parameters`](@ref), with
  `c` the [`DEFAULT_STEP_CEILING`](@ref) — `1`, i.e. never more than one full turn — settable per
  solve as `Optimizer(x, F; step_ceiling = …)` and disabled with `Inf`.

  On a set of parameters the one ``\alpha`` scales every block, so the ceiling is derived *per block*
  and the smallest over the blocks that live on a manifold is the one that binds
  ([`_manifold_αmax`](@ref)). A block that is an ordinary array contributes
  none — the ``2\pi`` is the turn of a rotation, and a vector space has no such scale either to
  impose or to tighten its neighbours with. A set with no manifold block therefore gets no ceiling at
  all, exactly as a plain vector does.

Over the eight starting points of `scripts/retraction_accuracy.jl`, counting the ones that end with
both factors still on ``\operatorname{St}(20,3)``:

| combination | ceiling off | ceiling on | worst `check` |
|---|---|---|---|
| `BFGS` `Quadratic` `Cayley` | 4 of 8 | **8 of 8** | `3.2e-1` → `6.1e-14` |
| `BFGS` `BierlaireQuadratic` `Cayley` | 4 of 8 | **8 of 8** | `5.5e-1` → `6.9e-14` |
| `BFGS` `Backtracking(expand)` `Geodesic` | 7 of 8 | **8 of 8** | `2.8e-12` → `6.0e-14` |
| `BFGS` `Backtracking` `Geodesic` | 7 of 8 | **8 of 8** | `2.8e-12` → `6.3e-14` |

All twenty combinations in that sweep are 8 of 8 with the ceiling. The last two rows are worth
reading twice: a *shrink-only* `Backtracking` that reaches an over-long step at all is surprising
until you notice that its expansion phase can exceed ``\alpha = 1`` and that ``\|\delta\|``
is what makes ``\alpha = 1`` too far. The mechanism is not confined to the searches that extrapolate
polynomials; it is confined to steps that are long relative to ``2\pi/\|\delta\|``.

!!! info "What the ceiling does not do"
    It costs nothing on a solve that never approaches it — on this problem every pinned figure under
    `Cayley` is reproduced to the digit with the ceiling on or off — and it is not a convergence
    safeguard. A step that is admissible but *uphill* is [`linesearch_rejected`](@ref)'s business, and
    a direction that ascends is [`ensure_descent!`](@ref)'s. The three are independent.

    One interaction is worth knowing, and it is why [`linesearch_rejected`](@ref) takes the ceiling
    as an argument. Where the ceiling binds hard, a search can report `LINESEARCH_FLOOR` — "no step
    along this direction decreases the merit" — when what it established is only that no *permitted*
    step does. Read literally that is a claim about the direction, and answering it by restarting
    ``Q`` would discard a sound quasi-Newton direction over a bound the caller imposed itself. So a
    `LINESEARCH_FLOOR` returned *at* the ceiling is exempt and its step is taken; below the ceiling
    it is a rejection as before. `LINESEARCH_EXHAUSTED` and `LINESEARCH_NO_DESCENT` are not exempt,
    being true of the direction whatever ceiling was in force.

[`DFP`](@ref) converges under the default expansion phase, but its direction stays under-scaled:
across eight starting points on the SVD problem it ranges over 385–1 118 iterations on `Geodesic`.

The curvature condition, and not the line search, keeps that range narrow. A ``Q`` built from secant
pairs with ``\delta^T\gamma \leq 0`` is badly conditioned on this problem, and how quickly the
expansion phase digs it back out is close to arbitrary. Enforcing the curvature condition — see
[`curvature_is_usable`](@ref) — takes a factor of 92 off the spread. `StrongWolfe(T; c₂ = 0.1)` is still somewhat faster and steadier — 296–868 iterations
across the same eight, at 18 127 objective evaluations against the default's 20 001 — and remains the
choice to pass explicitly on a DFP-heavy workload. It has to be ``c_2 = 0.1`` and not
`StrongWolfe`'s own default of ``0.9``: at ``0.9`` the strong Wolfe conditions already hold at
``\alpha = 1`` on 99.4% of iterations, so the bracketing phase never fires and the solve crawls just as
a shrink-only search does. ``0.1`` is the value [nocedal2006numerical](@cite) recommends where a more
accurate line search is needed, and it makes the expansion fire on 94.5% of iterations.

There is no initial-step heuristic that serves both cases. The common ones — [nocedal2006numerical](@cite)
equations 3.59 and 3.60, a unit first step, rescaling ``Q_0`` — all help the under-scaled method and
hurt the one whose direction is already well scaled, which is why the expansion phase, rather than a
cleverer ``\alpha_0``, is what fixes this.

### Keeping a fixed learning rate

`linesearch = Static(η)` remains the way to ask for a fixed learning rate, and it is what
[`Adam`](@ref) keeps as its default: Adam's direction ``-m_1/(\sqrt{m_2} + \delta)`` is a moving
average that is deliberately *not* required to descend on any individual step, so a
sufficient-decrease search has nothing to work with and would spend every step reporting that it found
no descent direction.

[`DecayingStatic`](@ref) is the weaker of the two ways to make such a solve terminate. It drives the step to zero geometrically, so
the solve does stop on a criterion, but the schedule is summable: it stops short at
``\|\nabla f\| \approx 10^{-3}`` with ``\|\Delta x\|`` already around ``10^{-13}``. A searching line
search reaches about ``10^{-7}`` on the same Adam problem.

## Quasi-Newton caches on manifold solutions

[`DFP`](@ref) runs on manifold solutions as well as on `AbstractVector` solutions: like
[`BFGS`](@ref) it is built from a secant pair and never needs a vector-valued point.
The cache operates at the [`OptimizerSolution`](@ref) level, with separate type parameters for the
solution and the gradient, because a manifold point and its horizontal lift need not have the same
shape. Its section may be a tree of sections rather than one, ``Q`` is sized by the *intrinsic*
dimension rather than by `length(x)`, and ``\gamma^\mathsf{T}Q\gamma`` uses the intrinsic pairing.

The same boundaries let both methods run on a bare [`Manifold`](@ref), not only on a whole set of
parameters holding several. For ``\operatorname{St}(3,1)`` the intrinsic dimension is 2 even though the gradient and the
direction are ambient ``3\times3`` lifts — which is exactly the mismatch the separate type parameters
exist for. The supporting pieces are [`_flat_scratch`](@ref) and `_flat_mul!` for
[`AbstractLieAlgHorMatrix`](@ref), `alloc_h` for a `Manifold`, and copying a point into a
[`GlobalSection`](@ref).

## Reproducibility

[`global_section`](@ref) draws from Julia's global RNG, and both [`OptimizerState`](@ref) and each
quasi-Newton cache construct a [`GlobalSection`](@ref). A manifold solve is therefore reproducible
only if the RNG is seeded *before* the state and the optimizer are built — not merely before the
starting point is drawn. Unseeded, `BFGS` with `Backtracking` varied between 17 and 18 iterations on
the sphere problem, with a final `check(x)` anywhere from ``2\times10^{-16}`` to
``4.5\times10^{-14}``.
