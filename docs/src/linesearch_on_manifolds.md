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
[`linesearch_problem`](@ref) therefore failed for every *searching* line search on a `NamedTuple` of
[`StiefelManifold`](@ref) points, and [`SimpleSolvers.Static`](@extref) only appeared to work because
it never evaluates the merit at all.

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
| SVD, best available | 1000 iterations, relative error ``10^{-2}`` | [`_BFGS`](@ref) in 93 iterations, relative error below ``10^{-8}`` |

The fixed-step column is not merely slower. With `Static(0.01)` the SVD solve reaches only
``\|\nabla f\| = 8\times10^{-2}`` after 1000 iterations, and ``1.9\times10^{-3}``,
``2.1\times10^{-4}``, ``4.0\times10^{-5}`` at 5000, 20 000 and 60 000; reaching the convergence gate
that way would take of the order of a million iterations.

!!! info "This changed in 0.2.0"
    [`GradientMethod`](@ref) and [`MomentumMethod`](@ref) used to default to a fixed
    `Static(1e-3)`. They could not do anything else, because until the trial point went through the
    retraction `Static` was the only line search that worked on manifold parameters at all.

## Pair gradients and directions intrinsically

`LinearAlgebra.dot` on an [`AbstractLieAlgHorMatrix`](@ref) is the *ambient* Frobenius product, and it
counts each off-diagonal block of the horizontal lift twice — so it comes out at exactly twice the
product of the lift's free parameters. Those free parameters are the coordinates everything else in
this package is expressed in: they size ``Q``, they are what [`outer!`](@ref) flattens before forming
its outer product, and they are what the line search's ``\alpha`` parameterizes. A central-difference
check showed the consequence: paired ambiently, the slope came out as ``2\varphi'(\alpha)`` where
``\varphi'(\alpha)`` was wanted.

[`_dot`](@ref) is the intrinsic pairing, and it is what three quantities need:

1. [`trial_slope`](@ref), so that the line-search derivative has the right scale.
2. The quasi-Newton denominator ``\delta^\mathsf{T}\gamma``, so that it agrees with the flattened
   ``T_1``, ``T_2`` and ``\gamma^\mathsf{T}Q\gamma`` it divides.
3. The predicted decrease ``\widetilde{\Delta f}``, so that it is comparable with the measured
   ``\Delta f``.

Getting that scale right improved [`_BFGS`](@ref) on the SVD problem from 176 to 113 iterations on
[`Geodesic`](@ref) with a shrink-only [`SimpleSolvers.Backtracking`](@extref), and from 197 to 93 on
[`Cayley`](@ref) with [`SimpleSolvers.Bisection`](@extref).

The two retractions are not equivalent here. For [`Geodesic`](@ref), [`trial_slope`](@ref) is the
exact derivative of the merit. [`Cayley`](@ref) is not a one-parameter subgroup, so its slope is exact
at ``\alpha = 0`` and drifts for finite steps — about 6% at ``\alpha = 0.5`` and 24% at
``\alpha = 1``. The merit *value* stays exact for either. A method that uses ``\varphi'``
quantitatively is therefore more sensitive to the choice of retraction than one that uses only its
sign.

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
right for a direction already scaled like a Newton step — [`_BFGS`](@ref) accepts ``\alpha = 1`` on
74% of its iterations — and wrong for one that is systematically *under*-scaled. [`_DFP`](@ref) wants
a median ``\alpha`` of 8, so under a shrink-only search it accepted the ceiling on 100% of its
iterations and crawled to the gradient gate in 49 679 of them. Raising only the initial trial step,
from 1 to 3, brought the same solve to 229 — which is what identifies the ceiling rather than DFP
itself as the cause.

The remedy is an expansion phase, which SimpleSolvers 0.11 added in response to
[SimpleSolvers issue #174](https://github.com/JuliaGNI/SimpleSolvers.jl/issues/174):
`Backtracking(T; expand = true)` lengthens an accepted *first* trial step for at most `nexpand = 3`
rounds of at most a factor `q = 10` each, while every longer trial still satisfies sufficient decrease
and strictly improves the merit. It is close to free — under 4% more objective evaluations per
iteration, and *exactly* nothing on a well-scaled problem, because the extrapolation reuses
``\varphi(0)``, ``\varphi'(0)`` and ``\varphi(\alpha)``, all of which are known once the trial step
has been accepted. That is why it is the default here for every method except [`Adam`](@ref); see
[`default_linesearch`](@ref) for the per-method choices and the evaluation counts behind them.

Two conclusions from those measurements are worth stating separately, because they are properties of
the manifold problem rather than of any one search:

- **Compare line searches by objective evaluations, not by iterations.**
  [`SimpleSolvers.Bisection`](@extref) needs the fewest iterations of any option here, and roughly 583
  objective evaluations per iteration to get them, against 26 for `Backtracking(expand = true)`.
  Bracketing methods refine a line *minimum*; a backtracking search returns the first ``\alpha`` that
  decreases ``f`` enough. Reach for a bracketing method when iterations rather than evaluations are
  what you pay for — a very expensive objective, or an outer loop bounded in iterations.
- **A search that uses ``\varphi'`` quantitatively inherits the retraction's slope error.**
  [`SimpleSolvers.Quadratic`](@extref) is competitive on `Geodesic` (189 iterations for `_DFP`) and
  falls apart on `Cayley` (555), because it fits a polynomial to a slope that is only first-order
  correct there. `Bisection` uses only the sign of ``\varphi'``, and
  [`SimpleSolvers.StrongWolfe`](@extref) only compares it against ``\varphi'(0)``; neither degrades
  that way.

[`_DFP`](@ref) converges under the default expansion phase, but its direction stays under-scaled, and
how quickly the expansion digs out a badly conditioned ``Q`` is close to arbitrary: across eight
starting points on the SVD problem it ranges over 512–77 890 iterations on `Geodesic`.
`StrongWolfe(T; c₂ = 0.1)` is both faster and far steadier — 201–624 iterations across the same eight
— and is the choice to pass explicitly on a DFP-heavy workload. It has to be ``c_2 = 0.1`` and not
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

[`DecayingStatic`](@ref) is the weaker of the two ways to make such a solve terminate — it is what the
`AdamWithDecay` method of v0.1.0 did with its own fields. It drives the step to zero geometrically, so
the solve does stop on a criterion, but the schedule is summable: it stops short at
``\|\nabla f\| \approx 10^{-3}`` with ``\|\Delta x\|`` already around ``10^{-13}``. A searching line
search reaches about ``10^{-7}`` on the same Adam problem.

## Quasi-Newton caches on manifold solutions

[`_DFP`](@ref) used to be restricted to `AbstractVector` solutions, and the restriction was
accidental: like [`_BFGS`](@ref) it is built from a secant pair and never needs a vector-valued point.
The cache operates at the [`OptimizerSolution`](@ref) level, with separate type parameters for the
solution and the gradient, because a manifold point and its horizontal lift need not have the same
shape. Its section may be a `NamedTuple`, ``Q`` is sized by the *intrinsic* dimension rather than by
`length(x)`, and ``\gamma^\mathsf{T}Q\gamma`` uses the intrinsic pairing.

The same boundaries let both methods run on a bare [`Manifold`](@ref), not only on a `NamedTuple` of
them. For ``\operatorname{St}(3,1)`` the intrinsic dimension is 2 even though the gradient and the
direction are ambient ``3\times3`` lifts — which is exactly the mismatch the separate type parameters
exist for. The supporting pieces are [`outer!`](@ref) and `_mul!` for
[`AbstractLieAlgHorMatrix`](@ref), `alloc_h` for a `Manifold`, and copying a point into a
[`GlobalSection`](@ref).

## Reproducibility

[`global_section`](@ref) draws from Julia's global RNG, and both [`OptimizerState`](@ref) and each
quasi-Newton cache construct a [`GlobalSection`](@ref). A manifold solve is therefore reproducible
only if the RNG is seeded *before* the state and the optimizer are built — not merely before the
starting point is drawn. Unseeded, `_BFGS` with `Backtracking` varied between 17 and 18 iterations on
the sphere problem, with a final `check(x)` anywhere from ``2\times10^{-16}`` to
``4.5\times10^{-14}``.
