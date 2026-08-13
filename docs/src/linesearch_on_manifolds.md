# Line searches on manifolds

This page records the theoretical and empirical findings from the manifold
line-search work.[^pr31] It started from one defect—line searches could not
evaluate a trial point on a manifold—and exposed several consequences for
coordinate pairings, quasi-Newton updates, and step-size selection.

## A line-search trial point must use the retraction

For an ordinary vector, a line search evaluates the one-dimensional merit
function

```math
\phi(\alpha) = f(x_k + \alpha p_k).
```

That construction is not valid when ``x_k`` lies on a manifold. In
particular, adding a tangent or horizontal-lift direction to a point on
``\operatorname{St}(N,n)`` generally leaves the manifold. In the SVD problem,
`linesearch_problem` therefore failed for every searching line search with a
`NamedTuple` of Stiefel points. `Static` appeared to work only because it
never evaluated the merit function.

The correct manifold trial point is formed by applying the selected
retraction to the current point and direction. `trial_iterate!` now follows
the same construction as `solver_step!` after the line search has selected
``\alpha``. The vector implementation still uses the allocation-free
`compute_new_iterate!` path.

The line-search derivative must use the same tangent-space representation as
the direction. `trial_slope` consequently pairs the gradient and direction
through `global_rep`; they do not otherwise necessarily have matching shapes
on a manifold.

This changes a fixed-step workaround into a genuine search:

| Problem | Fixed-step behavior | Searching behavior |
| --- | --- | --- |
| Sphere, `GradientMethod` | 28 iterations | 2 iterations, ``\|\nabla f\| = 2.7\times10^{-16}`` |
| Sphere, `MomentumMethod` | 23 iterations | 2 iterations, ``\|\nabla f\| = 1.7\times10^{-15}`` |
| SVD, best available | 1000 iterations, relative error ``10^{-2}`` | `_BFGS`: 113 iterations, relative error ``2.6\times10^{-11}`` |

With `Static(0.01)`, the SVD solve reaches only ``\|\nabla f\|=8\times10^{-2}``
after 1000 iterations. At 5000, 20000, and 60000 iterations the norm is
approximately ``1.9\times10^{-3}``, ``2.1\times10^{-4}``, and
``4.0\times10^{-5}``, respectively. Reaching the convergence gate by this
route would require roughly a million iterations.

## Pair gradients and directions intrinsically

The ambient Frobenius `dot` product on an `AbstractLieAlgHorMatrix` counts
each off-diagonal block of the horizontal lift twice. It is therefore exactly
twice the product of the free parameters. Those free parameters are the
coordinates used to size ``Q``, flatten with `outer!`, and parameterize the
line-search step ``\alpha``. A central-difference check showed that the old
`trial_slope` returned ``2\phi'(\alpha)`` rather than ``\phi'(\alpha)``.

The intrinsic pairing is now used consistently by `_dot` at three sites:

1. `trial_slope`, so the line-search derivative has the right scale.
2. The quasi-Newton denominator ``\delta^\mathsf{T}\gamma``, so it agrees with
   the flattened ``T_1``, ``T_2``, and ``\gamma^\mathsf{T}Q\gamma`` quantities.
3. The predicted decrease ``\widetilde{\Delta f}``, so it is comparable with
   the measured ``\Delta f``.

This improves `_BFGS` on the SVD problem from 176 to 113 iterations for
`Geodesic` plus `Backtracking`, and from 197 to 93 iterations for `Cayley`
plus `Bisection`.

There is an important retraction distinction. For `Geodesic`, `trial_slope` is
the exact derivative of the merit. `Cayley` is not a one-parameter subgroup,
so its slope is exact at ``\alpha=0`` but drifts for finite steps: the measured
error is about 6% at ``\alpha=0.5`` and 24% at ``\alpha=1``. The merit value
itself remains exact for either retraction. This explains why methods that
use the derivative quantitatively can be more sensitive to the retraction
than methods that use only its sign.

## Preserve symmetry in the DFP inverse Hessian

The DFP update contains a term of the form

```math
Q\,\gamma\gamma^\mathsf{T}\,Q,
```

which is symmetric in exact arithmetic. Computing its two matrix products
independently does not preserve symmetry in floating-point arithmetic. In
the observed run, the relative asymmetry grew from ``8\times10^{-16}`` after
five iterations to ``1.6\times10^{-11}`` after 20,000 iterations; eventually
`eigvals(Q)` returned complex values for a matrix that should be symmetric.

The DFP implementation now constructs the update symmetrically. The BFGS
cache did not exhibit the same problem because its second term is formed as
the exact transpose of the first.

## DFP needs expansion, not only backtracking

`Backtracking` begins at ``\alpha=1`` and only decreases the trial step. This
works well for a direction already scaled like a Newton step: `_BFGS` accepts
``\alpha=1`` on 74% of its iterations. DFP directions are systematically
under-scaled, however; DFP accepted ``\alpha=1`` on 100% of its iterations
and required 49,679 iterations to reach the gate. Changing only the initial
trial step from 1 to 3 reduced that solve to 229 iterations. The bottleneck
was the inability to expand the step, not DFP itself.

Line searches must be compared by objective evaluations as well as iteration
count. Bisection uses the fewest iterations but approximately 583 objective
evaluations per iteration, while Backtracking uses about 25:

| Search | Evaluations/iteration | `_BFGS`: iterations / evaluations | `_DFP`: iterations / evaluations |
| --- | ---: | ---: | ---: |
| `Backtracking` | 25 | 113 / 2,857 | 3000+ / 75,012, no convergence |
| `StrongWolfe` (``c_2=0.9``) | 36 | 159 / 5,708 | 3000+ / 105,054, no convergence |
| `StrongWolfe` (``c_2=0.1``) | 82 | 118 / 6,738 | 201 / 16,466 |
| `BierlaireQuadratic` | 102 | 170 / 17,340 | 322 / 27,484 |
| `Quadratic` | 129 | 173 / 22,267 | 189 / 18,313 |
| `Bisection` | 583 | 143 / 83,353 | 134 / 78,698 |

`StrongWolfe` already has an expansion/bracketing phase that Backtracking
lacks. With its default ``c_2=0.9``, however, the Wolfe conditions hold at
the first trial step on 99.4% of iterations, so expansion almost never runs.
With ``c_2=0.1``, expansion runs on 94.5% of iterations, the median accepted
step is 8, and it is the cheapest converging option for both tested
retractions. `Quadratic` is competitive on `Geodesic` but degrades to 555
iterations on `Cayley`, consistent with its greater dependence on the
quantitatively inaccurate finite-step slope.

The resulting defaults are consequently method-specific:

- `GradientMethod` and `MomentumMethod` use `Backtracking` instead of their
  former `Static(1e-3)` workaround.
- `_DFP` uses `StrongWolfe(c₂ = 0.1)` so it can lengthen under-scaled steps.
- Other well-scaled methods retain `Backtracking` because its low evaluation
  cost gives the smallest total work.
- `Adam` retains `Static`: its moving-average direction is not required to be
  a descent direction, so sufficient-decrease searches would reject its
  intended behavior.
- `linesearch = Static(η)` remains the explicit way to restore a fixed
  learning rate.

`DecayingStatic` is the weaker alternative for `AdamWithDecay`. It eventually
drives the step to zero and can terminate on a criterion, but its geometric
schedule is summable: the solve stops short at approximately
``\|\nabla f\|=10^{-3}`` while ``\|\Delta x\|`` is about ``10^{-13}``. A
searching line search reaches approximately ``10^{-7}`` for the same Adam
problem.

## Lift quasi-Newton caches to manifold solutions

DFP was previously restricted to `AbstractVector` solutions. That restriction
was accidental: DFP, like BFGS, is built from a secant pair and does not
require a vector-valued point. The cache now operates at the
`OptimizerSolution` level, with separate type parameters for the solution and
gradient, because a manifold point and its horizontal lift need not have the
same shape. Its section may be a `NamedTuple`, ``Q`` is sized by the intrinsic
dimension rather than `length(x)`, and ``\gamma^\mathsf{T}Q\gamma`` uses the
intrinsic pairing.

The same boundary fixes allow both BFGS and DFP to run on a bare `Manifold`.
For ``\operatorname{St}(3,1)`` the intrinsic dimension is 2 even though the
gradient and direction are ambient ``3\times3`` lifts. The required support
includes `outer!` and `_mul!` for `AbstractLieAlgHorMatrix`, `alloc_h` for a
`Manifold`, and copying a point into a `GlobalSection`. Both methods then
reach the sphere minimizer in a handful of iterations.

## Reproducibility and convergence diagnostics

`global_section` draws from Julia's global RNG. Each `OptimizerState` and each
cache creates a `GlobalSection`, so a manifold solve is reproducible only when
the RNG is seeded before constructing the state and optimizer. Without that
seed, `_BFGS` plus `Backtracking` varied between 17 and 18 iterations, and the
final `check(x)` varied from ``2\times10^{-16}`` to ``4.5\times10^{-14}``.
The initial point now seeds the solve, and the manifold tolerance is
``10^{-12}``, matching the corresponding SVD test.

The seven iteration-budget warnings had two causes. Six belonged to the
intentional 1000-iteration SVD algorithm-comparison budget; those tests keep
the budget and explicitly disable the warning. The remaining
`Float64`/`Cayley`/`Adam` warning represented a real convergence failure and
is fixed by selecting the step with a searching line search. Its distance to
the minimizer improves from ``1.6\times10^{-3}`` to ``1.3\times10^{-8}``.

## Consequences for tests and documentation

The new manifold line-search tests verify all four exported searching methods,
convergence versus a crawling fixed-step solve, both quasi-Newton methods on
both `NamedTuple` and bare-manifold solutions, and the `DecayingStatic`
schedule and its effect on a solve. The SVD tests separate convergence from
the fixed-budget algorithm comparison and stop shadowing `Base.error` with an
objective variable of the same name.

`Cayley`, `Geodesic`, and `AbstractRetraction` now have docstrings and
doctests. `StrongWolfe` is re-exported because it is now a public default
choice. The documentation build's page-size threshold was raised to account
for the additional API documentation. The complete suite and documentation
build were green with 4,310 assertions and no warnings.

## Upstream implications

The line-search findings were also reported upstream in
[SimpleSolvers issue #174](https://github.com/JuliaGNI/SimpleSolvers.jl/issues/174):

- Backtracking has no expansion phase.
- Strong Wolfe has an expansion phase, but its default ``c_2`` prevents it
  from firing for the under-scaled DFP direction.
- `Backtracking.α₀` is stored, documented, printed, and compared, but was not
  read, so the initial trial step could not actually be configured.
- Common initial-step heuristics (Nocedal--Wright 3.59 and 3.60, a unit first
  step, and rescaling ``Q_0``) help the under-scaled method but hurt a method
  whose direction is already well scaled. There is no universally good
  initial-step heuristic across these optimizers.

## A parsing caveat in the SVD example

The matrix `A` in `svd_optim.jl` visually resembles a 10×10 matrix but Julia
parses it as 20×5: a newline separates rows just as `;` does, even when the
source wraps a displayed row across two lines. The test is self-consistent
because it uses `size(A, 1)`, so this is not a correctness failure, but the
source does not describe the apparent matrix shape.

[^pr31]: See [the pull request](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/31).
