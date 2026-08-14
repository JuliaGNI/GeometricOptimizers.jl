# Changelog

All notable changes to GeometricOptimizers.jl are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0, so a minor bump is a
breaking release).

## [Unreleased] — targeting 0.2.0

v0.1.0 carried **two parallel optimizer hierarchies**: `src/optimizers/`, ported from
GeometricMachineLearning, which knew about manifolds and was driven by `optimization_step!`, and
`src/euclidean_optimizers/`, which had the newer `OptimizerState` / `solver_step!` / `solve!`
interface but only worked on `AbstractVector`s. This release collapses them into one. `Optimizer` is
now the single entry point and runs over anything the new `OptimizerSolution` union covers — an
`AbstractVector`, a `Manifold`, or a `NamedTuple` of arrays — with the same driver, the same states
and the same caches.

Unifying them meant reading both implementations against each other, which turned up a number of
defects that had been invisible: two of the three first-order methods computed a step that did not
match their own documented formula, two optimizer states were reading uninitialized memory, and
`mul!` on the special matrix types returned the wrong object. Those are fixed here, so **the iterates
this package produces change**, in most cases substantially for the better.

### Removed (breaking)

- **`EuclideanOptimizer` is gone.** Replace `EuclideanOptimizer(x, F; …)` with `Optimizer(x, F; …)`.
- **The old manifold entry points are gone**: `optimization_step!`, `init_optimizer_cache`,
  `setup_gradient_cache` / `setup_momentum_cache` / `setup_adam_cache` / `setup_bfgs_cache`,
  `perform_optimization!`, and the manifold `BFGS` method with its `_BFGSCache`. The
  `Optimizer(method, ps; retraction = cayley)` + `optimization_step!(o, λY, ps, dx)` sequence is
  replaced by `Optimizer` / `OptimizerState` / `solve!`:

  ```julia
  # before
  o = Optimizer(Momentum(), ps; retraction = cayley)
  λY = GlobalSection(ps)
  optimization_step!(o, λY, ps, dx)

  # now
  opt   = Optimizer(ps, L; algorithm = MomentumMethod(), retraction = Cayley())
  state = OptimizerState(MomentumMethod(), ps)
  solve!(ps, state, opt)
  ```

- **`AdamWithDecay` is removed.** It differed from `Adam` only in an exponentially decaying learning
  rate, and the learning rate is now the line search's business (see below); a decaying schedule
  belongs in a `LinesearchMethod`, which is what `DecayingStatic` is (see *Added*). Note that the
  decay it named was of the *learning rate*, not of the weights — `AdamWithEuclideanDecay` (see
  *Added*) is a different method and not a rename of this one.
- **`Adam`'s `η` field is removed.** It was never applied to the direction, so `Adam(1e-3)` and
  `Adam(1e2)` produced identical results.
- Two stranded test files (`test/optimizer_status_tests.jl`, `test/special_matrices/poisson_tensor.jl`)
  were deleted. Neither was included by `runtests.jl` and both referenced symbols that no longer exist
  (`QuasiNewtonOptimizer`, `PoissonTensor`).

### Changed (breaking)

- **Method types renamed.** `EuclideanOptimizerMethod` → `OptimizerMethod` (there is one abstract type
  now); `Gradient` / `_Gradient` → `GradientMethod`; `Momentum` → `MomentumMethod`. `_BFGS` and `_DFP`
  are unchanged. `GradientMethod`, `MomentumMethod`, `Adam` and their states are now exported.
- **`Adam`'s constructor changed shape.** Because `η` used to be the *first positional* argument, `β₁`,
  `β₂` and `δ` are keyword arguments now, so an old `Adam(1e-3)` call fails loudly instead of silently
  setting `β₁ = 1e-3`. The element type is what `Adam` takes positionally: `Adam(Float32)`. Unlike
  `MomentumMethod`, `Adam` is not converted by `Optimizer`, so the type has to match the parameters.
- **The learning rate moved to the line search.** Every `OptimizerMethod` now produces only a
  *direction*; how far the optimizer goes along it is the line search's `α`. A fixed learning rate `η`
  is `linesearch = Static(η)`, and `Static` is exported for that reason. `default_linesearch` makes
  `Static(1e-3)` the default for `Adam`, whose direction is a moving average that is deliberately not
  required to descend, and `Backtracking` the default for everything else: the (quasi-)Newton methods
  because they come with a scale of their own, `GradientMethod` and `MomentumMethod` because a
  searching line search is what makes them converge rather than merely descend.

  On the SVD problem, letting the line search pick the step takes the relative error after 1000
  iterations from 1.4e-2 to 2.3e-3 (`GradientMethod`) and from 1.4e-2 to 2.2e-3 (`MomentumMethod`).
  `Bisection` reaches 6.2e-7 in the same number of *iterations*, but it costs ≈583 objective
  evaluations per iteration against `Backtracking`'s ≈25, so on any fixed budget of work
  `Backtracking` is ahead; see the table in `default_linesearch`. Pass a searching method explicitly
  when iterations rather than evaluations are what you are paying for.

  The `Backtracking` is `Backtracking(T; expand = true)`, which is *not* SimpleSolvers' own default —
  see the next entry.
- **Requires SimpleSolvers 0.11, and the line search may now lengthen a step.** A shrink-only
  backtracking search starts its trial step at `α = 1` and can never exceed it. That is right for a
  direction already scaled like a Newton step — `_BFGS` accepts `α = 1` on 74% of its iterations — but
  wrong for one that is systematically under-scaled. `_DFP`'s wants a median `α` of 8, so on the SVD
  problem it accepted the ceiling on *every* iteration and needed 47 115 of them; handing the same
  search a trial step of 3 instead of 1 was worth a factor of 217, which is what identified the ceiling
  rather than the method as the cause.

  That measurement became JuliaGNI/SimpleSolvers.jl#174 and, in SimpleSolvers 0.11, the `expand` key:
  an accepted *first* trial step is lengthened while each longer trial still satisfies sufficient
  decrease and strictly improves the merit. `default_linesearch` switches it on. Iterations, then total
  objective evaluations, Geodesic / Cayley:

  | | `expand = false` | `expand = true` |
  |---|---|---|
  | `_BFGS` | 114 / 136 iters, 2 915 / 3 446 evals | **93 / 118**, **2 389 / 3 021** |
  | `_DFP` | 47 115 / 26 479, 1 177 919 / 662 019 | **702 / 1 366**, **18 258 / 35 329** |

  It costs under 4% per iteration and, on a well-scaled problem, exactly nothing: the extrapolation
  reuses `φ(0)`, `φ'(0)` and `φ(α)`, all known once the trial step is accepted, so declining to expand
  costs no evaluation. On the sphere problem the evaluation counts are identical with and without it.

  The `_DFP` figures are for one starting point, but they are now roughly representative: across eight,
  `_DFP` under the default ranges over 387–845 iterations (`Geodesic`) and 466–1 366 (`Cayley`), and
  `_BFGS` over 93–154 / 114–156. Those `_DFP` ranges read 512–77 890 and 465–3 834 when this entry was
  first written, because `Q` became badly conditioned and how fast the expansion dug it out was close to
  arbitrary; that was the missing curvature condition, fixed below. For a DFP-heavy workload
  `StrongWolfe(T; c₂ = 0.1)` is still somewhat faster and steadier — see the next entry.

  0.11 also removes `Backtracking.α₀`, the field that appeared to configure the trial step and was never
  read — unused here, so nothing in this package changes for it. The compat bound is `"0.11"` rather
  than `"0.10, 0.11"` because `expand` does not exist in 0.10.
- **`StrongWolfe` is re-exported.** It was the only one of SimpleSolvers' six line searches this
  package did not pass through. It is not a default, but it is the better explicit choice for a
  DFP-heavy workload: `StrongWolfe(T; c₂ = 0.1)` takes `_DFP` to 207 / 279 iterations and
  16 873 / 23 818 evaluations, 1.4× to 2.0× faster in wall clock than the expanding `Backtracking`
  default, and stays inside 207–755 / 215–447 across eight starting points. The "part that matters
  more" this entry originally claimed — that the default ranged over two orders of magnitude where
  `StrongWolfe` did not — no longer holds: the curvature-condition fix below brings the default to
  387–845 / 466–1 366, so the remaining case for `StrongWolfe` here is cost rather than reliability.
  `c₂ = 0.1` and not its own default of `0.9`, at which the Wolfe conditions already hold at `α = 1` on
  99.4% of iterations, so its bracketing phase never fires and it crawls too — 26 978 iterations
  against 207.
- **Retractions are passed as instances, not functions**: `retraction = Cayley()` / `Geodesic()`, not
  `retraction = cayley`. The default is `Cayley()`.
- **`MomentumMethod`'s recursion is fixed.** It accumulated `p ← p + α∇L`, which is not momentum but an
  undamped sum: for a constant gradient it grew linearly instead of saturating at `∇L/(1 - α)`, and it
  kept pushing after `∇L → 0`. It is `p ← αp + ∇L` now, with direction `-p`. On the SVD test problem
  the relative error drops from 1.9e-2 to 9.7e-3 — from *worse* than plain gradient descent to
  slightly better, as momentum should be. ([#18])
- **`Adam`'s update formulas are fixed.** Three defects, each of which changes the iterates:
  - the bias correction was evaluated at `t + 1` rather than `t`, making the very first direction
    `0.7425⋅sign(∇L)` instead of the `sign(∇L)` the correction is supposed to give;
  - the recursion factors were `β/(1 - βᵗ)` rather than `(β - βᵗ)/(1 - βᵗ)`, which is what the
    bias-corrected form requires — the former amplifies the moments by up to `1/(1 - β)` in the first
    iterations;
  - the square root was applied to `m₂` instead of to `m̃₂ = m₂ + δ`, i.e. the direction was
    `-m₁/(m₂ + δ)` instead of `-m₁/(√m₂ + δ)`, and it corrupted the stored second moment afterwards.

  Relative error on the SVD problem improves from 1.1e-3 to 1.7e-5 (`Geodesic`) and from 1.7e-3 to
  6.0e-5 (`Cayley`).
- **`_DFP` accepts a `Manifold` and a `NamedTuple`, like `_BFGS`.** Its cache was `AbstractVector`-only,
  so anything else fell through to a `NewtonOptimizerCache` and a `MethodError`. It is lifted to
  `OptimizerSolution` the way `BFGSCache` already was: the solution and the gradient get separate type
  parameters (on a manifold they are a point and a horizontal lift, of different shapes), the section
  may be a `NamedTuple` of sections, `Q` is sized by the length of the flattening rather than by
  `length(x)`, and the quadratic form `γᵀQγ` is taken in the flattened coordinates. `HessianDFP` and
  the `Hessian(::_DFP, …)` methods are widened to match.
- **`_BFGS` and `_DFP` run on a *bare* `Manifold`.** `Q` is sized by the intrinsic dimension — the
  length of the flattening, 2 for `St(3, 1)` — while the gradient and the direction are horizontal
  lifts of the ambient shape, `3 × 3`. Four methods that the `NamedTuple` case had and the bare case
  did not sat on that boundary: `outer!` and `_mul!` for an `AbstractLieAlgHorMatrix`, `alloc_h` for a
  `Manifold`, and `_copyto!` of a point into a `GlobalSection`. Without them `_BFGS` on a bare
  manifold died in `outer!` with `AssertionError: axes(O, 1) == axes(x, 1)`. `ParameterHandling.flatten`
  gains the `GrassmannLieAlgHorMatrix` method that the Stiefel lift already had.
- **`_BFGS` and `_DFP` now actually update their inverse Hessian.** `state.ḡ` was refreshed at the end
  of the iteration, i.e. at the very iterate the next secant difference `γ = ∇f(x⁽ᵏ⁾) - ∇f(x⁽ᵏ⁻¹⁾)`
  was formed at, so `γ` was identically zero, `δᵀγ` was zero with it, and the guard around the `Q`
  update skipped it on **every** iteration. `Q` stayed the identity for the whole solve, i.e. both
  methods silently ran as steepest descent. Both caches now advance `ḡ` themselves, right after they
  have used it. On `sum(sin.(x).^2)` from `0.5·ones(3)` with `Backtracking`, `_BFGS` went from
  `1.1e-3` after exhausting all 1000 iterations to machine precision in six.
- **The `_DFP` update uses `γγᵀ` where it used to use `δδᵀ`.** DFP is
  `Q ← Q - Qγγᵀ Q/(γᵀQγ) + δδᵀ/(δᵀγ)` (nocedal2006numerical, eq. 6.15); the rank-one term that is
  subtracted was built from `cache.ΔxΔx`, which also left `cache.ΔgΔg` computed on the line above and
  never read.
- **The (quasi-)Newton directions are checked for descent.** `ensure_descent!` replaces a direction
  with `∇f⋅δ ≥ 0` by the steepest-descent direction. A Newton step only descends where the Hessian is
  positive definite, and up to SimpleSolvers 0.8 the `Bisection` and `Quadratic` line searches hid
  this by returning a *negative* step; they no longer do. Without the safeguard,
  `sum(sin.(x).^2)` from `ones(3)` — where `sin²` has second derivative `-0.83` — converges to `π/2`,
  which is a *maximum*. `Adam` and `MomentumMethod` are deliberately excluded: their direction is a
  moving average and is allowed not to descend on an individual step.
- **`AdamState` and `MomentumState` are zero-initialized.** They used `_similar`, i.e. uninitialized
  memory, which is read on the first `update!` — so the first step depended on whatever happened to be
  in memory, and `Adam` could throw a `DomainError` from `√` of a negative value.
- **`l2norm` of a `NamedTuple` combines the block norms in quadrature** instead of summing them, which
  overestimated the ℓ² norm of the flattened parameters by up to `√k` for `k` blocks — and with it
  every stopping criterion computed from it. Stopping behaviour on `NamedTuple` parameters changes.
- **`mul!(C, A, α)` and `rmul!` on the special matrix types return `C`.** They used to return the inner
  field (`C.S` / `C.A` / `C.B`), violating the LinearAlgebra contract. This made `MomentumMethod` and
  `Adam` unusable on a bare `Manifold`: for a `StiefelManifold` of size (3, 1), `_mul` degraded a
  (3, 3) `StiefelLieAlgHorMatrix` into a (2, 1) `Matrix`, and the two algorithms then died with
  `DimensionMismatch` and `CanonicalIndexError`. ([#17])
- **`ParameterHandling.flatten(T, ::Manifold)` round-trips the manifold type.** It reconstructed a
  `StiefelManifold` for every kind of manifold, so a `GrassmannManifold` came back Stiefel — with a
  different `rgrad` and a different retraction, and no error anywhere.
- **`Float32` parameter `NamedTuple`s are no longer silently promoted to `Float64`** by
  `ParameterHandling.flatten`, which defaults to `Float64`; the flattened vector was then incompatible
  with the parameters.
- **`GlobalSection` is more tightly parameterized** (`GlobalSection{T,AT<:AbstractArray{T},λT}`) and
  copies its `Y` on construction. `update_section!` gained a four-argument form
  `update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B, retraction)` that writes into a separate destination; the
  three-argument form forwards to it in place.
- **An unimplemented retraction now errors** and names the retraction and the argument type. The
  fallback had an empty body, so it returned `nothing` and failed further downstream with a message
  about `Nothing`.

### Changed (breaking) — dependencies

- **Zygote is dropped** in favour of **ForwardDiff** — the unified interface differentiates a scalar
  objective of a small parameter set, which is what ForwardDiff is good at — and **ParameterHandling**,
  whose `flatten` / `unflatten` pair turns a `NamedTuple` of arrays into the vector the Euclidean
  methods want.
- **SimpleSolvers is now `0.10`** (was `0.8`), and the **minimum Julia version is 1.10** (was 1.0),
  which is SimpleSolvers 0.10's floor and the oldest version CI has ever tested.

  SimpleSolvers 0.9 reworked `Options`, so several keywords that used to be accepted here are gone:
  `g_restol`, `x_abstol_break`, `x_reltol_break`, `f_reltol_break` and `g_restol_break`. This does not
  change stopping behaviour at default settings — every removed `*_break` field defaulted to `Inf`,
  and the gradient residual is now gated on `f_reltol`, whose default `√eps(T)` is the number
  `g_restol` used to default to. The successive change in `f` is gated on the new `f_suctol`, which
  inherited `f_reltol`'s former default.

### Fixed

- **`Adam` and `MomentumMethod` no longer take the step a line search has rejected.** `solver_step!`
  restarts and searches along steepest descent when the outcome is `LINESEARCH_FLOOR`,
  `LINESEARCH_EXHAUSTED` or `LINESEARCH_NO_DESCENT`, and the two methods whose direction carries state
  were exempt from it. The reasoning was that `ensure_descent!` exempts them — a moving average is
  *allowed* not to descend on an individual step — and it does not carry over: a rejected search
  returns `α = 1` untouched, so the exemption did not permit a non-descent step, it took the
  **longest** step available along one.

  On Rosenbrock from `(-1.2, 1)` with `MomentumMethod(0.1)`, the solve reaches `f = 7.8e-5` by
  iteration 400 and is then ratcheted to `Inf` by thirteen such steps:

  | iteration | outcome | `α` | `f` |
  |---|---|---|---|
  | 441 | `LINESEARCH_NO_DESCENT` | 1.0 | 4.97e-2 → 4.65e3 |
  | 443 | `LINESEARCH_NO_DESCENT` | 1.0 | 8.16e1 → 5.33e9 |
  | 453 | `LINESEARCH_NO_DESCENT` | 1.0 | 1.46e3 → 4.41e19 |

  Each multiplies `f` by between `1e3` and `1e42`; the steps in between descend and cannot make it
  back. Iterations and final `f` on that problem, at a cap of 10 000:

  | | `Backtracking` | `Backtracking(expand)` | `BierlaireQuadratic` | `StrongWolfe(c₂ = 0.1)` |
  |---|---|---|---|---|
  | `MomentumMethod(0.1)` before | 379, **`Inf`** | 457, **`Inf`** | 6, **`Inf`** | 3 136, 2.7e-16 |
  | after | 2 921, **1.9e-16** | 2 693, **1.6e-16** | 355, **1.3e-16** | 3 254, 2.6e-16 |
  | `Adam` before | 10 000 (cap), 3.0e-5 | 275, 1.4e-16 | 1 941, 4.0e-17 | 2 846, 2.3e-16 |
  | after | **453**, 8.6e-18 | 256, 2.0e-20 | **285**, 1.7e-16 | **294**, 5.0e-17 |

  `GradientMethod` is bit-identical throughout — it is not in `FirstOrderMethodWithState`, so it was
  never exempt, which makes it the control. So is every `Static` figure, since that search reports no
  outcome to act on: `Static(0.1)` still diverges on Rosenbrock for both first-order methods, which is
  an over-large fixed step on a badly scaled problem and a different matter.

  **This also retires most of issue A9.** `Adam` + `BierlaireQuadratic` on the two-sphere problem of
  `test/manifold_linesearch_tests.jl` used to run out all 1 000 iterations under both retractions
  while sitting `6.8e-6` from the minimiser — at the answer, with no criterion it could meet. It now
  terminates in 251–286, and all fourteen (line search, retraction) combinations terminate on a
  criterion, in 251–331 iterations. The worst distance over the fourteen is `5.0e-7` and that one is
  `Static`; every searching line search is at `1.8e-8` or better. What survives of A9 is the cost
  argument, which is unchanged: `Adam` needs 251–331 iterations there where `GradientMethod` and
  `MomentumMethod` need 9–64, so `default_linesearch` keeps `Static` for `AdamFamily`.

  The momentum recursion is untouched: `p ← αp + ∇f` is evaluated in `update!(::MomentumState, …)`
  from `gradient_array(cache)` *after* the step, so which direction the step was taken along does not
  enter it. `ensure_descent!`, which acts on the direction before the search, still exempts both
  methods.

- **The line-search slope is the derivative of the line-search merit under `Cayley` too.**
  ``\varphi'(\alpha) = \langle\nabla{}f(x(\alpha)), B\rangle`` holds only where
  ``\alpha \mapsto \mathrm{retract}(\alpha{}B)`` is a one-parameter subgroup. `Geodesic` is one and
  `Cayley` is not, so pairing against the direction `B` regardless gave a slope that was exact at
  ``\alpha = 0`` and drifted from there. Against a central difference of the merit the search itself
  evaluates, on a `St(6, 3)` problem:

  | ``\alpha`` | 0 | 0.25 | 0.5 | 1 | 2 |
  |---|---|---|---|---|---|
  | paired with ``B`` | exact | 2.2% | 8.9% | 36% | **143%** |
  | with ``D(\alpha)`` | exact | 4e-10 | 1e-9 | 4e-9 | 4e-10 |

  The second row is the central difference's own truncation error at `h = 1e-6`, i.e. the slope is
  now exact to what the check can resolve. The new `retraction_differential` supplies the generator
  that turns with the step: with ``M = (\mathbb{I} - \frac{\alpha}{2}B)^{-1}``,
  ``\frac{d}{d\alpha}\mathrm{Cayley}(\alpha{}B) = MBM``, so the velocity of the trial curve is
  ``W(\alpha)(M^TBM)E`` where ``W(\alpha)`` is the frame the trial point was built in. Both inverses
  go through `lift_factors` and the Woodbury identity exactly as `cayley` does, so no ``N \times N``
  matrix is formed and the cost is ``O(Nn^2 + n^3)``. For `Geodesic` — and for an ordinary array under
  either retraction, where the retraction is addition — ``D(\alpha) = B`` and nothing is computed.

  **What is bit-identical**, verified over the ten (method, line search) combinations of
  `scripts/retraction_accuracy.jl` on the pinned seed and eight starting points: every `Geodesic`
  figure in the package, in every column; both `Backtracking` rows under `Cayley`, because
  `Backtracking` evaluates ``\varphi'`` at ``\alpha = 0`` only, where ``D(0) = B`` is returned
  untouched — so the default path costs nothing for this, in accuracy or in work; and, on the SVD
  problem, every `BierlaireQuadratic` figure, which is what shows that search never asks for
  ``\varphi'`` away from ``\alpha = 0`` there.

  **What moves** is `Bisection`, `StrongWolfe` and `Quadratic` under `Cayley`. Iterations and
  objective evaluations on the SVD problem, seed 1234:

  | | before | after |
  |---|---|---|
  | `_BFGS` + `Bisection` | 92 / 54 970 | 114 / 67 020 |
  | `_BFGS` + `StrongWolfe(c₂ = 0.1)` | 139 / 8 354 | 135 / 7 870 |
  | `_DFP` + `Bisection` | 96 / 56 106 | 110 / 64 306 |
  | `_DFP` + `Quadratic` | 550 / 54 176 | 529 / 50 656 |

  `Bisection` taking more iterations for a *correct* slope than for a wrong one is not a regression —
  it bisects ``\varphi'``, so a different slope is a different sequence of brackets — and its spread
  over the eight starting points tightens, 88–129 to 102–124 for `_DFP`. The clearest gain is
  `_DFP` + `Quadratic`, whose spread goes from 168–1 211 to 164–735.

  **It does not close issue A1b**, which is what it was written for, and that is the more useful
  result. A1b proposed an exact `Cayley` differential as the likely fix for `_BFGS` +
  `Quadratic`/`BierlaireQuadratic` running to the iteration cap and ending off the manifold on two of
  eight starting points. One of those four cases is fixed — `Quadratic` on seed 2 goes from the
  20 000-iteration cap at `check = 5.0e-2` to **90 iterations at `check = 6.4e-15`** — and three are
  not. See the rewritten A1b entry for where the cause actually is.

- **`GradientMethod` and `MomentumMethod` no longer throw on their own default line search.**
  `Optimizer(ones(3), x -> sum(x.^2); algorithm = GradientMethod())` followed by a `solve!` was a
  `MethodError`: `trial_slope`'s `AbstractVector` branch calls `gradient(cache)`, which only
  `NewtonOptimizerCache`, `BFGSCache` and `DFPCache` defined, so none of the three first-order
  methods could use *any* line search that evaluates ``\varphi'`` on Euclidean vector parameters.
  That became the *default* path in 0.2.0, when `default_linesearch` started returning
  `Backtracking(T; expand = true)` for everything outside `AdamFamily`. Manifold parameters were
  never affected — that branch of `trial_slope` allocates and never touches the cache.

  Defining the accessor is not on its own enough, and the second half is the interesting one.
  `trial_slope` evaluates the trial gradient *into* the cache — that is what makes it
  allocation-free — while `update!(::MomentumState, ...)` re-runs ``p \gets \alpha{}p + \nabla{}f(x_k)``
  from the same array afterwards. Sharing one array made the momentum accumulate the gradient at
  whatever trial step the search last probed. Worst relative error in the state's momentum over eight
  iterations of ``f(x) = \sum(x^2 + 0.1x^4 + 0.3\sin 3x)``:

  | | `Static` | `Backtracking` | `Bisection` | `Quadratic` | `BierlaireQuadratic` | `StrongWolfe` |
  |---|---|---|---|---|---|---|
  | shared array | 0 | 0 | **1.04** | **1.04** | **4.51** | **1.04** |
  | scratch array | 0 | 0 | 0 | 0 | 0 | 0 |

  `Backtracking` is exact by accident: it evaluates ``\varphi'`` once, at ``\alpha = 0``, where the
  trial gradient *is* ``\nabla{}f(x_k)``. So the three first-order caches carry a scratch array of
  their own, reached through `latest_gradient`, and `cache.g` stays what the direction and the state
  updates need it to be.

- **The gradient residual is measured at the iterate a solve returns.** `rg` was
  ``\|\nabla{}f(x_k)\|`` at the iterate the step *started* from, for every method. Under `Static`
  that is harmless — the direction is a scaled gradient, so a vanishing gradient means a vanishing
  step — and under a direction that carries momentum it is not. A line search accurate enough to
  drive ``\nabla{}f(x_1) \approx 0`` made `g_converged` fire while the momentum term was still moving
  the iterate: `MomentumMethod` + `Backtracking` on ``f(x) = 1 + x^2`` from ``x = 1`` stopped after
  two iterations at ``x = -0.2`` reporting `rg = 0`, where ``\|\nabla{}f(x)\| = 0.4`` and the momentum
  was ``2``. Final ``\|x\|`` from `ones(3)`, before and after:

  | objective | method | `Bisection` | `Quadratic` | `BierlaireQuadratic` |
  |---|---|---|---|---|
  | ``1 + \|x\|^2`` | `MomentumMethod` | 0.346 → **0** | 0.346 → **0** | 4.5e-4 → **1.1e-8** |
  | ``1 + \|x\|^2`` | `Adam` | 1.16 → **0** | 1.16 → **7.7e-16** | 1.2e-5 → **9.5e-13** |
  | ``\sum\sqrt{1+x^2}`` | `MomentumMethod` | 0.122 → **3.9e-16** | 0.122 → **7.9e-14** | 0.122 → **1.1e-9** |
  | ``\sum\sqrt{1+x^2}`` | `Adam` | 1.16 → **3.9e-16** | 1.16 → **7.7e-16** | 0.104 → **4.5e-9** |

  `Adam` was the worse of the two — at ``\|x\| = 1.16`` it had barely left `ones(3)` — and both
  reported convergence. `solver_step!` now refreshes `latest_gradient` at the accepted iterate. It
  never cost an *iteration*: across the 42 (objective, method, line search) combinations measured the
  count is equal or one lower.

  **And it costs no gradient evaluation either**, which is not obvious and was not free. The refresh
  computes ``\nabla{}f(x_{k+1})`` in the frame of `section(cache)`, and that is bit-for-bit what
  `update!(cache, state, …)` recomputes at the top of the next step — `update_section!`'s
  three-argument method has the body the two-argument one `update!(::MomentumState, …)` uses, so the
  cache's frame after a step *is* the state's frame after `update!`. `store_gradient!` therefore
  reuses it instead of evaluating again, guarded on `solution(cache) == x` and
  `section(cache) == section(state)` so that a caller which moves the iterate between steps falls
  back rather than silently reusing a stale value. On the SVD problem of
  `test/optimizer_convergence/svd_optim.jl`, `Adam` + `Static` over 2 000 iterations (`err = 1.7097`
  in every column, i.e. one trajectory):

  | | before this release | refresh, no reuse | as shipped |
  |---|---|---|---|
  | wall clock | 124 ms | 167 ms (1.35×) | **128 ms (1.03×)** |
  | ``\nabla{}f`` per iteration | 1 | 2 | **1** |

  `rgₐ` is formed from the same two arrays — ``\nabla{}f(x_{k+1}) - \nabla{}f(x_k)``, the successive
  difference `OptimizerStatus` prints it as — so the two `g` rows of a status are about one step
  rather than about two different ones. It used to come from `state.ḡ`, which is two iterates behind
  `cache.g` for these three methods and is uninitialised on the first: on
  ``f(x) = \sum(x^2 + 0.1x^4)`` that reported `rgₐ = 4.976` where the successive difference is
  `0.295`. The remaining half of that — `Δf̃` still reads `state.ḡ` — is open issue A10 below.

  `Newton`, `_BFGS` and `_DFP` are untouched — `latest_gradient` defaults to `gradient(cache)`,
  nothing refreshes it for them and `store_gradient!` is not on their path. Verified bit-for-bit: 266
  solves over Rosenbrock, the SVD problem and a two-sphere `NamedTuple`, across all seven line
  searches, both retractions and all six methods, reproduce `rg`, the iterate, the value and `check`
  to the last digit; `rgₐ` moves for the three first-order methods and for nothing else.

- **The `Geodesic` retraction no longer silently leaves the manifold for a large step.** `geodesic`
  built the exponential by summing ``\mathfrak{A}(X) = \sum_{n\geq1}X^{n-1}/n!`` directly. That series
  converges everywhere but is only *accurate* for a small argument, and the argument here is not
  small: the factorisation ``\bar{B} = B'(B'')^T`` puts ``\frac{1}{4}A^2 - B^TB`` in a block of
  ``X = (B'')^TB'``, so ``\|X\| \approx \|\bar{B}\|^2/4`` while its spectral radius is only
  ``\approx\|\bar{B}\|``. The terms cancel, and nothing reported it. `check(geodesic(B))` on a random
  `StiefelLieAlgHorMatrix(20, 3)`:

  | `‖B̄‖` | 5.8 | 17.8 | 36.5 | 78.8 | 160 | 361 | 767 |
  |---|---|---|---|---|---|---|---|
  | before | `2.1e-15` | `2.6e-12` | `4.4e-7` | `8.3e10` | `4.2e55` | `1.4e168` | `NaN` |
  | after | `1.1e-15` | `3.5e-15` | `7.9e-15` | `2.5e-14` | `9.0e-15` | `3.6e-14` | `7.7e-14` |

  At `‖B̄‖ = 79` the "retracted" point was not on the Stiefel manifold in any sense. `Cayley` was never
  affected — it inverts a `2n × 2n` matrix rather than summing a series.

  The fix is scaling and squaring, and it stays at `2n × 2n` throughout: the low-rank form is closed
  under squaring, `(I + B'W(B'')ᵀ)² = I + B'(2W + WXW)(B'')ᵀ`, so squaring the *exponential* is one
  application of `W ↦ 2W + WXW` and no `N × N` matrix is ever squared. It is also **faster** than what
  it replaces, by 1.6× at `N = 200, n = 10` and 4.6× at `N = 500, n = 50`, because the scaled series
  converges in a handful of terms where the unscaled one ground through hundreds.

  Consequences elsewhere. In `test/optimizer_convergence/svd_optim.jl` the worst `check` over eight
  starting points was `2.45e-5` (`_BFGS` + `Backtracking` + `Geodesic`), five orders of magnitude past
  that file's `MANIFOLD_TOLERANCE` of `1e-12`; it is `2.8e-12` now. Every `Geodesic` iteration and
  evaluation count in this package moved by a few percent, since a more accurate exponential is a
  different trajectory — the tables in `svd_optim.jl` and in `default_linesearch`'s docstring are
  re-measured, and `scripts/retraction_accuracy.jl` regenerates them. The `Cayley` columns are
  unchanged to the digit.

  `Geodesic` is now also the *cheaper* of the two retractions for `N ≳ 50`, which reverses what its
  docstring used to say: `cayley` finishes with a product of two `N × N` matrices at `O(N³)` where
  `geodesic` only assembles `I + B'𝔄(X)(B'')ᵀ` at `O(N²n)`. At `N = 1000, n = 20` that is 2.5 ms against
  39 ms.

- **`check` works on a `GrassmannManifold`.** It was defined for `StiefelManifold` only, so the one
  function that measures distance from the manifold — the assertion every manifold test rests on —
  was a `MethodError` for the other of the two manifolds this package provides. It is now
  `check(::Manifold)`, since the representative of a `GrassmannManifold` point satisfies `YᵀY = I`
  just as the Stiefel one does. This is why the accuracy loss above went unnoticed for so long: half
  the retraction paths had nothing that could have caught it.

- **`Optimizer`'s constructors no longer nest three levels of `kwargs...`, which cost Julia 1.12
  fifteen minutes of compile time per specialization.** The test suite ran for 31–42 minutes on 1.12
  across all three CI operating systems, against 3–5 minutes on 1.10, 1.13 and nightly, and
  `test/optimizer_convergence/svd_optim.jl` accounted for the whole difference
  (`Optimizer Convergence | 70 70 16m01.9s`, every other testset normal). It was never the numerical
  work: the identical solves driven from a script take 6.94 s on 1.12 and 7.04 s on 1.13, with the same
  iteration and evaluation counts.

  The trigger is having the `Optimizer` constructor and the `solve!` call in the *same* inferred
  function body — which is what any ordinary `function solve_my_problem(...)` does. Neither half is
  slow alone on 1.12: 0.99 s for the constructor, 2.35 s for `solve!`, 908 s for the two together.
  Inference has to resolve the `Core.kwcall` chain before it knows the constructor's return type, and
  on 1.12 propagating that into `solve!` goes superlinear.

  `Options` is now built once in the outermost method and passed positionally from there, so the
  return type does not depend on which keywords were given, and `Optimizer(x, F; …)` reaches the inner
  constructor through one level of splatting rather than three:

  | construction + solve in one body | 1.13 | 1.12 |
  |---|---|---|
  | before | 4.35 s | **940.86 s** |
  | before, behind a `@noinline` boundary | 4.57 s | 925.27 s |
  | after | 4.15 s | **6.71 s** |

  A 140× improvement on 1.12 and no change elsewhere. The second row is worth noting: a barrier around
  the construction does not help, and neither does `@nospecialize` on the enclosing function — only
  flattening the chain does, which is why the fix is where it is rather than at the call site. The
  regression itself is upstream and already fixed in 1.13 and nightly.

  No API change: every keyword `Optimizer` accepted, it still accepts.
- **A line search that reports it could not decrease the merit no longer gets its step taken.**
  `solver_step!` called `SimpleSolvers.solve`, which returns the step length and nothing else, so
  `LINESEARCH_FLOOR`, `LINESEARCH_EXHAUSTED` and `LINESEARCH_NO_DESCENT` were indistinguishable from a
  successful search. It now calls `solve_with_status`, and on any of those three outcomes restarts the
  inverse Hessian (`restart!`), sets the direction to steepest descent and searches once more. See
  `linesearch_rejected` for why the trigger is the outcome and not `φ > φ₀`.

  This was the `check(Y) = 1e200` divergence recorded in `test/optimizer_convergence/svd_optim.jl` and
  in commit `8673007`. On one of eight starting points, `_BFGS` + `Bisection` + `Geodesic` stopped
  after 4 iterations off the manifold altogether and reported *convergence*: `Bisection` bisects `φ'`,
  so on a non-convex ray it settled on a stationary point that was a maximum, said so
  (`LINESEARCH_FLOOR`, `φ(1) = φ(0)` exactly), and the step was taken anyway. `f` went from 3.38 to
  9.13, the corrupted secant pair gave the next direction `‖δ‖ = 345`, and retracting a lift that
  large left the manifold; `x_converged` then fired because `‖δ‖/‖x‖ ≈ 1e-98` once `‖x‖` was at
  `1e100`. That starting point now converges in 121 iterations at `check(Y) = 6e-14`.

  It also closes a second hole, which had been diagnosed as a tolerance problem. `@test status(result).rg < 1e-5`
  failed on CI at `1.354e-5`, and the assertion was unguaranteed rather than unlucky: over the ten
  (method, line search, retraction) combinations that test runs and eight starting points each,
  `g_converged` is `false` in **all eighty** — every solve terminates on `f_converged`, and `‖∇f‖`
  there ranged over `1.5e-8 .. 1.8e-5`. With the restart in place the worst case over the same eighty
  runs is `2.9e-7`, so `1e-5` has a factor of 35 of headroom and is now a real bound. ([#33])
- **The quasi-Newton update enforces the curvature condition.** The guard was
  `!iszero(ΔxΔg) && !isnan(ΔxΔg)`, which admits a negative `δᵀγ` as readily as a positive one — and
  admits denominators that are zero to within round-off, which it then divides a rank-two correction
  by. On the SVD problem `δᵀγ` took the values `-12.8`, `-4.5e-16` and `+1.5e-15` on consecutive
  iterations; `λmax(Q)` went from 3 to 442 as a result, and `λmin(Q)` to `-398` from another starting
  point. Both BFGS and DFP preserve positive definiteness of `Q` only for `δᵀγ > 0`, so
  `curvature_is_usable` now requires `δᵀγ > 1e-8 ⋅ ‖δ‖‖γ‖`. The threshold is relative because an
  absolute one cannot tell `1.5e-15` on a problem scaled to `1e0` from a legitimate pairing on a
  problem scaled to `1e-15`; its value barely matters, since `1e-8` and `eps(T)` behave identically.

  This is where `_DFP`'s notorious sensitivity to its starting point came from: an ill-conditioned `Q`
  on this problem is a `Q` built from pairs that should have been rejected. With an expanding
  `Backtracking` over eight starting points its iteration count goes from `512..77_890` to `512..845`
  on `Geodesic`, a factor of 92 less spread. It costs `_DFP` a factor of seventeen on Rosenbrock
  (50 iterations to 851, both reaching `f ≈ 3e-24`) because it had been exploiting those invalid
  updates to compensate for its under-scaled direction; `_BFGS` is unaffected at 22 either way.
- **A non-finite iterate stops the solve.** `meets_stopping_criteria` reported `NaN` through an
  `@error` and then carried on, so one starting point spent all 100 000 iterations of a raised cap
  emitting that message once per iteration. The flags are now part of the stopping condition, and none
  of the three convergence flags is set by it, so `isconverged` still distinguishes the two cases.
  `contains_nan` becomes `contains_nonfinite` and tests `isfinite`: `NaN` is the *last* thing a
  diverging solve produces, and the one on the SVD problem passed through `f = 1.2e169` and
  `check(Y) = 1.07e200` — neither of them `NaN` — two iterations before it got there. The
  `OptimizerStatus` fields `x_isnan`, `f_isnan` and `g_isnan` are renamed to `x_nonfinite`,
  `f_nonfinite` and `g_nonfinite` to match.
- **Gradients and directions are paired in the intrinsic coordinates, not the ambient ones.** `dot` on
  an `AbstractLieAlgHorMatrix` is the ambient Frobenius product, which counts each of the lift's
  off-diagonal blocks twice and so comes out *exactly twice* the product of its free parameters — the
  coordinates `Q` is sized by, `outer!` flattens to, and a line search's `α` parameterizes. The new
  `_dot` takes the pairing there instead, at three sites: `trial_slope`, which was returning
  `2φ′(α)`; the `δᵀγ` of the quasi-Newton update, whose value has to be consistent with the flattened
  `T₁`, `T₂` and `γᵀQγ` it divides; and the predicted decrease `Δf̃`, which has to be comparable with
  `Δf`. `_BFGS` on the SVD problem improves from 176 to 113 iterations (`Geodesic`, `Backtracking`)
  and from 197 to 93 (`Cayley`, `Bisection`).

  `trial_slope` remains exact only for `Geodesic`: `Cayley` is not a one-parameter subgroup, so
  `⟨∇f(x(α)), B⟩` is its derivative at `α = 0` and drifts from it with the step (about 6% at
  `α = 0.5`). This is documented on `trial_slope` rather than fixed.
- **`_DFP` keeps its inverse Hessian symmetric.** `Q γγᵀ Q` is symmetric in exact arithmetic, but
  forming it as two separate `mul!`s is not, and the error accumulated: `‖Q - Qᵀ‖/‖Q‖` grew from 8e-16
  after five iterations to 1.6e-11 after twenty thousand, at which point `eigvals(Q)` returned complex
  numbers for a matrix that is by definition symmetric. `BFGSCache` never had this, because it adds
  `T₁ + T₂` with `T₂` built as the exact transpose of `T₁`. Symmetrizing is not what makes `_DFP`
  converge, though — see the analysis in `test/optimizer_convergence/svd_optim.jl`.
- `zeros(SkewSymMatrix, n)` threw a `MethodError` about `zero(::Type{SkewSymMatrix})` — the
  non-parametric method had been dropped — including for its only in-repo caller,
  `zeros(::Type{StiefelLieAlgHorMatrix}, N, n)`. Both are public API.
- `Optimizer(x, problem)` sized its default gradient with `length(x)`, which for a `NamedTuple` is the
  number of *entries* and not the length of its flattening, so the first step failed with a
  `DimensionMismatch`. The choice now lives in `default_gradient`.
- `OptimizerCache` for a method with no matching cache — in practice an `Adam{Float64}` handed
  `Float32` parameters — reported a bare `MethodError`; it now says that `Adam` has to be constructed
  with the element type of the parameters.
- `GrassmannLieAlgHorMatrix` gained explicit `copy`, `zero`, `copyto!`, `fill!` and `similar` methods.
  The generic `AbstractArray` fallbacks route through `setindex!`, which it does not define, and its
  `similar` handed a two-parameter concrete type to `zeros`, which has no such method.

### Added

- **`Geodesic` takes an algorithm for the matrix exponential.** `Geodesic(ScaledSquaring())` is the
  default and is the fix described under *Fixed*; `AugmentedPade()` gets ``\mathfrak{A}`` out of a
  block of `Base.exp`'s Padé approximant, and `ProjectedSkew()` exponentiates the lift in an
  orthonormal basis of its own range, where it is a small skew-symmetric matrix and the exponential
  can be built from an eigendecomposition. All three compute the same map, so the one-parameter
  subgroup property `Geodesic` relies on holds for each.

  They differ in what they trade. `ProjectedSkew` is the only one whose orthogonality is
  *structural* rather than the outcome of a series, so it is the only one whose `check` does not
  degrade with the step: over the eight SVD starting points its worst case is `1.9e-13` against
  `ScaledSquaring`'s `2.8e-12`, and in `Float32` it holds `1.3e-6` where the others reach `2.3e-5`.
  It costs about an order of magnitude on the *typical* case and needs `qr` and `eigen`.
  `ScaledSquaring` is the default because it is the fastest, the most accurate against
  `exp(Matrix(B))`, and — using nothing but matrix products and norms — the only one of the three
  that runs on a `KernelAbstractions` GPU backend.

  `TaylorSeries()` is the pre-0.2.0 behaviour. It is kept, and documented as unusable, so that the
  regression stays reproducible from the test suite rather than only from this file.

  The algorithm reaches every entry point, not just `retraction(::Geodesic, ::AbstractLieAlgHorMatrix)`:
  `geodesic(Y, Δ, algorithm)` takes it as an optional third argument, so the tangent-vector form is
  selectable too rather than being pinned to the default.

- **`Options(store_trace = true)` does something.** `OptimizerResult` gains a `trace` of one
  `OptimizerTraceEntry` — `(iteration, f, rg)` — per iteration, reachable through `trace(result)` and
  empty unless the option asked for it. The option existed before and was accepted and ignored, by
  this package *and* by SimpleSolvers 0.11, where it is a field of `Options` that nothing reads: code
  that set it got neither a trace nor an error.

  The entries come from the `OptimizerStatus` that `solve!` already computes on every iteration, so
  the cost when the option is unset is one `Bool` test per iteration.

  What it is for is a statistic that does not depend on the *phase* of an orbit. `Adam` at a fixed
  learning rate does not converge to the minimizer, it circles it at a distance of order `α`, so the
  error at any single iteration is a sample of an arbitrary phase and moves with the last bits of the
  floating-point arithmetic — across Julia 1.10, 1.12 and 1.13 the final-iterate error on the SVD
  problem spans a factor of 3.0. Averaging over the last five hundred iterations measures the orbit's
  *radius* instead, which is a property of `α` and the problem: the same three versions span 1.06.
  That is what `test/optimizer_convergence/svd_optim.jl` now asserts on, at a tolerance with 1.8× of
  margin above the worst correct value and more than 1000× below the value the known `Adam` bugs
  produce. The tolerance it replaced had a factor of 1.9 to work in altogether.
- **`AdamWithEuclideanDecay`**, `Adam` with the *decoupled* weight decay of Loshchilov and Hutter
  (2019) — the method usually called AdamW. The decay `λx` is applied to the *direction* after the
  moments have been formed, so it never enters `m₁` or `m₂` and is scaled by the line search's `α`,
  exactly as the paper's schedule scales it. It shares `Adam`'s cache and state:
  `OptimizerState(AdamWithEuclideanDecay(), ps)` returns an `AdamState`.

  The name says *Euclidean* because `λx` is the gradient of `λ‖x‖²/2`, which is *constant* on both
  manifolds of this package (`‖Y‖²_F = tr(YᵀY) = n`, both being compact), so its Riemannian gradient
  `rgrad(Y, λY)` vanishes identically. The method therefore decays the ordinary arrays of a
  `NamedTuple` and leaves the manifolds in it alone — the case it exists for, a network with Stiefel
  weights next to unconstrained ones ([#14]) — and on a *bare* manifold it *is* `Adam`, for every `λ`.
  That last case warns at construction rather than being silently ignored. Whether a *Riemannian*
  weight decay should exist alongside it is [#28]; the name `AdamW` is held in reserve for the answer
  and, for now, raises an explanatory error rather than aliasing `Adam`. Derived in the new
  *Weight Decay on Manifolds* documentation page.

  This is **not** the replacement for the `AdamWithDecay` of v0.1.0 (see *Removed*) — that decayed the
  learning rate, and its replacement is `DecayingStatic`. The two are unrelated and compose: passing
  `linesearch = DecayingStatic(…)` decays both the step and, with it, the decay.
- **`OptimizerSolution`**, the union of `AbstractVector`, `Manifold` and `ArrayNamedTuple`, so that a
  single method signature covers all three — together with `named_tuple_wrapper.jl`, which supplies
  the elementwise arithmetic (`_copy`, `_zero`, `_add!`, `_mul`, `norm`) that the Euclidean code
  performs on plain vectors.
- **Heterogeneous and non-`Float64` parameter `NamedTuple`s.** `ArrayTuple` and `GlobalSectionTuple`
  were written as `Tuple{Vararg{AT}} where {AT<:…}`, which Julia's diagonal rule makes homogeneous — a
  `NamedTuple` holding a `StiefelManifold` and an ordinary `Matrix` at the same time was therefore not
  an `ArrayNamedTuple`. They are covariant now.
- `GradientFunction(F, ∇F!, nt::NamedTuple)`, so that a hand-written gradient can be used for
  parameters stored in a `NamedTuple`.
- `GlobalSection` now travels with the optimizer state rather than living inside the cache, with the
  accessors and in-place methods that `solver_step!` needs.
- `default_gradient` and `default_linesearch`, which name the two choices `Optimizer` makes when the
  caller does not supply a gradient or a line search.
- **`DecayingStatic`**, a `LinesearchMethod` that takes no search but whose step decays geometrically
  from `η₁` to `η₂` over `n` iterations, reading the iteration number out of the line search
  parameters. This is the replacement for the `AdamWithDecay` of v0.1.0 (see *Removed*), and being a
  line search it composes with `GradientMethod` and `MomentumMethod` too. It is the weaker of the two
  ways to make `Adam` converge: a geometric schedule is summable and so stops short (`‖∇f‖ ≈ 1e-3`
  once `‖Δx‖ ≈ 1e-13`), where letting a searching line search pick the step gets `Adam` to `≈1e-7`.
- **`AdamOptimizerWithDecay(n_epochs, T; η₁, η₂, kwargs...)`**, a convenience pairing of `Adam` with a
  `DecayingStatic` line search, returned as a `NamedTuple` to splat into `Optimizer`:
  `Optimizer(x, F; AdamOptimizerWithDecay(1000)...)`. It defines no type and no schedule of its own —
  `kwargs...` goes to `Adam`, so `β₁`, `β₂` and `δ` keep `Adam`'s defaults rather than a copy of them.
  It exists so that GeometricMachineLearning's method of the same name — which bundles Adam's `ρ₁`,
  `ρ₂`, `δ` with `η₁`, `η₂`, `n_epochs` and computes the identical
  `γ = exp(log(η₂/η₁)/n_epochs)` — has one name to migrate to now that the direction and the step
  size live in different halves of the optimizer. The name carries over but the call does not always:
  GML takes `T` from `η₁`, whose default is a `Float32` literal, so a `Float32` network needs
  `AdamOptimizerWithDecay(n_epochs, Float32)`; and GML's positional step sizes and its `ρ₁`/`ρ₂`
  spelling are keywords `η₁`, `η₂`, `β₁`, `β₂` here. Note that this decays the *learning rate*, not
  the weights; `AdamWithEuclideanDecay` is the unrelated one, and the *Two unrelated decays* section
  of the weight-decay page sets them side by side.
- **A line search can take its trial step through the retraction.** `linesearch_problem` built its
  merit with `SimpleSolvers.compute_new_iterate!`, i.e. `xₖ + α·pₖ`, which on a manifold is undefined —
  and would be wrong even if it were not, since a step has to go through the retraction and the
  direction is a horizontal lift of a different shape than the point. `Static`, the one method that
  never evaluates the merit, was therefore the only line search that worked on manifold parameters.
  `trial_iterate!` now builds the trial point the way `solver_step!` does, and `trial_slope` pairs the
  gradient with the direction through `global_rep`; both dispatch on the solution type, so the
  `AbstractVector` path keeps its allocation-free `compute_new_iterate!`.

### Known issues

Tracked issues open against this release. For the defects and gaps found *while* building it —
verified, measured, and not fixed here — see [Open Issues](#open-issues) at the end of this file.

- Type piracy in `l2norm`, `ParameterHandling.flatten`, `Gradient` and `outer!`, catalogued in the
  source. `ArrayNamedTuple` and `GlobalSectionNamedTuple` are type *aliases* for `NamedTuple`, so
  dispatching on them is dispatching on Base; a wrapper `struct` would make most of it legal. ([#16])
- Bare `Manifold` parameters are only partially supported ([#27]).
- `mode = :finitediff` throws a `MethodError` on `NamedTuple` parameters ([#24]).
- No documentation page describes the unified optimizer interface yet ([#25]).
- The `|g(x) - g(x')|` convergence *measure* is structurally zero for `Newton`. `solver_step!`
  refreshes `state.ḡ` at the same iterate the cache takes its gradient at, so the difference the
  status reports is always `0`. It is display-only — no stopping criterion reads it — but it is the
  same defect class as the quasi-Newton one fixed above, in the code path marked "this will have to be
  removed later".

## [0.1.0]

Initial release. Ports the manifold optimizer machinery from GeometricMachineLearning and the
Euclidean optimizer machinery from SimpleSolvers into one package.

## Open Issues

Defects and gaps found while working on this release that are **not** fixed by it, kept here so they
are tracked with the code rather than in a scratch file. A is correctness in this package, B its
observability, C its dead code and bookkeeping; D is upstream, E lists things reported during the
investigation that turned out not to be problems, and F the loose ends of the geodesic-retraction
review. Everything was verified directly — where a claim rests on a measurement, the measurement is
given. Entries A5, A6, C6 and D5 come from the review of [#36]; A7, A8, A9 and the second half of A1b
from [#38], and A10 from the review of [#38]; the rest from unifying the optimizer hierarchies.

A1, A2, A3, A7 and most of A9 are fixed and are kept for the record, since later entries refer to
them: A1 and A3 by [#36], A2 by [#38], A7 and A9 by the rejected-step fix in *Fixed* above. C6
describes text that [#36] introduced. A1b is still open, but the fix it proposed for itself has since
been implemented, measured and ruled out — read that entry before starting on it.

Two entries carried measurements that do not reproduce, both corrected in place: A1b's first-order
`check` table and A7's `194 of 200` outcome count. The *conclusion* held in each case and the
supporting figure did not, so treat a number here as reproducible only where the harness that
produced it is named.

This is the detailed catalogue. The short, issue-tracker-facing list is *Known issues* under
[Unreleased](#unreleased--targeting-020) above; the two do not overlap.

Ordered by severity within each section, except that entries found after the first pass are appended
to their section rather than interleaved, so that the labels stay stable references.

---

### A. This package — correctness

#### A1. `Geodesic` silently leaves the manifold for a lift of norm ≳ 50 — fixed in 0.2.0

Retained in full: A1b and A4 refer to it, and the measurement below is the only side-by-side
against `Cayley` on the identical lift, which is what shows the defect was specific to
``\mathfrak{A}`` and not to retraction on a manifold.

**Fixed** by the scaling-and-squaring rewrite of `𝔄`; kept here for the record. See `CHANGELOG.md`
and `src/retractions/exponential_algorithms.jl`. Two notes on what the investigation changed about
the diagnosis below:

- making the termination test relative to the partial sum, which the last paragraph proposes, was
  measured to change **nothing** at any lift norm. The loss is cancellation inside the sum, not the
  point at which the summation stops.
- the scaling and squaring does not need an `N × N` matrix: the low-rank form is closed under
  squaring, `(I + B̂WB̄ᵗ)² = I + B̂(2W + WXW)B̄ᵗ`, so it is a `2n × 2n` recurrence throughout.



**Severity: high.** The most serious thing found, and the amplifier of the divergence PR #35 fixes.

`geodesic(B)` builds the exponential from `𝔄(A) = Σₙ Aⁿ⁻¹/n!`
(`src/retractions/modified_exponential.jl:28-45`). That series is summed directly, with an
**absolute** termination test:

```julia
ε = eps(T)
while norm(Aⁿ) > ε      # `Aⁿ` here is the running term Aⁿ⁻¹/n!
```

For `‖A‖ ≫ 1` the partial sum is enormous and the terms cancel, so stopping when a *term* falls
below `2.2e-16` leaves a relative error of `eps ⋅ ‖𝔄A‖`, not `eps`. Measured on a random
`StiefelLieAlgHorMatrix(20, 3)` scaled up, against `Cayley` on the identical lift:

| `‖B‖` | `check(geodesic(B))` | `check(cayley(B))` |
|---|---|---|
| 6.06 | 2.8e-15 | 1.3e-15 |
| 30.3 | 5.4e-9 | 7.6e-15 |
| 60.6 | **4.31** | 1.7e-14 |
| 90.9 | **7.1e16** | 1.4e-14 |
| 303 | **6.5e134** | 5.9e-14 |
| 606 | **2.9e303** | 1.3e-13 |

At `‖B‖ = 60` the "retracted" point is not on the Stiefel manifold in any sense, and nothing
reports it. `Cayley` is unaffected at any of these norms — it inverts a `2n × 2n` matrix rather than
summing a series — so this is specific to `𝔄`, not to retraction on a manifold.

Two consequences already observed in the test suite:

- `_BFGS` + `Backtracking` + `Geodesic` reaches `check(Y) = 2.45e-5` on one of the eight starting
  points of `test/optimizer_convergence/svd_optim.jl`, against a `MANIFOLD_TOLERANCE` of `1e-12`.
  That starting point would fail the existing manifold assertion; the test passes only because it
  uses seed `1234`.
- It is why the divergence PR #35 fixes ended at `check(Y) = 1.07e200` rather than merely somewhere
  wrong. The bad step had `‖δ‖ = 345`; the table above says a `Geodesic` retraction of a lift that
  size produces garbage of roughly that magnitude. The line-search fix removes the cause of the
  large step, but the retraction will still do this to any large lift from any other cause.

Scaling and squaring is the standard remedy for a matrix exponential in this regime, and the
termination test should be relative to the partial sum rather than absolute regardless.

#### A1b. `Quadratic` and `BierlaireQuadratic` diverge on `Cayley` on two of eight starting points

**Severity: high.** Found while re-measuring the tables after fixing A1, and *not* caused by that fix
— `cayley` is bitwise unchanged by it (every `Cayley` figure in `svd_optim.jl` reproduces to the
digit).

**Still open, and the diagnosis below has been narrowed by ruling its own proposed fix out.** The
entry originally read "the honest fix is probably an exact `Cayley` differential in `trial_slope`".
That differential now exists (`retraction_differential`, see *Fixed* above) and is verified exact
against a central difference of the merit at every ``\alpha``; it closes **one of the four** cases
below and leaves three. So the slope error was real — 8.9% at ``\alpha = 0.5``, 143% at
``\alpha = 2`` — and was worth fixing on its own account, but it is not what this entry is about.
Anyone picking this up should not spend the effort a second time.

On the SVD problem of `test/optimizer_convergence/svd_optim.jl`, `_BFGS` with either polynomial line
search on the `Cayley` retraction runs to the iteration cap and ends off the manifold. Before the
differential, and after it:

| | seed | before | after |
|---|---|---|---|
| `Quadratic` | 2 | 20 000 (cap), `check` **5.0e-2** | **90 iterations, `check` 6.4e-15** |
| `Quadratic` | 8 | 20 000 (cap), `check` **1.7e-1** | 20 000 (cap), `check` **4.0e-1** |
| `BierlaireQuadratic` | 2 | 20 000 (cap), `check` **6.9e-2** | 20 000 (cap), `check` **6.9e-2** |
| `BierlaireQuadratic` | 8 | 20 000 (cap), `check` **2.0e-1** | 20 000 (cap), `check` **1.3e-1** |

The other six seeds converge in 90–159 iterations, and every one of these cases converges on
`Geodesic`. The `BierlaireQuadratic` rows are *bit-identical* across the differential, which is
itself informative: on this problem that search never asks for ``\varphi'`` away from
``\alpha = 0``, where the old and new slopes agree exactly. So for half these cases the slope was
never the mechanism, and it could not have been.

**Where the cause actually is: the line search returns an absurd ``\alpha``, and on a compact
manifold nothing tells it not to.** Tracing seed 8 under `Cayley` + `Quadratic` step by step, the
solve wanders rather than converging — `f` oscillates between 3 and 10 — and at iteration 8 the step
actually taken is

| iteration | 1 | 2 | … | 7 | 8 |
|---|---|---|---|---|---|
| ``\|\alpha\delta\|`` | 0.61 | 1.67 | … | 1.71 | **1.85e9** |
| ``\lambda_\mathrm{max}(Q)`` | 1.0 | 2.0 | … | 10.2 | **38.6** |
| `check` | 2.0e-14 | 3.4e-14 | … | 3.1e-14 | **9.6e-7** |

``Q`` is *well conditioned* at the step that does the damage — ``\lambda_\mathrm{max} = 38.6``, and it
had been restarted twice before this. The magnitude is entirely the line search's ``\alpha``, which
the two polynomial fits extrapolate to ``10^9``. **So damping the quasi-Newton update is not the
remedy**; that was the first hypothesis after the slope one, and the measurement above rules it out
as well.

What makes such an ``\alpha`` acceptable to a line search is the geometry, and it is specific to a
*compact* manifold. ``\varphi(\alpha) = f(\Lambda\mathrm{retract}(\alpha\bar{B})E)`` is **bounded**
in ``\alpha``: `Cayley` converges to a fixed rotation as ``\alpha \to \infty`` and `Geodesic` is
periodic in it. Measured on `St(10, 3)` against a random target:

| ``\alpha`` | 0 | 1 | 1e2 | 1e4 | 1e6 | 1e9 |
|---|---|---|---|---|---|---|
| ``\varphi``, `Cayley` | 4.8499 | 4.9648 | 5.4758 | 5.4861 | 5.4862 | 5.4862 |
| ``\varphi``, `Geodesic` | 4.8499 | 5.1399 | 5.0538 | 5.1601 | 4.9046 | **4.7299** |
| Euclidean ``f(x + \alpha{}p)`` | 1.1e1 | 1.1e1 | 3.3e4 | 3.4e8 | 3.4e12 | 3.4e18 |

The `Geodesic` row at ``\alpha = 10^9`` is *lower* than at ``\alpha = 0``, so a search that finds it
has found a genuine decrease and is right to report one. In the Euclidean row the same ``\alpha``
would be rejected by the merit alone, which is why no line search carries a guard against it. On a
manifold the merit gives the search no signal at all, and the only thing wrong with the step is that
retracting a lift of that norm loses the manifold.

That last part is *not* a defect in either retraction. `check` after a retraction grows like
``\varepsilon\|\bar{B}\|`` for both, which is conditioning and not a summation that cancels. On
`St(20, 3)`:

| ``\|\bar{B}\|`` | 1e2 | 1e4 | 1e6 | 1e7 | 1e8 | 1e9 |
|---|---|---|---|---|---|---|
| `Cayley` | 5.3e-14 | 3.0e-12 | 4.2e-10 | 1.1e-8 | 9.3e-8 | 7.7e-7 |
| `Geodesic` | 2.4e-14 | 1.7e-12 | 1.5e-10 | 2.7e-9 | 6.1e-9 | 3.7e-8 |

Both degrade; `Cayley` is about three to fifteen times worse across the range. That factor is the
whole of the `Geodesic`/`Cayley` split this entry opens with — `Geodesic` survives the same wild step
with a `check` an order of magnitude smaller, stays inside what the solve can recover from, and
converges. So the retraction is the *amplifier*, exactly as in A1, and the cause is upstream of it:
the quasi-Newton direction and the step the polynomial search accepts for it.

**What the remedy has to be.** A bound on the *step*, ``\|\alpha\delta\|``, and not on the
direction, on ``Q``, or on the outcome the search reports — none of which is wrong at the step that
does the damage. The scale is supplied by the geometry rather than invented: the retraction of a lift
is a rotation, so beyond ``\|\alpha\bar{B}\| \approx 2\pi`` a larger step adds nothing but the
round-off in the table above. A cap of a few multiples of that would leave every solve in this
package untouched — the largest step in a converging run measured here is ``5.5`` — while making the
``10^9`` one impossible.

It is still a behaviour change on every manifold solve and it needs its own measurement of every table
in the package, so it wants its own PR. It is *not* the same defect as A4 or A7, which this entry
previously grouped it with: A7 was the step of a *rejected* search being taken, and every search
involved here reports `LINESEARCH_DECREASED` or `LINESEARCH_FLOOR` on a step that genuinely decreases
the merit. Nothing is being overruled; the merit is simply not a bound on the step.

Coverage: `svd_optim.jl` now runs both polynomial searches in its convergence loop, but only on seed
`1234`, where they pass — so the loop covers the searches and does **not** guard this. Reproducing it
needs the eight-seed sweep of `scripts/retraction_accuracy.jl`.

**It is not specific to `_BFGS`.** On the same problem at seed `1234` under `Cayley`, 1 000
iterations, `check` at the end, before the differential and after:

| | `Backtracking(expand)` | `Bisection` | `Quadratic` | `BierlaireQuadratic` |
|---|---|---|---|---|
| `GradientMethod` | `5.8e-14` → `5.8e-14` | `7.1e-14` → `6.7e-14` | **`1.8e-3`** → **`7.1e-4`** | **`4.8e-3`** → **`4.8e-3`** |
| `MomentumMethod` | `6.0e-14` → `6.0e-14` | `5.7e-14` → `4.6e-14` | **`1.4e-3`** → **`5.8e-4`** | **`4.9e-3`** → **`4.9e-3`** |
| `Adam` | `8.5e-14` → `8.5e-14` | `8.7e-14` → `9.2e-14` | **`4.2e-3`** → **`9.7e-3`** | **`7.6e-4`** → **`7.6e-4`** |

Same split, six more cases, and the differential improves the `Quadratic` column by about 2.5× for
the two gradient methods while leaving `BierlaireQuadratic` untouched to the digit — none of them
anywhere near round-off either way. The `Backtracking` column is bit-identical, which is the control:
that search evaluates ``\varphi'`` at ``\alpha = 0`` only.

(The "before" columns here are a re-measurement on `main`. They do not match the numbers this entry
carried when it was written — `1.7e-14` / `5.6e-4` / `1.9e-3` for `GradientMethod` — which do not
reproduce from the harness described. The split they document is real and is what the re-measurement
shows; the individual figures were not. Take the table above as the record.)

(Context, and not itself a defect: no first-order method converges on this problem within 1 000
iterations. `Backtracking(expand)` and `Bisection` reach the optimal value — `err = 1.7097`, the same
one `_BFGS` converges to — but stop at `rg` between `4e-5` and `2e-3`, several orders off the gate.
That is what a first-order method on a badly conditioned problem does; it is why
`manifold_linesearch_tests.jl` uses a two-sphere problem for its first-order coverage.)

#### A2. `GradientMethod` and `MomentumMethod` throw on their own default line search — fixed in 0.2.0

**Fixed** by [#38]: `gradient` is defined for all three first-order caches, and `trial_slope` writes
into `latest_gradient` — a scratch array on those caches rather than an alias for `cache.g`. Three
notes on what the fix turned up that the diagnosis below did not anticipate:

- **defining the accessor is not enough.** `trial_slope`'s vector branch evaluates *into* the cache,
  and `update!(::MomentumState, …)` re-runs `p ← αp + ∇f(xₖ)` from the same array afterwards, so
  aliasing `gradient(cache)` onto `cache.g` made the momentum accumulate the gradient at whatever
  trial step the search last probed — 104% wrong under `Bisection`, `Quadratic` and `StrongWolfe`,
  451% under `BierlaireQuadratic`. `Backtracking` was exact only because it evaluates `φ'` once, at
  `α = 0`.
- **`rg` was one iteration stale, and that stopped these methods early.** It was `‖∇f(xₖ)‖` at the
  iterate the step started from — harmless under `Static`, and not harmless under a direction with
  momentum in it, where a line search accurate enough to drive `∇f(x₁) ≈ 0` made `g_converged` fire
  while the momentum still moved the iterate. That left `MomentumMethod` at `‖x‖ = 0.35` and `Adam`
  at `‖x‖ = 1.16` on the bracketing searches, both reporting convergence. `solver_step!` now
  refreshes the gradient at the accepted iterate. See the CHANGELOG entries above for the tables.
- **the refresh is free, but only because the value is reused.** `∇f` at the accepted iterate, in the
  frame the cache is in, is bit-for-bit the gradient the *next* `update!(cache, …)` recomputes, so
  refreshing without reusing simply doubles the gradient evaluations of a first-order step — 1.35× on
  `Adam` + `Static`. `store_gradient!` reuses it under a guard that compares the iterate and the
  frame, which brings that to 1.03×. Fixing `rgₐ` fell out of the same pairing; the half that did
  not is A10.

**Severity: high** (a documented entry point fails outright), **pre-existing on `main`**.

```julia
julia> Optimizer(ones(3), x -> sum(x.^2); algorithm = GradientMethod())
ERROR: MethodError: no method matching gradient(::GeometricOptimizers.GradientCache{…})
```

`_trial_slope`'s `AbstractVector` branch (`src/optimizers/linesearch_problem.jl:89-92`) calls
`gradient(cache)`. Only `BFGSCache` (`bfgs_cache.jl:50`), `DFPCache` and `NewtonOptimizerCache`
(`newton_optimizer_cache.jl:68`) define it; `GradientCache`, `MomentumCache` and `AdamCache` expose
the same field as `gradient_array` instead (`gradient_optimizer.jl:50`). So the three first-order
methods cannot use *any* line search that evaluates `φ'` on Euclidean vector parameters.

This became reachable by default in 0.2.0: `default_linesearch(::Type{T}, ::OptimizerMethod)` returns
`Backtracking(T; expand = true)`, and `GradientMethod`/`MomentumMethod` are not in `AdamFamily`, so
they get the searching default. Reproduced on `main` at `b2c20f0`, so PR #35 did not introduce it.

Note the reason the test suite does not catch it: `test/optimizer_tests.jl:95` checks only that
`default_linesearch` *returns* the right type for these methods, and the loops that actually solve
(`optimizer_tests.jl:30-33`, `descent_direction_tests.jl:14-16`) cover `Newton`, `_BFGS` and `_DFP`
only.

#### A3. `check(::GrassmannManifold)` does not exist — fixed in 0.2.0

**Fixed**: `check` is now generic over `Manifold`, in `src/manifolds/abstract_manifold.jl`.


**Severity: medium.** `check` is defined only for `StiefelManifold`
(`src/manifolds/stiefel_manifold.jl:85-87`), so the one function that measures distance from the
manifold — the assertion every manifold test rests on, and the thing that would have caught A1 — is
a `MethodError` for the other of the two manifolds this package provides. There is no generic
`check(::Manifold)` either.

The representative of a `GrassmannManifold` point satisfies `YᵗY = I` just as the Stiefel one does,
so the same expression is correct for it.

#### A4. `x_converged` cannot be trusted on a solve that has diverged

**Severity: medium** (a wrong "converged" report), **documented rather than fixed in PR #35**.

`rxᵣ = rxₐ / l2norm(cache.x)` (`src/optimizers/optimizer_status.jl:76`) measures "the iterate stopped
moving" only while `‖x‖` is bounded. On the diverging seed the step that left the manifold had
`‖δ‖ = 345` and the iterate was at `1e100`, so the relative change was `≈ 1e-98`, far under
`x_reltol = 2eps`, and `x_converged` fired. Confirmed directly: that run reported
`x_conv = true, f_conv = false, g_conv = false`.

PR #35 removes the cause and widens the non-finite check, so nothing measured still reaches this. It
is left open because every way of closing it inside `convergence_measures` needs a threshold on
`‖x‖` or on `‖x - x'‖` that no property of the problem supplies, and imposing one changes the
stopping behaviour of every Euclidean solve to guard a state that is now unreachable. It is
documented as a warning on `convergence_measures`.

**Re-checked against the two divergences that were still live**, since both are exactly the shape this
entry describes — an iterate running to `1e77`..`1e208` while the step stays large. Neither reaches
it. `_BFGS` + `Quadratic` + `Cayley` on seed 8 of the SVD problem, and `MomentumMethod(0.1)` on
Rosenbrock under each of the three line searches it used to diverge on, all report
`x_converged = false`, `f_converged = false` and `g_converged = false`; they stop on the iteration cap
and on `contains_nonfinite` respectively, so `isconverged` is `false` for all of them. A4 therefore
remains a hole that no measured solve falls into, and A7's fix removes three more of the candidates.
It stays open on the same terms, and it is *not* the same defect as A1b or A7: those are about which
step gets taken, this one only about how the stop is reported.

On a manifold this is not actually ambiguous: `‖Y‖_F = √n` exactly, so the denominator is a
*constant* and a large one is itself the divergence signal. The information is available; the
function just does not have it.

#### A5. `geodesic(Y, Δ)` and `cayley(Y, Δ)` are not deterministic and consume global RNG state

**Severity: medium.** Found in the PR #36 review, when an equality assertion between two calls of the
same retraction failed. Pre-existing on `main`; PR #36 neither causes nor fixes it.

Both tangent-vector entry points open with `GlobalSection(Y)`, and `global_section(::StiefelManifold)`
(`src/manifolds/stiefel_manifold.jl:126-133`) completes `Y` with a **random** basis of its orthogonal
complement:

```julia
A = KernelAbstractions.allocate(backend, T, N, N - n)
randn!(A)
A = A - Y.A * (Y.A' * A)
typeof(Y.A)(qr!(A).Q)
```

Two consequences, both measured on `St(20, 3)`:

- **The same call twice gives different answers.** Consecutive `geodesic(Y, Δ)` differ by `1.2e-14`
  in Frobenius norm. That is round-off, not a wrong answer — the retracted point genuinely does not
  depend on the section, and *that* is what the difference measures — but it means no test can assert
  equality between two retractions of the same input, only `isapprox`. Anything downstream that
  hashes, caches or bitwise-compares a retracted point is unsound.
- **It perturbs the global RNG stream.** After `Random.seed!(11)`, taking one retraction changes the
  next `rand()`. So a seeded run is reproducible only if the number of retractions taken is also
  fixed — which it is not, under any line search that adapts its trial count. Verified directly:
  `seed!(11); geodesic(Y, Δ); rand()` ≠ `seed!(11); rand()`.

The optimizer path is mostly insulated, because the section is built once at initialization and
thereafter parallel-transported by `update_section!`. The exposure is the direct `geodesic(Y, Δ)` /
`cayley(Y, Δ)` API and anything built on it.

A section is not unique, so drawing one is legitimate; taking it from the *global* RNG without the
caller being able to see or supply it is the problem. An `rng` argument threaded through
`GlobalSection`, or a deterministic completion (a Householder completion of `Y`, which needs no
randomness at all), would close it.

#### A6. `ScaledSquaring` takes about twice the squarings it needs

**Severity: low**, and a refinement rather than a defect — from the PR #36 review, where it was
deliberately left alone because changing it invalidates every table in that PR.

`ScaledSquaring`'s own docstring states the cause: `X = (B'')ᵀB'` has `‖X‖ ≈ ‖B̄‖²/4` while its
spectral radius is only `≈ ‖B̄‖`, because the eigenvalues of `X` are the nonzero (purely imaginary)
eigenvalues of the skew `B̄`. The halving count is taken from the norm, `s = ⌈log₂(‖X‖₁/θ)⌉`, so
`s ≈ 2log₂‖B̄‖` where `log₂‖B̄‖` would do.

Each squaring is an error amplification, so this costs both time and accuracy. Measured forward error
is `1e-14` in `Float64`, which is why it is not urgent — but `Float32` `check` reaches `2.3e-5` over
the sweep, and halving `s` is the obvious lever if that ever becomes the constraint.

The constraint on any replacement bound is that `ScaledSquaring` is the default *because* it is free
of scalar indexing and dense LAPACK, which is what lets it run on a GPU backend (see the `opnorm₁`
docstring). That rules out reaching for the spectral radius directly — an eigenvalue computation
would give the tighter bound and forfeit the reason the algorithm was chosen.

#### A7. `MomentumMethod` runs away when its direction ascends, and nothing stops it — fixed in 0.2.0

**Fixed**: `linesearch_rejected` now applies to `Adam` and `MomentumMethod` as well, so a rejected
search no longer gets its `α = 1` step taken. See the *Fixed* entry above for the tables. Two
corrections to the diagnosis below, both found while fixing it, and the second changes what the
remedy had to be:

- **the `194 of 200` figure does not reproduce.** Re-measured on the same problem, the same method
  and the same line search, `LINESEARCH_NO_DESCENT` is reported on **3 of the first 200** iterations
  and 13 of all 457. The direction does *not* ascend on almost every step; the searches
  overwhelmingly report `LINESEARCH_DECREASED` and the solve genuinely descends, reaching
  `f = 7.8e-5` by iteration 400. What the thirteen ascent steps do is multiply `f` by between `1e3`
  and `1e42` *each*, and the descending steps in between cannot make that back. The conclusion this
  entry draws is right; the picture of how it happens was not.
- **neither guard it proposes would have worked.** The first, "a bound on `‖p‖` relative to `‖∇f‖`",
  never fires: that ratio stays between `0.57` and `6.8` over the whole run, because it is the
  *gradient* that explodes and the momentum follows it. The second, "a count of consecutive
  `LINESEARCH_NO_DESCENT` outcomes", never fires either — the thirteen events are isolated, never
  consecutive. What was needed was neither: it is that a rejected search returns `α = 1`, so the
  question was never how many times the direction ascends but whether the *step* is taken when it
  does.

  Nor is the ascent outcome the whole of it. Under `BierlaireQuadratic` the same solve reaches `Inf`
  in six iterations through `LINESEARCH_EXHAUSTED`, also at `α = 1` every time and also multiplying
  `f` by ≈`1e21` a step. Acting on `LINESEARCH_NO_DESCENT` alone leaves that case diverging, which is
  why the fix is the whole of `linesearch_rejected`.

**Severity: high.** Found in [#38], while giving the first-order methods their first coverage on a
badly conditioned problem. Pre-existing; [#38] neither causes nor fixes it, but it made the affected
path the default.

On Rosenbrock from `(-1.2, 1)` with `MomentumMethod(0.1)`, the iterate reaches `Inf` rather than the
minimiser:

| line search | iterations | `f` | `‖x‖` at the end |
|---|---|---|---|
| `Backtracking` | 379 | **`Inf`** | `3.0e200` |
| `Backtracking(expand)` — the default | 457 | **`Inf`** | `8.7e77` |
| `BierlaireQuadratic` | 6 | **`Inf`** | `2.0e208` |
| `Bisection` | 636 | `3.3e-17` | ✓ |
| `Quadratic` | 577 | `2.3e-16` | ✓ |
| `StrongWolfe(c₂ = 0.1)` | 3 136 | `2.7e-16` | ✓ |

The mechanism, counted directly over the first 200 iterations of the `Backtracking(expand)` run:
**194 line searches report `LINESEARCH_NO_DESCENT`**, 4 report `LINESEARCH_DECREASED`. The direction
ascends almost every step, the search says so, and the step is taken anyway — `solver_step!` excludes
`FirstOrderMethodWithState` from both `ensure_descent!` and the `linesearch_rejected` restart. That
exclusion is right in principle, since a moving average is deliberately allowed not to descend on an
*individual* step, but nothing tells that apart from failing on 97% of them. `allow_f_increases`
defaults to `true`, so the outer loop does not stop it either, and `contains_nonfinite` only catches
it once the iterate is already past `1e200`.

`MomentumMethod` is the only method exposed this way, and the comparison is what shows it:

- `GradientMethod` is *not* in `FirstOrderMethodWithState`, so it keeps `ensure_descent!`, and its
  direction is `-∇f` and always descends. It never diverges under any of the seven searches — it is
  merely slow (10 000 iterations to `f = 6.3e-7`).
- `Adam` *is* excluded, and does not diverge either, because its direction is normalised to magnitude
  ≈1 per component. It converges under five of the seven.
- `MomentumMethod`'s direction is `-(αp + ∇f)`, unnormalised **and** unguarded, so an ascending step
  raises `‖∇f‖`, which raises `‖p‖`, which lengthens the next ascending step.

(`Static(0.1)` diverges here for `GradientMethod` too, in five iterations. That is an over-large fixed
step on a badly scaled problem and a different matter; the entry is about the searching line searches,
where the search correctly reports that no step decreases the merit and is overruled.)

Two candidate guards, neither costing an evaluation: a bound on `‖p‖` relative to `‖∇f‖`, or a count
of consecutive `LINESEARCH_NO_DESCENT` outcomes — which is exactly the counter B1 asks for, so the
two are worth doing together.

#### A8. `rg` for the (quasi-)Newton methods is whichever point the line search last probed

**Severity: medium**, and it errs in the safe direction. [#38] fixed this for the three first-order
caches and deliberately left the (quasi-)Newton ones alone, so that every table in
`default_linesearch`'s docstring and in `svd_optim.jl` stays bit-identical.

`latest_gradient` aliases `cache.g` for `NewtonOptimizerCache`, `BFGSCache` and `DFPCache`, and
`trial_slope` evaluates into it, so what `rg` reports depends on where the line search last evaluated
``\varphi'``. On Rosenbrock from `(-1.2, 1)`, `rg` against `‖∇f(x)‖` at the point the solve actually
returns:

| | `Backtracking(expand)` | `Bisection` | `Quadratic` | `StrongWolfe` |
|---|---|---|---|---|
| `Newton` | `5.0e-14` vs `0` | 1.0× | 1.0× | 1.0× |
| `_BFGS` | **5.8e4×** | 1.0× | 1.0× | 1.0× |
| `_DFP` | **299×** | 1.0× | 1.0× | 1.0× |

The three bracketing searches come out exact because their last ``\varphi'`` evaluation happens to
land at the accepted ``\alpha``; `Backtracking` — the default — evaluates ``\varphi'`` only at
``\alpha = 0``, so its `rg` is `‖∇f(xₖ)‖` at the iterate the step started from. That `rg` therefore
means a different point depending on which line search ran, and the same number is not comparable
across the rows of any of this package's tables.

It overestimates near a minimiser, so `g_converged` fires *late* rather than early — the opposite of
what the same staleness did to `MomentumMethod` and `Adam` before [#38], and the reason this is
medium and not high. The fix is the one line the first-order caches already use,
`refresh_latest_gradient!`; the cost is one gradient evaluation per iteration and a re-measurement of
every iteration count in the package, which is why it wants its own PR.

#### A9. A searching line search does not give `Adam` a criterion it can meet — mostly fixed in 0.2.0

**Mostly fixed**, and by A7's fix rather than by anything aimed at this. The case below — `Adam` +
`BierlaireQuadratic` exhausting all 1 000 iterations under both retractions while sitting `6.8e-6`
from the minimiser — was the `α = 1` step of a rejected search being taken, i.e. the same defect as
A7. All fourteen (line search, retraction) combinations now terminate on a criterion, in 251–331
iterations, and the worst distance over them is `5.0e-7` — attained by `Static`, i.e. by the orbit
this entry is really about, with every *searching* line search at `1.8e-8` or better.

What survives is the cost argument, which is the part `default_linesearch` rests on: `Adam` still
needs 251–331 iterations on that problem where `GradientMethod` and `MomentumMethod` need 9–64. So
`Static` remains the default for `AdamFamily`, and the sentence this entry asks for in `Adam`'s
docstring is still worth writing.

**Severity: low**, and a sharpening of something already documented rather than a new defect. From
[#38], where it is why the `Adam` testset in `manifold_linesearch_tests.jl` asserts less than the one
next to it.

`Adam`'s direction has magnitude ≈1 per component whatever the gradient is, so with a step that does
not shrink it circles the minimiser at that distance rather than settling on it. `DecayingStatic`
exists for this, and its docstring and testset record it for `Static`. What [#38] adds is that a
*searching* line search does not fix it either: on the two-sphere `NamedTuple` problem of
`manifold_linesearch_tests.jl`, `Adam` + `BierlaireQuadratic` runs out all 1 000 iterations under
both retractions while sitting `6.8e-6` (`Geodesic`) and `5.4e-7` (`Cayley`) from the minimiser — at
the answer, with no criterion it can meet. The other six searches terminate, but take 267–855
iterations where `GradientMethod` and `MomentumMethod` take 9–64.

So `default_linesearch`'s choice of `Static` for `AdamFamily` is not merely the cheaper one, and
`DecayingStatic` remains the only setting under which `Adam` terminates on a criterion by
construction. Worth a sentence in `Adam`'s docstring, which currently sends the reader to
`default_linesearch` for the cost argument and not for this one.

#### A10. `state.ḡ` is two iterates behind for the three first-order states

**Severity: low** — everything it still reaches is reported and not acted on. Found in the review of
[#38], which fixes the half of it that had become visible and leaves the rest. **Pre-existing on
`main`.**

`GradientState`, `MomentumState` and `AdamState` are advanced by `update!(state, opt, x)`, which runs
*after* the step. It writes the *post*-step iterate into `state.x` and the cache's *pre*-step gradient
into `state.g`, shifting the one before that into `state.ḡ`:

```
after update! at the end of step k:   state.x = xₖ   state.g = ∇f(xₖ₋₁)   state.ḡ = ∇f(xₖ₋₂)
```

So `state.g` does not belong to `state.x`, and `state.ḡ` is two iterates behind `cache.g` rather than
one. The quasi-Newton states do not have this: `update!(::BFGSCache, …)` advances `state.ḡ` itself,
inside the step, right after forming `γ` from it.

Two consumers, both in `OptimizerStatus`:

- `rgₐ = ‖cache.Δg‖`. **Fixed in [#38]**: the first-order caches now override `gradient_difference!`
  and take `latest_gradient - gradient`, which is the successive difference the status prints and
  needs no `state.ḡ`. On `f(x) = Σ(x² + 0.1x⁴)` from `[1.5, -0.8, 0.4]` with `MomentumMethod` +
  `Bisection` the old value was `4.976` at iteration three where the successive difference is
  `0.295`; on iteration one it differenced against the `_similar` memory these states never write,
  which is the same defect `test/optimizer_state_initialization.jl` exists to catch for the `Adam`
  moments.
- `Δf̃ = ⟨state.ḡ, δ⟩` (`optimizer_status.jl:82`), the first-order predicted decrease. **Not fixed.**
  It is a two-step-stale gradient paired with the current direction, so the prediction it makes is
  not one. Its only reader is `f_converged_strong`, which C1 records as computed and discarded — so
  whichever way C1 goes, this has to be settled with it, and settling it separately would be
  measuring a number nothing looks at.

The honest fix is upstream of both: `update!(state, opt, x)` should store the gradient that belongs
to the `x` it is storing. `latest_gradient` is exactly that gradient and is already in the cache. The
obstacle is that the same call site feeds the momentum recursion `p ← αp + ∇f(xₖ)`, which needs the
*pre*-step gradient and must keep getting `gradient_array(cache)` — so the two uses have to be
separated first, and `update!(::MomentumState, …)`'s argument list says they currently are not.

---

### B. This package — observability

#### B1. A line search failure is invisible in the returned status

PR #35 makes `solver_step!` act on `LINESEARCH_FLOOR` / `LINESEARCH_EXHAUSTED` /
`LINESEARCH_NO_DESCENT`, but nothing records that it happened. `OptimizerStatus` has no field for it
(verified: no reference to the outcome in `optimizer_status.jl`), so a solve that needed a
quasi-Newton restart on half its iterations is indistinguishable, in the object the caller gets,
from one that never needed one. Only a `verbosity ≥ 2` log message with `maxlog = 1` shows it.

A7 is no longer the argument for this that the entry claimed. That divergence was 13 rejected outcomes
over 457 iterations rather than 194 over 200, and it is fixed by *acting* on them rather than by
counting them — so the counter is not a guard against anything, it is what would have made the
thirteen visible. It is still worth having on its own terms: after A7's fix, a solve that needed a
steepest-descent substitution on a quarter of its iterations is *still* indistinguishable, in the
object the caller gets, from one that never needed one. `_BFGS` + `Quadratic` + `Cayley` on seed 8 of
the SVD problem reports `LINESEARCH_FLOOR` on 4 780 of its 20 000 iterations and says nothing about
any of them.

#### B2. `show_trace` and `extended_trace` are still accepted and ignored

PR #35 implements `store_trace`. The other two remain dead in this package *and* upstream (verified:
no reads of either in `SimpleSolvers/src/` outside `options.jl`'s struct, constructor and `show`).
Setting them gets neither output nor an error. Both are now cheap, since the per-iteration record
exists.

---

### C. This package — dead code and bookkeeping

#### C1. `f_converged_strong` is computed and thrown away

`convergence_measures` computes it at `src/optimizers/optimizer_status.jl:194` and returns it at
`:201`; the `OptimizerStatus` constructor destructures it at `:100` and never stores it. Nothing else
in the package mentions it. It is `Δf ≤ f_mindec ⋅ Δf̃`, i.e. an Armijo-style sufficient-decrease test
on the *outer* iteration, so it plausibly belongs with the stall detection that `Options.max_stalls`
and `Options.f_stall_window` were meant to drive — both of which are also unread here.

Whichever way this goes, it has to be settled together with A10: `Δf̃` is its only input, and for the
three first-order methods that is a two-step-stale gradient paired with the current direction. Using
`f_converged_strong` without fixing A10 would be acting on a prediction that is not one; deleting it
retires A10's second consumer along with it.

#### C2. `compute_direction!` for the quasi-Newton methods is dead

`src/optimizers/iterative_hessians/iterative_hessians_direction.jl:1-3` defines
`compute_direction!(opt, ::Union{BFGSState,DFPState})`. There is no call site (verified: the only
live callers are the `Newton` methods in `newton_optimizer_direction.jl`). The direction is formed
inline at the end of the cache `update!` instead — `bfgs_cache.jl:153`, `dfp_cache.jl:121`.

#### C3. `Random` and `LinearAlgebra` have no `[compat]` entries

`Project.toml` gives `Printf = "1"` but nothing for `Random` or `LinearAlgebra`, though all three are
stdlibs listed in `[deps]` and used the same way.

#### C4. The 0.2.0 release notes are not closed out

`Project.toml` says `version = "0.2.0"` while `CHANGELOG.md:9` still reads
`## [Unreleased] — targeting 0.2.0`, and the compare link at the bottom is `v0.1.0...main`.

#### C5. `_DFP` + `Backtracking(expand = true)` is documented rather than run, on stale grounds

`test/optimizer_convergence/svd_optim.jl` excludes that pair because its iteration count ranged
`512..77_890` over eight starting points. The curvature condition in PR #35 brings that to
`387..845` (`Geodesic`) and `466..1_366` (`Cayley`), comfortably inside the 5 000 cap the file
already uses, so the stated reason no longer holds. Left out only because there is no CI measurement
of the post-fix spread yet — the original surprise was a factor of four between platforms.

#### C6. Two figures in the 0.2.0 notes do not match the measurements they describe

From the PR #36 review. Four other mis-stated numbers in that PR were corrected in it; these two
were found afterwards, while re-running `scripts/retraction_accuracy.jl`, and are still wrong.

- **`ProjectedSkew` does not cost "about an order of magnitude".** The `Added` entry that [#36]
  writes for the exponential algorithms says it "costs about an order of magnitude on the *typical*
  case". Measured against `ScaledSquaring`, one retraction, minimum of 50, one BLAS thread:

  | `N`, `n` | 10, 2 | 20, 3 | 50, 5 | 100, 5 | 200, 10 | 500, 10 | 500, 50 | 1000, 20 |
  |---|---|---|---|---|---|---|---|---|
  | `ScaledSquaring` | 0.003 | 0.005 | 0.016 | 0.023 | 0.095 | 0.443 | 3.100 | 2.586 |
  | `ProjectedSkew` | 0.005 | 0.009 | 0.023 | 0.033 | 0.137 | 0.530 | 4.142 | 3.044 |
  | ratio | 1.7 | 1.8 | 1.4 | 1.4 | 1.4 | 1.2 | 1.3 | 1.2 |

  So 1.2×–1.8×, never above two. The algorithm's own docstring in
  `src/retractions/exponential_algorithms.jl` does *not* make the claim — it says only "a QR plus an
  eigendecomposition instead of matrix products", which is right — so this is the CHANGELOG line
  alone, and it understates how attractive `ProjectedSkew` is. A one-line fix.

- **`θ = 0.5` is not "the measured middle of that range".** `ScaledSquaring`'s docstring says so.
  Over `θ ∈ {0.125, 0.25, 0.5, 1, 2, 4}` at `‖B̄‖ = 155` the measured `check` is
  `9.94e-15, 1.76e-14, 2.82e-14, 2.11e-14, 1.18e-14, 4.97e-14` — `0.5` is the third largest of six,
  and `θ = 0.125` and `θ = 2` both beat it. The paragraph's actual point, that the choice barely
  matters across a 32× range, is correct and is what the numbers support; only the justification for
  the specific value is not.

---

### D. Upstream

#### D1. Julia 1.12: nested `kwargs...` feeding a call in the same inferred body

**Root-caused and worked around in this package by PR #35, but the behaviour is upstream.**

Constructing an `Optimizer` through three nested levels of `kwargs...` splatting and calling `solve!`
on it *in the same inferred function body* cost **940.86 s** of compile time on 1.12.6 against
**4.35 s** on 1.13.0-rc2. Neither half is slow alone (0.99 s for the constructor, 2.35 s for
`solve!`). Flattening to one level: 6.53 s. A `@noinline` barrier around the construction does not
help (925.27 s), and neither does `@nospecialize` on the enclosing function (965 s).

Effect on this project before the fix: the CI suite took 31–42 minutes on 1.12 on all three
operating systems, against 3–5 minutes on 1.10, 1.13 and nightly, with a single test file accounting
for the whole difference.

1.13 and nightly are unaffected, so this needs an upstream report only if 1.12 is still receiving
backports. `/tmp/go_diag/isolate2.jl` is a ~10-line reproducer (three variants, one per process).

#### D2. SimpleSolvers 0.11: three `Options` fields that nothing reads

`store_trace`, `show_trace` and `extended_trace` exist as fields of `SimpleSolvers.Options`
(`src/base/options.jl:458-460`), as constructor keywords (`:489-491`) and in its `show` (`:521-523`),
with **zero readers** anywhere in `SimpleSolvers/src/`. Anyone setting them on a SimpleSolvers solver
gets silence rather than a trace or an error. PR #35 implements `store_trace` at the
GeometricOptimizers level, which is arguably the right layer since `solve!` is ours, but the upstream
options remain misleading.

#### D3. SimpleSolvers `Bisection` reports success when it cannot bracket

`_bisection_core` (`src/linesearch/bisection.jl:81-84`):

```julia
if y₀ * y₁ > zero(y₀)                                  # no sign change
    report_bisection_nobracket(α₀, α₁, y₀, y₁, config)
    return (abs(y₀) ≤ abs(y₁) ? α₀ : α₁), true, n      # ← converged = true
end
```

The intent is that a flat derivative at a minimum is benign, but the effect is that the core cannot
distinguish "found the root" from "gave up", and this sits upstream of the `LINESEARCH_FLOOR` path
that produced the divergence in A1/PR #35.

#### D4. `maxlog` on the SimpleSolvers line-search warnings is per session, not per solve

Documented upstream at `src/linesearch/linesearch_status.jl:221-229`: `maxlog` is keyed on source
location, so a genuine line-search failure in the tenth solve of a run is silent if the first solve
already used the budget. Relevant because it is the only channel by which a rejected line search was
visible before PR #35, and still the only channel for B1.

#### D5. Documenter does not catch an `@ref` to a method signature that no longer exists

From the PR #36 review. That PR replaced `geodesic(::StiefelLieAlgHorMatrix)` and
`geodesic(::GrassmannLieAlgHorMatrix)` with one method on `AbstractLieAlgHorMatrix`, and left a
docstring pointing at `[`geodesic(::StiefelLieAlgHorMatrix)`](@ref)` — a method the same PR deletes.

I expected the docs build to fail on it. It does not: `makedocs` completes with exit 0 and no
warning, because Documenter falls back to the **binding**-level docs for `geodesic` when the
signature matches no documented method. Verified by reintroducing the dead reference and rebuilding.

So the guarantee is weaker than it looks. A `@ref` to a *name* that does not exist is caught; a
`@ref` to a name that exists with a signature that does not is silently redirected to some other
method's page. This is worth knowing here specifically because the commit immediately before that one
on the same branch was "mend four dead doc links" — the build cannot be what certifies that work.
`checkdocs = :all` does not help: it checks that docstrings are *included*, not that references
resolve to what they name.

---

### E. Reported and then withdrawn

Three things reported as problems during the investigation that are **not**:

- `test/grassmann_test_help.jl` is not orphaned — it is `include`d by
  `test/global_sections/global_sections.jl:8`, `test/global_sections/omega_functions.jl:8` and
  `test/retractions/retractions.jl:9`.
- `test/optimizers_problems.jl` is not dead code — `test/optimizer_tests.jl:12` includes it.
- `NaNMath` is used, at `test/optimizer_tests.jl:2`.

---

### F. Loose ends from the geodesic-retraction review

Neither is a defect in the code; both are things a later reader would otherwise have to rediscover.

- **The PR #36 description still says the MNIST scripts use the geodesic retraction.** The claim was
  corrected in `docs/src/manifold_optimizers.md`, but it also appears in the pull request body, where
  it is the stated justification for making `ScaledSquaring` the default. None of the five MNIST
  scripts passes `retraction`, so they all take the `Optimizer` default, which is `Cayley()`. The
  default is still the right choice — being free of dense LAPACK is reason enough — but if that PR
  body becomes a squashed commit message, the wrong reason goes into the history with it.
- **`svd_tables()` was not independently reproduced.** The PR #36 review re-ran
  `scripts/retraction_accuracy.jl`'s `exponential_tables()` and confirmed every accuracy figure to
  the digit. The iteration and evaluation counts in `test/optimizer_convergence/svd_optim.jl` and in
  `default_linesearch`'s docstring come from `svd_tables()`, which was not re-run — it is the slow
  half, and the `Geodesic` columns all moved in that PR. They rest on the author's measurement alone.

[#14]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/14
[#16]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/16
[#17]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/17
[#18]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/18
[#24]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/24
[#25]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/25
[#27]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/27
[#28]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28
[#33]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/33
[#36]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/36
[#38]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/38
[0.1.0]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.1.0
[Unreleased]: https://github.com/JuliaGNI/GeometricOptimizers.jl/compare/v0.1.0...main
