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
  problem it accepted the ceiling on *every* iteration and needed 49 679 of them; handing the same
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

[#14]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/14
[#16]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/16
[#17]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/17
[#18]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/18
[#24]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/24
[#25]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/25
[#27]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/27
[#28]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28
[0.1.0]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.1.0
[Unreleased]: https://github.com/JuliaGNI/GeometricOptimizers.jl/compare/v0.1.0...main
