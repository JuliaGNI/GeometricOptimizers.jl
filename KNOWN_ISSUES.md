# Known issues

What is known to be wrong in GeometricOptimizers.jl and is not fixed yet. An entry leaves this
file when its fix merges, and the CHANGELOG entry of the fix names its ID. IDs are never reused.

## A. This package — correctness

### A5 · `geodesic(Y, Δ)` and `cayley(Y, Δ)` are not deterministic and consume global RNG state

- location: `src/manifolds/stiefel_manifold.jl:126-133`
- kind: defect
- found: #36
- evidence:

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

### A6 · `ScaledSquaring` takes about twice the squarings it needs

- location: —
- kind: defect
- found: #36
- evidence:

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

  `NativePade`, added in [#54], takes `s` from ``\|X\|_1`` in exactly the same way and inherits the
  whole of this, so the entry now covers two algorithms and a fix would apply to both at once.

### A12 · The `Cayley` differential is recomputed per `φ'`, and its cost in a solve is unmeasured

- location: `svd_optim.jl`
- kind: not verified
- found: #40
- evidence:

  **Severity: low**, and not a defect — a cost this release introduced and did not measure. Found in
  the review of [#40], where `retraction_differential` was added.

  Under `Cayley`, `trial_slope` calls `retraction_differential!` on every evaluation of ``\varphi'``.
  That is `lift_factors!` and two ``2n\times{}2n`` solves in the optimizer's workspace —
  ``O(Nn^2 + n^3)``, the same order as the retraction itself, and no allocation on a host `Matrix` —
  where before it was a `_dot` against an array the cache already held. On `St(6, 3)` at
  ``\alpha = 0.5`` one call takes about 1.1 μs (`scripts/in_place_retraction_cost.jl`). `Geodesic`
  copies ``\bar{B}`` at every ``\alpha`` and `Cayley` does at ``\alpha = 0``, so every `Geodesic` solve
  and the `Backtracking` default pay a copy and no solve; what is unmeasured is a search that
  evaluates ``\varphi'`` many times per iteration, which on this problem is `Bisection` at ≈580
  objective evaluations per iteration.

  **The iteration and evaluation counts in `svd_optim.jl` do not answer this.** They moved under the
  change — `_BFGS + Bisection` under `Cayley` from 92 to 114 iterations — but they moved because the
  trajectory changed, so they measure a different solve rather than the cost of a step. Nothing here
  is a wall-clock measurement.

  The obvious remedy if it does turn out to matter is not a cache but a shared factorisation:
  `linesearch_problem`'s `d(α, params)` calls `trial_iterate!` and then `trial_slope` with the *same*
  ``\alpha``, and both go through `lift_factors!` — the first on ``\alpha\bar{B}`` and the second on
  ``\bar{B}`` — so one line search evaluation factors the same lift twice. Fusing them would need
  `trial_iterate!` to hand its factors on, which is a wider change to that interface than a cost
  nobody has measured justifies.

### A14 · `x_converged` still cannot see a Euclidean solve that diverges downhill

- location: —
- kind: defect
- found: 2026-08-14
- evidence:

  **Severity: low** — no measured solve reaches it. The remainder of A4, which is otherwise fixed
  above; kept here because the hole is real and because the next person to look at
  `convergence_measures` should find it named rather than have to re-derive it.

  The two guards A4's fix installed are a *scale* (`solution_scale`, exact on a manifold because
  ``\|Y\|_F = \sqrt{n}``) and the *objective* (`x_converged` requires `!f_increased`). A Euclidean
  solve whose ``\|x\|`` grows without bound while `f` decreases at every step defeats both: nothing
  bounds ``\|x\|``, so ``\|x - x'\|/\|x'\|`` still goes to zero for a step that is not small, and the
  objective never gives the second guard anything to act on. It would be reported as converged.

  The divergences this package has actually produced all went *uphill* (`3.38 → 9.13 → 1.2e169`) or
  straight to non-finite, so the guards cover them; and both of those causes are fixed at their source
  (`linesearch_rejected`, `curvature_is_usable`). What would close it is a scale fixed at the start of
  the solve rather than read off the current iterate — and the obvious candidate, ``\|x_0\|``, fails on
  the two commonest starting points: it is `0` at the origin, and it is the wrong scale entirely for a
  solve that legitimately travels a long way. That is the same "no property of the problem supplies a
  threshold" this entry inherits from A4; it is narrower now, and it is not gone.

### A16 · `DEFAULT_STEP_CEILING = 1` is calibrated against one problem

- location: `svd_optim.jl`
- kind: not verified
- found: 2026-08-15
- evidence:

  **Severity: low.** From the step-ceiling work that closed A1b.

  The *shape* of the ceiling is not in question — `c⋅2π/‖δ‖` is what the geometry gives, and `c = 1`
  says "never more than one full turn", which is an argument rather than a fit. What is thin is the
  evidence for that particular `c`. It rests on the SVD problem: over the converging solves there the
  largest `‖αδ‖` is `2.03`, comfortably inside `2π`, and every one of the twenty combinations in the
  sweep is unmoved or improved by the ceiling. Upstream's own table brackets it from the other side —
  `αmax = 10` leaves 7 of 8 seeds on the manifold and `αmax = 1` leaves 8 of 8 — but that is the same
  problem again.

  The per-block ceiling that closed A15 makes this *thinner*, not less thin, and that is the reason to
  keep the entry open. Since the ceiling stopped being tightened by a block's neighbours, it no longer
  binds anywhere on the pinned seed at all — every figure in `svd_optim.jl`'s table is now identical
  with the ceiling on and off. So the SVD problem no longer bounds `c` from below in any way: it says
  only that `c = 1` is loose enough there, and nothing about where it would start to cost.

  So there is no measurement of a problem on which `c = 1` is too *tight*, and one plausibly exists: a
  solve whose direction is systematically under-scaled wants a large `α`, which is exactly the `_DFP`
  story that motivated `Backtracking(expand = true)`. `_DFP` passes here, so the two do not collide on
  this problem; nothing says they cannot.

  **What to do**: measure `c ∈ {0.5, 1, 2, 4, Inf}` across the sweep and record where the iteration
  counts start to move. If they move at `c = 1` on any row, the default is too tight and the entry
  becomes a real defect; if they move only below it, the default is justified and this entry closes with
  a table. `svd_tables(step_ceiling = …)` takes the keyword already, so this is a loop and not new code.
  Note that the sweep can no longer answer the *upper* half of that question — see the paragraph above.

### A17 · `_manifold_αmax` pairs solution and direction blocks positionally

- location: —
- kind: defect
- found: #44
- evidence:

  **Severity: low**, and an unasserted assumption rather than a live defect. From the review of [#44],
  where the per-block ceiling that closed A15 was written.

  `_manifold_αmax(values(sol), values(direction(cache)), c)` walks the two tuples in step and decides
  from `sol`'s block whether `δ`'s block is a manifold direction. It therefore assumes the two
  `NamedTuple`s have the same keys in the same order, and it checks nothing.

  The assumption holds, and holds by construction: the cache's direction is built from the solution by
  `_similar`, which is `Base.map` over the `NamedTuple`, and `map` throws
  `ArgumentError: Named tuple names do not match.` on keys that differ *or are merely reordered*, then
  rebuilds the result under the first argument's keys. Verified directly on a mixed
  `(Y::StiefelManifold, W::Matrix, b::Vector)` problem: the direction comes back as
  `(Y::StiefelLieAlgHorMatrix, W::Matrix, b::Vector)` with `keys` equal.

  That argument is stronger than it was when this entry was written. It used to rest on `apply_toNT`'s
  hand-rolled `@assert keys(ps[1]) == keys(p)`, and `@assert`'s own docstring warns that it "might be
  disabled at various optimization levels"; Base's check is part of how `map` constructs the result and
  cannot be compiled out. `apply_toNT` was
  deleted in [0.5.0](#050), which is where all 30 of its call sites became
  `map`.

  What is not good about it is that this is the one place in the package that pairs two block
  structures *without* going through the construction that checks. Everything else — `_copyto!`,
  `_difference!`, the `GlobalSection` copies — is a `map` and would fail loudly. If a future cache
  ever built its direction some other way, the ceiling would silently be derived from the wrong block,
  which is a wrong `αmax` and not an error.

  **What to do**: cheapest is one `@assert keys(sol) == keys(δ)` where the cache is constructed, so the
  invariant is stated once and costs nothing per solver step. Routing `_manifold_αmax` itself through
  `map` is the *obviously* correct version and is why it was not done: it builds a `NamedTuple`
  of per-block ceilings, i.e. an allocation on every line-search call, to compute one scalar.

### A18 · `𝔄` accepts an `AbstractExponentialAlgorithm` it cannot serve

- location: `docs/src/retractions.md`
- kind: defect
- found: #45
- evidence:

  **Severity: low**, and a signature that is wider than the implementation rather than a wrong answer.
  From the review of [#45].

  `𝔄(X, algorithm)` is implemented for [`TaylorSeries`](@ref), [`ScaledSquaring`](@ref) and
  [`AugmentedPade`](@ref). [`ProjectedSkew`](@ref) is the fourth `AbstractExponentialAlgorithm` and has
  no `𝔄` method at all: it specialises `geodesic` directly, because it exponentiates the lift in an
  orthonormal basis of its range rather than going through ``\mathfrak{A}`` — see the *Disadvantages*
  paragraph on `docs/src/retractions.md`, which already tells readers that `𝔄(X, ProjectedSkew())` does
  not exist.

  The signature `𝔄(B̂, B̄, ::AbstractExponentialAlgorithm)` nevertheless accepts it, so
  `𝔄(B̂, B̄, ProjectedSkew())` dispatches, forwards, and dies one frame in with a `MethodError`
  naming the two-argument `𝔄` — not the method that was called, and not the fact that this
  algorithm lives a level up.

  **What to do**: either give `𝔄` a `ProjectedSkew` method that errors with the explanation — that it
  is a `geodesic`-level algorithm, and to call `geodesic(B, ProjectedSkew())` — which costs one
  method; or introduce the subtype of `AbstractExponentialAlgorithm` that the three
  ``\mathfrak{A}``-level algorithms share and narrow the signature to it, which makes it a
  `MethodError` at the call site instead of a frame in. The first is cheaper and says more; the
  second is the one that makes the type hierarchy match what is implemented.

### A22 · The symplectic SR decomposition is not backward stable, and fails three ways

- location: `src/decompositions/symplectic_sr.jl`
- kind: defect
- found: 2026-09-18
- evidence:

  **Severity: medium**, and it is a property of the algorithm rather than a defect in the port. Opened
  by the change that added `sr!` and `SymplecticStiefelManifold`; the two test files point here.

  ``S`` is symplectic and therefore not orthogonal, so its condition number is unbounded, and this
  implementation has no re-orthogonalization step — the one the literature adds for exactly this
  reason. The residual of a drawn point,
  ``\|U^T\mathbb{J}U - \mathbb{J}\|``, therefore grows with the size instead of staying at machine
  precision. The measured table is under *Added* in [Unreleased](#unreleased); the short form is a
  median of `1.3e-14` at `Float64` 6×4 and `6.2e-8` at 40×20, and `0.014` at `Float32` 20×10.

  **Three outcomes, and they are not equally visible:**

  - a large residual, which a caller who checks will see;
  - a `NaN`, which makes `check(U) < tol` silently **false** rather than large — 2 draws in 200 at
    `Float32` 40×20;
  - a `DomainError` from `sqrt` of a negative argument at `src/decompositions/symplectic_sr.jl`, in
    `symplectic_householder!`, where ``\sqrt{\|b\|^2 - b_1^2 - \nu^2}`` cancels — 1 draw in 200 at
    `Float32` 20×10 and 6 at 40×20.

  **Only the medians reproduce.** The maxima and the throw counts move by orders of magnitude with
  the draw order: `Float64` 40×20 has been measured at 0.2 and at 54.0 in two runs of the same sweep.
  Any figure quoted from the tail of this distribution is a sample, not a bound.

  A related sharp edge, measure-zero for a random draw but part of the same class:
  `ρ = sign(a[1]) * norm(a)` gives `ρ = 0` and then `c₁ = Inf` silently when `a[1] == 0`. Standard
  Householder picks a sign that cannot vanish.

  `test/decompositions/symplectic_sr.jl` runs in `Float32` too, at 4×2, 6×4 and 10×6. The median of
  `‖SᵀJS - J‖` for `sr(randn(T, N2, N2))` is 8.0 to 8.5, 136 to 141 and 9820 to 10400 `eps(T)` in
  both precisions, over 20000 draws at each of three seeds (Julia 1.13.1). The draws that throw or
  return a non-finite value are 0, 17 to 23 and 111 to 122 in 20000 in `Float32`, and none in
  `Float64`. The test's rate bound in `Float32` is ten times the measured exceedance, 7 % at 10×6,
  where the `Float64` bound is 1 %.

  Closing this means implementing the stabilized variant, which is numerical work rather than a
  repair. Until then the type is documented as `Float64`-only at small sizes, in both the
  `SymplecticStiefelManifold` and `sr!` docstrings, and nothing in the package depends on it.

### A23 · `SymplecticStiefelManifold` has no `Ω` and no `zero`, so it cannot be optimized over

- location: `src/lie_algebras/`
- kind: defect
- found: 2026-09-18
- evidence:

  **Severity: medium.** Opened by the change that added the type.

  `StiefelManifold` implements `Ω` and `zero` in addition to `rand`, `rgrad`, `metric` and
  `global_section`; `SymplecticStiefelManifold` implements neither. `Ω(U, Δ)` is a `MethodError`, and
  `zero(U)` is worse than an error: it falls through to `zero(::AbstractMatrix)` and returns a plain
  `Matrix`, where `zero(::StiefelManifold)` returns a `StiefelLieAlgHorMatrix`. An optimizer reaching
  for either gets a wrong answer or a failure at its first step.

  Both need the **symplectic horizontal Lie algebra**, which this package does not have:
  `src/lie_algebras/` holds the Stiefel and Grassmann horizontal components and no symplectic one. A
  draft of the tangent-space projection `πₑ` exists in `GeometricMachineLearning`'s history at
  `6a8e19a7:legacy/arrays/sympl_st_E_ts.jl`, written against two Lie algebra types that were never
  committed anywhere; it is a starting point, not a solution.

  Until this closes, the type is a geometric object with a metric, a Riemannian gradient and a global
  section — not an optimization target.

### A26 · `symplectic_normalize` and `symplectic_gram_schmidt!` are real-only by construction

- location: —
- kind: defect
- found: #111
- evidence:

  **Severity: medium.** Found in the review of [#111], the PR that fixed
  `symplectic_form` for complex operands.

  Both scale a pair by `sign(fac)/sqrt(abs(fac))`, which gives `eᵀJf = 1` only for a
  real `fac`; for a complex one the form comes out as `sign(fac)²`. They still use `'`
  for the form, and a switch to `transpose` alone would not make them correct. The
  functions are unchanged and will not work with complex element types. The package
  documents that it supports real element types only.

  **What to do**: Write a complex-valued algorithm that computes the sign correctly,
  or replace both with a method that accepts only real input by type constraint.

### A28 · `SymplecticStiefelManifold` has no `rebuild`, so `changebackend` refuses it

- location: `src/parameter_protocol.jl`
- kind: defect
- found: 2026-09-24
- evidence:

  **Severity: low.** `src/parameter_protocol.jl` gives the symplectic Stiefel manifold a
  `freeparameters` and no `rebuild`, so `mapstorage` raises an `ArgumentError` for it, and
  `changebackend`, which walks the parameter protocol, raises with it. The Stiefel and Grassmann
  points and every structured matrix move between backends. Found by `scripts/device_products.jl`.

### K7 · The two `zero_tangent` bodies differ only in the lift type

- location: `src/manifolds/stiefel_manifold.jl`, `src/manifolds/grassmann_manifold.jl`
- kind: found late
- found: 2026-09-27
- evidence: `zero_tangent(::StiefelManifold)` and `zero_tangent(::GrassmannManifold)` are the same
  three lines with `StiefelLieAlgHorMatrix` and `GrassmannLieAlgHorMatrix`; one method on the lift
  type would serve both.

### K24 · In `Float32`, `global_section` of a `SymplecticStiefelManifold` point raises the `DomainError` of A22 on a few draws in 2000

- location: `src/manifolds/symplectic_stiefel_manifold.jl`, `global_section`
- kind: found late
- found: 2026-10-09; one issue with A22
- evidence: over 2000 points from `rand(rng, SymplecticStiefelManifold{T}, N2, n2)` per size, with
  the point's generator `Random.Xoshiro(987_654)` independent of the global one that the completion
  draws from (`Random.seed!(1)`), Julia 1.13.1: in `Float32`, `global_section` raises the
  `DomainError` of A22 on 0, 1 and 5 points at 4×2, 6×4 and 10×6 (and `rand` itself on 0, 3 and 1);
  in `Float64` on none. Every section that returns has `‖UᵀJΛ‖ / (‖U‖² ‖Λ‖²)` under `1120 eps(T)`
  in both precisions. The section is built with `sr!`, which has no re-orthogonalization step, so
  the stabilized SR decomposition of A22 closes this too.

### K25 · `global_section` of a Stiefel or Grassmann point loses orthogonality to the point, up to 2e4 eps, when its completion draws the numbers the point was drawn from

- location: `src/manifolds/stiefel_manifold.jl`, `_complement_columns`
- kind: found late
- found: 2026-10-09
- evidence: 400 seeds per shape, the point from `rand(Random.Xoshiro(seed), …)`, Julia 1.13.1.
  With `Random.seed!(seed)` before the draw, the completion that `global_section` draws from the
  global generator repeats the point's own numbers: on a `JLArray` the median `‖Yᵀλ‖` is 15 to 25
  `eps(T)`, 39 to 63 of the 400 draws exceed `100 eps(T)`, and the largest is 7.4e3 `eps(T)` in
  `Float32` and 2.1e4 `eps(T)` in `Float64`; on the host the median is 23 to 34 `eps(T)`, 49 to 76
  draws exceed `100 eps(T)`, the largest is 1.7e4 `eps(T)`, and at `Float32` 12×4 one draw raises
  `ArgumentError: No value arguments present` from the `something` of the second `_cholesky_qr2`.
  With `Random.seed!(seed + 10^6)`, so that the two streams are independent, the median is 0.5 to
  0.97 `eps(T)` and the largest 1.43 `eps(T)` on a `JLArray`, with no draw above `100 eps(T)`.
  Several test files seed the global generator and a `Xoshiro` of the point with one seed.

### K26 · `DFP` with `Backtracking(expand = true)` does not converge on Rosenbrock in `Float32` from a start two ulps from the standard one

- location: `src/optimizers/iterative_hessians/quasi_newton_cache.jl`
- kind: found late
- found: 2026-10-09
- evidence: over the 49 `Float32` starts within ±3 ulps of `(-1.2, 1)`, `DFP` from the start
  `(-2, -2)` ulps stops after 300 iterations at `x = [1.0223, 1.0453]`, `f = 5.0e-4`,
  `rg = 0.0348`; it converges from the other 48, and `BFGS` from all 49. The test in
  `test/optimizers/quasi_newton_secant_tests.jl` uses the standard start, from which both converge.
  Whether the cause is the update, the line search or the conditioning of `Float32` is not
  investigated.

### K27 · A solve whose polynomial line search finds no decrease takes a zero step and reports convergence far from the optimum

- location: `src/optimizers/optimizer.jl`, `solver_step!` (the `at_round_off_floor(…) ? zero(T)` step)
- kind: defect
- found: 2026-10-09
- evidence: on the SVD problem of `test/verification/svd_optim.jl`, `DFP` with `Quadratic` under
  `Geodesic` from seed 8 stops after 3 iterations with `rg = 3.5269928303258307`, a relative error
  of 1.3277 against the best rank-3 approximation, `x_converged = true`, `f_converged = true` and
  `g_converged = false`, in `Float64` and in `Float32`; `scripts/retraction_accuracy.jl`'s
  `solve_once` gives the same numbers. SimpleSolvers' `Quadratic` (`src/linesearch/quadratic.jl:111`,
  the last return of `solve_with_status`) classifies every step that does not decrease the merit as
  `LINESEARCH_FLOOR`, also one whose merit rose (`φ(α) = 6.37` against `φ₀ = 6.22` on a probe of the
  same mechanism), and `quadratic_bierlaire.jl:243` has the same rule. The steepest-descent retry
  returns the floor again, `solver_step!` takes a zero step, and `x_converged` reports convergence.
  From `Float32`'s own draw of the starting point, the same happens on seed 6 for `BFGS` with
  `Quadratic` and with `BierlaireQuadratic`, and for `DFP` with `Quadratic`. The eight-seed sweep of
  `svd_optim.jl` marks the three assertions of that one cell broken with this ID and issue #153.
  `GradientMethod` with `BierlaireQuadratic` on `Fsmooth` of `test/integration/optimizer_tests.jl`
  stops the same way in `Float32` after 2 iterations, 172 to 370 `√eps` from the minimiser, on 3 of
  the starts tried (not the test's), with `LINESEARCH_FLOOR` at `φ(0) = 3.001769` and `φ(1) = 3.0`.

### K28 · In `Float32`, a fixed-step `Adam` solve stops on `f_converged` far from the minimiser and reports convergence

- location: `src/optimizers/optimizer_status.jl`, `convergence_measures` (`f_relchange ≤ f_suctol`)
- kind: defect
- found: 2026-10-09
- evidence: `Adam` with `Static(0.1)` on `Fsmooth` of `test/integration/optimizer_tests.jl` stops at
  iteration 43, 0.116 from the minimiser (336 `√eps(Float32)`), with `isconverged` true and the flags
  `(x, f, g) = (false, true, false)`, `rfᵣ = 1.59e-7`. Two successive iterates of the fixed-step
  orbit agree in `f` to one ulp, and `f_converged` fires on that. `Float64` runs the same orbit down
  to 8.8 `√eps(Float64)`. The two distance assertions of that case in `Float32` are `@test_broken`
  with this ID and issue #154.

### K29 · The default gradient gate `f_reltol = √eps(T)` of `g_converged` is out of reach in `Float32` on a problem whose gradient at the round-off floor of `f` is of the order of `√eps(T)`

- location: `src/optimizers/optimizer_status.jl`, `convergence_measures` (`g_converged`)
- kind: defect
- found: 2026-10-09
- evidence: the `Backtracking` solve of `test/integration/manifold_linesearch_tests.jl` stops in
  `Float32` after 5 iterations on the change of `x` and `f`, at `rg = 6.4e-4 = 1.9 √eps(Float32)`,
  above the gate; `Float64` meets it at 0.72 `√eps(Float64)`. The `g_converged` assertion of that
  solve in `Float32` is `@test_broken` with this ID and issue #155.

### K30 · In `Float32`, `BFGS` and `DFP` with `Backtracking` stop after one iteration from a start near the minimiser of a quadratic

- location: `src/optimizers/optimizer.jl`, `solve!`
- kind: found late
- found: 2026-10-09
- evidence: on `F` of `test/integration/optimizer_tests.jl` from `x₀ = 0.0172`, both stop after 1
  iteration at 50 `√eps(Float32)` from the minimiser. The test starts in `[0.5, 1.5]`, where both
  converge to under 10 `√eps(T)`.

### K31 · `Adam` with `Static(0.1)` on two Stiefel spheres runs out its iteration budget in `Float64` on some seeds of the global generator

- location: `test/integration/manifold_linesearch_tests.jl`, testset `Adam runs on a manifold NamedTuple under a searching line search`
- kind: found late
- found: 2026-10-10
- evidence: the `Adam` + `Static(T(0.1))` solve of that testset (`ps₀`, `two_spheres`,
  `max_iterations = 1000`), with `Random.seed!(s)` before the state is built, on Julia 1.13.1:

  | `T` | seed | `Geodesic` | `Cayley` |
  |:--|:--|:--|:--|
  | `Float64` | 1 | 1000 iterations, not converged, 3.9e-3 | 1000, not converged, 4.5e-3 |
  | `Float64` | 2 | 1000, not converged, 2.9e-3 | 1000, not converged, 2.3e-3 |
  | `Float64` | 1234 | 334, converged, 2.0e-8 | 331, converged, 4.4e-8 |
  | `Float32` | 1, 2, 1234 | 104 to 127, converged, 4.3 to 4.9 `√eps` | 119 to 127, converged, 4.3 to 6.7 `√eps` |

  The distance is the larger of the two leaves' distances to their minimisers. The seed draws the
  completion of each `GlobalSection`; a fixed step lets `Adam` circle the minimiser, and on these
  seeds the orbit does not shrink to a stopping criterion within the budget. The test runs at seed
  1234.

### K32 · `storage_gradient` of a device `StiefelLieAlgHorMatrix` for an `Adjoint` of a lift-typed cotangent indexes the device array one entry at a time

- location: `src/parameter_protocol.jl`, `storage_gradient`
- kind: found late
- found: 2026-10-10
- evidence: with `S` and `H` `StiefelLieAlgHorMatrix(SkewSymMatrix(JLArray(rand(rng, T, 3)), 3),
  JLArray(rand(rng, T, 2, 3)), 5, 3)` under `allowscalar(false)`, `storage_gradient(S, H')` raises
  `Scalar indexing is disallowed.` in `Float32` and `Float64`; on host arrays it returns a
  `StiefelLieAlgHorMatrix`, and `storage_gradient(S, H)` (the block method for a lift-typed
  cotangent) runs on both backends (Julia 1.13.1).

### K33 · `storage_gradient` of a device `GrassmannLieAlgHorMatrix` for a cotangent of the lift's own type indexes the device array one entry at a time

- location: `src/parameter_protocol.jl`, `storage_gradient`
- kind: found late
- found: 2026-10-10
- evidence: with `A` and `G` `GrassmannLieAlgHorMatrix(JLArray(rand(rng, T, 3, 2)), 5, 2)` under
  `allowscalar(false)`, `storage_gradient(A, G)` raises `Scalar indexing is disallowed.` in
  `Float32` and `Float64`; on host arrays it returns a `GrassmannLieAlgHorMatrix` (Julia 1.13.1).
  The lift-typed cotangent takes the dense `AbstractMatrix` method; only the Stiefel lift has a
  block method for a cotangent of its own type.

## B. This package — observability

### B1 · A line search failure is invisible in the returned status

- location: `optimizer_status.jl`
- kind: defect
- found: 2026-08-14
- evidence:

  PR #35 makes `solver_step!` act on `LINESEARCH_FLOOR` / `LINESEARCH_EXHAUSTED` /
  `LINESEARCH_NO_DESCENT`, but nothing records that it happened. `OptimizerStatus` has no field for it
  (verified: no reference to the outcome in `optimizer_status.jl`), so a solve that needed a
  quasi-Newton restart on half its iterations is indistinguishable, in the object the caller gets,
  from one that never needed one. Only a `verbosity ≥ 2` log message with `maxlog = 1` shows it.

  The `MomentumMethod` runaway is no longer the argument for this that this entry claimed when that was
  open as A7. That divergence was 13 rejected outcomes
  over 457 iterations rather than 194 over 200, and it is fixed by *acting* on them rather than by
  counting them — so the counter is not a guard against anything, it is what would have made the
  thirteen visible. It is still worth having on its own terms: after that fix, a solve that needed a
  steepest-descent substitution on a quarter of its iterations is *still* indistinguishable, in the
  object the caller gets, from one that never needed one.

  **This entry has now lost its worked example, and did not lose its point.** It read that `_BFGS` +
  `Quadratic` + `Cayley` on seed 8 of the SVD problem takes the steepest-descent branch on 4 780 of its
  20 000 iterations and says nothing about any of them. That solve no longer exists: the step ceiling
  that closed A1b brings that whole row inside 176 iterations. No replacement figure is quoted here
  because there is no
  way to get one from outside `solver_step!` — which *is* the defect, one level down, and it is why the
  fix below would be its own instrument. Do not replace the number by reasoning about it.

  **What to do**: one counter, `linesearch_restarts`,
  and not one field per outcome — the question a caller has is "did this solve need help". Accumulate
  it on the *state*, which persists across iterations as `iterations` already does, rather than
  widening the `OptimizerStatus(state, cache, f; config)` constructor, which is where `solver_step!`
  would otherwise have to thread it.

### B2 · `show_trace` and `extended_trace` are still accepted and ignored

- location: `SimpleSolvers/src/`
- kind: dead code
- found: 2026-08-14
- evidence:

  PR #35 implements `store_trace`. The other two remain dead in this package *and* upstream (verified:
  no reads of either in `SimpleSolvers/src/` outside `options.jl`'s struct, constructor and `show`).
  Setting them gets neither output nor an error. Both are now cheap, since the per-iteration record
  exists.

  **What to do**: `show_trace` prints the record every `config.show_every`
  iterations; `extended_trace` adds `rxₐ` and `Δf` to `OptimizerTraceEntry`. Small, and it retires the
  last two silently-ignored options. Belongs in one PR with B1 and C1 — all three are the same subject,
  the status object not saying enough.

## C. This package — dead code and bookkeeping

### C1 · `f_converged_strong` is computed and thrown away

- location: `src/optimizers/optimizer_status.jl:312`
- kind: dead code
- found: 2026-08-14
- evidence:

  `convergence_measures` computes it at `src/optimizers/optimizer_status.jl:312` and returns it at
  `:319`; the `OptimizerStatus` constructor destructures it at `:135` and never stores it. Nothing else
  in the package mentions it. It is `Δf ≤ f_mindec ⋅ Δf̃`, i.e. an Armijo-style sufficient-decrease test
  on the *outer* iteration, so it plausibly belongs with the stall detection that `Options.max_stalls`
  and `Options.f_stall_window` were meant to drive — both of which are also unread here.

  `Δf̃` is its only input, and it pairs `previous_gradient(state)`, the gradient the step was built
  from, with the direction of that step, for every method.

  **What to do**: either *use* it as the stall detector `Options.max_stalls` and
  `Options.f_stall_window` were meant to drive — count consecutive iterations that fail it, stop after
  `max_stalls` — or *delete* it from `convergence_measures`' return tuple and say in the docstring that
  the outer-iteration Armijo test is not implemented. Deleting is the honest default. Using it is worth
  more but is a behaviour change that needs its own measurement over the eight starting points, so it
  must not ride along in an observability PR; split it out if that is the choice.

### C7 · `ensure_descent!` is vacuous for `GradientMethod`

- location: `gradient_optimizer.jl:82`
- kind: dead code
- found: 2026-08-14
- evidence:

  **Severity: low**, and currently harmless. Found while writing `steepest_descent!`, which exists
  because of the same aliasing.

  `ensure_descent!` tests `dot(rhs(cache), direction(cache)) > 0`. That works because `rhs` is
  ``-\nabla{}f`` on the (quasi-)Newton caches — but on the three first-order caches `rhs` is defined as
  an *alias* for `direction` (`gradient_optimizer.jl:82`, `momentum_optimizer.jl:63`,
  `adam_optimizer.jl:72`), so the test reads `dot(δ, δ) > 0` and is true for every `δ` that is neither
  zero nor `NaN`.

  Of the three, only `GradientCache` reaches it: `MomentumMethod` and `Adam` are `FirstOrderMethodWithState`
  and `solver_step!` skips the call for them deliberately. So `GradientMethod` runs a safeguard that
  cannot fire. It is harmless *today* because that method's direction already is ``-\nabla{}f``, which
  always descends — the safeguard has nothing to catch. It is worth recording because the harmlessness
  is a property of the direction and not of the guard: anything that changes what `GradientCache` puts
  in `direction` would silently lose the check rather than start failing it.

  The same aliasing had a second consequence that *was* live, and is fixed in this release: the
  steepest-descent substitution after a rejected line search was written as
  `_copyto!(direction(cache), rhs(cache))`, which is a no-op on these three caches. See
  `steepest_descent!`.

### C9 · Most of the harnesses these figures come from are not in the repository

- location: `/tmp/go_diag/`
- kind: not verified
- found: 2026-08-14
- evidence:

  **Severity: low**, and the direct cause of every stale figure this catalogue has had to correct. The
  sequencing plan this catalogue absorbed listed five measurement scripts under `/tmp/go_diag/` and
  said they "should be moved somewhere durable before relying on them". One of the five was:
  `scripts/retraction_accuracy.jl`, which regenerates the SVD tables and the exponential-accuracy
  tables. The rest are gone, and each round of work since has added another.

  What is quoted somewhere and has no committed harness:

  - **the flag-and-residual probe** — 264 solves over Rosenbrock, ``\sum\sin^2``, two Euclidean
    objectives, the `St(3,1)` sphere and a manifold `NamedTuple`, across six methods and six line
    searches, reporting iterations, gradient-evaluation counts, `rg` against ``\|\nabla{}f\|`` computed
    outside the optimizer, and the four status flags. It is what showed A4 to be numerically inert and
    what measured A8's `5.8e4×` table, and it is the only thing that would catch either regressing.
  - **the Rosenbrock iteration counts** that `ROSENBROCK_MAX_ITERATIONS` is set against.
  - **the `Adam` cross-version statistic**, quoted in `svd_optim.jl` as a 1.06× spread over three Julia
    versions and reproducible only by running three Julia versions.
  - **the 1.12 compile-time reproducer** (D1), which no longer exists at all.
  - **the wall-clock timings** in `default_linesearch` — `0.155 s` against `0.246 s` on `Geodesic`,
    `0.205 s` against `0.451 s` on `Cayley`, and the "1.6× to 2.2× faster" they support. `svd_tables`
    reports iterations and evaluations and not time, so the step-ceiling round regenerated every other
    figure in that docstring and left these four untouched. They now say so in place, which is the
    minimum this entry asks for and not a fix.
  - **the MNIST run**: the 6 h 53 min RTX 4090 figures, the
    ``\sqrt{1.8} \approx 1.342`` plateau and the per-configuration losses are
    still quoted here, while `distill_mnist_results.jl` and the five scripts that produced them are now
    in GMLDatasets.jl. This is the one entry on the list whose harness *exists* and is merely elsewhere,
    which makes it the mildest case and the easiest to get wrong: a figure quoted in this repository
    and regenerated in another one goes stale exactly as quietly, and nothing here will notice.

  The pattern is the one of the `DFP  Backtracking` row of `svd_optim.jl`'s table, which went wrong
  unnoticed because no sweep regenerates it: a number that no committed script regenerates is a number that
  goes stale silently, and the ones this catalogue has caught were all of that kind. Either put them
  under `scripts/` next to `retraction_accuracy.jl`, or accept the rule the preamble already states and
  stop quoting figures that nothing can re-run.

### C11 · The `Quadratic` `Geodesic`/`Cayley` gap is unexplained, and now explicitly so

- location: —
- kind: docs
- found: 2026-08-15
- evidence:

  **Severity: low**, an explanatory gap and not a defect. Recorded because a wrong explanation for it
  was removed this round and nothing replaced it.

  `_DFP` + `Quadratic` takes 175 iterations under `Geodesic` and 529 under `Cayley`. `default_linesearch`
  used to attribute that to `trial_slope` being only first-order correct under `Cayley`; the exact
  `retraction_differential` disproved it (the figure moved 550 → 529 and the gap stayed). This round
  removed the second candidate too: the step ceiling turned out to have nothing to do with it either —
  `175 vs 529` with the ceiling on is the same pair as with it off, since the per-block ceiling does not
  bind on this seed. Both docstring and docs page now say the remainder is unexplained rather than
  offering a third guess.

  An intermediate version of the ceiling *did* move the `Geodesic` figure, to 308, and it would have
  been easy to read that as the third explanation. It was an artefact of combining the two manifold
  blocks in quadrature (issue A15) and went away when the ceiling became per-block. Worth recording as a
  near miss of exactly the kind the A1b preamble warns about.

  That is the right state to be in — see the preamble on A1b, where two plausible-looking explanations
  in a row were the expensive part — but it is a loose end, and the pattern that resolved A1b applies:
  stop proposing mechanisms and instrument the quantity that differs. Here that is the sequence of
  brackets the fit is built on, which is not observable from outside SimpleSolvers.

  **What to do**: nothing, unless the gap starts to matter. `Quadratic` is not a default under either
  retraction and both figures converge. If it does matter, the measurement is the per-iteration `α`,
  bracket width and fit residual under the two retractions from one starting point — the same
  instrumentation A1b needed, and the same reason it is not in the package.

### C12 · `step_αmax` takes its element type from the ceiling alone

- location: `manifold_linesearch_tests.jl`
- kind: defect
- found: #44
- evidence:

  **Severity: low**, a sharp edge on an internal helper. From the review of [#44].

  ```julia
  function step_αmax(c::T, δ) where {T}
      n = l2norm(δ)
      (isfinite(n) && n > zero(n)) ? c * T(2π) / T(n) : T(Inf)
  end
  ```

  `T` comes from `c` and from nothing else, so `step_αmax(1, δ)` on an integer ceiling throws an
  `InexactError` at `T(2π)` rather than promoting. `T(n)` is the mirror image and is the worse of the
  two, because it does not throw: with `c` a `Float32` and `δ` a `Float64` direction whose norm exceeds
  `3.4e38` — finite in `Float64`, `Inf32` on conversion — the guard above sees a finite `n`, the
  division underflows and `step_αmax` returns **`0.0f0`**. Measured: `step_αmax(1.0f0, [1e39, 1e39])` is
  `0.0`. Upstream then raises an `ArgumentError`, correctly, because a ceiling of zero asks for a step
  that violates `α > 0`; so the failure is loud, but it is raised for the wrong reason and blames the
  caller for what is a conversion in this function.

  Neither is reachable through an `Optimizer`. The struct stores `step_ceiling::T` and the constructor
  writes `T(step_ceiling)`, so `c` arrives in the element type of the problem and `l2norm(δ)` is already
  in it — the two types cannot differ; `manifold_linesearch_tests.jl` pins exactly that (`step_ceiling =
  1` gives a `Float64` field). So this is about calling the helper directly, which the tests do.

  **What to do**: `promote_type(typeof(c), typeof(l2norm(δ)))` and take the one `T` from that, which is
  a one-line change, kills both halves, and leaves every existing assertion true — the `Float32` test
  promotes to `Float32`. What is not worth doing is guarding the narrowing separately; the promotion is
  the guard.

### C15 · The compile-time figures cover the first-order caches only

- location: `scripts/`
- kind: not verified
- found: #45
- evidence:

  **Severity: low**, and a gap in evidence rather than in code. From the review of [#45].

  The measurements in [0.2.1](#021) — 14.25 s / 14.48 s cold against a run that did not finish in seven
  or in ten minutes — were taken on `GeometricMachineLearning`'s symplectic-autoencoder test against a
  branch on which only `GradientCache`/`GradientState`, `MomentumCache`/`MomentumState` and
  `AdamCache`/`AdamState` had been unbound. That test drives `Adam`, so those six are the whole of what
  it exercises.

  `BFGSCache`, `BFGSState`, `DFPCache` (`QuasiNewtonCache` replaces the two caches), `NewtonOptimizerCache`,
  `NewtonOptimizerState` and the `VT` of `OptimizerResult` were unbound afterwards on the strength of their *inferred types* — for the
  quasi-Newton three, `Base.return_types` showed the same coupled shape the six had, and worse for
  `BFGSState`, which carried a free `T` across four parameters and the three-parameter `GlobalSection`
  `UnionAll` under a `Vararg`; after, all five parameters are independent. That is a sound argument from
  the same root cause, and it is not a measurement: no quasi-Newton or Newton solve was ever timed
  through a function, before or after, so the claim that they hung is an inference and the claim that
  they no longer do is untested.

  **What to do**: the cheap version is the one already written — the repro in the release notes with
  `algorithm = _BFGS()` and with `Newton()` on Euclidean parameters, cold, before and after, which is
  two runs of an existing harness. Do it before quoting these numbers for anything but `Adam`. The
  version worth more is C9's: a compile-time measurement that lives in `scripts/` rather than in a
  `/tmp` file that is gone by the time anyone asks.

### C16 · `NativePade`'s threshold rests on a measurement, not on a backward-error criterion

- location: —
- kind: not verified
- found: #54
- evidence:

  **Severity: low**, and a gap in evidence rather than in code — the same shape as C15. From the review
  of [#54], and it is what keeps [#52] open now that an algorithm has been added.

  [#52] asked for the "proposed algorithm, Padé degree, scaling threshold, and backward-error criterion
  … documented from primary references", and for a criterion "appropriate for this structured, strongly
  non-normal argument rather than importing a threshold without validation". Three of those four are
  done: the degree, the threshold, and the provenance of the coefficients as the ``[7/6]`` Padé
  approximant of ``\exp`` rearranged, cited to [higham2005scaling, higham2008functions](@cite). The
  fourth is not, and the docstring says so rather than implying otherwise.

  What `θ = 1/2` actually rests on is (a) the Newton--Schulz residual bound
  ``\|\mathbb{I} - q_6\|_1 \leq \sum_k|q_k|\theta^k = 0.256``, hence ``0.256^{32} \approx 2e{-}19``,
  and (b) a measured forward error — the eight-point norm sweep on both manifolds in both formats, plus
  400 random ``6\times6`` arguments per ``\theta``. Both are forward statements about this family of
  arguments. Neither is a backward-error criterion, and the ``\theta_m`` tables that would supply one
  [higham2005scaling, almohy2010new](@cite) are derived for ``\exp`` rather than ``\varphi_1`` and are
  stated in ``\|X\|`` — which is precisely the norm A6 records as uninformative here, since ``X``'s
  lower-left block gives ``\|X\| \approx \|\bar{B}\|^2/4`` against a spectral radius of only
  ``\approx\|\bar{B}\|``. Importing such a table without adapting it is the thing [#52] asked not to
  do, so it was not done; the honest position is that the threshold is validated and not derived.

  Two smaller pieces of [#52] are open with it. Its comparison of candidate designs is empirical —
  `NativePade` against `AugmentedPade` on cost, allocations, `check` and forward error — but the
  theoretical comparison of direct ``\varphi_1`` evaluation against the low-rank and augmented
  alternatives was not written; option (a) was chosen on the argument that it needs no solve, which is a
  portability argument rather than a numerical one. On a GPU backend, `scripts/metal_check.jl` runs
  `geodesic` with `NativePade` and with `ScaledSquaring` on Metal, on
  `60 * rand(StiefelLieAlgHorMatrix{Float32}, 20, 3)`, and matches the host; no other GPU backend
  has been exercised.

  **What to do**: derive a ``\varphi_1`` backward-error bound for a non-normal argument, or state
  plainly in the docstring that no such bound is claimed and that the threshold is a measured one — the
  latter is what is written today, and it is enough to use the algorithm honestly but not enough to
  close [#52]'s first acceptance criterion.

## D. Upstream

### D1 · Julia 1.12: nested `kwargs...` feeding a call in the same inferred body

- location: `scripts/nested_kwargs_cost.jl`
- kind: upstream
- found: 2026-08-14
- evidence:

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
  backports; if it is not, the value is documentation rather than a fix, and the warning on
  `Optimizer(x, F)` already carries it.

  **The reproducer is written down now — `scripts/nested_kwargs_cost.jl` — and it does not reproduce.**
  That was the outstanding action here ("the reproducer this was measured with lived in `/tmp` and is
  gone; it has to be rewritten, which is the case for writing it into `test/` or `scripts/` this time"),
  and doing it produced a negative rather than a bug report. Four reconstructions, on both 1.12 patch
  releases:

  | | 1.12.6 | 1.12.7 | 1.13.0-rc3 |
  |---|---|---|---|
  | real `Optimizer`, one level | 3.44 | 3.13 | 3.36 |
  | real `Optimizer`, three levels | 3.44 | 3.12 | 3.39 |
  | real `Optimizer`, three levels, 3 algorithms × 2 retractions | — | 5.05 | 5.37 |
  | real `Optimizer`, one level, same breadth | — | 5.05 | 5.34 |
  | `Base`-only synthetic, 800-deep tree, three levels | — | 0.21 | 0.19 |
  | `Base`-only synthetic, one level | — | 0.21 | 0.19 |

  Against the 4.35 s / **940.86 s** this entry records for the first two rows. The nested and flat
  columns are equal to the hundredth of a second everywhere, and the 1.13 figures agree with the
  recorded 4.35 s to within a cold measurement's spread — so the harness is measuring the right thing;
  it is the 940 that will not come back.

  **So the description above is not sufficient**, four ways: "a constructor reached through N nested
  `kwargs...` levels whose result is passed to a second function with a large call tree, both in one
  inferred body" does not by itself produce the cliff. That is what anyone would work from, so knowing
  it is incomplete is the useful part of this. The patch-release explanation is *ruled out* — 1.12.6 and
  1.12.7 agree to within 0.3 s on every control — which was the cheap hypothesis and worth eliminating
  first.

  **What to do**, in order:

  1. Nothing goes to JuliaLang yet. A bug report needs a reproducer and four negatives are not one.
  2. Try the two ingredients none of the four varies: `Options(T; options_kwargs...)`, whose keyword
     defaults are computed from other keywords so inference has a dependency order to resolve, and the
     objective's own call tree (the SVD closure over a captured `A`, at the original's problem size
     rather than the harness's `St(20, 3)`).
  3. If those are also negative, the remaining hypothesis is that something in this package between
     PR #35 and now removed the sensitivity — in which case **D1 is closeable** and the artefact worth
     having is whichever commit did it. Testing that means bisecting `#35..HEAD` with the nested shape
     reinstated at each step, which `scripts/nested_kwargs_cost.jl --with-package` is written to make
     mechanical.

  The warning on `Optimizer(x, F)` stays either way: it records a measurement that was taken, and
  nothing here shows the flattening is safe to undo — only that the cost it avoids cannot currently be
  demonstrated.

### D2 · SimpleSolvers 0.12: three `Options` fields that nothing reads

- location: `src/base/options.jl:458-460`
- kind: upstream
- found: 2026-08-15; one issue with SimpleSolvers' *Open Issues* entry "`store_trace`, `show_trace` and `extended_trace` have no readers"
- evidence:

  `store_trace`, `show_trace` and `extended_trace` exist as fields of `SimpleSolvers.Options`
  (`src/base/options.jl:458-460`), as constructor keywords (`:489-491`) and in its `show` (`:521-523`),
  with **zero readers** anywhere in `SimpleSolvers/src/`. Anyone setting them on a SimpleSolvers solver
  gets silence rather than a trace or an error. PR #35 implements `store_trace` at the
  GeometricOptimizers level, which is arguably the right layer since `solve!` is ours, but the upstream
  options remain misleading.

  **What to do**: file against JuliaGNI/SimpleSolvers.jl, naming which of the two
  defensible resolutions is preferred — implement the trace in SimpleSolvers' own solvers, or remove
  the three fields and let each caller own its trace, which is what GeometricOptimizers now does.
  Either is better than a field that accepts a value and discards it. If the first, this package's
  local implementation should be retired in favour of it.

  **Still open in 0.12**, and now confirmed from both sides: upstream's own open-issues list carries it
  under *"Reported by GeometricOptimizers.jl, not addressed in 0.12.0"*, naming the same three fields
  and the same two resolutions, and noting that both are breaking so neither belonged in a release
  driven by something else. The header of this entry moved from 0.11 to 0.12 and nothing else in it did.

### D5 · Documenter does not catch an `@ref` to a method signature that no longer exists

- location: —
- kind: upstream
- found: #36
- evidence:

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

  The `@extref` half behaves the *opposite* way and D8 below is the case: an external link that does not
  resolve is a hard error and fails the build. So the two halves of the same feature have opposite
  failure modes — a dead internal reference is silently redirected, a dead external one stops the
  build — which is worth knowing when a docs build suddenly fails after an upstream release.

### D7 · A step ceiling that binds can be reported as `LINESEARCH_FLOOR`

- location: —
- kind: upstream
- found: 2026-08-15; one issue with SimpleSolvers' *Open Issues* entry "A ceiling that binds can be reported as `LINESEARCH_FLOOR`"
- evidence:

  **Severity: medium**, inherited from SimpleSolvers 0.12 with the step ceiling and carried in
  upstream's own open-issues list. **Open upstream and without a consequence here**: the downstream
  half was B3, and it is closed — see *Fixed* above. This is the part that is not this package's to
  fix, and it stays listed because any *other* consumer of a capped search inherits it.

  `SimpleSolvers.capped_status` classifies the step at `αmax` by the same round-off rule `τ` as any
  other returned step. A merit that is *still falling* at the ceiling — which is what the capped case
  means — but has fallen by less than `τ` over the whole admissible range therefore comes back as
  `LINESEARCH_FLOOR`. That outcome is a claim about the **direction**, that no line search can make
  progress along it, and consumers act on it: SimpleSolvers' own solver through `flag_stall!` and
  `max_stalls`, and this package through `linesearch_rejected`, which throws `Q` away and re-searches
  along steepest descent. What was actually established is only that no step the *caller permits*
  decreases the merit measurably.

  Upstream is explicit that this is the same shape as the two unearned floors 0.12 removed from
  `Bisection`, reached through a third door, and that it is reachable exactly where the caller's ceiling
  is tightest — the case the ceiling exists for. It was left standing because the alternatives are not
  obviously better: `LINESEARCH_EXHAUSTED` would say "no step was found" about a step that was found and
  returned, and closing it properly means a boolean on the `LinesearchStatus`, which upstream declined
  on the grounds that the struct is copied per solver step and a caller who set the ceiling can compare
  it against `steplength`.

  Nothing measured here is affected at `DEFAULT_STEP_CEILING = 1`: all twenty sweep combinations
  reproduce their no-ceiling iteration counts or improve on them, so the path is not being taken.

  **What to do**: nothing, on either side. Upstream is where it is filed and the reasoning for leaving
  it there is sound; downstream it is closed, because `linesearch_rejected` now takes the ceiling
  `solver_step!` passed and exempts a `LINESEARCH_FLOOR` returned at it. That is what upstream meant by
  "a caller who set the ceiling can compare it against `steplength`", and it needed no new field on the
  status. The entry stays here as the record of *why* that comparison is in `linesearch_rejected` — a
  reader who finds the exemption and not this will think it is guarding against nothing.

### D8 · `SimpleSolvers.solve_with_status` has no binding-level entry to link to

- location: `docs/make.jl`
- kind: upstream
- found: 2026-08-15
- evidence:

  **Severity: low**, and it cost this package two documentation links this round.

  SimpleSolvers documents `solve_with_status` per *method*, so its `objects.inv` carries entries like
  `SimpleSolvers.solve_with_status-Union{Tuple{T}, Tuple{Linesearch{...}}}` and no bare
  `SimpleSolvers.solve_with_status` binding. `solve_with_status!` does have one. A binding-level
  `[`SimpleSolvers.solve_with_status`](@extref)` therefore cannot resolve, and — unlike the internal
  `@ref` case of D5 — DocumenterInterLinks makes that a **hard error** that terminates the build.

  This surfaced when SimpleSolvers' published docs were rebuilt for 0.12, and is *not* caused by
  depending on 0.12: the inventory is fetched from the `stable` URL regardless of which version is
  resolved, so `main` had the same broken build the moment those docs went up. Two docstrings here
  carried the reference — `solver_step!` and `linesearch_rejected` — and both are now plain code, which
  fixes the build and loses the link.

  **What to do**: ask upstream for a binding-level docstring on `solve_with_status`, as
  `solve_with_status!` already has; it is one `@docs` entry and it restores the link for every
  downstream package. Failing that, this package can pin the local fallback inventory that
  `docs/make.jl` already names — `docs/inventories/SimpleSolvers.toml`, which does not currently
  exist — so that an upstream docs rebuild cannot break this build again. The second is worth doing
  regardless: relying on a fetched inventory means a docs build that passes today can fail tomorrow with
  no commit here.

### K10 · `SimpleSolvers.alloc_h(::NetworkParameters)` sizes a manifold leaf by its dense storage

- location: SimpleSolvers `ext/SimpleSolversNeuralNetworkParametersExt.jl`, `alloc_h(ps::NetworkParameters)`
- kind: upstream
- found: 2026-09-27
- evidence: it sizes by `mapparameters(zero, ·)`, and `zero` of a Stiefel or Grassmann point is the
  point's own shape, not its lift (#21). For `NetworkParameters((a = St(6,3),)))` it returns
  `(18, 18)` where the intrinsic dimension is 12 (`origin/main` gave `(12, 12)`). This package does not
  call it for a non-vector: `BFGSState` sizes `Q` through `_alloc_q`, and `alloc_h(::Manifold)` is this
  package's own. The fix is upstream: size by the zero tangent, as `_alloc_q` does.

### K12 · Revise prints EMFILE errors in the test log

- location: `test/quality/jet.jl:18` (`using JET`)
- kind: upstream
- found: 2026-10-01
- evidence: JET 0.12 loads Revise, and Revise's file watcher runs out of file handles. Each
  failure prints an `UNHANDLED TASK ERROR: IOError: FolderMonitor: too many open files (EMFILE)`
  block into the log; no test fails and the totals do not change. A full run
  (`Pkg.test()`, Julia 1.13.1, JET 0.12.2) prints 7 such blocks and passes
  16343 tests; the same full run at `f353f10`, which has no JET testset, prints 0 and passes 16261,
  and the difference is the 82 tests of the JET testset.

### K15 · A sandboxed run with no matching Metal cache fails at the Metal precompile instead of skipping

- location: `test/devices/metal.jl:17` (`using Metal`)
- kind: upstream
- found: 2026-10-02
- evidence: Metal.jl 1.11.1's precompile workload (`src/precompile.jl:18`) calls
  `mtlfunction(identity, Tuple{Nothing})`, which calls `device()`. In a macOS sandbox that hides
  the GPU, `Metal.devices()` is empty, so Metal cannot precompile there, and
  `test/devices/metal.jl` never reaches its device skip. A default run on Apple silicon includes the `metal` group, so a
  sandboxed `Pkg.test()` is red whenever no Metal cache for its flags exists, as after a Metal
  update. A cache that a process outside the sandbox builds for the same flags loads in the sandbox,
  and the file then records its skip. Measured with Julia 1.13.1: the first sandboxed
  run of the `metal` group gave `Error 1`, with `Failed to precompile Metal` and
  `BoundsError: attempt to access 0-element Vector{Metal.MTL.MTLDevice} at index [1]` from
  `device()`; after one run of the same command outside the sandbox it gave `Broken 1`. The fix is
  upstream: a workload that skips the kernel compilation where no device exists.

## F. Loose ends from the geodesic-retraction review

Not a defect in the code; a thing a later reader would otherwise have to rediscover.

### K1 · The PR #36 description still says the MNIST scripts use the geodesic retraction.

- location: `docs/src/manifold_optimizers.md`
- kind: docs
- found: 2026-08-14
- evidence:

  The claim was
  corrected in `docs/src/manifold_optimizers.md`, but it also appears in the pull request body, where
  it is the stated justification for making `ScaledSquaring` the default. None of the five MNIST
  scripts passed `retraction`, so they all took the `Optimizer` default, which is `Cayley()`. The
  default is still the right choice — being free of dense LAPACK is reason enough — but if that PR
  body becomes a squashed commit message, the wrong reason goes into the history with it. The
  scripts have since moved to GMLDatasets.jl (see [0.3.1](#031)), which changes nothing
  about the PR body this entry is about.

## G. Found when this file was split from the CHANGELOG

### K2 · Two comments point at the *Open Issues* preamble of `CHANGELOG.md`, which does not hold it; the rule they cite is in neither file

- location: `scripts/optimizer_allocations.jl:8`
- kind: docs
- found: 2026-09-26
- evidence: `scripts/optimizer_allocations.jl:8` says "because of the rule the *Open Issues* preamble
  states" and `scripts/retraction_step_allocations.jl:9` says "reason the *Open Issues* preamble
  states". That preamble rule ("treat a number here as reproducible only where the harness that
  produced it is named") did not move into this file. Found by
  `git grep -n -i 'open issues' -- ':!CHANGELOG.md'`.

### K3 · The ID A22 was used for two different issues

- location: `KNOWN_ISSUES.md`
- kind: docs
- found: 2026-09-26
- evidence: commit 4d2ed06 (2026-08-20) added `#### A22. The only independent exponential
  implementation was CPU-only`, and 7f66695 (2026-08-23) removed it. Commit 4e9eb5f (#90,
  2026-09-18) used `A22` again for the SR decomposition. A commit message that cites A22 before
  2026-09-18 means the first issue.

### K4 · Three reference links in `CHANGELOG.md` have no definition

- location: `CHANGELOG.md:166`
- kind: docs
- found: 2026-09-26
- evidence: `[0.8.0]`, `[0.6.1]` and `[B]` (`CHANGELOG.md:166`, `:1186`, `:1330`, `:2091`) have no
  `[label]: url` line, and `[Unreleased]` compares `v0.6.0...main`.

## H. The test suite

### K22 · Three `[Unreleased]` bullets of `CHANGELOG.md` give test paths from before the move to `test/integration/`

- location: `CHANGELOG.md:33`
- kind: docs
- found: 2026-10-08
- evidence: `grep -n 'test/device_solve.jl\|test/manifold_linesearch_tests.jl' CHANGELOG.md` gives
  `:33` (`test/manifold_linesearch_tests.jl`), `:49`, `:51` and `:162` (`test/device_solve.jl`),
  and `:4347` (`test/manifold_linesearch_tests.jl`). Line 4347 is in the released 0.2.0 section and
  stays. Since #145 the files are `test/integration/manifold_linesearch_tests.jl` and
  `test/integration/device_solve.jl`. The bullets are not released, so the paths may be corrected.

### K23 · The `Downgrade` job cannot resolve the test environment: its `LLVM` 10.1.0 excludes every `Metal` that `test/Project.toml` allows

- location: `test/Project.toml:32` (`Metal = "1.10"`)
- kind: defect
- found: 2026-10-03
- evidence: the advisory `Downgrade - ubuntu-latest` job fails before any test runs. CI run
  37951076527 at `2114a3b` (Julia 1.12.7):

  ```
  ERROR: LoadError: Unsatisfiable requirements detected for package LLVM [929cbde3]:
   LLVM [929cbde3] log:
   ├─possible versions are: 0.9.0 - 10.1.0 or uninstalled
   ├─restricted to versions 10.1.0 by an explicit requirement, leaving only versions: 10.1.0
   └─restricted by compatibility requirements with Metal [dde4c033] to versions: 7.2.0 - 9.13.2 — no versions left
     └─Metal [dde4c033] log:
       ├─possible versions are: 0.0.1 - 1.11.1 or uninstalled
       └─restricted to versions 1.10.0 - 1 by project [fff0a915], leaving only versions: 1.10.0 - 1.11.1
  ```

  The job resolves the direct dependencies of the root `Project.toml` to their floors and every
  indirect dependency to its newest version, and keeps that manifest (`allow_reresolve: false`).
  No floor pins `LLVM`: 10.1.0 is its newest version. The General registry's compat entries give
  the chain: the `KernelAbstractions` floor 0.9.38 allows `Atomix` 1, `Atomix` 1.6.0 needs
  `UnsafeAtomics` 0.4, and `UnsafeAtomics` 0.4.0 needs `LLVM` 10. The log does not print the
  manifest, so the chain is read from the registry, not from the run. The test environment then
  adds `Metal`, and `Metal` 1.11.1 needs `LLVM` 7.2 to 9. CI run 37122615676 at `0aa622b` (Julia
  1.11.9) fails the same way; there the explicit requirement is `UnsafeAtomics` 0.3.3, its newest
  version at that time, which restricts `LLVM` to 10.0.0. The required jobs resolve without the
  floor manifest and pass.

### K34 · The exact-zero allocation assertion of `retraction_workspace.jl` failed in two rows of one count run under heavy machine load

- location: `test/retractions/retraction_workspace.jl:105`
- kind: not verified
- found: #156
- evidence: a per-testset count run on `417eff0`, at a machine load of 12 to 32, reported one failed
  assertion in each of two rows of the testset "a workspace built for its retraction, for a
  $(nameof(LT)){$T}": `StiefelLieAlgHorMatrix{Float32}` with 27 passed and 1 failed, and
  `GrassmannLieAlgHorMatrix{Float64}` with 27 passed and 1 failed. The only allocation assertion
  in that testset is the exact zero `@test _measured_differential!(D, own, Cayley(), B, 0.5) == 0`
  at line 105, which is unchanged since `main`. The failing assertion was not captured. Both full
  suites on `4a6883c` (Julia 1.13.1 and the 1.12 floor, at a load of 3 to 5) pass the file 572/572.
  The cause is not established.

[#38]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/38
[#40]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/40
[#44]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/44
[#45]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/45
[#52]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/52
[#54]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/54
[#111]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/111
