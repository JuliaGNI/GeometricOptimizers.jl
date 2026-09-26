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

### A10 · `state.ḡ` is two iterates behind for the three first-order states

- location: `test/optimizer_state_initialization.jl`
- kind: defect
- found: #38
- evidence:

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

### A24 · `Δf` spans two iterations for the three first-order states

- location: `optimizer_status.jl:103`
- kind: defect
- found: 2026-09-21
- evidence:

  **Severity: low** — it moves when convergence fires, not whether a solve is correct. Found in the
  review of the `value`/`previous_value` work in 0.6.0, and it is the `f̄` half of A10: same states,
  same root cause, `update!(state, opt, x)` running after the step. **Pre-existing on `main`.**

  `OptimizerStatus` computes `Δf = f - state.f̄` (`optimizer_status.jl:103`), reading the field and not
  the accessor. `scripts/optimizer_status_delta_f.jl` asks, per optimizer method, whether that `Δf`
  equals the one-step difference `f[end] - f[end-1]` or the two-step difference `f[end] - f[end-2]`,
  taking the objective values from a stored trace. Exact equality, so there is no tolerance to tune.
  On `f(x) = Σ(x⁴ + x²)` from `[1.0, 2.0, 3.0]`, six iterations:

  | method | `Δf` spans |
  |:--|:--|
  | `GradientMethod` | **two steps** |
  | `MomentumMethod` | **two steps** |
  | `Adam` | **two steps** |
  | `BFGS` | one step |
  | `DFP` | one step |
  | `Newton` | one step |

  Two consumers, both stale for the first three:

  - `rfₐ = norm(Δf)` and `rfᵣ = rfₐ / norm(f)` (`optimizer_status.jl:109-110`), which are what
    `f_converged` tests (`:397`). During monotone descent a two-step `Δf` overstates the decrease, so
    `f_converged` fires late rather than early — the safe direction, which is why this is low and not
    medium.
  - `f_increased = f > state.f̄` (`:130`), which reads the same stale field and is one of the two
    guards on `x_converged` (`:391`). Here the comparison is against an objective two iterations old,
    so an iterate that rose against its immediate predecessor can still read as a decrease.

  `BFGSState` holds one iterate and one objective rather than a pair, which is why it spans one step
  and why it gains `previous_value` and no `value` in 0.6.0. `NewtonOptimizerState` holds a pair and
  shifts like the first-order states, and still spans one step only because `optimizer.jl:431` calls
  `update!` a second time inside `solver_step!` and re-synchronises `f̄` before the status reads it.
  That line carries the comment `# this will have to be removed later`; removing it moves `Newton`
  into the two-step column, and the script is what would catch that.

  **What to do.** The fix is the one A10 names, one level up: `update!(state, opt, x)` should store the
  objective and the gradient that belong to the `x` it is storing. It cannot be taken for `f̄` alone,
  because `f_increased` and `Δf` read the same field and would move together, and because
  `INITIAL_BFGS_F` — the first-iteration sentinel `optimizer_status.jl:124-129` describes — is
  calibrated against the present ordering. Settle it with A10 and with issue #108, which is the third
  face of the same ordering: a state read after `solve!` returns lags the returned iterate by one.

### A12 · The `Cayley` differential is recomputed per `φ'`, and its cost is unmeasured

- location: `svd_optim.jl`
- kind: not verified
- found: #40
- evidence:

  **Severity: low**, and not a defect — a cost this release introduced and did not measure. Found in
  the review of [#40], where `retraction_differential` was added.

  Under `Cayley`, `trial_slope` now calls `retraction_differential` on every evaluation of ``\varphi'``.
  That is `lift_factors`, a `StiefelProjection`, two ``2n\times{}2n`` solves and about six allocations —
  ``O(Nn^2 + n^3)``, the same order as the retraction itself — where before it was a `_dot` against an
  array the cache already held. `Geodesic` returns ``\bar{B}`` untouched at every ``\alpha`` and
  `Cayley` does at ``\alpha = 0``, so every `Geodesic` solve and the `Backtracking` default pay nothing;
  what is unmeasured is a search that evaluates ``\varphi'`` many times per iteration, which on this
  problem is `Bisection` at ≈580 objective evaluations per iteration.

  **The iteration and evaluation counts in `svd_optim.jl` do not answer this.** They moved under the
  change — `_BFGS + Bisection` under `Cayley` from 92 to 114 iterations — but they moved because the
  trajectory changed, so they measure a different solve rather than the cost of a step. Nothing here
  is a wall-clock measurement.

  The obvious remedy if it does turn out to matter is not a cache but a shared factorisation:
  `linesearch_problem`'s `d(α, params)` calls `trial_iterate!` and then `trial_slope` with the *same*
  ``\alpha``, and both go through `lift_factors` — the first on ``\alpha\bar{B}`` and the second on
  ``\bar{B}`` — so one line search evaluation factors the same lift twice. Fusing them would need
  `trial_iterate!` to hand its factors on, which is a wider change to that interface than a cost
  nobody has measured justifies.

### A13 · `Newton`'s state advances its frame by the gradient, not by the step

- location: `src/optimizers/optimizer.jl`
- kind: defect
- found: 2026-08-14
- evidence:

  **Severity: low** — it costs an evaluation and not an answer. Found while fixing A8, which is what
  made the difference visible: with the gradient reuse in place, `Newton` is the one method that cannot
  have it.

  `update!(state::NewtonOptimizerState, opt, x)` (`src/optimizers/optimizer.jl`, the line already
  marked "this will have to be removed later") ends with

  ```julia
  update_section!(state.section, gradient_array(cache(opt)), x -> retraction(opt.retraction, x))
  ```

  i.e. it advances the state's `GlobalSection` by ``\nabla{}f`` where every other state
  advances it by the direction the step was taken along. For Euclidean parameters `update_section!` is
  ``\Lambda^t.Y \gets \Lambda^{t-1}.Y + B``, so this is not a formality: after a step the cache's frame
  holds ``Y + \delta`` and the state's holds ``Y + \nabla{}f``.

  Two consequences, both of them about `store_gradient!`'s reuse guard, which requires
  `section(cache) == section(state)`:

  - **`Newton` pays one gradient evaluation per iteration that `_BFGS` and `_DFP` do not.** The frames
    do not match, so the guard declines the reuse and the cache evaluates ``\nabla{}f`` afresh at the
    point `refresh_latest_gradient!` has just evaluated it at. Measured on Rosenbrock from
    ``(-1.2, 1)`` with `Backtracking(expand)`: 103 gradient evaluations over 26 iterations before A8,
    124 over 25 after.
  - **The reuse comes back by accident once the iteration has nowhere left to go.** Once ``\nabla{}f``
    and ``\delta`` have both gone to zero the two frames agree again and the guard fires. That is
    *correct* — on Euclidean parameters `global_rep` is the identity, so the value depends only on
    `solution(cache) == x`, which the guard also checks — but it means the branch taken is not a
    property anything should assert on. `test/optimizer_tests.jl` asserts on the gradient the direction
    is built from instead, and says so. Measured by calling `latest_gradient_is_current` at the top of
    each of 30 forced `solver_step!`s: the guard fires on 3 of the 30 on Rosenbrock — all of them past
    the iteration the solve stops at, which is why the evaluation count above does not move — and on 28
    of the 30 on ``\sum{}x^2``, where `Newton` reaches the minimiser in one step and both quantities are
    zero from then on.

  The fix is to advance the state's section by `direction(cache(opt))` like every other state, after
  which the reuse is available to `Newton` too and the extra evaluation goes away. It needs a
  re-measurement of the `Newton` rows and nothing else — `NewtonOptimizerCache` is `AbstractArray`-only,
  so no manifold path reaches this.

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

### A18 · `𝔄` and `𝔄exp` accept an `AbstractExponentialAlgorithm` they cannot serve

- location: `docs/src/retractions.md`
- kind: defect
- found: #45
- evidence:

  **Severity: low**, and a signature that is wider than the implementation rather than a wrong answer.
  From the review of [#45], where `𝔄exp` was added.

  `𝔄(X, algorithm)` is implemented for [`TaylorSeries`](@ref), [`ScaledSquaring`](@ref) and
  [`AugmentedPade`](@ref). [`ProjectedSkew`](@ref) is the fourth `AbstractExponentialAlgorithm` and has
  no `𝔄` method at all: it specialises `geodesic` directly, because it exponentiates the lift in an
  orthonormal basis of its range rather than going through ``\mathfrak{A}`` — see the *Disadvantages*
  paragraph on `docs/src/retractions.md`, which already tells readers that `𝔄(X, ProjectedSkew())` does
  not exist.

  The signatures `𝔄(B̂, B̄, ::AbstractExponentialAlgorithm)` and `𝔄exp(B̂, B̄, ::AbstractExponentialAlgorithm)`
  nevertheless accept it, so `𝔄exp(B̂, B̄, ProjectedSkew())` dispatches, forwards, and dies one frame in
  with a `MethodError` naming `𝔄` — not the function that was called, and not the fact that this
  algorithm lives a level up. `𝔄exp` inherits the hole rather than adding one, and narrowing only
  `𝔄exp` would put the two out of step, which is why it was left as it is.

  **What to do**: either give `𝔄` a `ProjectedSkew` method that errors with the explanation — that it
  is a `geodesic`-level algorithm, and to call `geodesic(B, ProjectedSkew())` — which fixes both
  entry points at once and costs one method; or introduce the subtype of `AbstractExponentialAlgorithm`
  that the three ``\mathfrak{A}``-level algorithms share and narrow both signatures to it, which makes
  it a `MethodError` at the call site instead of a frame in. The first is cheaper and says more; the
  second is the one that makes the type hierarchy match what is implemented.

### A19 · `ScaledSquaring`'s GPU claim is untested here

- location: `docs/src/retractions.md`
- kind: not verified
- found: #45
- evidence:

  **Severity: unknown, which is the point.** From the review of [#45]. **Half of it is fixed** — see
  *Fixed* under [0.4.2](#042) — and this is the half that is not.

  The documentation states the property in three places and rests the default on it:
  `docs/src/retractions.md` calls [`ScaledSquaring`](@ref) one of the two usable algorithms that run
  unchanged on a `KernelAbstractions` GPU backend and the default because it is the cheaper of them,
  and `docs/src/manifold_optimizers.md` repeats it. `GeometricOptimizers.opnorm₁` exists solely to keep
  it: its docstring says `LinearAlgebra.opnorm(X, 1)` "is a scalar-indexing double loop, and scalar
  indexing is exactly what a GPU array cannot serve."

  Two things sat against that, both verified by reading. The first turned out to be a defect rather
  than a doubt, and is fixed:

  - `𝔄(A)` — the series `ScaledSquaring` sums, on the ``2n\times{}2n`` argument — opened with
    `Aⁿ = one(A)` and `𝔄A = one(A)`. `Base.one(::AbstractMatrix)` is `Base._one` in
    `base/abstractarray.jl`, which does `similar`, `fill!` and then **a scalar-indexed loop over the
    diagonal**. So the path was not "nothing but matrix products and norms"; it reached the same
    construct `opnorm₁` was written to avoid, one level down. `AbstractLieAlgHorMatrix` had a
    `Base.one` of its own that is a KernelAbstractions kernel precisely to avoid this — but `𝔄`'s
    argument is a bare matrix, so that method did not apply to it. (It was written for
    `StiefelLieAlgHorMatrix` alone and moved to the abstract type in [0.2.2](#022), which is why the
    Grassmann retraction was on the scalar-indexed path as well until then; that was *a piece of* this
    entry and not a fix for it.) **Fixed in [#54]**, which is where the claim also stopped being
    untested against an array type that errors on scalar indexing. What remains is the second bullet,
    and a `JLArray` does not settle it — it reproduces the *failure mode* of a GPU backend, not the
    backend.
  - **No run in this repository exercises it**, and there is no GPU code left here to change that:
    `mnist_cuda.jl`, `mnist_metal.jl`, `mnist_metal_short.jl` and `metal_memory_probe.jl` were all of
    it, and they moved to [GMLDatasets.jl](https://github.com/JuliaGNI/GMLDatasets.jl) with the rest of
    the MNIST material. None of the five MNIST scripts passed `retraction`, so all of them took the
    `Optimizer` default, which is `Cayley()` — the same fact recorded under F below. The 6 h 53 min
    RTX 4090 run whose figures that package's documentation carries therefore never called `geodesic`,
    `𝔄` or `ScaledSquaring` at all.

  What is *not* established is whether this actually breaks. CUDA.jl and Metal.jl error on disallowed
  scalar indexing, which would make it a hard failure rather than a slow one, but no GPU was available
  to the review and the claim is not being called false — only unverified, with a specific reason to
  doubt it and a one-line way to find out.

  **What to do**: run `geodesic(60 * rand(StiefelLieAlgHorMatrix{Float32}, 20, 3) |> gpu)` on a CUDA or
  Metal backend, with `NativePade` alongside it. The identity half of this is done — `𝔄` builds it the
  way `one(::AbstractLieAlgHorMatrix)` does, through the shared `unit_matrix` — so what is left is the
  transcript, which is worth having given that three documentation passages depend on it and that the
  one thing already found here was found by reading rather than by running. Do it together with C14,
  which decides where the identity is assembled.

### A20 · `default_gradient` has no `Manifold` method and silently takes the `AbstractArray` one

- location: `src/optimizers/optimizer.jl:175-176`
- kind: defect
- found: #46
- evidence:

  **Severity: medium**, and narrow — it is unreachable through the constructor anyone actually calls.
  From the review of [#46], found by checking that PR's claim about what it left open against merged
  `main` instead of against the wording of [#27]. Pre-existing; [#46] neither causes nor fixes it.

  `default_gradient` has two methods (`src/optimizers/optimizer.jl:175-176`):

  ```julia
  default_gradient(problem::OptimizerProblem{T}, x::AbstractArray) where {T} = GradientAutodiff{T}(problem.F, length(x))
  default_gradient(problem::OptimizerProblem, x::ArrayNamedTuple) = GradientAutodiff(problem.F, x)
  ```

  `Manifold <: AbstractMatrix`, so a bare manifold takes the first. That builds the gradient from the
  *length* and composes `problem.F` with a flat vector, where the point of `GradientAutodiff(F, ::Manifold)`
  — added in [0.2.2](#022) at `src/utils.jl:30` — is that it rebuilds the manifold before
  calling `F`. The `NamedTuple` method is already the delegating form, and the reason it exists is
  recorded in `default_gradient`'s own docstring one screen up: a `Gradient` for a `NamedTuple` "has to
  be constructed from `x` itself". A bare manifold is the same argument and did not get the same
  treatment.

  It does not raise where it is built. It raises at the first gradient evaluation, and only on an
  objective that names its argument type — so an `F(Y) = -tr(Y'MY)` written without the annotation
  computes something plausible off the flattened vector and never says anything. Reproduced on `main`
  at 6166479:

  ```julia
  F(Y::GrassmannManifold) = -tr(Y' * M * Y)
  x = rand(GrassmannManifold{Float64}, 3, 1)

  default_gradient(OptimizerProblem(F, x), x)(x)
  # MethodError: no method matching F(::Vector{ForwardDiff.Dual{…}})

  GradientAutodiff(F, x)(x)          # the method [#46] added, for comparison
  # [-0.28344900891714625; -0.660038737272924; -0.4099740609429401;;]
  ```

  The same on a `StiefelManifold`, which is why this is not a Grassmann entry: it is the half of [#27]'s
  second bullet that survived A11.

  Nothing in the test suite reaches it. `Optimizer(x, F; …)` builds its own gradient at
  `src/optimizers/optimizer.jl:244` and never consults `default_gradient`; the constructor that does is
  the lower-level `Optimizer(algorithm, problem, hessian, cache, linesearch; gradient = default_gradient(problem, cache.x), …)`
  at `:156`, and `_optimizer` at `:201`. So the gap is real and unreachable by the documented entry
  point at the same time, which is the reason it outlived a PR that fixed everything around it.

  **What to do**: one method,

  ```julia
  default_gradient(problem::OptimizerProblem, x::Manifold) = GradientAutodiff(problem.F, x)
  ```

  next to the two above, and a test that goes through the `:156` constructor with an objective that
  annotates its argument — the annotation is the part that matters, since without it the wrong gradient
  is silent rather than loud. Worth doing together with the other half of [#27], `mode = :finitediff`
  (`:246`), which has no `Manifold` method either and no `NamedTuple` one ([#24]); the three are one
  subject, which is "every entry point that builds a gradient should agree about what a manifold is".

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

### A21 · The optimizer interface cannot hold GPU arrays

- location: `MNIST_PORT.md`
- kind: defect
- found: #14
- evidence:

  **Severity: medium**, and a regression rather than a gap: `GeometricMachineLearning`'s optimizers ran
  on `CUDABackend()`, and the port of [#14] could not keep that. Found by that port, recorded in the
  `MNIST_PORT.md` it wrote, and moved here when the MNIST material left (see [0.3.1](#031)
  above) — this entry is that file's finding restated against current `src/`, not a new measurement.

  **This entry loses the sharpest half of its first bullet in
  [0.5.0](#050).** The scalar-indexing fallback it described was
  `ParameterHandling`'s, and that dependency is gone: `NeuralNetworkParameters.flatten!` writes each leaf
  with one `copyto!(v, doffs, x, firstindex(x), n)` over a range the layout already knows, and indexes no
  element, so there is nothing left for a device array to fall through *to*. What is restated below is
  what survives, which is a transfer per step rather than an error.

  The parameters of a GPU run stay on the host. Two independent things put them there:

  - **The per-step flattening, now as a transfer rather than a scalar-indexing fallback.**
    `(grad::Gradient{T})(nt::ArrayNamedTuple{T})` (`src/optimizers/named_tuple_wrapper.jl:21`) flattens
    on *every* gradient evaluation, and `flatten` allocates its destination as a
    `Vector{T}(undef, length(layout))` — a **host** vector, whatever the leaves are. `unflatten` is the
    same boundary in reverse: it slices `v[l.range]`, reshapes, and hands `rebuild` a host array, so the
    parameter set it returns is host-resident even if the one it was built from was not. A device run
    therefore pays a download and an upload of the whole parameter set per step, which is the cost
    measured below — a transfer that works rather than scalar indexing that does not, but the parameters
    still do not stay resident.
  - **The state.** `_similar(a::Manifold{T}) = rand(manifold_constructor(a){T}, size(a)...)` at `:46`
    goes to `rand(manifold_type, N, n)` (`src/manifolds/abstract_manifold.jl:110`) and from there to
    `rand(CPU(), …)` at `:43` — always the host, whatever `a` is. It backs `x̄` and the `BFGS`/`DFP`
    caches, so even a flattening that stayed on the device would leave the state mixing host and device
    arrays. Untouched by the dependency change.

  What this cost the run it was found in is small: the optimizer touches only the parameters — 154938
  of them, 620 kB in `Float32`, so ≈1.2 MB uploaded and downloaded per step — against ≈3 GB of
  device-side activations in the forward and backward passes, which stayed on the device throughout.
  That is why the port left it, and it is a statement about that network and not about the interface.
  A parameter set large enough to be worth keeping resident would pay the transfer on every step.

  The rest of `src/` is written against `KernelAbstractions` and *looks* backend-agnostic; whether it
  is has not been established, and A19 above is one specific reason to doubt it.

  **What to do**: the flattening is now `NeuralNetworkParameters`' to fix rather than this package's —
  what it needs is a `flatten` that allocates its destination on the backend of the parameters, and a
  `FlatParameters` that keeps it there, which is an upstream request and not a method here. On this side
  what is left is to thread the backend through `_similar` —
  `rand(backend, MT{T}, N, n)` already exists (`src/manifolds/abstract_manifold.jl:70`, allocating
  through `KernelAbstractions` at `:28`), and `KernelAbstractions.get_backend` is how `global_section`
  and `Base.zero` already find the backend of a point they are given
  (`src/manifolds/stiefel_manifold.jl:128,143`). Neither is large, and neither is worth doing blind:
  the check is a GPU run of the optimizer, which nothing in this repository does any more, so this
  should be closed together with A19 — one backend, one session, both claims settled.

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

### A27 · `ProjectTo` is pinned as the Frobenius projection, and the storage gradient is not yet derived from it

- location: —
- kind: defect
- found: #111
- evidence:

  **Severity: medium.** Found in the review of [#111], in investigation of why a
  code path that reads a `ProjectTo` cotangent's storage as `∂L/∂S` gives the wrong
  gradient.

  `ProjectTo` on a `SymmetricMatrix` or `SkewSymMatrix` gives the natural cotangent
  (the Frobenius projection `½(Ā ± Āᵀ)` of a dense cotangent). A new test holds it to
  that: its pairing with every storage direction matches a central difference, and a
  weight used twice (two cotangents added, then projected again) gets the projection of
  the sum. That representation is kept because AD can add and re-project it.

  Returning `∂L/∂S` from `ProjectTo` instead was tried and rejected: Zygote adds the
  two cotangents of a weight used twice as dense matrices and projects the sum again,
  which doubled the off-diagonal entries (ratio 2.0 against finite differences), and a
  dense Zygote-native cotangent mixed in gave ratios from −1.6 to 3.2.

  A caller that forms the flat gradient of a parameter set through Zygote and passes
  its storage as `∂L/∂S` gets half of each off-diagonal entry: the ratio of the storage
  gradient, by finite differences, to the storage of the Zygote cotangent is 1 on the
  diagonal and 2 off it for a `SymmetricMatrix`, and 2 for every entry of a
  `SkewSymMatrix`, whose storage holds only off-diagonal entries (n = 3, Zygote 0.7.13).
  A `GradientMethod` or `MomentumMethod` step taken on that storage therefore moves
  those entries by half; this follows from the cotangent and was not measured through
  an optimizer step. The conversion to `∂L/∂S` (the lower triangle of
  `G + Gᵀ` with the diagonal counted once, or of `G − Gᵀ`) belongs where an AD
  cotangent becomes a parameter gradient, not in `ProjectTo` itself.

  **What to do**: The mechanism is a step in the optimizer's parameter update after
  Zygote adds cotangents, and before the gradient is applied. Which package owns it
  (GeometricOptimizers or NeuralNetworkParameters) is a design decision.

### A28 · `SymplecticStiefelManifold` has no `rebuild`, so `changebackend` refuses it

- location: `src/parameter_protocol.jl`
- kind: defect
- found: 2026-09-24
- evidence:

  **Severity: low.** `src/parameter_protocol.jl` gives the symplectic Stiefel manifold a
  `freeparameters` and no `rebuild`, so `mapstorage` raises an `ArgumentError` for it, and
  `changebackend`, which walks the parameter protocol, raises with it. The Stiefel and Grassmann
  points and every structured matrix move between backends. Found by `scripts/device_products.jl`.

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

  Whichever way this goes, it has to be settled together with A10: `Δf̃` is its only input, and for the
  three first-order methods that is a two-step-stale gradient paired with the current direction. Using
  `f_converged_strong` without fixing A10 would be acting on a prediction that is not one; deleting it
  retires A10's second consumer along with it.

  **What to do**: either *use* it as the stall detector `Options.max_stalls` and
  `Options.f_stall_window` were meant to drive — count consecutive iterations that fail it, stop after
  `max_stalls` — or *delete* it from `convergence_measures`' return tuple and say in the docstring that
  the outer-iteration Armijo test is not implemented. Deleting is the honest default. Using it is worth
  more but is a behaviour change that needs its own measurement over the eight starting points, so it
  must not ride along in an observability PR; split it out if that is the choice.

### C2 · `compute_direction!` for the quasi-Newton methods is dead

- location: `src/optimizers/iterative_hessians/iterative_hessians_direction.jl:1-3`
- kind: dead code
- found: 2026-08-14
- evidence:

  `src/optimizers/iterative_hessians/iterative_hessians_direction.jl:1-3` defines
  `compute_direction!(opt, ::Union{BFGSState,DFPState})`. There is no call site (verified: the only
  live callers are the `Newton` methods in `newton_optimizer_direction.jl`). The direction is formed
  inline at the end of the cache `update!` instead — `bfgs_cache.jl`, `dfp_cache.jl`.

  **What to do**: delete the file and its `include`. The alternative — routing
  `solver_step!` through `compute_direction!` for symmetry with `Newton` — is a refactor with no
  behavioural gain, and the inline form is what the `ḡ`-ordering fix depends on. Deleting is the
  smaller and clearer change.

### C5 · `_DFP` + `Backtracking(expand = true)` is documented rather than run, on stale grounds

- location: `test/optimizer_convergence/svd_optim.jl`
- kind: missing test
- found: 2026-08-14
- evidence:

  `test/optimizer_convergence/svd_optim.jl` excludes that pair because its iteration count ranged
  `512..77_890` over eight starting points. The curvature condition in PR #35 brings that to
  `385..1_118` (`Geodesic`) and `466..1_177` (`Cayley`), comfortably inside the 5 000 cap the file
  already uses, so the stated reason no longer holds. Left out only because there is no CI measurement
  of the post-fix spread yet — the original surprise was a factor of four between platforms.

  **What to do**: add the pair to the driver loop in `svd_optim.jl` at the existing 5 000 cap and
  rewrite the comment that explains why it is not run. Gate it on having seen one green CI run on
  Linux *and* Windows, because the local spread does not measure the thing that surprised us.

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

### C8 · `svd_optim.jl`'s table and the script's `COMBINATIONS` are not the same ten rows

- location: `svd_optim.jl`
- kind: docs
- found: #40
- evidence:

  **Severity: low**, bookkeeping. Found in the review of [#40], while making the "regenerated by
  `scripts/retraction_accuracy.jl`" claim in that table true.

  Both are ten (method, line search) pairs and eight of the ten agree. The two that do not:

  - **`_DFP + Backtracking`** is in the table and not in `COMBINATIONS`. That is deliberate — at 48 322
    iterations on the pinned seed it would dominate the runtime of every sweep, and its only purpose in
    the table is the `α = 1` ceiling argument below it — and it now says so in place. Its spread
    (`10_448..114_116`) is an older measurement at a cap high enough not to bind, which is why it
    exceeds the `SVD_MAX_ITERATIONS = 20_000` the rest of the column is quoted against.
  - ~~**`_BFGS + StrongWolfe(c₂ = 0.1)`** is in `COMBINATIONS` and not in the table.~~ **Closed.** It
    was measured on every run of the sweep and printed to nobody; the row is in the table now
    (`135 / 135` iterations, `7 893 / 7 880` evaluations), written down while re-measuring for the step
    ceiling. The two options this entry offered were to add the row or drop it from `COMBINATIONS`, and
    adding it turned out to cost one line rather than the sweep time the entry assumed — it was already
    being computed.

  So one of the two discrepancies is closed and the other is documented rather than removed. The general
  point stands: a table that names a script as its source should be checkable against that script row by
  row, or say which rows are exceptions and why. `svd_optim.jl` now marks its three non-regenerated
  cells explicitly, which is what makes the remaining gap safe.

  **The first bullet is why this is worth doing rather than filing.** Running the A8 sweep found that
  the `_DFP + Backtracking` row had been *wrong* on `Geodesic` since the curvature-condition fix —
  47 115 iterations and 1 177 919 evaluations where the same harness measures 48 322 and 1 208 157, and
  where `default_linesearch`'s own table said 48 322 all along. Neither table is regenerated by the
  sweep, so nothing compared them. Both are corrected and both now name the harness and the cap; a row
  that no script regenerates is a row that goes stale silently.

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
  - **the MNIST run**, as of [0.3.1](#031) above: the 6 h 53 min RTX 4090 figures that A19
    and A21 rest on, the ``\sqrt{1.8} \approx 1.342`` plateau and the per-configuration losses are
    still quoted here, while `distill_mnist_results.jl` and the five scripts that produced them are now
    in GMLDatasets.jl. This is the one entry on the list whose harness *exists* and is merely elsewhere,
    which makes it the mildest case and the easiest to get wrong: a figure quoted in this repository
    and regenerated in another one goes stale exactly as quietly, and nothing here will notice.

  The pattern is C8's worked example: a number that no committed script regenerates is a number that
  goes stale silently, and the ones this catalogue has caught were all of that kind. Either put them
  under `scripts/` next to `retraction_accuracy.jl`, or accept the rule the preamble already states and
  stop quoting figures that nothing can re-run.

### C10 · The eight-seed sweep is now inside `MANIFOLD_TOLERANCE` and is still not a test

- location: `svd_optim.jl`
- kind: missing test
- found: 2026-08-15
- evidence:

  **Severity: low**, and it is the cheapest open item here.

  `svd_optim.jl` used to say that enabling the eight-seed sweep as a test would need either
  `ProjectedSkew` or a tolerance of `1e-11`, because the worst `check` over the eight was `2.8e-12`
  against a `MANIFOLD_TOLERANCE` of `1e-12`. The step ceiling removed that outlier — it was one
  over-long step and not the accumulation the file attributed it to — and the worst `check` over all
  twenty combinations and all eight seeds is now `2.5e-13`. **The stated obstacle is gone.**

  What is in the suite instead is the four A1b cases at seeds 2 and 8, added with the ceiling. That
  covers the defect that was found and not the twenty-by-eight surface the sweep measures, which is
  where both A1b and the previously unnoticed `Geodesic` 7-of-8 rows were found in the first place —
  neither by the pinned seed the suite otherwise runs.

  **What to do**: run the sweep in CI, not in `Pkg.test()` — at `SVD_MAX_ITERATIONS = 20_000` it is
  minutes rather than seconds, which is why it is not simply added to the driver loop. A scheduled
  workflow asserting `on_the_manifold(...) == 8` for every row and `rg < CONVERGED_GRADIENT_TOLERANCE`
  would have caught A1b years earlier than a sweep someone remembered to run. Gate on one green run per
  platform first, as C5 asks for the same reason.

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

### C13 · `MANIFOLD_TOLERANCE` is defined three times

- location: `test/optimizer_convergence/svd_optim.jl:19`
- kind: defect
- found: #44
- evidence:

  **Severity: low**, and the one on this list with a way to go wrong quietly. From the review of [#44].

  `const MANIFOLD_TOLERANCE = 1e-12` appears in `test/optimizer_convergence/svd_optim.jl:19`,
  `test/manifold_linesearch_tests.jl:45` and — added with the step ceiling —
  `scripts/retraction_accuracy.jl:184`. Three copies of one number with no import path between them: a
  script cannot `include` a test file that runs a suite as a side effect, and the constant is a property
  of the tests rather than of the package, so it does not belong in `src/`.

  The reason it matters more than ordinary duplication is what the third copy does. `on_the_manifold`
  counts seeds against it, and that count is what every "8 of 8" in this release means. If the script's
  copy and the suite's copy ever drift, the sweep and the tests will disagree about whether a solve is
  on the manifold and nothing will say so — the sweep is not run in CI (see C10), so the disagreement
  would surface as a table that no longer matches a passing suite.

  **What to do**: one `test/manifold_tolerance.jl` holding the constant and a comment, `include`d by all
  three. That the script reaches into `test/` is already true — it takes its matrix from
  `test/optimizer_convergence/svd_matrix.jl` — so this adds no new coupling, only removes two copies.

### C14 · `geodesic` and `𝔄exp` assemble the same product independently

- location: `retractions.jl:128`
- kind: dead code
- found: #45
- evidence:

  **Severity: low**, and a duplication that was created deliberately rather than found. From the review
  of [#45], where `𝔄exp` was added.

  Both compute ``\mathbb{I} + B'\mathfrak{A}(B', B'')(B'')^T``:

  ```julia
  geodesic(B, algorithm) = manifold_type(B)(one(B) + B̂ * 𝔄(B̂, B̄, algorithm) * B̄')   # retractions.jl:128
  𝔄exp(B̂, B̄, algorithm) = I + B̂ * 𝔄(B̂, B̄, algorithm) * B̄'                          # modified_exponential.jl
  ```

  `𝔄exp` was added as a name for what `geodesic` already did, not as a replacement for the inline
  expression, on the grounds that `geodesic` also takes the lift apart and wraps the result. Both of
  those are one call each, so `manifold_type(B)(𝔄exp(lift_factors(B)..., algorithm))` is the whole of
  it, and the reason not to fold them is thinner than it looked when the two lines were written a
  commit apart.

  Two things now live in two places rather than one. The **default algorithm** is the first: both say
  `ScaledSquaring`, and they have to, because a lift retracted through `geodesic` and the same lift
  exponentiated through `𝔄exp` are meant to agree — a testset asserts exactly that, which is a test
  existing to catch a duplication rather than a defect. The **identity** is the second, and the two do
  not spell it the same way: `geodesic` uses `one(B)`, the KernelAbstractions kernel on
  `StiefelLieAlgHorMatrix`, and `𝔄exp` uses `I + …`, whose `LinearAlgebra` method writes the diagonal
  by scalar indexing. Whether that difference costs anything is A19's question.

  **What to do**: decide A19 first, since it decides how the identity should be built, then have
  `geodesic` call `𝔄exp` and delete the inline expression. The default then has one home, and the
  testset that pins the two together can go with it.

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

  `BFGSCache`, `BFGSState`, `DFPCache`, `NewtonOptimizerCache`, `NewtonOptimizerState` and the `VT` of
  `OptimizerResult` were unbound afterwards on the strength of their *inferred types* — for the
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
  portability argument rather than a numerical one. And no *actual* GPU backend has been exercised: the
  `JLArray` test settles scalar-index freedom, which is the failure mode, but A19 above is still open
  for the backend itself.

  **What to do**: derive a ``\varphi_1`` backward-error bound for a non-normal argument, or state
  plainly in the docstring that no such bound is claimed and that the threshold is a measured one — the
  latter is what is written today, and it is enough to use the algorithm honestly but not enough to
  close [#52]'s first acceptance criterion. Whichever, it belongs with A19: one session that runs the
  thing on a GPU and settles a bound is two of these entries.

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

### K2 · Four comments point at the *Open Issues* section of `CHANGELOG.md`, which is now this file

- location: `scripts/optimizer_allocations.jl:8`
- kind: docs
- found: 2026-09-26
- evidence: `scripts/optimizer_allocations.jl:8` and `scripts/retraction_step_allocations.jl:9` say "the
  *Open Issues* preamble states"; `test/decompositions/symplectic_sr.jl:32` and
  `test/manifolds/symplectic_stiefel_manifold.jl:14` say "*Open Issues* in `CHANGELOG.md`". Found by
  `git grep -n -i 'open issues' origin/main -- ':!CHANGELOG.md'`.

### K3 · The ID A22 was used for two different issues

- location: `KNOWN_ISSUES.md`
- kind: docs
- found: 2026-09-26
- evidence: commit 4d2ed06 (2026-08-20) added `#### A22. The only independent exponential
  implementation was CPU-only`, and 7f66695 (2026-08-23) removed it. Commit 4e9eb5f (#90,
  2026-09-18) used `A22` again for the SR decomposition. A commit message that cites A22 before
  2026-09-18 means the first issue.

### K4 · Three reference links in `CHANGELOG.md` have no definition

- location: `CHANGELOG.md:71`
- kind: docs
- found: 2026-09-26
- evidence: `[0.8.0]`, `[0.6.1]` and `[B]` (`CHANGELOG.md:71`, `:1091`, `:1235`, `:1996`) have no
  `[label]: url` line, and `[Unreleased]` compares `v0.6.0...main`.

[#14]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/14
[#24]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/24
[#27]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/27
[#38]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/38
[#40]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/40
[#44]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/44
[#45]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/45
[#46]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/46
[#52]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/52
[#54]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/54
[#111]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/111
