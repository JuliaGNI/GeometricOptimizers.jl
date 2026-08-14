using GeometricOptimizers
using GeometricOptimizers: StiefelManifold, Cayley
using SimpleSolvers: Static, Backtracking, Bisection, Quadratic, BierlaireQuadratic, StrongWolfe
using LinearAlgebra: norm, svd
using Test
import Random
Random.seed!(1234)

# The matrix lives in its own file so that `scripts/retraction_accuracy.jl`, which regenerates the
# tables below, measures the same problem by construction rather than by a copied literal.
A = include("svd_matrix.jl")

# named `objective` and not `error`, which is what it used to be called: that shadows `Base.error`
# for the whole file, so a genuine `error("...")` anywhere in it would have been a `MethodError`
objective(ps::NamedTuple) = norm(A - ps.w₁ * ps.w₂' * A)

# Both iterates stay on the Stiefel manifold, so this is a round-off tolerance and nothing
# else; the values actually observed are of the order of `1e-14`.
const MANIFOLD_TOLERANCE = 1e-12

# How close to the best rank-`n` approximation a *converged* solve has to get, as a relative error.
#
# This is deliberately not tighter. Nothing bounds the error in the objective at the point a solve
# stops in a platform-independent way: across eight starting points the worst case here is 2.6e-11,
# but CI has produced 1.3e-10 on the same seed this file uses, on a different Julia version. The
# previous value of `1e-10` therefore passed by luck of the platform rather than by anything the
# stopping criterion guarantees.
#
# It still discriminates: the fixed-step runs in `svd_test` above reach 1e-2 at best, so a converged
# solve is separated from an unconverged one by six orders of magnitude either way.
const CONVERGED_ERROR_TOLERANCE = 1e-8

# How small `‖∇f‖` is at the point a solve stops.
#
# Not `1e-5`, which is what this used to be, and not because `1e-5` was unlucky. The reasoning behind
# it was that a solve stops when `‖∇f‖ ≤ f_reltol = √eps ≈ 1.5e-8`, and that is simply not what
# happens on this problem: over the ten (method, line search, retraction) combinations run below and
# eight starting points each, `g_converged` is `false` in all eighty. Every one of them terminates on
# `f_converged` instead -- the successive relative change in `f` falling to `f_suctol = 2eps` -- and
# `‖∇f‖` at that point is whatever it is.
#
# Which is not arbitrary, just not `√eps`. Near a minimizer `f - f_min ≈ ‖∇f‖²/2λ`, so `f` stops
# changing in double precision once `‖∇f‖ ≈ √(eps ⋅ f ⋅ 2λ)`, i.e. around `1e-8` for a well-scaled
# problem and higher where the curvature is poor. Before the fixes described further down the worst
# case over those eighty runs was `1.8e-5` (`_DFP` + `StrongWolfe(c₂ = 0.1)` + `Cayley`), so `1e-5`
# sat *inside* the natural spread of the quantity it was bounding and CI's `1.354e-5` on
# Julia 1.13/Linux was unremarkable rather than a regression.
#
# With `linesearch_rejected` and `curvature_is_usable` in place the worst case over the same eighty
# runs is `2.9e-7`, so `1e-5` now has a factor of 35 of headroom and is a real bound rather than a
# coin flip. It is kept at `1e-5` for exactly that reason: it is the value that fails if the
# line-search handling regresses.
#
# The headroom grew again when `rg` became the residual at the iterate a solve returns rather than at
# whichever point the line search last probed (open issue A8): the stale value overestimated the
# residual near a minimiser. Over the eighteen converging combinations of the eight-seed sweep in
# `scripts/retraction_accuracy.jl` the worst `rg` goes from `2.5e-7` to `2.0e-7`, i.e. a factor of 50
# of headroom. (The two that do not converge -- `_BFGS` with either polynomial search under `Cayley`,
# open issue A1b -- report `rg` of order `1e0` on two of their eight seeds, which is the divergence
# and not the tolerance.) `g_converged` is still `false` in all eighty: `1e-7` is four orders above
# the `f_reltol` gate, and every one of these solves still terminates on `f_converged`.
const CONVERGED_GRADIENT_TOLERANCE = 1e-5

# How close `GradientMethod` and `MomentumMethod` get to the best rank-`n` approximation, as a
# relative error at iteration `1000` with `Static(0.01)` and seed `1234`:
#
#                  Geodesic   Cayley
#     GradientMethod  1.0e-2   9.8e-3
#     MomentumMethod  9.7e-3   9.3e-3
#
# Both leave roughly a factor of two, and both are *bit-identical* on Julia 1.10, 1.12 and 1.13
# (1.016e-2 and 9.688e-3 to every digit measured), so the final iterate is a perfectly good statistic
# for them. It is not one for `Adam`; see `ADAM_MEAN_ORBIT_TOLERANCE` below.
#
# `MomentumMethod` used to land at `1.9e-2` / `1.7e-2` here, i.e. *worse* than plain gradient
# descent, which is what issue #18 was about: it accumulated `p ← p + α∇L` and thereby kept
# pushing after `∇L → 0`. With the classic `p ← αp + ∇L` it is slightly better than gradient
# descent, as momentum should be.
#
# On the review comment "it shouldn't be necessary to increase the iteration number": correct,
# and it is back to `main`'s 1000. An intermediate version of this branch ran 1500 steps, but
# that was never a property of the unified interface — it was only needed to satisfy a
# `gradient` tolerance of `3e-3`, which 1000 steps does not reach. Measured from an identical
# starting point (seed `1234`, Geodesic), the unified interface and the old `optimization_step!`
# code agree to every digit printed:
#
#                       old            new
#     1000 steps   0.0101613780334933   0.0101613780334933
#     1500 steps   0.00151325729788261  0.00151325729788277
#
# So there is no per-step convergence regression to paper over here; `Static(0.01)` interacts
# with the retraction exactly as it used to.
const RELATIVE_ERROR_TOLERANCE = (gradient=2e-2, momentum=2e-2)

# The number of trailing iterations the `Adam` statistic averages over, out of `1000`.
const ADAM_ORBIT_WINDOW = 500

# How large `Adam`'s orbit around the minimizer is, as a relative error averaged over the last
# `ADAM_ORBIT_WINDOW` iterations.
#
# The point of averaging: with a fixed `α`, `Adam`'s direction has magnitude ≈1 per component
# whatever the gradient is, so it does not converge to the minimizer -- it circles it at a distance
# of order `α`. The error at iteration 1000 is a sample of an arbitrary *phase* on that orbit, and the
# last bits of the floating-point arithmetic move the phase. Averaging over a stretch of the orbit
# measures its *radius*, which is a property of `α` and the problem and not of the platform.
#
# Measured for identical code and seed on three Julia versions, Geodesic / Cayley:
#
#                            1.13             1.12             1.10        spread
#     iteration 1000     1.45e-5 / 2.88e-5  1.40e-5 / 1.24e-5  2.17e-5 / 9.7e-6   3.0x
#     min over 901:1000  5.8e-6 / 6.7e-6    9.3e-6 / 5.3e-6    5.9e-6 / 5.4e-6    1.76x
#     mean over 901:1000 3.50e-5 / 3.44e-5  3.12e-5 / 2.92e-5  2.88e-5 / 2.85e-5  1.21x
#     mean over 501:1000 2.24e-5 / 2.24e-5  2.12e-5 / 2.15e-5  2.13e-5 / 2.17e-5  1.06x
#
# So this is not the `min` an earlier version of this comment proposed: `min` is a lower envelope,
# it is attained at whichever single iteration happened to fall nearest the minimizer, and it is
# measurably *less* stable than the mean. The longer the window, the more of the orbit is averaged
# and the tighter the spread -- hence `501:1000` rather than `901:1000`.
#
# The margin, which is what the old snapshot statistic did not have. Worst correct value measured is
# 2.24e-5, so `4e-5` is 1.8x above it; and it is a real guard on the `Adam` bugs the CHANGELOG
# records (bias correction at `t + 1`, factors `β/(1 - βᵗ)` instead of `(β - βᵗ)/(1 - βᵗ)`, `√`
# applied to `m₂` rather than to `m̃₂`). Reintroducing them makes this statistic read 1.04e-1
# (Geodesic) and 4.5e-2 (Cayley), i.e. more than 1000x over the tolerance. The blanket `1e-1` this
# file once applied to all three algorithms is what let those bugs through in the first place.
#
# Getting the trace needs `Options(store_trace = true)`, which is now implemented -- see `trace`. It
# used to be accepted and silently ignored, by this package and by SimpleSolvers alike.
const ADAM_MEAN_ORBIT_TOLERANCE = 4e-5

"""
    starting_point(n)

The starting point every solve in this file uses, on `St(size(A, 1), n)²`.

Seeded on each call, and not once at the top of the file, so that every solve starts from the *same*
point: the solves in between draw from the global RNG themselves (each `GlobalSection` does), so
without this a later run would start somewhere that depends on how much randomness an earlier one
happened to consume. The tolerances here are calibrated for one starting point, so that has to be
pinned.
"""
function starting_point(n)
    Random.seed!(1234)
    (w₁=rand(StiefelManifold, size(A, 1), n), w₂=rand(StiefelManifold, size(A, 1), n))
end

"""
    best_rank_n_error(n)

The error of the best rank-`n` approximation of `A`, i.e. what `LinearAlgebra.svd` gives.
"""
function best_rank_n_error(n)
    U, Σ, Vt = svd(A)
    U_result = U[:, 1:n]
    norm(A - U_result * U_result' * A)
end

"""
    relative_error(ps, err_best)

How far `ps` is from the best rank-`n` approximation, relative to it.
"""
relative_error(ps, err_best) = norm((objective(ps) - err_best) / err_best)

"""
    mean_orbit_error(entries, err_best)

The mean relative error over `entries` of a [`GeometricOptimizers.trace`](@ref).

Spelled out rather than taken from `Statistics.mean`, which is not a test dependency and is not worth
becoming one for a mean over a fixed-length window.
"""
mean_orbit_error(entries, err_best) =
    sum(abs((entry.f - err_best) / err_best) for entry in entries) / length(entries)

# `warn_iterations = 0` silences "Optimizer took 1000 iterations", which is true and is the point:
# this is a fixed-budget comparison of three first-order methods at one learning rate, not a
# convergence test. None of them can converge here — with `Static(0.01)` the gradient is 8.4e-2 after
# these 1000 steps against a gate of 1.5e-8, and it is not stuck but slow (1.9e-3 / 2.1e-4 / 4.0e-5 at
# 5000 / 20000 / 60000 steps), so reaching the gate this way would take of the order of a million. The
# convergence test is `svd_convergence_check` below.
#
# `store_trace = true` because the `Adam` statistic is an average over the last `ADAM_ORBIT_WINDOW`
# iterations rather than the final iterate; see `ADAM_MEAN_ORBIT_TOLERANCE`.
const FIXED_BUDGET_STEPS = 1000

"""
    svd_check(relative_errors, mean_orbit_errors)

Compare the three first-order methods against each other and against their tolerances.
"""
function svd_check(relative_errors, mean_orbit_errors)
    for name in keys(RELATIVE_ERROR_TOLERANCE)
        @test relative_errors[name] < RELATIVE_ERROR_TOLERANCE[name]
    end

    @test mean_orbit_errors.adam < ADAM_MEAN_ORBIT_TOLERANCE

    # The ordering is the part of this that does not depend on the exact starting point:
    # bias-corrected `Adam` beats plain gradient descent on this problem by a wide margin — a factor
    # of 310 on the averaged statistic, against the factor of 10 asserted here.
    @test mean_orbit_errors.adam < mean_orbit_errors.gradient / 10
end

# The `Optimizer` is constructed and `solve!` called from *this loop* rather than from inside a helper
# that takes the retraction and the algorithm as arguments, which is how this file used to read. That
# shape cost 951 s to run on Julia 1.12 -- one single method compilation of 908 s, against 0.04 s on
# 1.13 -- because inference had to propagate the type of a constructor reached through three nested
# levels of `kwargs...` into a `solve!` call in the same inferred body.
#
# That is fixed in `Optimizer`'s constructors now, so this loop no longer *has* to look like this: the
# helper shape would be fast again. It is left flat because it costs nothing and because the failure
# mode was so quiet -- the tests all passed, they just took sixteen minutes. See the warning on
# `Optimizer(x, F)` for the measurements and for what does not work as a fix (`@noinline`,
# `@nospecialize`).
for retraction in (GeometricOptimizers.Geodesic(), GeometricOptimizers.Cayley())
    err_best = best_rank_n_error(3)
    relative_errors = Float64[]
    mean_orbit_errors = Float64[]

    for algorithm in (GradientMethod(), MomentumMethod(), GeometricOptimizers.Adam())
        ps = starting_point(3)
        state = OptimizerState(algorithm, ps)
        optimizer = Optimizer(ps, objective; retraction=retraction, algorithm=algorithm,
            linesearch=Static(0.01), max_iterations=FIXED_BUDGET_STEPS, warn_iterations=0,
            store_trace=true)
        result = solve!(ps, state, optimizer)

        for Y in values(ps)
            @test GeometricOptimizers.check(Y) < MANIFOLD_TOLERANCE
        end

        push!(relative_errors, relative_error(ps, err_best))

        # the trace records `f`, so the relative error per iteration comes straight out of it
        window = @view GeometricOptimizers.trace(result)[(end-ADAM_ORBIT_WINDOW+1):end]
        push!(mean_orbit_errors, mean_orbit_error(window, err_best))
    end

    names = (:gradient, :momentum, :adam)
    svd_check(NamedTuple{names}(Tuple(relative_errors)), NamedTuple{names}(Tuple(mean_orbit_errors)))
end

"""
    svd_convergence_check(n, ps, state, result, max_iterations)

The same problem as the fixed-budget loop above, solved to convergence rather than to a fixed budget.

`_BFGS` needs a line search that actually searches, and until the line search learned to take its
trial step through the retraction that was impossible on manifold parameters — `Static` was the only
one that worked, because it is the only one that never evaluates the merit. So this problem had no
algorithm that converged on it at all: the three first-order methods above exhaust 1000 iterations at
a relative error of 1e-2.

The `Optimizer` is built and solved by the caller rather than here; see the comment at the
fixed-budget loop above.
"""
function svd_convergence_check(n, ps, state, result, max_iterations)
    err_best = best_rank_n_error(n)

    # it stops on a convergence criterion, not on the iteration cap
    @test GeometricOptimizers.iteration_number(state) < max_iterations
    @test GeometricOptimizers.status(result).rg < CONVERGED_GRADIENT_TOLERANCE

    # and it gets to the answer, which the fixed-step runs above reach to 1e-2 at best
    @test relative_error(ps, err_best) < CONVERGED_ERROR_TOLERANCE

    for Y in values(ps)
        @test GeometricOptimizers.check(Y) < MANIFOLD_TOLERANCE
    end
end

# `_DFP` needed the same lift to `OptimizerSolution` that `_BFGS` already had; before it, its cache was
# `AbstractVector`-only, so a `NamedTuple` fell through to a `NewtonOptimizerCache` and a `MethodError`.
# Every combination of retraction, method and line search converges on this problem now, but the cost
# is uneven, and the ordering by *iterations* is not the ordering by *work* -- a `Bisection` iteration
# spends ≈580 objective evaluations against ≈25 for a `Backtracking` one. Iterations, then total
# evaluations, Geodesic / Cayley:
#
#                                     iterations          evaluations       iters over 8 seeds
#     _BFGS  Backtracking(expand)     95 /   118        2_441 /  3_031     104..161 /  91..156
#     _BFGS  Backtracking            136 /   136        3_457 /  3_456     114..192 / 118..203
#     _BFGS  Bisection               133 /   114       78_658 / 67_030       87..133 / 102..137
#     _BFGS  Quadratic               111 /    98       15_377 / 12_213       89..159 /  90..cap
#     _BFGS  BierlaireQuadratic      130 /   119       13_781 / 12_776      108..261 /  96..cap
#     _DFP   Backtracking(expand)   768 / 1_366       20_001 / 35_339     385..1_118 / 466..1_177
#     _DFP   Backtracking        48_322 / 26_479    1_208_157 / 662_029  10_448..114_116 / 5_596..26_479
#     _DFP   StrongWolfe(c₂=0.1)    218 /   279       18_127 / 23_828      296..868 / 198..515
#     _DFP   Bisection               136 /   110       80_001 / 64_316       87..141 / 102..124
#     _DFP   Quadratic               175 /   529       18_122 / 50_666       92..868 / 164..735
#
# Every evaluation count here is ten higher than it was before `rg` became the residual at the iterate
# a solve returns (open issue A8), and every iteration count and seed spread is unchanged under it
# (the one iteration count that does move is the `_DFP  Backtracking` correction below). Ten is one
# gradient evaluation on this problem -- `GradientAutodiff` costs exactly ten objective calls for these
# 60 parameters, and the counter above counts those too -- and it is the refresh at the *last* iterate,
# the one no `update!` follows and so the one nothing reuses. Per solve and not per iteration: the
# reuse in `store_gradient!` is what makes the difference `10` rather than `10 x iterations`.
#
# Re-measuring also corrected the `_DFP  Backtracking` row, whose `Geodesic` figures read 47_115 and
# 1_177_919 -- a state of the code that predates `curvature_is_usable`, and which `default_linesearch`'s
# own table already disagreed with. Both columns are now `solve_once` from `scripts/retraction_accuracy.jl`
# at a cap of 200_000, like everything else here; its *spread* is still the older measurement it has
# always been.
#
# Every `Geodesic` figure here moved when the retraction was fixed (see `ScaledSquaring`): a more
# accurate exponential is a different trajectory, so the counts shift by a few percent in both
# directions. Every row but `_DFP  Backtracking` is regenerated by `svd_tables()` in
# `scripts/retraction_accuracy.jl`, which measures the same matrix this file does -- it is
# `svd_matrix.jl` for both -- and whose default cap is the 20_000 the last column reports against.
# That is what "cap" in that column means: those two combinations do not converge on two of the
# eight, and that is open issue A1b.
#
# `_DFP  Backtracking` is deliberately not one of the script's `COMBINATIONS` -- it is the shrink-only
# search whose only purpose here is the ceiling argument three paragraphs down, and at 48_322
# iterations on the pinned seed it would dominate the runtime of every sweep. Its spread is an older
# measurement at a cap high enough not to bind, which is why it exceeds 20_000.
#
# The `Cayley` column moved again when `trial_slope` got an exact `Cayley` differential; the
# `Geodesic` column is bit-identical under that change, and so are both `Backtracking` rows, which
# evaluate `φ'` at `α = 0` only. What moved: `_BFGS  Bisection` 92 -> 114 iterations (54_970 ->
# 67_020 evaluations), `_DFP  Bisection` 96 -> 110 (56_106 -> 64_306), `_DFP  Quadratic` 550 -> 529
# (54_176 -> 50_656, and its spread 168..1_211 -> 164..735), and `_BFGS  Quadratic` 101 -> 98. A more
# accurate `φ'` is a different trajectory in the same way a more accurate exponential is; `Bisection`
# needing a few more iterations for a *correct* slope than for a wrong one is not a regression, it is
# a different sequence of brackets. (Those "after" evaluation counts are what the differential left
# behind; each is ten below the table above, which was measured after A8 added the final gradient.)
#
# Four stale `Cayley` bounds in the table above are corrected here -- three lower and one upper:
# `_BFGS  Backtracking(expand)` read 114 where it measures 91, `_BFGS  Backtracking` 131 where it
# measures 118, `_DFP  StrongWolfe` 215 where it measures 198, and `_DFP  Backtracking(expand)`'s
# upper bound read 1_366 -- the pinned value -- where the spread is 466..1_177. All four are
# unchanged between `main` and the differential, so they are bookkeeping and not a behaviour change.
#
# What the fix bought, over the same eight starting points: the worst `check` on `Geodesic` was
# `2.45e-5` -- `_BFGS` + `Backtracking` on seed 2, five orders of magnitude past
# `MANIFOLD_TOLERANCE`, and passing here only because this file uses seed `1234`. That same solve is
# `2.8e-12` now, a factor of 10^7. It is still the worst of the eight by two orders of magnitude, and
# the residue is accumulation over 147 iterations rather than any one retraction -- per-seed `check`
# for that combination:
#
#     seed                1        2        3        4        5        6        7        8
#     ScaledSquaring   2.8e-14  2.8e-12  1.9e-14  4.2e-15  4.1e-15  3.7e-15  6.2e-14  2.1e-14
#     AugmentedPade    2.9e-14  2.4e-13  1.9e-14  4.4e-15  3.2e-15  3.2e-15  6.2e-14  2.1e-14
#     ProjectedSkew    1.7e-13  9.8e-14  6.7e-14  1.4e-13  2.5e-14  7.8e-14  1.9e-13  4.4e-14
#
# `ProjectedSkew` is the only one whose worst case stays inside `MANIFOLD_TOLERANCE` (1.9e-13 against
# 2.8e-12), which is the trade its docstring describes -- structural orthogonality bounds the *worst*
# seed while costing about an order of magnitude on the typical one. Enabling the eight-seed sweep as
# a test would therefore need either `ProjectedSkew` or a tolerance of `1e-11`.
#
# The `_BFGS` + `Bisection` + `Geodesic` row used to read "see note below", because on one of those
# eight starting points that combination *diverged*: it stopped after 4 iterations with
# `check(Y) = 1e200`, i.e. off the manifold altogether, and reported convergence while doing it. That
# is fixed. `Bisection` bisects `φ'`, so on a non-convex ray it can settle on a stationary point of
# the ray that is a *maximum*; it said so (`LINESEARCH_FLOOR`, `φ(1) = φ(0)` exactly) and
# `solver_step!` took the step anyway, because it called `solve` and saw only the step length. See
# `linesearch_rejected`, `curvature_is_usable` and `restart!`. That starting point now converges in
# 121 iterations, and the worst `‖∇f‖` over all of these rows and all eight starting points went from
# `NaN` to `2.9e-7` -- see `CONVERGED_GRADIENT_TOLERANCE`, which is the other issue the same fix
# closed.
#
# The `_DFP  Backtracking` row is a property of the *line search*, not of DFP. A shrink-only backtracking
# search starts its trial step at `α = 1` and can never exceed it; measuring the `α` it returns settles
# what happens:
#
#                       fraction α == 1   fraction α > 1   median α   iterations
#     _BFGS  Backtracking         73.5%             0%          1.0          113
#     _BFGS  Bisection               0%          67.8%          1.42         143
#     _DFP   Backtracking        100.0%             0%          1.0       49_679
#     _DFP   Bisection               0%          94.8%         11.1          134
#
# (The `α` columns were measured on the code as it stood before `linesearch_rejected` and
# `curvature_is_usable`, which is why the iteration column here does not quite match the table above
# -- it read 113 / 143 / 49_679 / 134 then and 136 / 133 / 48_322 / 136 now. What the columns
# characterise is the *line search*, and that has not changed; the point they make about the ceiling
# at `α = 1` stands either way.)
#
# `_BFGS` produces a direction already scaled like a Newton step, so `α = 1` is the right answer and
# accepting it is not a failure. `_DFP` produces a systematically *under-scaled* direction that wants a
# median `α` of 8, and a shrink-only search cannot get there: it accepts `α = 1` on every single
# iteration and the solve crawls -- steps of `‖Δx‖ ≈ 1e-5` against a gradient of `≈ 1e-4`, the gradient
# falling by less than a factor of two over 19_500 iterations. It is not stuck (it terminates on a
# criterion, not on the cap), just pinned at the ceiling. Changing *nothing* but the initial trial step
# handed to the same search, which can shrink from it but not grow past it, is worth a factor of 217:
#
#     α₀ = 1  →  49_679 iterations        α₀ = 100  →  936
#     α₀ = 3  →     229 iterations        α₀ = 1000 →  2_281
#     α₀ = 10 →     268 iterations
#
# That measurement became JuliaGNI/SimpleSolvers.jl#174 and, in SimpleSolvers 0.11, the `expand` key
# that `default_linesearch` now switches on: an accepted *first* trial step is lengthened while each
# longer trial still satisfies sufficient decrease and strictly improves the merit. It costs under 4%
# per iteration, and it takes `_DFP` from no practical convergence to 702 and 1_366 iterations on the
# seed used here.
#
# That pair is still *not* run below, but the reason has weakened considerably. Its iteration count
# used to be extraordinarily sensitive to the starting point -- over eight seeds it ranged
#
#     Geodesic   512 .. 77_890        Cayley   465 .. 3_834
#
# against 201..624 for `StrongWolfe(c₂ = 0.1)` and 103..143 for `Bisection`. DFP's `Q` became badly
# conditioned (κ ≈ 1e9, see the trace referenced above) and how quickly the expansion phase dug it out
# was close to arbitrary. CI found this the honest way: the case converged in 830 iterations locally
# and exceeded a 3_000 cap on Julia 1.10 / Linux.
#
# `curvature_is_usable` is what that sensitivity was: an ill-conditioned `Q` on this problem is a `Q`
# built from secant pairs it should have rejected. With the condition enforced the same eight starting
# points give
#
#     Geodesic   387 .. 845           Cayley   466 .. 1_366
#
# i.e. a factor of 92 less spread on `Geodesic`. That is well inside the 5_000 cap used below, so this
# pair could reasonably be run now. It is left documented rather than run only because the earlier CI
# surprise was a factor of four between one platform and another and there is no CI measurement of the
# post-fix spread yet -- worth revisiting once there is. `default_linesearch` still says what it says
# about `StrongWolfe` being the better *explicit* choice for a DFP-heavy workload, and that is now a
# statement about cost (16_873 evaluations against 18_258) rather than about reliability.
#
# At `StrongWolfe`'s own `c₂ = 0.9` the Wolfe conditions already hold at `α = 1` on 99.4% of iterations,
# its bracketing phase never fires, and it crawls just as the shrink-only search does.
const CONVERGENCE_MAX_ITERATIONS = 5000

# As in the fixed-budget loop above, the `Optimizer` is constructed here rather than inside
# `svd_convergence_check`; see the comment there for why that used to matter on Julia 1.12.
for retraction in (GeometricOptimizers.Geodesic(), GeometricOptimizers.Cayley())
    for (algorithm, linesearch) in ((GeometricOptimizers._BFGS(), Backtracking(Float64)),
        (GeometricOptimizers._BFGS(), Backtracking(Float64; expand=true)),
        (GeometricOptimizers._BFGS(), Bisection(Float64)),
        # The two polynomial searches, which had no coverage in this loop at all until now.
        #
        # Read what they do and do not guard. On *this* starting point they converge, and did before
        # the exact `Cayley` differential as well (`Quadratic` 101 -> 98 iterations, and
        # `BierlaireQuadratic` unchanged at 119). What they do not guard is open issue A1b: on seeds
        # 2 and 8 this same pair runs to the cap under `Cayley` and ends off the manifold, and three
        # of those four cases still do. So this is coverage of the polynomial searches, not a
        # regression test for A1b -- that one needs the eight-seed sweep in
        # `scripts/retraction_accuracy.jl`, which is where the figures in the table above come from.
        (GeometricOptimizers._BFGS(), Quadratic(Float64)),
        (GeometricOptimizers._BFGS(), BierlaireQuadratic(Float64)),
        # `_DFP` with the two searches whose cost on this problem is stable across starting points
        (GeometricOptimizers._DFP(), Bisection(Float64)),
        (GeometricOptimizers._DFP(), StrongWolfe(Float64; c₂=0.1)))
        ps = starting_point(3)
        state = OptimizerState(algorithm, ps)
        optimizer = Optimizer(ps, objective; retraction=retraction, algorithm=algorithm,
            linesearch=linesearch, max_iterations=CONVERGENCE_MAX_ITERATIONS, warn_iterations=0)
        result = solve!(ps, state, optimizer)
        svd_convergence_check(3, ps, state, result, CONVERGENCE_MAX_ITERATIONS)
    end
end
