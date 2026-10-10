using GeometricOptimizers
using GeometricOptimizers: StiefelManifold, Cayley
using SimpleSolvers: Static, Backtracking, Bisection, Quadratic, BierlaireQuadratic,
                     StrongWolfe
using LinearAlgebra: norm, svd
using Test
import Random
include("../helpers/eltypes.jl")
include("../helpers/manifold_tolerance.jl")
Random.seed!(1234)

# The matrix lives in its own file so that `scripts/retraction_accuracy.jl`, which regenerates the
# tables below, measures the same problem by construction rather than by a copied literal.
A = include("../helpers/svd_matrix.jl")

# `A` rounded to `T`: the problem a solve in `T` is given, and the one its optimum is computed for.
svd_matrix(::Type{T}) where {T} = Matrix{T}(A)

# named `objective` and not `error`, which is what it used to be called: that shadows `Base.error`
# for the whole file, so a genuine `error("...")` anywhere in it would have been a `MethodError`
svd_objective(B) = ps -> norm(B - ps.w₁ * ps.w₂' * B)

# How close to the best rank-`n` approximation a *converged* solve has to get, as a relative error in
# the objective.
#
# Nothing bounds the error in the objective at the point a solve stops in a platform-independent way:
# CI on another Julia version has measured 1.3e-10 on the seed this file uses, against a local worst
# of 2.6e-11 over eight seeds, so a bound near the measured values passes by luck of the platform.
# Measured over seed 1234 and the eight seeds of the sweep below, all twenty combinations,
# `Float64`: worst 1.4e-10, 1/54 of this bound; `Float32`: worst 5.6e-5, 1/3 of it.
#
# It still discriminates: the fixed-step runs above reach 6e-3 at best, so a converged solve is
# separated from an unconverged one by orders of magnitude in both precisions.
converged_error_tolerance(::Type{T}) where {T} = sqrt(eps(T)) / 2

# How small `‖∇f‖` is at the point a solve stops.
#
# Every one of these solves terminates on `f_converged` -- the successive relative change in `f`
# falling to `f_suctol = 2eps` -- and not on `g_converged`, so `‖∇f‖` at that point is not `f_reltol`.
# Near a minimizer `f - f_min ≈ ‖∇f‖²/2λ`, so `f` stops changing once `‖∇f‖ ≈ √(eps ⋅ f ⋅ 2λ)`: a
# multiple of `√eps`, and the measurement says so. Over seed 1234 and the eight seeds of the sweep
# below, all twenty combinations, the worst `rg` is 3.0e-7 = 20√eps in `Float64` and 7.2e-3 = 21√eps
# in `Float32`, so this bound has a factor of 25 of headroom in both.
#
# It is a real bound and not a coin flip: before `linesearch_rejected` and `curvature_is_usable` the
# worst `Float64` case was 1.8e-5 (`DFP` + `StrongWolfe(c₂ = 0.1)` + `Cayley`) and CI saw 1.354e-5,
# both above the 7.6e-6 this gives in `Float64`, so it fails if the line-search handling regresses.
converged_gradient_tolerance(::Type{T}) where {T} = 512 * sqrt(eps(T))

# How close the subspace a converged solve finds is to the optimum: `‖w₁w₁ᵀ - UₙUₙᵀ‖`, with `Uₙ` the
# top `n` left singular vectors of the matrix. The objective is `‖A - w₁w₂ᵀA‖`, minimised by
# Eckart-Young exactly where `w₁w₂ᵀA = UₙUₙᵀA`, and that forces `w₁w₁ᵀ = UₙUₙᵀ` (and `w₂ = w₁`, but
# along `w₂` the objective is flat to fourth order, so `w₂` is only determined to `eps^(1/4)` and is
# not asserted). A minimiser is accurate to the root of the objective's precision, so this is a
# multiple of `√eps`: measured over seed 1234 and the eight sweep seeds, all twenty combinations, the
# worst is 8.6√eps in `Float64` and 24√eps in `Float32` (both `BFGS` + `BierlaireQuadratic`).
projector_tolerance(::Type{T}) where {T} = 128 * sqrt(eps(T))

# How close `GradientMethod` and `MomentumMethod` get to the best rank-`n` approximation, as a
# relative error at iteration `1000` with `Static(0.01)` and seed `1234`, measured on 1.13:
#
#                  Geodesic   Cayley
#     GradientMethod  5.9e-3   5.9e-3
#     MomentumMethod  5.6e-3   5.6e-3
#
# in `Float32` and in `Float64` alike, to the two digits printed: this is the truncation error of a
# fixed budget, which round-off does not set, so the bound is the same number in both precisions and
# not a multiple of `eps`. Both leave a factor of three. The final iterate is a perfectly good
# statistic for these two. It is not one for `Adam`; see `adam_mean_orbit_tolerance` below.
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
function relative_error_tolerance(::Type{T}) where {T}
    (gradient = T(2e-2), momentum = T(2e-2))
end

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
# The margin, which is what the old snapshot statistic did not have. Measured on 1.13, Geodesic /
# Cayley: 2.82e-5 / 2.89e-5 in `Float64` and 2.87e-5 / 2.84e-5 in `Float32`, so `4e-5` is 1.4x above
# the worst. The radius is a property of `α` and the problem, not of round-off, so it is the same
# number in both precisions and not a multiple of `eps`. It is a real guard on the `Adam` bugs the
# CHANGELOG records (bias correction at `t + 1`, factors `β/(1 - βᵗ)` instead of
# `(β - βᵗ)/(1 - βᵗ)`, `√` applied to `m₂` rather than to `m̃₂`). Reintroducing them makes this
# statistic read 1.04e-1 (Geodesic) and 4.5e-2 (Cayley), i.e. more than 1000x over the tolerance. The
# blanket `1e-1` this file once applied to all three algorithms is what let those bugs through in the
# first place.
#
# Getting the trace needs `Options(store_trace = true)`, which is now implemented -- see `trace`. It
# used to be accepted and silently ignored, by this package and by SimpleSolvers alike.
adam_mean_orbit_tolerance(::Type{T}) where {T} = T(4e-5)

"""
    starting_point(T, n, seed = 1234)

The starting point of a solve in `T`, on `St(size(A, 1), n)²`.

Seeded on each call, and not once at the top of the file, so that every solve starts from the *same*
point: the solves in between draw from the global RNG themselves (each `GlobalSection` does), so
without this a later run would start somewhere that depends on how much randomness an earlier one
happened to consume. The tolerances here are calibrated for one starting point, so that has to be
pinned.

Drawn in `Float64` and rounded to `T`, rather than drawn in `T`, for the same reason: a `Float32` draw
from the same seed is a different point, and so a different problem. From its own draw `Float32`
`GradientMethod` ends the fixed budget at a relative error of 0.18 where `Float64` ends at 5.9e-3,
which is the starting point and not the precision; from the rounded one both end at 5.9e-3. Every
operation of the solve is still in `T`, which the `eltype` assertions check.
"""
function starting_point(::Type{T}, n, seed::Integer = 1234) where {T}
    Random.seed!(seed)
    w₁ = rand(StiefelManifold{Float64}, size(A, 1), n)
    w₂ = rand(StiefelManifold{Float64}, size(A, 1), n)
    NetworkParameters((
        w₁ = StiefelManifold(Matrix{T}(w₁)), w₂ = StiefelManifold(Matrix{T}(w₂))))
end

"""
    best_rank_n(T, n)

The best rank-`n` approximation of `svd_matrix(T)`, i.e. what `LinearAlgebra.svd` gives: the matrix,
its error, and the projector onto its top `n` left singular vectors.

An SVD and not the code under test, and in `Float64`, so that the reference adds no round-off of `T`
to what is measured.
"""
function best_rank_n(::Type{T}, n) where {T}
    B = Matrix{Float64}(svd_matrix(T))
    U = svd(B).U[:, 1:n]
    (matrix = B, error = norm(B - U * U' * B), projector = U * U')
end

"""
    relative_error(ps, best)

How far `ps` is from the best rank-`n` approximation, relative to it, evaluated in `Float64`.
"""
function relative_error(ps, best)
    w₁, w₂ = Matrix{Float64}(ps.w₁), Matrix{Float64}(ps.w₂)
    abs((norm(best.matrix - w₁ * w₂' * best.matrix) - best.error) / best.error)
end

"""
    projector_distance(ps, best)

`‖w₁w₁ᵀ - UₙUₙᵀ‖`, the distance of the subspace `ps` found from the optimal one; see
`projector_tolerance`.
"""
function projector_distance(ps, best)
    w₁ = Matrix{Float64}(ps.w₁)
    norm(w₁ * w₁' - best.projector)
end

"""
    mean_orbit_error(entries, err_best)

The mean relative error over `entries` of a [`GeometricOptimizers.trace`](@ref).

Spelled out rather than taken from `Statistics.mean`, which is not a test dependency and is not worth
becoming one for a mean over a fixed-length window.
"""
function mean_orbit_error(entries, err_best)
    sum(abs((entry.f - err_best) / err_best) for entry in entries) / length(entries)
end

# `warn_iterations = 0` silences "Optimizer took 1000 iterations", which is true and is the point:
# this is a fixed-budget comparison of three first-order methods at one learning rate, not a
# convergence test. None of them can converge here — with `Static(0.01)` the gradient is 8.4e-2 after
# these 1000 steps against a gate of 1.5e-8, and it is not stuck but slow (1.9e-3 / 2.1e-4 / 4.0e-5 at
# 5000 / 20000 / 60000 steps), so reaching the gate this way would take of the order of a million. The
# convergence test is `svd_convergence_check` below.
#
# `store_trace = true` because the `Adam` statistic is an average over the last `ADAM_ORBIT_WINDOW`
# iterations rather than the final iterate; see `adam_mean_orbit_tolerance`.
#
# `min_iterations` makes the budget the budget in `Float32` too: there `Adam`'s change in `f` falls
# under `f_suctol = 2eps(Float32)` after 230 steps, `f_converged` stops the solve, and the window of
# the last 500 iterations does not exist. In `Float64` it changes nothing; that solve runs 1000 anyway.
const FIXED_BUDGET_STEPS = 1000

"""
    svd_check(T, relative_errors, mean_orbit_errors)

Compare the three first-order methods against each other and against their tolerances.
"""
function svd_check(::Type{T}, relative_errors, mean_orbit_errors) where {T}
    for name in keys(relative_error_tolerance(T))
        @test relative_errors[name] < relative_error_tolerance(T)[name]
    end

    @test mean_orbit_errors.adam < adam_mean_orbit_tolerance(T)

    # The ordering is the part of this that does not depend on the exact starting point:
    # bias-corrected `Adam` beats plain gradient descent on this problem by a wide margin — a factor
    # of 1_600 on the averaged statistic in both precisions, against the factor of 10 asserted here.
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
@testset "fixed budget, three first-order methods, $T" for T in REAL_ELTYPES
    objective = svd_objective(svd_matrix(T))
    best = best_rank_n(T, 3)

    for retraction in (GeometricOptimizers.Geodesic(), GeometricOptimizers.Cayley())
        relative_errors = Float64[]
        mean_orbit_errors = Float64[]

        # no `Newton`: `starting_point` returns a parameter set, which `Newton` is out of scope for,
        # and `Optimizer` rejects it there — see `test/integration/optimizer_tests.jl`
        for algorithm in (GradientMethod(), MomentumMethod(), GeometricOptimizers.Adam())
            ps = starting_point(T, 3)
            state = OptimizerState(algorithm, ps)
            optimizer = Optimizer(
                ps, objective; retraction = retraction, algorithm = algorithm,
                linesearch = Static(T; α = T(0.01)), max_iterations = FIXED_BUDGET_STEPS,
                min_iterations = FIXED_BUDGET_STEPS, warn_iterations = 0, store_trace = true)
            result = solve!(ps, state, optimizer)

            @test all(Y -> eltype(Y) == T, values(ps))
            for Y in values(ps)
                @test GeometricOptimizers.check(Y) < manifold_tolerance(T)
            end

            push!(relative_errors, relative_error(ps, best))

            # the trace records `f`, so the relative error per iteration comes straight out of it
            window = @view GeometricOptimizers.trace(result)[(end - ADAM_ORBIT_WINDOW + 1):end]
            push!(mean_orbit_errors, mean_orbit_error(window, best.error))
        end

        names = (:gradient, :momentum, :adam)
        svd_check(T, NamedTuple{names}(Tuple(relative_errors)),
            NamedTuple{names}(Tuple(mean_orbit_errors)))
    end
end

"""
    svd_convergence_check(T, ps, state, result, best, max_iterations)

The same problem as the fixed-budget loop above, solved to convergence rather than to a fixed budget.

`BFGS` needs a line search that actually searches, and until the line search learned to take its
trial step through the retraction that was impossible on manifold parameters — `Static` was the only
one that worked, because it is the only one that never evaluates the merit. So this problem had no
algorithm that converged on it at all: the three first-order methods above exhaust 1000 iterations at
a relative error of 6e-3.

The `Optimizer` is built and solved by the caller rather than here; see the comment at the
fixed-budget loop above.

The eight-seed sweep at the end of this file counts the same criteria with `svd_converged` instead of asserting them per solve.
"""
function svd_convergence_check(::Type{T}, ps, state, result, best, max_iterations) where {T}
    @test all(Y -> eltype(Y) == T, values(ps))

    # it stops on a convergence criterion, not on the iteration cap
    @test GeometricOptimizers.iteration_number(state) < max_iterations
    @test GeometricOptimizers.isconverged(GeometricOptimizers.status(result))
    @test GeometricOptimizers.status(result).rg < converged_gradient_tolerance(T)

    # and it gets to the answer, which the fixed-step runs above reach to 6e-3 at best: the objective,
    # and the subspace that attains it
    @test relative_error(ps, best) < converged_error_tolerance(T)
    @test projector_distance(ps, best) < projector_tolerance(T)

    for Y in values(ps)
        @test GeometricOptimizers.check(Y) < manifold_tolerance(T)
    end
end

# `DFP` carries the same lift to `OptimizerSolution` that `BFGS` does; without it its cache is
# `AbstractVector`-only, so a `NamedTuple` falls through to a `NewtonOptimizerCache` and a `MethodError`.
# Every combination of retraction, method and line search converges on the pinned seed, but the cost
# is uneven, and the ordering by *iterations* is not the ordering by *work* -- a `Bisection` iteration
# spends ≈580 objective evaluations against ≈25 for a `Backtracking` one. Iterations, then total
# evaluations, Geodesic / Cayley:
#
#                                     iterations          evaluations       iters over 8 seeds
#     BFGS  Backtracking(expand)     95 /   118        2_441 /  3_031     104..161 /  91..170
#     BFGS  Backtracking            136 /   136        3_457 /  3_456     146..192 / 118..201
#     BFGS  Bisection               133 /   114       78_658 / 67_030       93..147 / 102..137
#     BFGS  StrongWolfe(c₂=0.1)     135 /   135        7_893 /  7_880       91..146 / 107..152
#     BFGS  Quadratic               111 /    98       15_377 / 12_213       99..159 /  99..176
#     BFGS  BierlaireQuadratic      130 /   119       13_781 / 12_776      102..182 / 107..281
#     DFP   Backtracking(expand)   768 / 1_366       20_001 / 35_339     385..1_118 / 466..1_177
#     DFP   Backtracking        48_322 / 26_479    1_208_157 / 662_029  10_448..114_116 / 5_596..26_479
#     DFP   StrongWolfe(c₂=0.1)    218 /   279       18_127 / 23_828      296..868 / 198..515
#     DFP   Bisection               136 /   111       80_001 / 65_447       99..141 / 102..124
#     DFP   Quadratic               175 /   529       18_122 / 50_666       92..868 / 164..735
#
# `DFP  Backtracking` is the one row not in the script's `COMBINATIONS`: 48_322 iterations on one seed.
#
# These figures are not those of the code this file runs: on 1.13 the script measures for example
# 131 / 131 iterations for `BFGS  Backtracking(expand)` and `226..2_948 / 413..1_609` over the seeds
# for `DFP  Backtracking(expand)`, and `DFP  Quadratic  Geodesic` stops with a zero step on seed 8
# (see `SWEEP_MIN_CONVERGED`). The eight-seed sweep testset at the end of this file asserts the
# behaviour, and this table does not.
#
# **All twenty are now 8/8 on the manifold**, which is the column that matters and the one the sweep
# now prints (`on_the_manifold`). It was 8/8 in sixteen of them and not in four: `BFGS` with either
# polynomial search under `Cayley` was 4/8 -- open issue A1b, now closed -- and `BFGS` with either
# `Backtracking` under `Geodesic` was 7/8, which nothing had noticed. Both are the same defect and the
# same fix, the step ceiling of `DEFAULT_STEP_CEILING`; see the paragraphs below the `α` table.
#
# The `BFGS  StrongWolfe(c₂=0.1)` row is new *here* and not new to the measurement: it has always been
# one of the script's `COMBINATIONS` and was simply missing from this table.
#
# **What the step ceiling cost on this starting point: nothing.** Both columns come from `svd_tables()`,
# the second as `svd_tables(step_ceiling = Inf)` -- the knob is a keyword on `solve_once` precisely so
# that the comparison is regenerated by the harness rather than recalled -- and every *pinned* figure
# above, on both retractions, is reproduced to the digit with the ceiling on and off. The ceiling does
# not bind on seed 1234 at all. That is the design: what it buys is on the other seven starting points.
#
# It is worth recording that this was *not* true of the ceiling as first written, because the reason is
# instructive. Deriving the bound from `2π` over the norm of the whole direction combined the two
# `StiefelManifold` blocks of this problem in quadrature, which tightened it by up to `√2` -- enough to
# bind on three cells (`BFGS  Quadratic` 111 -> 120 iterations, `BFGS  BierlaireQuadratic` 130 -> 113,
# `DFP   Quadratic` 175 -> 308, all under `Geodesic`) and on nothing under `Cayley`. Those three looked
# like the price of bounding the step and were the price of a sloppy norm; deriving the ceiling per
# block, which is what the geometry says, removes all three. That was issue A15.
#
# The seed spreads are where the fix lives. The four rows that were not 8/8:
#
#                                        before          after
#     BFGS  Quadratic     Cayley     90..cap  (4/8)   99..176  (8/8)   worst check 3.2e-1 -> 6.1e-14
#     BFGS  Bierlaire     Cayley     93..cap  (4/8)  107..281  (8/8)   worst check 5.5e-1 -> 6.9e-14
#     BFGS  Backtr(exp)   Geodesic  104..161  (7/8)  104..161  (8/8)   worst check 2.8e-12 -> 6.0e-14
#     BFGS  Backtracking  Geodesic  114..192  (7/8)  146..192  (8/8)   worst check 2.8e-12 -> 6.3e-14
#
# The first two are A1b as it was catalogued. The second two were not catalogued at all and are the
# same defect: that `2.8e-12` is the `BFGS` + `Backtracking` + `Geodesic` seed 2 the paragraph on
# `ProjectedSkew` below already singles out as "the worst of the eight by two orders of magnitude".
# It was read there as accumulation over 147 iterations, and that reading was wrong -- it is one
# over-long step, of exactly the kind A1b describes, and bounding the step removes it. `Backtracking`
# reaching it at all is worth noting, since a shrink-only search cannot exceed `α = 1`: the expansion
# phase can, and `‖δ‖` is what makes `α = 1` too far.
#
# Note the third row: the *spread* is unchanged and the row still moved from 7/8 to 8/8. The seed that
# was off the manifold took the same number of iterations to get there; the ceiling changed what one of
# them did, not how many there were. This is why `on_the_manifold` is the column to read and the
# iteration spread is not.
#
# That has a consequence for this file. The `1e-11` tolerance the `ProjectedSkew` paragraph says an
# eight-seed sweep would need is no longer needed: the worst `check` over all twenty combinations and
# all eight seeds is `2.5e-13` (`DFP  Backtracking(expand)  Cayley`) and over the twelve `BFGS`
# rows it is `6.9e-14`, so a bound of `1e-12` clears that sweep with a factor of 4. Measured on 1.13
# the worst is `3.8e-14` = 173 eps (`DFP  Backtracking(expand)  Geodesic`), and 34 eps in `Float32`;
# `manifold_tolerance(T)` is 4096 eps, and the sweep testset below asserts it.
#
# Worst `rg` over all twenty and all eight seeds is `3.8e-07` (`BFGS  BierlaireQuadratic  Cayley`),
# against `3.1e-01` with the ceiling off -- that one being the diverging solve rather than a tolerance.
# See `converged_gradient_tolerance`.
#
# Every evaluation count here is ten higher than it was before `rg` became the residual at the iterate
# a solve returns (issue A8), and every iteration count and seed spread is unchanged under it
# (the one iteration count that does move is the `DFP  Backtracking` correction below). Ten is one
# gradient evaluation on this problem -- `GradientAutodiff` costs exactly ten objective calls for these
# 60 parameters, and the counter above counts those too -- and it is the refresh at the *last* iterate,
# the one no `update!` follows and so the one nothing reuses. Per solve and not per iteration: the
# reuse in `store_gradient!` is what makes the difference `10` rather than `10 x iterations`.
#
# Re-measuring also corrected the `DFP  Backtracking` row, whose `Geodesic` figures read 47_115 and
# 1_177_919 -- a state of the code that predates `curvature_is_usable`, and which `default_linesearch`'s
# own table already disagreed with. Both columns are now `solve_once` from `scripts/retraction_accuracy.jl`
# at a cap of 200_000, like everything else here; its *spread* is still the older measurement it has
# always been.
#
# Every `Geodesic` figure here moved when the retraction was fixed (see `ScaledSquaring`): a more
# accurate exponential is a different trajectory, so the counts shift by a few percent in both
# directions. Every row but `DFP  Backtracking` is regenerated by `svd_tables()` in
# `scripts/retraction_accuracy.jl`, which measures the same matrix this file does -- it is
# `svd_matrix.jl` for both -- and whose default cap is the 20_000 the last column reports against.
# No entry in that column reads "cap" any more: it used to, for the two combinations of issue A1b that
# ran out of iterations on two of their eight seeds, and with the step ceiling all twenty converge.
#
# `DFP  Backtracking` is deliberately not one of the script's `COMBINATIONS` -- it is the shrink-only
# search whose only purpose here is the ceiling argument three paragraphs down, and at 48_322
# iterations on the pinned seed it would dominate the runtime of every sweep. Its spread is an older
# measurement at a cap high enough not to bind, which is why it exceeds 20_000.
#
# The `Cayley` column moved again when `trial_slope` got an exact `Cayley` differential; the
# `Geodesic` column is bit-identical under that change, and so are both `Backtracking` rows, which
# evaluate `φ'` at `α = 0` only. What moved: `BFGS  Bisection` 92 -> 114 iterations (54_970 ->
# 67_020 evaluations), `DFP  Bisection` 96 -> 110 (56_106 -> 64_306), `DFP  Quadratic` 550 -> 529
# (54_176 -> 50_656, and its spread 168..1_211 -> 164..735), and `BFGS  Quadratic` 101 -> 98. A more
# accurate `φ'` is a different trajectory in the same way a more accurate exponential is; `Bisection`
# needing a few more iterations for a *correct* slope than for a wrong one is not a regression, it is
# a different sequence of brackets. (Those "after" evaluation counts are what the differential left
# behind; each is ten below the table above, which was measured after A8 added the final gradient.)
#
# Four stale `Cayley` bounds in the table above are corrected here -- three lower and one upper:
# `BFGS  Backtracking(expand)` read 114 where it measures 91, `BFGS  Backtracking` 131 where it
# measures 118, `DFP  StrongWolfe` 215 where it measures 198, and `DFP  Backtracking(expand)`'s
# upper bound read 1_366 -- the pinned value -- where the spread is 466..1_177. All four are
# unchanged between `main` and the differential, so they are bookkeeping and not a behaviour change.
#
# What the fix bought, over the same eight starting points: the worst `check` on `Geodesic` was
# `2.45e-5` -- `BFGS` + `Backtracking` on seed 2, seven orders of magnitude past
# `manifold_tolerance(Float64)`, which a test of seed `1234` alone does not see. That same solve was
# `2.8e-12` after it, a factor of 10^7, and is inside `6.3e-14` now -- that being the worst of the
# eight for the combination, which is the resolution `svd_tables` reports. Per-seed `check` for it,
# as measured *before* the step ceiling:
#
#     seed                1        2        3        4        5        6        7        8
#     ScaledSquaring   2.8e-14  2.8e-12  1.9e-14  4.2e-15  4.1e-15  3.7e-15  6.2e-14  2.1e-14
#     AugmentedPade    2.9e-14  2.4e-13  1.9e-14  4.4e-15  3.2e-15  3.2e-15  6.2e-14  2.1e-14
#     ProjectedSkew    1.7e-13  9.8e-14  6.7e-14  1.4e-13  2.5e-14  7.8e-14  1.9e-13  4.4e-14
#
# The conclusion drawn from that table does not survive the step ceiling, and it is worth saying which
# half of it was wrong. The *trade* it describes is real and is `ProjectedSkew`'s docstring's to make:
# structural orthogonality bounds the worst seed at about an order of magnitude on the typical one.
# What was wrong is the diagnosis of the seed-2 outlier. It was read as accumulation over 147
# iterations -- i.e. as the retraction's problem, which is why the remedy looked like a choice of
# exponential -- and it is not: it is one over-long step, and bounding the step takes that entry inside
# `6.3e-14` with `ScaledSquaring` untouched. The retraction was the amplifier here exactly as it is in
# A1b.
#
# So the sentence this paragraph used to end with -- that enabling the eight-seed sweep as a test would
# need either `ProjectedSkew` or a tolerance of `1e-11` -- is no longer true. The worst `check` over
# the whole sweep is `2.5e-13`, and `manifold_tolerance(Float64)` (9.1e-13) clears it with
# `ScaledSquaring`.
#
# The `BFGS` + `Bisection` + `Geodesic` row used to read "see note below", because on one of those
# eight starting points that combination *diverged*: it stopped after 4 iterations with
# `check(Y) = 1e200`, i.e. off the manifold altogether, and reported convergence while doing it. That
# is fixed. `Bisection` bisects `φ'`, so on a non-convex ray it can settle on a stationary point of
# the ray that is a *maximum*; it said so (`LINESEARCH_FLOOR`, `φ(1) = φ(0)` exactly) and
# `solver_step!` took the step anyway, because it called `solve` and saw only the step length. See
# `linesearch_rejected`, `curvature_is_usable` and `restart!`. That starting point now converges in
# 121 iterations, and the worst `‖∇f‖` over all of these rows and all eight starting points went from
# `NaN` to `2.9e-7` -- see `converged_gradient_tolerance`, which is the other issue the same fix
# closed.
#
# The `DFP  Backtracking` row is a property of the *line search*, not of DFP. A shrink-only backtracking
# search starts its trial step at `α = 1` and can never exceed it; measuring the `α` it returns settles
# what happens:
#
#                       fraction α == 1   fraction α > 1   median α   iterations
#     BFGS  Backtracking         73.5%             0%          1.0          113
#     BFGS  Bisection               0%          67.8%          1.42         143
#     DFP   Backtracking        100.0%             0%          1.0       49_679
#     DFP   Bisection               0%          94.8%         11.1          134
#
# (The `α` columns were measured on the code as it stood before `linesearch_rejected` and
# `curvature_is_usable`, which is why the iteration column here does not quite match the table above
# -- it read 113 / 143 / 49_679 / 134 then and 136 / 133 / 48_322 / 136 now. What the columns
# characterise is the *line search*, and that has not changed; the point they make about the ceiling
# at `α = 1` stands either way.)
#
# `BFGS` produces a direction already scaled like a Newton step, so `α = 1` is the right answer and
# accepting it is not a failure. `DFP` produces a systematically *under-scaled* direction that wants a
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
# per iteration, and it takes `DFP` from no practical convergence to 702 and 1_366 iterations on the
# seed used here.
#
# That pair is run, over the eight seeds, in the sweep testset at the end of this file. Its
# iteration count used to be extraordinarily sensitive to the starting point -- over eight seeds it
# ranged
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
# i.e. a factor of 92 less spread on `Geodesic`. Measured on 1.13, at the 5_000 cap below, seeds
# 1..8:
#
#     Float64  Geodesic   815, 1_288, 226, 1_138, 556, 636, 498, 2_948
#              Cayley     491, 413, 432, 682, 833, 1_609, 433, 793
#     Float32  Geodesic   47, 84, 34, 41, 36, 51, 46, 45
#              Cayley     40, 67, 38, 44, 50, 58, 58, 38
#
# All inside the cap, the worst (`Float64`, `Geodesic`, seed 8) at 59% of it. A factor of four
# between platforms, which CI has shown on this pair, would take that seed past the cap; if CI finds
# it, the count is the thing to report, not the cap to raise. `default_linesearch` still says what
# it says about `StrongWolfe` being the better *explicit* choice for a DFP-heavy workload, and that
# is now a statement about cost (16_873 evaluations against 18_258) rather than about reliability.
#
# At `StrongWolfe`'s own `c₂ = 0.9` the Wolfe conditions already hold at `α = 1` on 99.4% of iterations,
# its bracketing phase never fires, and it crawls just as the shrink-only search does.
const CONVERGENCE_MAX_ITERATIONS = 5000

# As in the fixed-budget loop above, the `Optimizer` is constructed here rather than inside
# `svd_convergence_check`; see the comment there for why that used to matter on Julia 1.12.
@testset "converges to the best rank-3 approximation, seed 1234, $T" for T in REAL_ELTYPES
    objective = svd_objective(svd_matrix(T))
    best = best_rank_n(T, 3)

    for retraction in (GeometricOptimizers.Geodesic(), GeometricOptimizers.Cayley())
        for (algorithm, linesearch) in ((GeometricOptimizers.BFGS(), Backtracking(T)),
            (GeometricOptimizers.BFGS(), Backtracking(T; expand = true)),
            (GeometricOptimizers.BFGS(), Bisection(T)),
        # The two polynomial searches. On *this* starting point they converge and always did, so
        # this loop is coverage of the searches; the starting points that failed are covered by
        # `A1B_SEEDS` below, which is the regression test for A1b proper.
            (GeometricOptimizers.BFGS(), Quadratic(T)),
            (GeometricOptimizers.BFGS(), BierlaireQuadratic(T)),
        # `DFP` with the two searches whose cost on this problem is stable across starting points
            (GeometricOptimizers.DFP(), Bisection(T)),
            (GeometricOptimizers.DFP(), StrongWolfe(T; c₂ = T(0.1))))
            ps = starting_point(T, 3)
            state = OptimizerState(algorithm, ps)
            optimizer = Optimizer(
                ps, objective; retraction = retraction, algorithm = algorithm,
                linesearch = linesearch, max_iterations = CONVERGENCE_MAX_ITERATIONS,
                warn_iterations = 0)
            result = solve!(ps, state, optimizer)
            svd_convergence_check(T, ps, state, result, best, CONVERGENCE_MAX_ITERATIONS)
        end
    end
end

# The regression test for issue A1b.
#
# A line search bounds the step it returns by the merit. On a *compact* manifold the merit is bounded,
# so that test never fires: `Quadratic` returned `α = 4.3e7` on a direction of norm 5.54 -- a step of
# `‖αδ‖ = 2.4e8` -- and reported a decrease it had genuinely measured. Retracting a lift that large
# leaves `St(20, 3)`, and the solve then reported *convergence* from a point that was no longer on it.
#
# The bound that does exist is geometric: `2π/‖δ‖`, since retracting a lift is a rotation. It changes
# at every step and nothing about `φ` reveals it, which is why SimpleSolvers 0.12 takes it per call
# through `params.αmax` and leaves the value here. See `DEFAULT_STEP_CEILING` and `linesearch_parameters`.
#
# These are the starting points that failed, not the one the loop above uses. Seed `1234` converged
# throughout and so could never have caught this -- which is the whole reason A1b needed a sweep to
# find and a sweep to confirm. `check` is the assertion that matters: A1b's failure is a solve that
# claims success from off the manifold, so a test on `rg` or on the objective alone would have passed
# while diverging.
const A1B_SEEDS = (2, 8)

@testset "a bounded merit does not produce an unbounded step (issue A1b), $T" for T in REAL_ELTYPES
    objective = svd_objective(svd_matrix(T))
    best = best_rank_n(T, 3)

    for linesearch in (Quadratic(T), BierlaireQuadratic(T)), seed in A1B_SEEDS

        algorithm = GeometricOptimizers.BFGS()
        ps = starting_point(T, 3, seed)
        state = OptimizerState(algorithm, ps)
        optimizer = Optimizer(ps, objective; retraction = GeometricOptimizers.Cayley(),
            algorithm = algorithm, linesearch = linesearch,
            max_iterations = CONVERGENCE_MAX_ITERATIONS, warn_iterations = 0)
        result = solve!(ps, state, optimizer)

        # Both of these ran to a 20_000 cap before, at `check` of 3.2e-1 and 5.5e-1. They now take
        # 90..297 iterations, so `CONVERGENCE_MAX_ITERATIONS` is a real bound here and not the cap.
        svd_convergence_check(T, ps, state, result, best, CONVERGENCE_MAX_ITERATIONS)
    end
end

# The eight-seed sweep of `scripts/retraction_accuracy.jl` as a test. These are that
# script's `COMBINATIONS`, row by row, and so the table above without `DFP  Backtracking`: the
# shrink-only search takes tens of thousands of iterations on every seed (`10_448..114_116` on
# `Geodesic`), which no cap of this file allows and no test can afford. Each combination is held to a
# rate: of its eight solves, at least `SWEEP_MIN_CONVERGED` pass the criteria of the pinned seed
# (`svd_converged`), and every one of them ends with both factors on the manifold.
const SWEEP_COMBINATIONS = (
    ("BFGS  Backtracking(expand)", GeometricOptimizers.BFGS(),
        T -> Backtracking(T; expand = true)),
    ("BFGS  Backtracking", GeometricOptimizers.BFGS(), T -> Backtracking(T)),
    ("BFGS  Bisection", GeometricOptimizers.BFGS(), T -> Bisection(T)),
    ("BFGS  StrongWolfe(c₂=0.1)", GeometricOptimizers.BFGS(),
        T -> StrongWolfe(T; c₂ = T(0.1))),
    ("BFGS  Quadratic", GeometricOptimizers.BFGS(), T -> Quadratic(T)),
    ("BFGS  BierlaireQuadratic", GeometricOptimizers.BFGS(), T -> BierlaireQuadratic(T)),
    ("DFP   Backtracking(expand)", GeometricOptimizers.DFP(),
        T -> Backtracking(T; expand = true)),
    ("DFP   Bisection", GeometricOptimizers.DFP(), T -> Bisection(T)),
    ("DFP   StrongWolfe(c₂=0.1)", GeometricOptimizers.DFP(),
        T -> StrongWolfe(T; c₂ = T(0.1))),
    ("DFP   Quadratic", GeometricOptimizers.DFP(), T -> Quadratic(T)))

const SWEEP_SEEDS = 1:8

# How many of the eight seeds each combination converges from, as `svd_converged` decides it.
#
# Whether one solve converges is a rounding-path outcome, so the sweep asserts a rate. Measured on
# aarch64 under `--check-bounds=yes` (1.12.7, 1.13.1, 1.14-DEV) and `auto` (1.13.1), both precisions:
# every combination converges from at least 7 of the 8 seeds, and at most one seed misses:
#
# - `DFP  Quadratic  Geodesic`, seed 8, in both precisions, here and in CI on x64: it stops
#   after 3 iterations at `rg = 3.5`, a relative error of 1.3, with `x_converged` set. Its line search
#   returns `LINESEARCH_FLOOR` on a step whose merit *rose*, the steepest-descent retry does the same,
#   and `solver_step!` takes a zero step, which `x_converged` reads as convergence (issue #153, K27 in
#   KNOWN_ISSUES.md).
# - `DFP  Backtracking(expand)  Geodesic`, seed 2, `Float64`, 1.13 and 1.14 under `--check-bounds=yes`
#   only: it reaches the 5_000 cap, where it takes 1_288 iterations under `auto`.
#
# CI on x64 1.12.7 (Linux and Windows) misses one more solve in `Float32`, which stops at `rg = 3.9`
# and a relative error of 1.26 without reaching the cap; which combination it belongs to is not
# known, so one combination may converge from 6 of 8 there, and no measured run gives fewer. The rate
# still fails on a regression: with the guard that `curvature_is_usable` replaced
# (`!iszero(ΔxΔg) && !isnan(ΔxΔg)`), `DFP  Backtracking(expand)` converges from 5 of 8 seeds in
# `Float64` under either retraction, and a combination that breaks outright converges from none.
const SWEEP_MIN_CONVERGED = 6

"""
    svd_converged(T, ps, state, result, best, max_iterations)

Whether a solve passes the convergence assertions of `svd_convergence_check`: it stops before the
cap on a convergence criterion, at `rg`, objective and subspace inside their tolerances.
"""
function svd_converged(::Type{T}, ps, state, result, best, max_iterations) where {T}
    status = GeometricOptimizers.status(result)
    GeometricOptimizers.iteration_number(state) < max_iterations &&
        GeometricOptimizers.isconverged(status) &&
        status.rg < converged_gradient_tolerance(T) &&
        relative_error(ps, best) < converged_error_tolerance(T) &&
        projector_distance(ps, best) < projector_tolerance(T)
end

@testset "eight-seed sweep, $T" for T in REAL_ELTYPES
    objective = svd_objective(svd_matrix(T))
    best = best_rank_n(T, 3)

    for retraction in (GeometricOptimizers.Geodesic(), GeometricOptimizers.Cayley()),
        (label, algorithm, linesearch) in SWEEP_COMBINATIONS

        converged = 0
        on_the_manifold = 0
        for seed in SWEEP_SEEDS
            ps = starting_point(T, 3, seed)
            state = OptimizerState(algorithm, ps)
            optimizer = Optimizer(
                ps, objective; retraction = retraction, algorithm = algorithm,
                linesearch = linesearch(T), max_iterations = CONVERGENCE_MAX_ITERATIONS,
                warn_iterations = 0)
            result = solve!(ps, state, optimizer)
            converged += svd_converged(
                T, ps, state, result, best, CONVERGENCE_MAX_ITERATIONS)
            on_the_manifold += all(
                Y -> GeometricOptimizers.check(Y) < manifold_tolerance(T), values(ps))
        end
        @testset "$label, $(nameof(typeof(retraction)))" begin
            @test converged >= SWEEP_MIN_CONVERGED
            @test on_the_manifold == length(SWEEP_SEEDS)
        end
    end
end
