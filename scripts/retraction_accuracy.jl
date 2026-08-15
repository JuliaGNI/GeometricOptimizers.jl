# Regenerates every figure this package quotes about the retractions and about the SVD problem.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/retraction_accuracy.jl
#
# Two independent parts, both cheap enough to run on a laptop:
#
#   * `exponential_tables()` — the accuracy and cost of the four `AbstractExponentialAlgorithm`s.
#     Feeds the tables in `src/retractions/exponential_algorithms.jl` and the note on `Cayley` in
#     `src/retractions/retraction_types.jl`. `docs/src/retractions.md` recomputes the accuracy
#     tables when the documentation is built, from the same seed and the same `SCALES`, so this
#     script and that page print the same rows.
#   * `svd_tables()` — iterations, objective evaluations, `‖∇f‖` and `check` for every (method, line
#     search, retraction) combination of `test/optimizer_convergence/svd_optim.jl`, on the seed that
#     file uses and across eight starting points. Feeds the tables in that file and in
#     `default_linesearch`'s docstring. Its `max_iterations` is the cap those tables report against,
#     so it belongs here and not at a call site: `_BFGS` with either polynomial search does not
#     converge on two of the eight under `Cayley` (open issue A1b), and what the tables print for
#     those is the cap itself.
#
# The timings are a `minimum` over repetitions with a single BLAS thread, which is the only form that
# is stable enough to quote. They are still machine-dependent; the accuracy figures are not.

using GeometricOptimizers
using GeometricOptimizers: geodesic, cayley, check, 𝔄, lift_factors, Geodesic, Cayley
using GeometricOptimizers: ScaledSquaring, AugmentedPade, ProjectedSkew, TaylorSeries
using GeometricOptimizers: _BFGS, _DFP, iteration_number, status
using SimpleSolvers: Static, Backtracking, Bisection, Quadratic, BierlaireQuadratic, StrongWolfe
using LinearAlgebra
using Printf
import Random

const ALGORITHMS = (TaylorSeries(), ScaledSquaring(), AugmentedPade(), ProjectedSkew())
const ALGORITHM_NAMES = ("TaylorSeries", "ScaledSquaring", "AugmentedPade", "ProjectedSkew")

"""
    best(f, repetitions = 20)

The fastest of `repetitions` runs of `f`, in milliseconds, after one warm-up call.
"""
function best(f, repetitions::Integer=20)
    f()
    minimum(1:repetitions) do _
        t₀ = time_ns()
        f()
        (time_ns() - t₀) / 1e6
    end
end

# The scales every accuracy table below sweeps over. All three use the *same* eight, drawn from the
# same seed, so a row of one is the same lift as the row of another — and so that
# `docs/src/retractions.md`, which recomputes these tables when the documentation is built, gets the
# figures this script prints rather than figures that merely resemble them.
const SCALES = (0.1, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0, 120.0)

"A sweep of horizontal lifts of increasing norm, all drawn from the same seed."
function sweep(T, N, n)
    Random.seed!(1234)
    [T(s) * rand(StiefelLieAlgHorMatrix{T}, N, n) for s in SCALES]
end

function exponential_tables(; N::Integer=20, n::Integer=3)
    BLAS.set_num_threads(1)

    println("== check(geodesic(B, algorithm)), St($N, $n), Float64 ==")
    print(rpad("‖B̄‖", 10))
    foreach(name -> print(lpad(name, 16)), ALGORITHM_NAMES)
    println(lpad("Cayley", 16))
    for B in sweep(Float64, N, n)
        @printf("%9.2f", norm(Matrix(B)))
        foreach(a -> @printf("%16.2e", check(geodesic(B, a))), ALGORITHMS)
        @printf("%16.2e\n", check(cayley(B)))
    end

    println("\n== relative error against exp(Matrix(B)) ==")
    print(rpad("‖B̄‖", 10))
    foreach(name -> print(lpad(name, 16)), ALGORITHM_NAMES[2:end])
    println()
    for B in sweep(Float64, N, n)
        reference = exp(Matrix(B))
        @printf("%9.2f", norm(Matrix(B)))
        foreach(ALGORITHMS[2:end]) do a
            @printf("%16.2e", norm(Matrix(geodesic(B, a)) - reference) / norm(reference))
        end
        println()
    end

    println("\n== Float32 ==")
    print(rpad("‖B̄‖", 10))
    foreach(name -> print(lpad(name, 16)), ALGORITHM_NAMES[2:end])
    println()
    for B in sweep(Float32, N, n)
        @printf("%9.2f", norm(Matrix(B)))
        foreach(a -> @printf("%16.2e", check(geodesic(B, a))), ALGORITHMS[2:end])
        println()
    end

    println("\n== ScaledSquaring: sensitivity to θ ==")
    Random.seed!(99)
    B = 30 * rand(StiefelLieAlgHorMatrix{Float64}, N, n)
    reference = exp(Matrix(B))
    @printf("‖B̄‖ = %.1f\n", norm(Matrix(B)))
    for θ in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
        Y = geodesic(B, ScaledSquaring(θ))
        @printf("θ = %5.3f   check %9.2e   error %9.2e\n", θ, check(Y),
            norm(Matrix(Y) - reference) / norm(reference))
    end

    println("\n== cost of one retraction, ms, minimum of 50, 1 BLAS thread ==")
    print(rpad("N", 7) * rpad("n", 6))
    foreach(name -> print(lpad(name, 16)), ALGORITHM_NAMES)
    println(lpad("Cayley", 16))
    Random.seed!(7)
    for (N, n) in ((10, 2), (20, 3), (50, 5), (100, 5), (200, 10), (500, 10), (500, 50), (1000, 20))
        B = rand(StiefelLieAlgHorMatrix{Float64}, N, n)
        @printf("%-7d%-6d", N, n)
        foreach(a -> @printf("%16.3f", best(() -> geodesic(B, a), 50)), ALGORITHMS)
        @printf("%16.3f\n", best(() -> cayley(B), 50))
    end
end

# ---------------------------------------------------------------------------------------------
# The SVD problem of `test/optimizer_convergence/svd_optim.jl`, which is where every iteration and
# evaluation count this package quotes comes from.

const A = include(joinpath(@__DIR__, "..", "test", "optimizer_convergence", "svd_matrix.jl"))

const COMBINATIONS = (
    ("_BFGS  Backtracking(expand)", _BFGS(), () -> Backtracking(Float64; expand=true)),
    ("_BFGS  Backtracking        ", _BFGS(), () -> Backtracking(Float64)),
    ("_BFGS  Bisection           ", _BFGS(), () -> Bisection(Float64)),
    ("_BFGS  StrongWolfe(c₂=0.1) ", _BFGS(), () -> StrongWolfe(Float64; c₂=0.1)),
    ("_BFGS  Quadratic           ", _BFGS(), () -> Quadratic(Float64)),
    ("_BFGS  BierlaireQuadratic  ", _BFGS(), () -> BierlaireQuadratic(Float64)),
    ("_DFP   Backtracking(expand)", _DFP(), () -> Backtracking(Float64; expand=true)),
    ("_DFP   Bisection           ", _DFP(), () -> Bisection(Float64)),
    ("_DFP   StrongWolfe(c₂=0.1) ", _DFP(), () -> StrongWolfe(Float64; c₂=0.1)),
    ("_DFP   Quadratic           ", _DFP(), () -> Quadratic(Float64)),
)

# The cap the "iters over 8 seeds" column of `svd_optim.jl` reports against, and the value that makes
# its "cap" entries mean something: `_BFGS` with either polynomial search runs out of iterations on
# two of the eight under `Cayley` rather than converging (open issue A1b). It has to be a constant
# rather than a call-site keyword, because a sweep run at a different cap prints a different table for
# those rows and the table does not say which cap it was.
const SVD_MAX_ITERATIONS = 20_000

"""
    solve_once(algorithm, linesearch, retraction, seed; max_iterations, step_ceiling)

One solve of the SVD problem, returning its iteration count, objective evaluations, final `‖∇f‖`,
worst `check` over the two factors, and relative error against the best rank-3 approximation.

`step_ceiling` is the knob of issue A1b, in multiples of `2π` (see `DEFAULT_STEP_CEILING`). It is a
keyword rather than a constant because the *comparison* between a ceiling and none is the measurement
the entry rests on, and both halves of it have to come from this harness rather than from a REPL:
`svd_tables(step_ceiling = Inf)` is the "before" column of every table this script feeds.
"""
function solve_once(algorithm, linesearch, retraction, seed::Integer; max_iterations::Integer=SVD_MAX_ITERATIONS,
    step_ceiling=GeometricOptimizers.DEFAULT_STEP_CEILING)
    evaluations = Ref(0)
    objective(ps::NamedTuple) = (evaluations[] += 1; norm(A - ps.w₁ * ps.w₂' * A))

    Random.seed!(seed)
    ps = (w₁=rand(StiefelManifold, size(A, 1), 3), w₂=rand(StiefelManifold, size(A, 1), 3))
    state = OptimizerState(algorithm, ps)
    optimizer = Optimizer(ps, objective; retraction=retraction, algorithm=algorithm,
        linesearch=linesearch, max_iterations=max_iterations, warn_iterations=0,
        step_ceiling=step_ceiling)
    result = solve!(ps, state, optimizer)

    U, _, _ = svd(A)
    err_best = norm(A - U[:, 1:3] * U[:, 1:3]' * A)

    (iterations=iteration_number(state), evaluations=evaluations[],
        rg=status(result).rg, check=maximum(check, values(ps)),
        error=abs((objective(ps) - err_best) / err_best))
end

# The tolerance `test/optimizer_convergence/svd_optim.jl` and `test/manifold_linesearch_tests.jl` both
# use for "still on the manifold". Both iterates stay on `St(N, 3)` when a solve behaves, so this is a
# round-off bound and nothing else; the values observed are of the order of `1e-14`.
const MANIFOLD_TOLERANCE = 1e-12

"""
    on_the_manifold(results)

How many of `results` ended with both factors still on the Stiefel manifold.

This is the statistic issue A1b was actually about, and the one the `worst check` column only implies:
A1b's failure is a solve that *reports success* from a point that is no longer on `St(20, 3)`, so what
matters is the count of seeds that end inside `MANIFOLD_TOLERANCE` rather than how far the worst one
strayed. The two say different things -- one bad seed and four bad seeds can give the same worst
`check`, and the seed count is what moved from 4 to 8 when the step ceiling was passed.
"""
on_the_manifold(results) = count(r -> r.check ≤ MANIFOLD_TOLERANCE, results)

function svd_tables(; seeds=1:8, step_ceiling=GeometricOptimizers.DEFAULT_STEP_CEILING)
    for (name, retraction) in (("Geodesic", Geodesic()), ("Cayley", Cayley()))
        println("\n== $name: seed 1234, then the spread over seeds $(first(seeds))..$(last(seeds)), " *
                "cap $(SVD_MAX_ITERATIONS), step_ceiling $(step_ceiling) ==")
        println(rpad("combination", 30) * lpad("iters", 8) * lpad("evals", 11) *
                lpad("rg", 11) * lpad("check", 11) * "   iters over the seeds")
        for (label, algorithm, linesearch) in COMBINATIONS
            pinned = solve_once(algorithm, linesearch(), retraction, 1234; step_ceiling=step_ceiling)
            over_seeds = [solve_once(algorithm, linesearch(), retraction, seed; step_ceiling=step_ceiling)
                          for seed in seeds]
            @printf("%-30s%8d%11d%11.2e%11.2e   %d..%d  (%d/%d on the manifold, worst rg %.2e, worst check %.2e)\n",
                label, pinned.iterations, pinned.evaluations, pinned.rg, pinned.check,
                minimum(r -> r.iterations, over_seeds), maximum(r -> r.iterations, over_seeds),
                on_the_manifold(over_seeds), length(over_seeds),
                maximum(r -> r.rg, over_seeds), maximum(r -> r.check, over_seeds))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exponential_tables()
    svd_tables()
end
