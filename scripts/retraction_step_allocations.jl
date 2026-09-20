# What a retraction and an optimizer iteration allocate on a manifold, with the retraction workspace
# and without it.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/retraction_step_allocations.jl
#
# This is the harness behind the CHANGELOG entry on `RetractionWorkspace`, and it exists for the
# reason the *Open Issues* preamble states: a number is reproducible only where the harness that
# produced it is named. `test/flat_buffer_allocations.jl` is the complement — it asserts the
# *property* the second table shows, that a retraction taken in a workspace costs the same at every
# ambient dimension, where this says what that is worth.
#
# ## What is compared, and why both columns come from the shipped code
#
# `update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B, retraction, workspace)` takes the retraction in `workspace` when
# one is passed and in fresh arrays when `nothing` is. Both columns below are that one function, so
# neither can go stale against an implementation it copies — which is how a sibling script put four
# exact-looking byte figures into `CHANGELOG.md` that had stopped describing the code.
#
# ## Three traps this script is written around
#
#   * **Compilation.** Every call is made once at every size before anything is recorded, and the
#     `@allocated` is inside a function with its arguments passed in. A `@allocated` written in a
#     loop at top level boxes its loop variables and reports the box; `test/flat_buffer_allocations.jl`
#     has the measurement of what that costs.
#   * **BLAS threads.** Pinned to one. They do not move a byte count, but every other script here
#     pins them and a script that does not invites its figures to be compared with one that does.
#   * **The line search.** The per-iteration figure in the third table is a *median*, because how
#     many trials a search takes is a property of the problem: the same step over 21 repeats ran
#     from 50 112 to 95 824 bytes.

using GeometricOptimizers
using GeometricOptimizers: GlobalSection, global_rep, update_section!, retraction_workspace,
                           solver_step!, increase_iteration_number!, initialize_state!,
                           OptimizerStatus, cache, config, problem, value, update!
using NeuralNetworkParameters: NetworkParameters
using LinearAlgebra: BLAS
using Printf
using Statistics: median
import Random

BLAS.set_num_threads(1)

const SIZES = ((6, 3), (20, 3), (100, 5), (400, 5))
const REPEATS = 21

# The three shapes of solution the per-iteration table covers, and the problem each is solved on.
const N, n, m = 6, 3, 4
Random.seed!(1234)
const TARGET = randn(N, m)

vector_problem() = (randn(Random.Xoshiro(3), 12), v -> sum(abs2, v))

function manifold_problem()
    (rand(Random.Xoshiro(4), StiefelManifold{Float64}, N, n),
        Y -> sum(abs2, Y * ones(n, m) .- TARGET) / 2)
end

function container_problem()
    (
        NetworkParameters((Y = rand(Random.Xoshiro(1), StiefelManifold{Float64}, N, n),
            W = randn(Random.Xoshiro(2), n, m), b = zeros(N))),
        ps -> sum(abs2, ps.Y * ps.W .+ ps.b .- TARGET) / 2)
end

measured(f, args...) = (f(args...); @allocated f(args...))

function fixture(N, n)
    Y = rand(Random.Xoshiro(N + n), StiefelManifold{Float64}, N, n)
    Λ = GlobalSection(Y)

    (Λ = Λ, Λ₂ = GlobalSection(Y), ws = retraction_workspace(Y),
        B = global_rep(Λ, randn(Random.Xoshiro(N * n), N, n) ./ 100))
end

function retraction_table(R)
    println("\n", nameof(typeof(R)), ": bytes per `update_section!` of a StiefelManifold")
    @printf("  %6s %4s %14s %14s %8s\n", "N", "n", "no workspace", "workspace", "ratio")
    for (N, n) in SIZES
        f = fixture(N, n)
        # both variants warmed at this size before either is recorded
        update_section!(f.Λ₂, f.Λ, f.B, R, nothing)
        update_section!(f.Λ₂, f.Λ, f.B, R, f.ws)

        bare = measured(update_section!, f.Λ₂, f.Λ, f.B, R, nothing)
        held = measured(update_section!, f.Λ₂, f.Λ, f.B, R, f.ws)
        @printf("  %6d %4d %14d %14d %8.1f\n", N, n, bare, held, bare / held)
    end
end

# The body of `solve!`'s `while`, minus the trace push and the stopping test: one iteration of the
# real loop, not a `solver_step!` on its own, because the status and the state update are part of
# what an iteration costs.
function step!(x, state, opt)
    increase_iteration_number!(state)
    solver_step!(x, state, opt)
    f = value(problem(opt), x)
    OptimizerStatus(state, cache(opt), f; config = config(opt))
    update!(state, opt, x)

    f
end

measured_step!(x, state, opt) = @allocated step!(x, state, opt)

function step_table()
    println("\nbytes per iteration of the `solve!` loop body, median of ", REPEATS)
    @printf("  %-22s %-16s %10s %10s %10s\n", "solution", "algorithm", "median", "min",
        "max")
    for (name, make) in (("Vector", vector_problem), ("StiefelManifold", manifold_problem),
            ("NetworkParameters", container_problem)),
        algorithm in (BFGS(), GradientMethod())

        x, F = make()
        Random.seed!(1234)
        opt = Optimizer(x, F; algorithm = algorithm, max_iterations = 10_000)
        state = OptimizerState(algorithm, x)
        initialize_state!(state)
        step!(x, state, opt)

        bytes = [measured_step!(x, state, opt) for _ in 1:REPEATS]
        @printf("  %-22s %-16s %10d %10d %10d\n", name, nameof(typeof(algorithm)),
            median(bytes), minimum(bytes), maximum(bytes))
    end
end

function main()
    retraction_table(Cayley())
    retraction_table(Geodesic())
    step_table()
end

main()
