# What one manifold solver step allocates, and what the dense-to-packed projections, `Ω`, the skew
# product and the symplectic decompositions allocate.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --check-bounds=auto --project=. scripts/solver_step_allocations.jl
#
# This is the harness behind the figures of the CHANGELOG entries on the line-search buffers, the
# packed projections and the symplectic views. It calls only functions that the base of those
# entries has too, so the "before" column is this script run against that commit, in a tree of its
# own, with the same command.
#
# ## The solver step
#
# `StiefelManifold{Float64}` at `40 × 5`. The objective and its gradient allocate nothing, so the
# figure is what the optimizer allocates and not what the user's functions do. Each row is three
# warm iterations and then 21 measured ones, and the figure is their median, because the number of
# trials a line search takes is a property of the iterate.
#
# ## Traps this script is written around
#
#   * **Compilation.** Every call is made once before it is measured, and each `@allocated` is in a
#     function that takes its arguments as parameters.
#   * **BLAS threads.** Pinned to one, as in every other script here.
#   * **Time.** The times are medians over repeated calls in one process, on a machine that is never
#     idle. Quote the byte counts; quote a time only as a direction.

using GeometricOptimizers
using GeometricOptimizers: solver_step!, increase_iteration_number!, initialize_state!,
                           problem,
                           value, update!, map_to_S, map_to_Skew, Ω, apply_S_left
using LinearAlgebra: BLAS, I
using Printf
import Random

BLAS.set_num_threads(1)

const REPEATS = 21
const W = collect(range(0.1, 2.0; length = 200))

F(Y) = sum(i -> W[i] * parent(Y)[i]^3, eachindex(parent(Y)))
∇F!(g, v) = (g .= 3 .* W .* v .^ 2; g)

median_of(v) = sort(v)[(length(v) + 1) ÷ 2]

measured(f, args...) = (f(args...); @allocated f(args...))

function timed(f, args...)
    f(args...)
    median_of([@elapsed(f(args...)) for _ in 1:REPEATS])
end

function step!(x, state, opt)
    increase_iteration_number!(state)
    solver_step!(x, state, opt)
end

function advance!(x, state, opt)
    f = value(problem(opt), x)
    update!(state, opt, x, f)
end

measured_step!(x, state, opt) = @allocated step!(x, state, opt)

function step_table()
    println("bytes per `solver_step!`, StiefelManifold{Float64} 40 × 5, median of ", REPEATS)
    @printf("  %-34s %10s %10s %10s\n", "algorithm and line search", "median", "min", "max")
    for (name, algorithm, linesearch) in (("BFGS, default search", BFGS(), nothing),
        ("GradientMethod, default search", GradientMethod(), nothing),
        ("GradientMethod, fixed step 1/50", GradientMethod(), 1 / 50))
        x = rand(Random.Xoshiro(7), StiefelManifold{Float64}, 40, 5)
        Random.seed!(1234)
        kwargs = linesearch === nothing ? (;) : (; linesearch)
        opt = Optimizer(x, F; (∇F!) = ∇F!, algorithm, max_iterations = 10_000, kwargs...)
        state = OptimizerState(algorithm, x)
        initialize_state!(state)
        for _ in 1:3
            step!(x, state, opt)
            advance!(x, state, opt)
        end

        bytes = Int[]
        for _ in 1:REPEATS
            push!(bytes, measured_step!(x, state, opt))
            advance!(x, state, opt)
        end
        @printf("  %-34s %10d %10d %10d\n", name, median_of(bytes), minimum(bytes),
            maximum(bytes))
    end
end

function kernel_table()
    println("\nthe other figures")
    @printf("  %-44s %12s %10s\n", "call", "bytes", "ms")
    rng = Random.Xoshiro(11)

    A = randn(rng, 1024, 1024)
    for (name, f) in (("map_to_S, n = 1024", map_to_S), (
        "map_to_Skew, n = 1024", map_to_Skew))
        @printf("  %-44s %12d %10.2f\n", name, measured(f, A), 1e3 * timed(f, A))
    end

    Y = rand(rng, StiefelManifold{Float64}, 60, 5)
    Δ = rgrad(Y, randn(rng, 60, 5))
    @printf("  %-44s %12d\n", "Ω(Y, Δ), StiefelManifold 60 × 5", measured(Ω, Y, Δ))

    S₁, S₂ = SkewSymMatrix(randn(rng, 6, 6)), SkewSymMatrix(randn(rng, 6, 6))
    @printf("  %-44s %12d\n", "SkewSymMatrix * SkewSymMatrix, n = 6", measured(*, S₁, S₂))

    J = [zeros(40, 40) I(40); -I(40) zeros(40, 40)]
    B = randn(rng, 80, 80)
    gram_schmidt(B, J) = symplectic_gram_schmidt!(copy(B), J)
    @printf("  %-44s %12d\n", "symplectic_gram_schmidt! on a copy, 80 × 80",
        measured(gram_schmidt, B, J))

    C = randn(rng, 80, 6)
    @printf("  %-44s %12d\n", "sr, 80 × 6", measured(sr, C))
    Λ = sr(C).S.Λ
    @printf("  %-44s %12d\n", "apply_S_left, 80 × 3 against sr(80 × 6)",
        measured(apply_S_left, Λ, randn(rng, 80, 3)))
end

println("GeometricOptimizers at ", pkgdir(GeometricOptimizers), ", Julia ", VERSION,
    ", check-bounds ", Base.JLOptions().check_bounds, "\n")
step_table()
kernel_table()
