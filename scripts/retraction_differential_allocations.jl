# What `retraction_differential` allocates on the line-search path, which the retraction workspace
# does not reach.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/retraction_differential_allocations.jl
#
# This is the harness behind the CHANGELOG entry saying that `retraction_differential` is the
# largest per-trial manifold allocation left after `RetractionWorkspace`. It is the companion of
# `scripts/retraction_step_allocations.jl`, which measures the part the workspace does reach, and it
# uses that script's sizes and its fixture so the two tables can be read side by side.
#
# ## What is measured, and why α = 0 is a column of its own
#
# `retraction_differential(::Cayley, ::AbstractLieAlgHorMatrix, α)` returns `B` unchanged when `α`
# is zero and otherwise builds `lift_factors(B)` and `StiefelProjection(B)` fresh. The default
# `Backtracking` evaluates φ′ only at α = 0, so it pays nothing; a line search that evaluates φ′
# away from zero pays the figure in the other column, once per trial. Both columns are therefore
# load-bearing: one says the default is unaffected, the other says what such a search costs.
#
# ## Two traps this script is written around
#
#   * **Compilation.** Every call is made once at every size before anything is recorded, and the
#     `@allocated` sits inside a function with its arguments passed in, for the reason
#     `test/flat_buffer_allocations.jl` documents: a `@allocated` written at top level boxes its
#     variables and reports the box.
#   * **BLAS threads.** Pinned to one, as every other script here pins them. They do not move a byte
#     count, but a figure produced without the pin invites comparison with one produced with it.

using GeometricOptimizers
using GeometricOptimizers: GlobalSection, global_rep, retraction_differential
using LinearAlgebra: BLAS
using Printf
import Random

BLAS.set_num_threads(1)

# The sizes `scripts/retraction_step_allocations.jl` uses, so the two tables line up.
const SIZES = ((6, 3), (20, 3), (100, 5), (400, 5))

measured(f, args...) = (f(args...); @allocated f(args...))

# The same fixture `scripts/retraction_step_allocations.jl` builds, minus the parts only
# `update_section!` needs: one Stiefel point, its global section, and a horizontal lift.
function fixture(N, n)
    Y = rand(Random.Xoshiro(N + n), StiefelManifold{Float64}, N, n)
    Λ = GlobalSection(Y)

    global_rep(Λ, randn(Random.Xoshiro(N * n), N, n) ./ 100)
end

function differential_table()
    println("\nCayley: bytes per `retraction_differential` of a StiefelManifold lift")
    @printf("  %6s %4s %14s %14s\n", "N", "n", "α = 0", "α = 1/2")
    for (N, n) in SIZES
        B = fixture(N, n)
        # both values warmed at this size before either is recorded
        retraction_differential(Cayley(), B, 0.0)
        retraction_differential(Cayley(), B, 0.5)

        at_zero = measured(retraction_differential, Cayley(), B, 0.0)
        at_half = measured(retraction_differential, Cayley(), B, 0.5)
        @printf("  %6d %4d %14d %14d\n", N, n, at_zero, at_half)
    end
end

differential_table()
