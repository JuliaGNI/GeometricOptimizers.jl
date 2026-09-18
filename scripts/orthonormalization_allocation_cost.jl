# What `GeometricOptimizers._cholesky_qr2(A)` costs in bytes against the Householder QR it replaced.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/orthonormalization_allocation_cost.jl
#
# This is the check behind the allocation table in the CHANGELOG entry for CholeskyQR2. That table
# is quoted from this script and from nothing else.
#
# ## What is compared, and why the baseline is spelled this way
#
# The expression CholeskyQR2 replaced is `typeof(A)(qr!(A).Q)`, on the freshly drawn `A` and in
# place — that is what `rand(backend, MT, N, n)` and `global_section` both used to run. The
# baseline below is therefore that expression and not `typeof(A)(qr!(copy(A)).Q)`.
#
# The difference is the whole point of this file. `qr!` destroys its argument, so a baseline
# measured over several repetitions has to get a fresh `A` from somewhere. Putting a `copy(A)`
# inside the measured expression is the obvious way and it is wrong: it charges the baseline for an
# allocation the production path never made, which flatters CholeskyQR2 by the size of one argument
# at every row. An earlier round of these figures did exactly that. The refresh here is a
# `copyto!` into a scratch matrix allocated once, outside `@allocated`.
#
# The `copy(A)` column is printed so that the two ratios can be read side by side and the trap stays
# visible rather than being a sentence someone has to believe.
#
# ## The shape
#
# `N × (N-n)` with `n = 3`, which is what `global_section` factorizes: the complement of an
# `N × n` point. That shape is square inside the complement, and it is the one on the retraction
# path, so it is the shape whose cost matters.
#
# ## Three traps this script is written around
#
#   * **Compilation.** Timing or measuring two variants in one process runs the first cold and the
#     second warm. Both are run once at every size before anything is recorded.
#   * **A single `@allocated` call.** One call can report a figure the neighbouring sizes
#     contradict. Every byte column is the minimum of `ALLOC_REPS` calls, and the point is the shape
#     of the column rather than any one row.
#   * **Dead code.** No result escapes, so all of them are consumed into a checksum that is printed.

using GeometricOptimizers: _cholesky_qr2
using LinearAlgebra: qr!
using Printf
using Random

const SIZES = [20, 50, 100, 200, 400]
const N_SMALL = 3
const ALLOC_REPS = 30
const SEED = 4242

"The argument `global_section` hands the orthonormalization: `N × (N-n)` with the point projected out."
function complement_draw(::Type{T}, N::Int, n::Int) where {T}
    Y = Matrix(qr!(randn(T, N, n)).Q)
    A = randn(T, N, N - n)
    A - Y * (Y' * A)
end

"Minimum of `ALLOC_REPS` `@allocated` calls, so one outlying figure cannot set the row."
function cholesky_qr2_bytes(A)
    minimum(1:ALLOC_REPS) do _
        @allocated _cholesky_qr2(A)
    end
end

"""
The same, for the in-place Householder QR.

`W` is allocated by the caller and refreshed with `copyto!` *outside* the measured expression,
because `qr!` destroys what it is given and the production path never paid for a copy.
"""
function householder_bytes(A, W)
    minimum(1:ALLOC_REPS) do _
        copyto!(W, A)
        @allocated typeof(W)(qr!(W).Q)
    end
end

"What a `copy(A)` costs on its own — the term an in-expression refresh would wrongly add to the baseline."
function copy_bytes(A)
    minimum(1:ALLOC_REPS) do _
        @allocated copy(A)
    end
end

function main()
    Random.seed!(SEED)
    checksum = 0.0

    @printf("%5s  %6s  %14s  %14s  %12s  %8s  %10s\n",
        "N", "cols", "_cholesky_qr2", "qr! in place", "copy(A)", "ratio", "if copied")
    println("-"^82)

    for N in SIZES
        A = complement_draw(Float64, N, N_SMALL)
        W = similar(A)

        # Warm both paths at this size before anything is recorded.
        checksum += sum(abs, something(_cholesky_qr2(A), zero(A)))
        copyto!(W, A)
        checksum += sum(abs, typeof(W)(qr!(W).Q))

        chol = cholesky_qr2_bytes(A)
        house = householder_bytes(A, W)
        cp = copy_bytes(A)

        @printf("%5d  %6d  %14d  %14d  %12d  %8.2f  %10.2f\n",
            N, N - N_SMALL, chol, house, cp, chol / house, chol / (house + cp))
    end

    println()
    println("`ratio` is the honest one: both columns measure what the code actually runs.")
    println("`if copied` is what the baseline reads when a `copy(A)` is folded into it by mistake.")
    @printf("checksum %.6e\n", checksum)
end

main()
