# How often `GeometricOptimizers._cholesky_qr2` breaks down on the draw `global_section` gives it,
# and how often the redraw behind it fails as well.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/orthonormalization_breakdown_rate.jl
#
# This is the check behind the rate quoted in the docstrings of
# `GeometricOptimizers._cholesky_qr2` and `GeometricOptimizers.orthonormal_columns`, and behind the
# `ORTHONORMALIZATION_ATTEMPTS` bound. Those three quote figures from this script and from nothing
# else.
#
# ## Why the rate is worth measuring rather than estimating
#
# Forming `AᵀA` squares the condition number, so a draw that a Householder QR handles without
# comment can leave the Gram matrix numerically indefinite. `global_section` factorizes `N × (N-n)`
# Gaussian columns with the span of the point projected out — square inside that complement — and a
# square Gaussian's condition number has a heavy tail. In `Float32` that tail reaches the failure
# threshold often enough to matter, which is what `orthonormal_columns` exists for.
#
# ## Why three seeds and not one
#
# The quantity is itself a rate estimated from a sample, so a single run reports one draw from a
# binomial and reads as more precise than it is. Three independent seeds are run and all three
# counts are printed. The docstring quotes the range they span, not their mean.
#
# ## What the redraw column establishes
#
# `orthonormal_columns` replaces a failed draw rather than repairing it, so the relevant question
# is not the failure rate alone but whether *consecutive* draws fail. The second column counts the
# draws whose immediate replacement also failed. It has been zero in every run so far, which is
# what makes the `ORTHONORMALIZATION_ATTEMPTS = 8` bound comfortable rather than lucky.

using GeometricOptimizers: _cholesky_qr2
using LinearAlgebra: qr!
using Printf
using Random

const SHAPES = [(50, 3), (100, 3), (200, 3)]
const DRAWS_PER_SHAPE = 1000
const SEEDS = [4242, 271828, 31415]
const T = Float32

"The argument `global_section` hands the orthonormalization, for a fresh random point each time."
function complement_draw(::Type{S}, N::Int, n::Int) where {S}
    Y = Matrix(qr!(randn(S, N, n)).Q)
    A = randn(S, N, N - n)
    A - Y * (Y' * A)
end

"""
Failures, and failures whose immediate redraw also failed, over `DRAWS_PER_SHAPE` draws per shape.

Returns `(breakdowns, consecutive, total)`.
"""
function count_breakdowns(seed::Int)
    Random.seed!(seed)
    breakdowns = 0
    consecutive = 0
    total = 0

    for (N, n) in SHAPES, _ in 1:DRAWS_PER_SHAPE

        total += 1
        if _cholesky_qr2(complement_draw(T, N, n)) === nothing
            breakdowns += 1
            # The redraw is what `orthonormal_columns` would do next. Count it failing too.
            _cholesky_qr2(complement_draw(T, N, n)) === nothing && (consecutive += 1)
        end
    end

    breakdowns, consecutive, total
end

function main()
    @printf("%8s  %11s  %7s  %12s  %14s\n",
        "seed", "breakdowns", "of", "rate", "redraw failed")
    println("-"^58)

    rates = Float64[]
    for seed in SEEDS
        breakdowns, consecutive, total = count_breakdowns(seed)
        push!(rates, breakdowns / total)
        @printf("%8d  %11d  %7d  %12s  %14d\n",
            seed, breakdowns, total,
            breakdowns == 0 ? "-" : @sprintf("1 in %.0f", total / breakdowns),
            consecutive)
    end

    println()
    @printf("element type %s, shapes %s, %d draws each\n", T, SHAPES, DRAWS_PER_SHAPE)
    @printf("rate across seeds: 1 in %.0f to 1 in %.0f\n",
        1 / maximum(rates), 1 / minimum(rates))
end

main()
