# What `*(::AbstractLieAlgHorMatrix, ::AbstractMatrix)` costs on the host against the generic
# `AbstractMatrix` product it shadows, in time and in bytes.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/lift_multiply_cost.jl
#
# This is the check behind the table in `CHANGELOG.md`. That table quotes figures from this script
# and from nothing else.
#
# ## Why the measurement is needed at all
#
# The block method exists for the device: the generic product reads its argument one entry at a
# time, which no device array serves. On the host it is easy to argue it should also be faster —
# a lift stores `n(n-1)/2 + (N-n)n` numbers where the generic path reads `N²`, and the `(N-n)²`
# bottom-right block is zeros `getindex` manufactures. That is a counting argument and not a
# measurement: `scripts/triangular_multiply_cost.jl` measures the triangulars, where the same
# argument predicts a factor of two and the ratio is 0.76x to 1.72x.
#
# Two things make the lift a different measurement from the triangulars, and both cut the same way.
# The block path hands its two off-diagonal products to BLAS, where the generic path cannot use
# BLAS at all — a lift is not a `StridedArray`, so `invoke` reaches `generic_matmatmul!`. And the
# `A` block of a Stiefel lift is a `SkewSymMatrix`, whose product is a `KernelAbstractions` launch
# with a fixed cost that a small block cannot amortize. So the ratio is expected to be large at
# large `N` and to say nothing useful at small `N`, which is exactly why both ends are in the table.
#
# ## Why `invoke` and not a second implementation
#
# The baseline has to be *the method this one shadows*, not a reimplementation of it.
# `invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, B, C)` reaches the generic product with the
# same argument, in the same process, with no source change and nothing to revert.
#
# ## Five traps this script is written around
#
#   * **BLAS threads.** The new path calls BLAS and the baseline does not, so the thread count
#     decides the ratio outright. It is pinned to one below and printed with the table.
#   * **Compilation.** Timing two variants in one process measures the first cold and the second
#     warm. Both are called once at every size before anything is recorded.
#   * **A single sample.** The time column is a median over many samples, and the point is the
#     shape of the column across `N` rather than any one entry.
#   * **The clock.** `time_ns` advances in steps of about 42 ns on an Apple Silicon host, so timing
#     one call per sample at `N = 6` measures the clock and not the product. `calls_per_sample`
#     folds enough calls into one sample that a tick is a rounding error.
#   * **Dead code.** Neither result escapes, so both are folded into a checksum that is printed.

using GeometricOptimizers: GrassmannLieAlgHorMatrix, StiefelLieAlgHorMatrix
using LinearAlgebra: BLAS
using Printf
using Random

const SIZES = [6, 32, 128, 512]
const T = Float64

"Samples at a given size: enough for a stable median, few enough to finish at 512."
samples(N) = N ≤ 32 ? 201 : 31

"The shortest a timing sample may be, in seconds. About five hundred ticks of a 42 ns clock."
const SAMPLE_FLOOR = 2e-5

block_product(B, C) = B * C
generic_product(B, C) = invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, B, C)

"Calls to fold into one timing sample, so that `SAMPLE_FLOOR` is reached. One pilot call sizes it."
function calls_per_sample(f, B, C)
    f(B, C)
    t = @elapsed f(B, C)
    max(1, ceil(Int, SAMPLE_FLOOR / max(t, 1e-9)))
end

"Median seconds per call over `k` samples of `c` calls each, after one warm-up call."
function median_time(f, B, C, k, c)
    f(B, C)
    times = Vector{Float64}(undef, k)
    for i in 1:k
        t = time_ns()
        for _ in 1:c
            f(B, C)
        end
        times[i] = (time_ns() - t) * 1e-9 / c
    end
    sort!(times)
    times[(k + 1) ÷ 2]
end

function main()
    Random.seed!(20260921)
    BLAS.set_num_threads(1)
    checksum = zero(T)

    @printf("%-26s %6s %6s %7s %12s %12s %8s %10s %10s\n",
        "type", "N", "n", "calls", "block [s]", "generic [s]", "ratio", "block [B]",
        "generic [B]")
    println("-"^108)

    for LT in (StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix), N in SIZES

        n = N ÷ 2
        B = rand(LT{T}, N, n)
        C = rand(T, N, n)

        # both warmed at this size before either is recorded
        checksum += sum(block_product(B, C)) + sum(generic_product(B, C))

        # one call count for both, so that the two columns are folded the same way
        c = max(calls_per_sample(block_product, B, C), calls_per_sample(generic_product, B, C))
        k = samples(N)
        t_block = median_time(block_product, B, C, k, c)
        t_generic = median_time(generic_product, B, C, k, c)

        b_block = @allocated block_product(B, C)
        b_generic = @allocated generic_product(B, C)

        @printf("%-26s %6d %6d %7d %12.3e %12.3e %8.2f %10d %10d\n",
            nameof(LT), N, n, c, t_block, t_generic, t_generic / t_block, b_block,
            b_generic)
    end

    println()
    @printf("element type %s, %d samples up to N = 32 and %d above, medians per call\n",
        T, samples(32), samples(128))
    @printf("BLAS threads %d; ratio > 1 means the block path is faster; checksum %.6e\n",
        BLAS.get_num_threads(), checksum)
end

main()
