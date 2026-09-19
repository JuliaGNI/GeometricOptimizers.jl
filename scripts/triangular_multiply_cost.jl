# What `*(::AbstractTriangular, ::AbstractMatrix)` costs on the host against the generic
# `AbstractMatrix` product it shadows, in time and in bytes.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/triangular_multiply_cost.jl
#
# This is the check behind the table in `CHANGELOG.md` and behind the paragraph in the docstring of
# `*(::AbstractTriangular, ::AbstractMatrix)` that tells a reader not to expect a host speed-up.
# Both quote figures from this script and from nothing else.
#
# ## Why the measurement is needed at all
#
# The kernel method exists for the device: the generic product reads its argument one entry at a
# time, which no device array serves. On the host it is easy to argue it should also be faster —
# the packed vector holds `n(n-1)/2` entries where the generic path reads `n²`, the other
# `n(n+1)/2` being zeros `getindex` manufactures — and that argument predicts about a factor of
# two. It is a counting argument, not a measurement, and it is wrong: a `KernelAbstractions` launch
# has a fixed cost that a small product cannot amortize, and the small product is the common case
# here. The retraction tests use `n = 6`.
#
# ## Why `invoke` and not a second implementation
#
# The baseline has to be *the method this one shadows*, not a reimplementation of it.
# `invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)` reaches the generic product with the
# same argument, in the same process, with no source change and nothing to revert.
#
# ## Four traps this script is written around
#
#   * **Compilation.** Timing two variants in one process measures the first cold and the second
#     warm. Both are called once at every size before anything is recorded.
#   * **A single sample.** The time column is a median over many samples, and the point is the
#     shape of the column across `n` rather than any one entry.
#   * **The clock.** `time_ns` advances in steps of about 42 ns on an Apple Silicon host, and an
#     `n = 6` product takes about three of those steps. Timing one call per sample therefore
#     measures the clock: every median is a small whole number of ticks, and the ratio comes out as
#     3/4 or 4/4 depending on where the noise falls. `calls_per_sample` folds enough calls into one
#     sample that a tick is a rounding error, which is what makes the `n = 6` row mean anything.
#   * **Dead code.** Neither result escapes, so both are folded into a checksum that is printed.

using GeometricOptimizers: LowerTriangular, UpperTriangular
using Printf
using Random

const SIZES = [6, 32, 128, 512]
const T = Float64

"Samples at a given size: enough for a stable median, few enough to finish at 512."
samples(n) = n ≤ 32 ? 201 : 31

"The shortest a timing sample may be, in seconds. About five hundred ticks of a 42 ns clock."
const SAMPLE_FLOOR = 2e-5

kernel_product(A, B) = A * B
generic_product(A, B) = invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)

"Calls to fold into one timing sample, so that `SAMPLE_FLOOR` is reached. One pilot call sizes it."
function calls_per_sample(f, A, B)
    f(A, B)
    t = @elapsed f(A, B)
    max(1, ceil(Int, SAMPLE_FLOOR / max(t, 1e-9)))
end

"Median seconds per call over `k` samples of `c` calls each, after one warm-up call."
function median_time(f, A, B, k, c)
    f(A, B)
    times = Vector{Float64}(undef, k)
    for i in 1:k
        t = time_ns()
        for _ in 1:c
            f(A, B)
        end
        times[i] = (time_ns() - t) * 1e-9 / c
    end
    sort!(times)
    times[(k + 1) ÷ 2]
end

function main()
    Random.seed!(20260919)
    checksum = zero(T)

    @printf("%-16s %6s %7s %12s %12s %8s %10s %10s\n",
        "type", "n", "calls", "kernel [s]", "generic [s]", "ratio", "kernel [B]",
        "generic [B]")
    println("-"^90)

    for MT in (LowerTriangular, UpperTriangular), n in SIZES

        A = MT(rand(T, n * (n - 1) ÷ 2), n)
        B = rand(T, n, n)

        # both warmed at this size before either is recorded
        checksum += sum(kernel_product(A, B)) + sum(generic_product(A, B))

        # one call count for both, so that the two columns are folded the same way
        c = max(calls_per_sample(kernel_product, A, B), calls_per_sample(generic_product, A, B))
        k = samples(n)
        t_kernel = median_time(kernel_product, A, B, k, c)
        t_generic = median_time(generic_product, A, B, k, c)

        b_kernel = @allocated kernel_product(A, B)
        b_generic = @allocated generic_product(A, B)

        @printf("%-16s %6d %7d %12.3e %12.3e %8.2f %10d %10d\n",
            nameof(MT), n, c, t_kernel, t_generic, t_generic / t_kernel, b_kernel,
            b_generic)
    end

    println()
    @printf("element type %s, %d samples up to n = 32 and %d above, medians per call\n",
        T, samples(32), samples(128))
    @printf("ratio > 1 means the kernel is faster; checksum %.6e\n", checksum)
end

main()
