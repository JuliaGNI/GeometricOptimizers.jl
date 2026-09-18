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
# ## Three traps this script is written around
#
#   * **Compilation.** Timing two variants in one process measures the first cold and the second
#     warm. Both are called once at every size before anything is recorded.
#   * **A single sample.** The time column is a median over many repetitions, and the point is the
#     shape of the column across `n` rather than any one entry.
#   * **Dead code.** Neither result escapes, so both are folded into a checksum that is printed.

using GeometricOptimizers: LowerTriangular, UpperTriangular
using Printf
using Random

const SIZES = [6, 32, 128, 512]
const T = Float64

"Repetitions at a given size: enough to be stable, few enough to finish at 512."
reps(n) = n ≤ 32 ? 2000 : 30

kernel_product(A, B) = A * B
generic_product(A, B) = invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)

"Median seconds over `k` calls, after one warm-up call."
function median_time(f, A, B, k)
    f(A, B)
    times = Vector{Float64}(undef, k)
    for i in 1:k
        t = time_ns()
        f(A, B)
        times[i] = (time_ns() - t) * 1e-9
    end
    sort!(times)
    times[(k + 1) ÷ 2]
end

function main()
    Random.seed!(20260919)
    checksum = zero(T)

    @printf("%-16s %6s %12s %12s %8s %10s %10s\n",
        "type", "n", "kernel [s]", "generic [s]", "ratio", "kernel [B]", "generic [B]")
    println("-"^82)

    for MT in (LowerTriangular, UpperTriangular), n in SIZES

        A = MT(rand(T, n * (n - 1) ÷ 2), n)
        B = rand(T, n, n)

        # both warmed at this size before either is recorded
        checksum += sum(kernel_product(A, B)) + sum(generic_product(A, B))

        k = reps(n)
        t_kernel = median_time(kernel_product, A, B, k)
        t_generic = median_time(generic_product, A, B, k)

        b_kernel = @allocated kernel_product(A, B)
        b_generic = @allocated generic_product(A, B)

        @printf("%-16s %6d %12.3e %12.3e %8.2f %10d %10d\n",
            nameof(MT), n, t_kernel, t_generic, t_generic / t_kernel, b_kernel, b_generic)
    end

    println()
    @printf("element type %s, %d reps below n = 128 and %d above, medians\n",
        T, reps(32), reps(128))
    @printf("ratio > 1 means the kernel is faster; checksum %.6e\n", checksum)
end

main()
