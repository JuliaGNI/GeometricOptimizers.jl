# What the grouping of `metric(::SymplecticStiefelManifold, ::AbstractMatrix, ::AbstractMatrix)`
# costs, against the grouping it replaced.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/symplectic_metric_cost.jl
#
# With no argument both groupings are measured in one process, each warmed at every size before
# anything is recorded. Pass `old` or `new` to measure one of them alone, which is the
# one-process-per-variant form:
#
#     julia --startup-file=no --project=. scripts/symplectic_metric_cost.jl old
#     julia --startup-file=no --project=. scripts/symplectic_metric_cost.jl new
#
# This is the check behind the *Implementation* paragraph in the docstring of `metric` and behind
# the `CHANGELOG.md` entry. Both quote figures from this script and from nothing else.
#
# ## What the two groupings are
#
# Both evaluate `g_U(Δ₁, Δ₂) = tr(P Δ₁ᵀ(𝕀_2N - ½𝕁ᵀUPUᵀ𝕁)Δ₂)` with `P = (UᵀU)⁻¹`. They differ in
# what they build on the way:
#
#   * `old` — the expression as the definition writes it. The middle factor is a dense `2N × 2N`
#     matrix, `𝕁ᵀUPUᵀ𝕁`, and forming it costs `O(N³)`: the last of its four products multiplies two
#     matrices of the ambient dimension. The `2N × 2N` identity is built to subtract from it, and
#     `P` is inverted twice, once in the trace and once inside the middle factor.
#   * `new` — `tr(P(Δ₁ᵀΔ₂ - ½(Δ₁ᵀ𝕁ᵀU)P(Uᵀ𝕁Δ₂)))`. Multiplying the middle factor out first leaves
#     three `2n × 2n` matrices under one trace. Nothing larger than `𝕁Δ₂` is formed, the identity
#     is not built at all, and `P` is inverted once.
#
# `old` is written out here rather than reached through `invoke`, because the new method does not
# shadow it — it replaced it.
#
# ## Why the accuracy of the points does not matter here
#
# The draws at the larger sizes are far from the manifold — `SymplecticStiefelManifold`'s docstring
# carries the residuals and `CHANGELOG.md` the table. Nothing below reads the value either grouping
# returns; that the two agree is asserted in `test/manifolds/symplectic_stiefel_manifold.jl`, on
# points small enough for the question to mean something. What is timed is arithmetic on matrices
# of the right shapes, and that is insensitive to where the point lies.
#
# ## Four traps this script is written around
#
#   * **Compilation.** Timing two variants in one process measures the first cold and the second
#     warm. Both are called once at every size before anything is recorded, and the `old`/`new`
#     arguments exist for anyone who would rather not rely on that.
#   * **A single sample.** The time column is a median over many samples, and the point is the
#     shape of the column across `2N` rather than any one entry.
#   * **BLAS threading.** The `2N × 2N` products in `old` are large enough to be threaded, and the
#     thread count is printed with the table because the ratio moves with it.
#   * **Dead code.** Neither result escapes, so both are folded into a checksum that is printed.

using GeometricOptimizers: SymplecticStiefelManifold, _poisson_tensor, rgrad, unit_matrix
using LinearAlgebra
using Printf
using Random

const SIZES = [20, 40, 80, 160]
const N_SMALL = 6
const T = Float64

"Samples at a given size: enough for a stable median, few enough to finish at 2N = 160."
samples(N2) = N2 ≤ 80 ? 201 : 101

"The shortest a timing sample may be, in seconds. About five hundred ticks of a 42 ns clock."
const SAMPLE_FLOOR = 2e-5

"The metric as its definition writes it, through the `2N × 2N` middle factor."
function old_metric(U::SymplecticStiefelManifold{T}, Δ₁, Δ₂) where {T}
    J = _poisson_tensor(U, size(U, 1))
    tr(inv(U' * U) * Δ₁' *
       (unit_matrix(J) - (T(1) / 2) * J' * U * inv(U' * U) * U' * J) * Δ₂)
end

"The metric as this package evaluates it, in the `2n × 2n` factors."
function new_metric(U::SymplecticStiefelManifold{T}, Δ₁, Δ₂) where {T}
    J = _poisson_tensor(U, size(U, 1))
    P = inv(U' * U)
    tr(P * (Δ₁' * Δ₂ - (T(1) / 2) * (Δ₁' * (J' * U)) * P * ((U' * J) * Δ₂)))
end

"Calls to fold into one timing sample, so that `SAMPLE_FLOOR` is reached. One pilot call sizes it."
function calls_per_sample(f, args)
    f(args...)
    t = @elapsed f(args...)
    max(1, ceil(Int, SAMPLE_FLOOR / max(t, 1e-9)))
end

"Median seconds per call over `k` samples of `c` calls each, after one warm-up call."
function median_time(f, args, k, c)
    f(args...)
    times = Vector{Float64}(undef, k)
    for i in 1:k
        t = time_ns()
        for _ in 1:c
            f(args...)
        end
        times[i] = (time_ns() - t) * 1e-9 / c
    end
    sort!(times)
    times[(k + 1) ÷ 2]
end

function main(variants)
    Random.seed!(20260919)
    checksum = zero(T)

    @printf("%6s %6s %7s %12s %12s %8s %12s %12s\n",
        "2N", "2n", "calls", "old [s]", "new [s]", "ratio", "old [B]", "new [B]")
    println("-"^88)

    for N2 in SIZES
        U = rand(SymplecticStiefelManifold{T}, N2, N_SMALL)
        Δ₁ = rgrad(U, randn(T, N2, N_SMALL))
        Δ₂ = rgrad(U, randn(T, N2, N_SMALL))
        args = (U, Δ₁, Δ₂)

        # both warmed at this size before either is recorded
        checksum += old_metric(args...) + new_metric(args...)

        c = max(calls_per_sample(old_metric, args), calls_per_sample(new_metric, args))
        k = samples(N2)
        t_old = :old in variants ? median_time(old_metric, args, k, c) : NaN
        t_new = :new in variants ? median_time(new_metric, args, k, c) : NaN

        @printf("%6d %6d %7d %12.3e %12.3e %8.2f %12d %12d\n",
            N2, N_SMALL, c, t_old, t_new, t_old / t_new,
            :old in variants ? (@allocated old_metric(args...)) : 0,
            :new in variants ? (@allocated new_metric(args...)) : 0)
    end

    println()
    @printf("element type %s, %d samples up to 2N = 80 and %d above, medians per call\n",
        T, samples(80), samples(160))
    @printf("BLAS threads %d; ratio > 1 means the new grouping is faster; checksum %.6e\n",
        BLAS.get_num_threads(), checksum)
end

main(isempty(ARGS) ? (:old, :new) : (Symbol(only(ARGS)),))
