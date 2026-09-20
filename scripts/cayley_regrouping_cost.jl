# What the grouping of `cayley(::AbstractLieAlgHorMatrix)` costs, against the grouping it replaced
# and against `geodesic`, which is the same retraction family at the same sizes.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/cayley_regrouping_cost.jl
#
# Both groupings are measured in one process, each warmed at every size before anything is
# recorded. `old` or `new` as an argument suppresses the *timing loop* of the other one and prints
# a `NaN` in its column — it does not keep the other grouping out of the process, because the
# warm-up and `calls_per_sample` call both either way, and taking those out would remove the
# warming that makes one process valid in the first place:
#
#     julia --startup-file=no --project=. scripts/cayley_regrouping_cost.jl old
#
# This is the check behind the paragraph in the docstring of `cayley(::StiefelLieAlgHorMatrix)` and
# behind the `CHANGELOG.md` entry. Both quote figures from this script and from nothing else.
#
# ## What the two groupings are
#
# Both evaluate `Cayley(B̄) = (𝕀 - ½B̄)⁻¹(𝕀 + ½B̄)` through the `N × 2n` factors `B̂`, `B̄` of
# `lift_factors`, with `𝔠 = (𝕀_2n - ½B̄ᵀB̂)⁻¹` the only inverse either takes. They differ in where
# the parentheses fall:
#
#   * `old` — `(𝕀 + ½B̂𝔠B̄ᵀ)(𝕀 + ½B̄)`, the Woodbury form of the left factor times the right one.
#     Both factors are dense `N × N`, so the last multiplication alone is `O(N³)`.
#   * `new` — `𝕀 + B̂𝔠B̄ᵀ`. Writing `𝕀 + ½B̄ = (𝕀 - ½B̄) + B̄` cancels the right factor, and the
#     `2n × 2n` matrix that leaves, `𝕀_2n + ½𝔠B̄ᵀB̂`, is `𝔠` itself. Nothing larger than `N × 2n`
#     enters a product, and the result is still the `N × N` retraction.
#
# `new` is the shipped `cayley` itself, so the two cannot drift. `old` is what the package
# evaluated before, written out here rather than reached through `invoke`, because nothing shadows
# it any more — it was replaced. It wraps its result in the same manifold type the method does, so
# that the byte columns compare like with like.
#
# ## What the result cannot be
#
# The retraction of a lift is an `N × N` matrix, so neither grouping can cost less than `O(N²n)`,
# and `new` reaches that floor. The claim this script checks is therefore not that the cost stops
# depending on `N` — it is that the `O(N³)` term is gone, which shows up as `new` tracking
# `geodesic` across the `N` column where `old` pulled away from it.
#
# ## Four traps this script is written around
#
#   * **Compilation.** Timing two variants in one process measures the first cold and the second
#     warm. Both are called once at every size before anything is recorded, and the `old`/`new`
#     arguments exist for anyone who would rather not rely on that.
#   * **A single sample.** The time column is a median over many samples, and the point is the
#     shape of the column across `N` rather than any one entry.
#   * **BLAS threading, which is the trap that matters here.** `main` pins BLAS to one thread, and
#     the ratio is meaningless without that. The `O(N³)` product the old grouping makes is a square
#     `gemm`, which parallelizes almost perfectly; the `O(N²n)` products that replace it are skinny
#     and do not. Multithreaded BLAS therefore hides most of the difference, by an amount that
#     depends on what else the machine is doing. Measured on one host at `N = 200`, twelve threads:
#     3.3 on one occasion, 2.1 on another, and **756** with twelve other Julia processes running.
#     Single-threaded the same column reproduces to about 10% under all three conditions. One
#     thread measures the arithmetic, which is what the grouping changed.
#   * **Dead code.** No result escapes, so all of them are folded into a checksum that is printed.

using GeometricOptimizers: GrassmannLieAlgHorMatrix, SkewSymMatrix, StiefelLieAlgHorMatrix,
                           cayley, geodesic, lift_factors, manifold_type
using LinearAlgebra
using Printf
using Random

const SIZES = [50, 100, 200, 400]
const T = Float64
const N_SMALL = 3

"Samples at a given size: enough for a stable median, few enough to finish at N = 400."
samples(N) = N ≤ 200 ? 201 : 101

"The shortest a timing sample may be, in seconds. About five hundred ticks of a 42 ns clock."
const SAMPLE_FLOOR = 2e-5

"The `2n × 2n` identity and the two lift factors, which both groupings need and neither is timed on."
function pieces(B)
    𝕀_small = one(B isa StiefelLieAlgHorMatrix ? B.A : zeros(T, B.n, B.n))
    𝕆 = zero(𝕀_small)
    (hcat(vcat(𝕀_small, 𝕆), vcat(𝕆, 𝕀_small)), one(B))
end

"`(𝕀 + ½B̂𝔠B̄ᵀ)(𝕀 + ½B̄)` — the grouping this package evaluated before, wrapped as the method wraps it."
function old_cayley(B)
    𝕀_small2, 𝕀_big = pieces(B)
    B̂, B̄ = lift_factors(B)
    manifold_type(B)((𝕀_big + T(0.5) * B̂ * inv(𝕀_small2 - T(0.5) * B̄' * B̂) * B̄') *
                     (𝕀_big + T(0.5) * B))
end

"`𝕀 + B̂𝔠B̄ᵀ` — the shipped method itself, so the two cannot drift."
new_cayley(B) = cayley(B)

"Calls to fold into one timing sample, so that `SAMPLE_FLOOR` is reached. One pilot call sizes it."
function calls_per_sample(f, B)
    f(B)
    t = @elapsed f(B)
    max(1, ceil(Int, SAMPLE_FLOOR / max(t, 1e-9)))
end

"Median seconds per call over `k` samples of `c` calls each, after one warm-up call."
function median_time(f, B, k, c)
    f(B)
    times = Vector{Float64}(undef, k)
    for i in 1:k
        t = time_ns()
        for _ in 1:c
            f(B)
        end
        times[i] = (time_ns() - t) * 1e-9 / c
    end
    sort!(times)
    times[(k + 1) ÷ 2]
end

function lift(::Type{StiefelLieAlgHorMatrix}, N)
    StiefelLieAlgHorMatrix(SkewSymMatrix(rand(T, N_SMALL, N_SMALL)),
        rand(T, N - N_SMALL, N_SMALL), N, N_SMALL)
end
function lift(::Type{GrassmannLieAlgHorMatrix}, N)
    GrassmannLieAlgHorMatrix(rand(T, N - N_SMALL, N_SMALL), N, N_SMALL)
end

function main(variants)
    Random.seed!(20260919)
    BLAS.set_num_threads(1)                       # see the note on BLAS threading in the header
    checksum = zero(T)

    @printf("%-26s %5s %7s %12s %12s %8s %12s %12s %12s\n",
        "lift", "N", "calls", "old [s]", "new [s]", "ratio", "geodesic [s]", "old [B]",
        "new [B]")
    println("-"^126)

    for LT in (StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix), N in SIZES

        B = lift(LT, N)

        # every variant warmed at this size before any of them is recorded
        for f in (old_cayley, new_cayley, geodesic)
            checksum += sum(f(B))
        end

        c = maximum(calls_per_sample(f, B) for f in (old_cayley, new_cayley, geodesic))
        k = samples(N)
        t_old = :old in variants ? median_time(old_cayley, B, k, c) : NaN
        t_new = :new in variants ? median_time(new_cayley, B, k, c) : NaN
        t_geo = median_time(geodesic, B, k, c)

        @printf("%-26s %5d %7d %12.3e %12.3e %8.2f %12.3e %12d %12d\n",
            nameof(LT), N, c, t_old, t_new, t_old / t_new, t_geo,
            :old in variants ? (@allocated old_cayley(B)) : 0,
            :new in variants ? (@allocated new_cayley(B)) : 0)
    end

    println()
    @printf("element type %s, n = %d, %d samples up to N = 200 and %d above, medians per call\n",
        T, N_SMALL, samples(200), samples(400))
    @printf("BLAS threads %d; ratio > 1 means the new grouping is faster; checksum %.6e\n",
        BLAS.get_num_threads(), checksum)
end

main(isempty(ARGS) ? (:old, :new) : (Symbol(only(ARGS)),))
