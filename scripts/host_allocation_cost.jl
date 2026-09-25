# What `KernelAbstractions.zeros(CPU(), T, m)` costs against the host spelling `zeros(T, m)`, in bytes
# and in time.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/host_allocation_cost.jl
#
# This is the check behind the comment on `_zeros` in
# `src/allocators.jl` and behind the matching CHANGELOG entry. Both quote figures from
# it. An earlier round of those figures was quoted from a measurement nobody had archived, and two
# independent re-runs then disagreed about the byte number — which is the whole reason this file exists.
#
# ## What is compared, and why it is these two expressions
#
# `KernelAbstractions.zeros(backend, T, dims)` is `allocate` followed by `Base.fill!`
# (`KernelAbstractions.jl`), and `allocate(::CPU, T, dims)` is `Array{T}(undef, dims)`. So on a `CPU` the
# device spelling returns the same `Vector{T}` with the same values as `zeros(T, m)` and charges for it:
# it writes the zeros itself instead of taking a page the operating system has already zeroed.
#
# The triangular allocators hold `n*(n-1)÷2` elements for an `n × n` matrix, so the `m` column below is
# the array that is really allocated and the `n` column is the matrix it belongs to.
#
# ## Three traps this script is written around
#
#   * **Compilation.** Timing two variants in one process measures the first one cold and the second one
#     warm. Both are therefore run once at every size before anything is recorded.
#   * **A single `@allocated` call.** One call can report a figure the neighbouring sizes contradict. The
#     byte column is the minimum of several calls, and the point is the *shape* of the column, not any
#     one row.
#   * **Dead code.** Neither result escapes, so both are consumed into a checksum that is printed.

using KernelAbstractions
using Printf

const BACKEND = CPU()

# `m` is the length of the flat buffer an `n × n` triangular matrix holds.
const LENGTHS = [
    1, 3, 6, 10, 28, 45, 100, 190, 300, 500, 1024, 2048, 4096, 16384, 65536, 262144]

const ALLOC_REPS = 5
const TIME_BATCHES = 25

host(::Type{T}, m::Int) where {T} = zeros(T, m)
device(::Type{T}, m::Int) where {T} = KernelAbstractions.zeros(BACKEND, T, m)

"Minimum of `ALLOC_REPS` `@allocated` calls, so one outlying figure cannot set the row."
function allocated_min(f, ::Type{T}, m::Int) where {T}
    minimum(1:ALLOC_REPS) do _
        @allocated f(T, m)
    end
end

"""
Seconds per call, as the minimum over `TIME_BATCHES` batches of `k` calls each.

One call at a small `m` is shorter than the clock's resolution, so a per-call `@elapsed` reads either
zero or one tick and a ratio built from it is meaningless. Timing a *batch* and dividing amortises the
resolution away. The minimum over batches is taken because noise from the garbage collector and the
scheduler only ever adds time.
"""
function timed(f, ::Type{T}, m::Int) where {T}
    k = clamp(2^22 ÷ (m + 64), 20, 20000)
    best = Inf
    acc = zero(T)
    for _ in 1:TIME_BATCHES
        t = @elapsed for _ in 1:k
            x = f(T, m)
            acc += @inbounds x[1]
        end
        best = min(best, t / k)
    end
    (best, k, acc)
end

function main(::Type{T} = Float64) where {T}
    checksum = zero(T)

    # Warm up both paths at every size before recording anything.
    for m in LENGTHS
        checksum += @inbounds host(T, m)[1]
        checksum += @inbounds device(T, m)[1]
    end

    @printf("%8s %8s %12s %12s %12s %12s %12s %8s %8s\n",
        "n", "m", "host B", "KA B", "delta B", "host s", "KA s", "ratio", "calls")
    for m in LENGTHS
        n = round(Int, (1 + sqrt(1 + 8m)) / 2)

        bh = allocated_min(host, T, m)
        bd = allocated_min(device, T, m)

        (hs, k, a1) = timed(host, T, m)
        (ds, _, a2) = timed(device, T, m)
        checksum += a1 + a2

        @printf("%8d %8d %12d %12d %12d %12.3e %12.3e %8.2f %8d\n",
            n, m, bh, bd, bd - bh, hs, ds, ds / hs, k)
    end

    println()
    println("checksum (must be 0.0): ", checksum)
    println("Julia ",
        VERSION,
        ", check-bounds=",
        Base.JLOptions().check_bounds == 0 ? "auto" :
        Base.JLOptions().check_bounds == 1 ? "yes" : "no")
    println("KernelAbstractions ", pkgversion(KernelAbstractions))
    return nothing
end

main()
