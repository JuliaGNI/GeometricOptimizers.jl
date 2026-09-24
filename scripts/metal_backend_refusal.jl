# The backend guard against a **real** device, not the `JLArrays` stand-in.
#
# Run with the repository as the active project, in an environment that also carries `Metal`, on a
# machine with an Apple GPU:
#
#     julia --startup-file=no --project=. scripts/metal_backend_refusal.jl
#
# Every other mixed-backend figure in this repository — `scripts/mixed_backend_seam.jl`,
# `scripts/mixed_backend_plain_matrix.jl`, `test/mixed_backend_refusal.jl` — is measured with
# `JLArrays`, which is a *host-backed* device. That is what makes it useful there: it can fall back
# to the host, so an unguarded operation returns an answer and the defect is visible as a wrong
# backend rather than as a crash. It also means none of those figures says what a real device does.
# This script is that check.
#
# ## What it reports, and why the two columns differ
#
# On `JLArrays` an unguarded pair **answers on the wrong backend**. On Metal the same pair
# **fails to compile**, inside
#
#     GPU compilation of MethodInstance for (::Metal.var"#broadcast_2d#_copyto!##2")(…) failed
#     KernelError: passing non-bitstype argument
#
# because the host-side structured operand is carried into the broadcast as a non-`isbits`
# `Base.Broadcast.Extruded{SymmetricMatrix{Float32, Vector{Float32}}, …}`. That message names
# neither backend, neither operand type in a form a caller reads, and not the mismatch — which is
# the whole reason `_check_same_backend` exists. So a refusal is an improvement on Metal too, but
# for a different reason: not "an answer from the wrong place" but "an unreadable failure".
#
# ## Float32 everywhere
#
# Metal has no `Float64`. `MtlArray(rand(Float64, 2, 2))` raises, so every fixture here is `Float32`
# and the element type is a constant rather than a parameter. Metal also has no `qr`, which is why
# the device `StiefelManifold` is built by moving a host point across rather than by calling
# `rand(StiefelManifold{Float32}, …)` on the device.
#
# ## It skips itself where there is no device
#
# `Metal.functional()` is `false` on Linux, on Windows, on a Mac without Apple silicon and on a
# macOS runner older than 15, and `Metal.device()` can return a null device inside a sandbox even
# where the hardware is present. The script says so and exits 0 rather than failing, so it is safe
# to run anywhere.

using GeometricOptimizers
using GeometricOptimizers: StrictlyLowerTriangular, StiefelProjection, add!
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: mul!
using Metal
using Printf
import Random

if !Metal.functional()
    println("Metal is not functional here; nothing measured.")
    exit(0)
end

Random.seed!(1234)

const T = Float32
const N, n = 6, 3

println("device  ", Metal.device())
println("backend ", KernelAbstractions.get_backend(MtlArray(zeros(T, 1))))
println()

host_skew = SkewSymMatrix(rand(T, N, N))
host_sym = SymmetricMatrix(rand(T, N, N))
host_lo = StrictlyLowerTriangular(rand(T, N, N))
host_mat = rand(T, N, N)
host_tall = rand(T, N, n)
host_lift = rand(StiefelLieAlgHorMatrix{T}, N, n)
host_E = StiefelProjection(T, N, n)
host_Y = rand(StiefelManifold{T}, N, n)

device_skew = SkewSymMatrix(MtlArray(rand(T, N, N)))
device_sym = SymmetricMatrix(MtlArray(rand(T, N, N)))
device_lo = StrictlyLowerTriangular(MtlArray(rand(T, N, N)))
device_mat = MtlArray(rand(T, N, N))
device_tall = MtlArray(rand(T, N, n))
device_lift = StiefelLieAlgHorMatrix(
    SkewSymMatrix(MtlArray(Matrix(host_lift.A))), MtlArray(Matrix(host_lift.B)), N, n)
device_E = StiefelProjection(MetalBackend(), T, N, n)
device_Y = StiefelManifold(MtlArray(Matrix(host_Y.A)))

# One host operand and one Metal operand in every case, across the five guarded operations and the
# owned types that carry them. Both argument orders appear wherever the package has a method for
# both, because an unguarded pair picks its side by argument order.
const CASES = (
    ("SkewSym(host) + SkewSym(mtl)", () -> host_skew + device_skew),
    ("SkewSym(mtl) + SkewSym(host)", () -> device_skew + host_skew),
    ("SkewSym(host) - SkewSym(mtl)", () -> host_skew - device_skew),
    ("SkewSym(host) + Matrix(mtl)", () -> host_skew + device_mat),
    ("SkewSym(host) - Matrix(mtl)", () -> host_skew - device_mat),
    ("Matrix(mtl) - SkewSym(host)", () -> device_mat - host_skew),
    ("SkewSym(host) * Matrix(mtl)", () -> host_skew * device_mat),
    ("SkewSym(mtl) * Matrix(host)", () -> device_skew * host_mat),
    ("add!(host, host, mtl)",
        () -> add!(SkewSymMatrix(rand(T, N, N)), host_skew, device_skew)),
    ("Sym(host) + Sym(mtl)", () -> host_sym + device_sym),
    ("Sym(host) - Sym(mtl)", () -> host_sym - device_sym),
    ("Sym(host) + Matrix(mtl)", () -> host_sym + device_mat),
    ("Sym(host) - Matrix(mtl)", () -> host_sym - device_mat),
    ("Sym(host) * Matrix(mtl)", () -> host_sym * device_mat),
    ("Lower(host) + Lower(mtl)", () -> host_lo + device_lo),
    ("Lower(host) - Lower(mtl)", () -> host_lo - device_lo),
    ("Lower(host) + Matrix(mtl)", () -> host_lo + device_mat),
    ("Lower(host) - Matrix(mtl)", () -> host_lo - device_mat),
    ("Lower(host) * Matrix(mtl)", () -> host_lo * device_mat),
    ("mul!(Lower(mtl), Lower(host), 2)", () -> mul!(device_lo, host_lo, T(2))),
    ("mul!(Lower(host), Lower(mtl), 2)", () -> mul!(host_lo, device_lo, T(2))),
    ("Lift(host) + Lift(mtl)", () -> host_lift + device_lift),
    ("Lift(host) - Lift(mtl)", () -> host_lift - device_lift),
    ("Lift(host) + Matrix(mtl)", () -> host_lift + device_mat),
    ("Lift(host) - Matrix(mtl)", () -> host_lift - device_mat),
    ("E(host) - Matrix(mtl)", () -> host_E - device_tall),
    ("Matrix(mtl) * E(host)", () -> device_mat * host_E),
    ("Y(host) * Matrix(mtl)", () -> host_Y * MtlArray(rand(T, n, n))),
    ("Y(mtl) * Matrix(host)", () -> device_Y * rand(T, n, n)),
    ("rowvec(host) * Y(mtl)", () -> rand(T, N)' * device_Y),
    ("Y(host)' * Matrix(mtl)", () -> host_Y' * device_mat),
    ("Y(host) - Matrix(mtl)", () -> host_Y - device_tall))

function report(cases)
    refused = 0
    answered = 0
    other_error = 0

    @printf("%-34s  %-9s  %s\n", "operation", "outcome", "detail")
    for (label, f) in cases
        try
            result = f()
            answered += 1
            backend = try
                string(KernelAbstractions.get_backend(result))
            catch
                "unplaceable"
            end
            @printf("%-34s  %-9s  on %s\n", label, "ANSWERED", backend)
        catch err
            message = first(split(sprint(showerror, err), '\n'))
            if err isa ArgumentError && occursin("mixed backends", message)
                refused += 1
                @printf("%-34s  %-9s\n", label, "REFUSED")
            else
                other_error += 1
                @printf("%-34s  %-9s  %s\n", label, "ERROR", message[1:min(52, end)])
            end
        end
    end

    @printf("\n%d cases: %d refused by name, %d answered anyway, %d other error\n",
        length(cases), refused, answered, other_error)
end

report(CASES)

# What this reported when it was written, on an `AGXG16CDevice`: **30 refused by name, 0 answered
# anyway, 2 other error**.
#
# Nothing answered. That is the part that differs from `JLArrays`, where the same sweep leaves a
# dozen operations returning a value from a backend the caller did not choose. A real device cannot
# fall back to the host, so the unguarded cases crash instead.
#
# The two errors are `Sym(host) + Matrix(mtl)` and `Lower(host) + Matrix(mtl)`, and they are open
# issue A25 in `CHANGELOG.md`: `SymmetricMatrix` and `AbstractTriangular` own no
# `+(X, ::AbstractMatrix)` method, so the pair reaches `Base`'s generic `+` and, through it, Metal's
# broadcast kernel, which cannot take a host-side structured operand as a non-`isbits` argument.
# Their `-` counterparts are refused by name, which is the `-` half of that gap closed.
