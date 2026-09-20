# What a computation does when its two operands are on different backends.
#
# Run with the repository as the active project, in an environment that also carries `JLArrays`:
#
#     julia --startup-file=no --project=. scripts/mixed_backend_seam.jl
#
# This is the check behind the *Fixed* entry in `CHANGELOG.md` for `_check_same_backend`, and it is
# written to run on either side of that change. On a tree without the guard it reports how many of
# the cases answer anyway; on a tree with it, how many are refused by name. Nothing here is timed, so
# no warm-up and no cold process are needed.
#
# `JLArrays` is the device stand-in, as it is in `test/device_multiply.jl`. It matters that it is a
# *host-backed* device: it can fall back to the host, which is exactly what lets an unguarded
# operation return an answer instead of failing. A real device fails the whole set instead — on Metal
# inside `GPU compilation of MethodInstance for …broadcast_linear…` — so the counts below are
# `JLArrays`' and the qualitative finding is the portable part.
#
# `allowscalar(false)` is what makes this a measurement rather than a description: without it a
# scalar index on a `JLArray` merely warns and more cases come back with an answer.

using GeometricOptimizers
using GeometricOptimizers: LowerTriangular, StiefelProjection, add!
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: transpose
using Printf
import Random

Random.seed!(1234)
allowscalar(false)

const T = Float32
const N, n = 6, 3

const HOST = KernelAbstractions.get_backend(zeros(T, 1))

host_skew = SkewSymMatrix(rand(T, N, N))
dev_skew = SkewSymMatrix(JLArray(rand(T, N, N)))
host_sym = SymmetricMatrix(rand(T, N, N))
dev_sym = SymmetricMatrix(JLArray(rand(T, N, N)))
host_lo = LowerTriangular(rand(T, N, N))
dev_lo = LowerTriangular(JLArray(rand(T, N, N)))
host_mat = rand(T, N, N)
dev_mat = JLArray(rand(T, N, N))
host_Y = rand(StiefelManifold{T}, N, n)
dev_Y = StiefelManifold(JLArray(Matrix(host_Y.A)))
host_E = StiefelProjection(T, N, n)
dev_E = StiefelProjection(KernelAbstractions.get_backend(dev_mat), T, N, n)
host_lift = rand(StiefelLieAlgHorMatrix{T}, N, n)
dev_lift = StiefelLieAlgHorMatrix(
    SkewSymMatrix(JLArray(Matrix(host_lift.A))), JLArray(Matrix(host_lift.B)), N, n)
host_U = rand(SymplecticStiefelManifold{T}, N, 4)
dev_U = SymplecticStiefelManifold(JLArray(Matrix(host_U.A)))
host_small = rand(T, 4, 4)
dev_small = JLArray(rand(T, 4, 4))

# Every pair below has one operand on the host and one on the device. The enumeration is the point:
# a count quoted without it cannot be re-measured, and the denominator depends entirely on which
# operations are listed.
const CASES = (
    ("SkewSym(host) + SkewSym(dev)", () -> host_skew + dev_skew),
    ("SkewSym(dev) + SkewSym(host)", () -> dev_skew + host_skew),
    ("SkewSym(host) - SkewSym(dev)", () -> host_skew - dev_skew),
    ("SkewSym(host) + Matrix(dev)", () -> host_skew + dev_mat),
    ("SkewSym(dev) + Matrix(host)", () -> dev_skew + host_mat),
    ("SkewSym(host) * Matrix(dev)", () -> host_skew * dev_mat),
    ("SkewSym(dev) * Matrix(host)", () -> dev_skew * host_mat),
    ("add!(host, host, dev)",
        () -> add!(SkewSymMatrix(rand(T, N, N)), host_skew, dev_skew)),
    ("Sym(host) + Sym(dev)", () -> host_sym + dev_sym),
    ("Sym(host) - Sym(dev)", () -> host_sym - dev_sym),
    ("Sym(host) * Matrix(dev)", () -> host_sym * dev_mat),
    ("Sym(dev) * Matrix(host)", () -> dev_sym * host_mat),
    ("Lower(host) + Lower(dev)", () -> host_lo + dev_lo),
    ("Lower(host) - Lower(dev)", () -> host_lo - dev_lo),
    ("Lower(host) * Matrix(dev)", () -> host_lo * dev_mat),
    ("Lift(host) + Lift(dev)", () -> host_lift + dev_lift),
    ("Lift(host) - Lift(dev)", () -> host_lift - dev_lift),
    ("Y(host) * Matrix(dev)", () -> host_Y * JLArray(rand(T, n, n))),
    ("Y(dev) * Matrix(host)", () -> dev_Y * rand(T, n, n)),
    ("rowvec(host) * Y(dev)", () -> rand(T, N)' * dev_Y),
    ("rowvec(dev) * Y(host)", () -> JLArray(rand(T, N))' * host_Y),
    ("Y(host)' * Matrix(dev)", () -> host_Y' * dev_mat),
    ("Y(dev)' * Matrix(host)", () -> dev_Y' * host_mat),
    ("Y(host)' * Y(dev)", () -> host_Y' * dev_Y),
    ("E(host) * Matrix(dev)", () -> host_E * JLArray(rand(T, n, n))),
    ("Matrix(dev) * E(host)", () -> dev_mat * host_E),
    ("rowvec(host) * E(dev)", () -> rand(T, N)' * dev_E),
    ("rowvec(dev) * E(host)", () -> JLArray(rand(T, N))' * host_E),
    # `SymplecticStiefelManifold` carries ten of the guard sites, more than any other file, so its
    # twenty pairs are the largest block here. Two of them — the adjoint against an adjoint — no
    # point satisfies dimensionally, and a `DimensionMismatch` is what they raised before.
    ("U(host) * Matrix(dev)", () -> host_U * dev_small),
    ("U(dev) * Matrix(host)", () -> dev_U * host_small),
    ("Matrix(host) * U(dev)", () -> host_mat * dev_U),
    ("Matrix(dev) * U(host)", () -> dev_mat * host_U),
    ("rowvec(host) * U(dev)", () -> rand(T, N)' * dev_U),
    ("rowvec(dev) * U(host)", () -> JLArray(rand(T, N))' * host_U),
    ("transpose(host) * U(dev)", () -> transpose(rand(T, N)) * dev_U),
    ("U(host)' * Matrix(dev)", () -> host_U' * dev_mat),
    ("U(dev)' * Matrix(host)", () -> dev_U' * host_mat),
    ("Matrix(host) * U(dev)'", () -> host_small * dev_U'),
    ("Matrix(dev) * U(host)'", () -> dev_small * host_U'),
    ("rowvec(host) * U(dev)'", () -> rand(T, 4)' * dev_U'),
    ("rowvec(dev) * U(host)'", () -> JLArray(rand(T, 4))' * host_U'),
    ("transpose(host) * U(dev)'", () -> transpose(rand(T, 4)) * dev_U'),
    ("U(host)' * U(dev)", () -> host_U' * dev_U),
    ("U(dev)' * U(host)", () -> dev_U' * host_U),
    ("U(host)' * U(dev)'", () -> host_U' * dev_U'),
    ("U(dev)' * U(host)'", () -> dev_U' * host_U'),
    ("U(host) * U(dev)", () -> host_U * dev_U),
    ("U(dev) * U(host)", () -> dev_U * host_U))

function report(cases)
    named = 0
    other_error = 0
    answered = 0

    @printf("%-30s  %-9s  %s\n", "operation", "outcome", "detail")
    for (label, f) in cases
        try
            result = f()
            answered += 1
            backend = try
                string(KernelAbstractions.get_backend(result))
            catch
                "unplaceable"
            end
            @printf("%-30s  %-9s  on %s\n", label, "ANSWERED", backend)
        catch err
            message = first(split(sprint(showerror, err), '\n'))
            if err isa ArgumentError && occursin("mixed backends", message)
                named += 1
                @printf("%-30s  %-9s\n", label, "REFUSED")
            else
                other_error += 1
                @printf("%-30s  %-9s  %s\n", label, "ERROR", message[1:min(48, end)])
            end
        end
    end

    @printf("\n%d cases: %d refused by name, %d answered anyway, %d other error\n",
        length(cases), named, answered, other_error)
    @printf("host backend %s\n", HOST)
end

report(CASES)

# An `ANSWERED` row is the finding this script exists for. It means the operation returned a value
# computed on a backend the caller did not choose — and which backend that is follows the argument
# order, not the types. An `ERROR` row is the next-best outcome: the operation failed, but with a
# message that names neither operand.
