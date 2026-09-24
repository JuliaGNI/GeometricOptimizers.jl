# What a sum or a difference does when an owned matrix meets a *plain* `AbstractMatrix` on another
# backend.
#
# Run with the repository as the active project, in an environment that also carries `JLArrays`:
#
#     julia --startup-file=no --project=. scripts/mixed_backend_plain_matrix.jl
#
# This is the second enumeration, for the case `scripts/mixed_backend_seam.jl` does not reach. That
# one pairs owned types with each other and with a plain array only where the package already had a
# method; this one asks the question the other way round — for every owned type, what happens when
# the operand on the far side is an ordinary `Matrix` or `JLArray` on a different backend. It is the
# check behind the `-` entry in `CHANGELOG.md` and behind open issue A25, and it runs on either side
# of that change. Nothing here is timed, so no warm-up and no cold process are needed.
#
# ## Why a plain matrix is a separate question
#
# `_check_same_backend` guards a pair only where this package owns the method. Where it does not, the
# call reaches `Base`'s generic `+`/`-` at `arraymath.jl:6`, which broadcasts and takes its backend
# from the argument order. So the coverage depends on which methods happen to exist, not on which
# types are "owned" — and that is exactly what a count over owned *types* hides.
#
# ## The enumeration
#
# Ten owned types × two operators × two argument orders × which side carries the device operand.
# That is 80 cases and the sweep is closed rather than hand-picked, which is what lets the count be
# re-measured. Ten is the membership of `OwnedMatrix` in `src/ambiguities.jl` counted out — the two
# triangulars, the two horizontal lifts, the three manifolds, `SkewSymMatrix`, `SymmetricMatrix` and
# `StiefelProjection` — so a type added to that union and not to `TYPES` below shows up as a sweep
# that no longer covers the guard it measures.
# `JLArrays` is the device stand-in, as it is in `test/mixed_backend_refusal.jl`, and it
# matters that it is a *host-backed* device: it can fall back to the host, which is what lets an
# unguarded operation return an answer instead of failing. `allowscalar(false)` is what makes this a
# measurement rather than a description.
#
# `scripts/metal_backend_refusal.jl` asks the same question of a real device.

using GeometricOptimizers
using GeometricOptimizers: StrictlyLowerTriangular, StrictlyUpperTriangular,
                           StiefelProjection
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using Printf
import Random

Random.seed!(1234)
allowscalar(false)

const T = Float32
const N, n = 6, 3

host_mat = rand(T, N, N)
dev_mat = JLArray(rand(T, N, N))
host_tall = rand(T, N, n)
dev_tall = JLArray(rand(T, N, n))
host_wide = rand(T, N, 4)
dev_wide = JLArray(rand(T, N, 4))

host_U = rand(SymplecticStiefelManifold{T}, N, 4)
host_Y = rand(StiefelManifold{T}, N, n)
host_G = rand(GrassmannManifold{T}, N, n)
host_lift = rand(StiefelLieAlgHorMatrix{T}, N, n)
host_grass = rand(GrassmannLieAlgHorMatrix{T}, N, n)

# Each row is one owned type: its host instance, its device instance, and the plain matrix of the
# shape that type adds to. The shape differs — `StiefelProjection` and `StiefelManifold` are `N × n`
# and `SymplecticStiefelManifold` is `N × 4` — so the plain operand travels with the type rather than
# being shared, and a `DimensionMismatch` cannot be mistaken for a refusal.
const TYPES = (
    ("SkewSym", SkewSymMatrix(rand(T, N, N)),
        SkewSymMatrix(JLArray(rand(T, N, N))), host_mat, dev_mat),
    ("Sym", SymmetricMatrix(rand(T, N, N)),
        SymmetricMatrix(JLArray(rand(T, N, N))), host_mat, dev_mat),
    ("Lower", StrictlyLowerTriangular(rand(T, N, N)),
        StrictlyLowerTriangular(JLArray(rand(T, N, N))), host_mat, dev_mat),
    ("Upper", StrictlyUpperTriangular(rand(T, N, N)),
        StrictlyUpperTriangular(JLArray(rand(T, N, N))), host_mat, dev_mat),
    ("Lift", host_lift,
        StiefelLieAlgHorMatrix(SkewSymMatrix(JLArray(Matrix(host_lift.A))),
            JLArray(Matrix(host_lift.B)), N, n), host_mat, dev_mat),
    ("Grassmann", host_grass,
        GrassmannLieAlgHorMatrix(JLArray(Matrix(host_grass.B)), N, n), host_mat, dev_mat),
    ("E", StiefelProjection(T, N, n),
        StiefelProjection(KernelAbstractions.get_backend(dev_mat), T, N, n), host_tall, dev_tall),
    ("Y", host_Y, StiefelManifold(JLArray(Matrix(host_Y.A))), host_tall, dev_tall),
    ("G", host_G, GrassmannManifold(JLArray(Matrix(host_G.A))), host_tall, dev_tall),
    ("U", host_U, SymplecticStiefelManifold(JLArray(Matrix(host_U.A))),
        host_wide, dev_wide))

function outcome(f)
    try
        result = f()
        backend = try
            string(KernelAbstractions.get_backend(result))
        catch
            "unplaceable"
        end
        ("ANSWERED", "on " * backend)
    catch err
        message = first(split(sprint(showerror, err), '\n'))
        if err isa ArgumentError && occursin("mixed backends", message)
            ("REFUSED", "")
        else
            ("ERROR", message[1:min(48, end)])
        end
    end
end

function report()
    refused = 0
    answered = 0
    other_error = 0

    @printf("%-32s  %-9s  %s\n", "operation", "outcome", "detail")
    for (name, host, device, host_plain, device_plain) in TYPES, op in (+, -)

        symbol = string(Symbol(op))
        cases = (("$name(host) $symbol Matrix(dev)", () -> op(host, device_plain)),
            ("Matrix(dev) $symbol $name(host)", () -> op(device_plain, host)),
            ("$name(dev) $symbol Matrix(host)", () -> op(device, host_plain)),
            ("Matrix(host) $symbol $name(dev)", () -> op(host_plain, device)))

        for (label, f) in cases
            (verdict, detail) = outcome(f)
            verdict == "REFUSED" && (refused += 1)
            verdict == "ANSWERED" && (answered += 1)
            verdict == "ERROR" && (other_error += 1)
            @printf("%-32s  %-9s  %s\n", label, verdict, detail)
        end
    end

    total = refused + answered + other_error
    @printf("\n%d cases: %d refused by name, %d answered anyway, %d other error\n",
        total, refused, answered, other_error)
    @printf("host backend %s\n", KernelAbstractions.get_backend(host_mat))
end

report()

# An `ANSWERED` row is the finding. It means the operation returned a value computed on a backend the
# caller did not choose, and which backend that is follows the argument order rather than the types.
# An `ERROR` row here is `Scalar indexing is disallowed`, which names neither operand: it is what the
# same pair gives when the *structured* operand is the one on the device, because the fallback
# broadcast reaches `getindex` on a type whose entries are packed.
#
# Both outcomes are the same defect seen from the two argument orders, and neither is a regression
# from anything: on a tree without the `-` guard this script reports 12 refused, 34 answered and 34
# scalar-indexing errors, and with it 52, 14 and 14. **All 40 `-` cases are refused; the 28 that
# remain are `+`** — 14 answering and 14 erroring — on the seven types that own no
# `+(X, ::AbstractMatrix)` method. That is open issue A25.
