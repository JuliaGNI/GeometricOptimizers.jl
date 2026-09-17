# `copyto!` -- and `assign!`, which shares its contract here -- writes the storage of the package's
# structured matrices and of `Manifold` through `copyto!` and not through a broadcast (`.=`).
#
# A device array with no `BroadcastStyle` rule combining it with a host `Array` accepts a direct
# `copyto!` from one and refuses that broadcast. So a write spelled as a broadcast makes a
# host-to-device transfer of a structured type fail while the same transfer of the bare storage
# array succeeds -- the backwards outcome for a package whose point is those types. A real device
# shows it on `SkewSymMatrix`'s storage and on `StiefelLieAlgHorMatrix` (`Close the
# GeometricOptimizers audit findings.md`, section 20).
#
# `JLArrays` cannot stand in for the device here: a `JLArray` accepts `.=` from a host `Array`.
# `_NoBroadcastVector` and `_NoBroadcastMatrix` below carry the restriction on the CPU instead,
# through the same mechanism a real device array uses to enforce it: they define
# `copyto!(dest, src::AbstractArray)` but not `copyto!(dest, ::Broadcasted)`, so a direct `copyto!`
# works and `.=` throws. What this file pins is that every site below spells the write as the
# former.

using GeometricOptimizers
using GeometricOptimizers: assign!
using Random
using Test

Random.seed!(2026)

struct _NoBroadcastVector{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(A::_NoBroadcastVector) = size(A.data)
Base.getindex(A::_NoBroadcastVector, i::Int) = A.data[i]
Base.setindex!(A::_NoBroadcastVector, v, i::Int) = (A.data[i] = v)
Base.copyto!(dest::_NoBroadcastVector, src::AbstractArray) = (copyto!(dest.data, src); dest)
function Base.copyto!(dest::_NoBroadcastVector, ::Base.Broadcast.Broadcasted)
    error("broadcast assignment is disallowed on _NoBroadcastVector")
end

struct _NoBroadcastMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
end
Base.size(A::_NoBroadcastMatrix) = size(A.data)
Base.getindex(A::_NoBroadcastMatrix, i::Int, j::Int) = A.data[i, j]
Base.setindex!(A::_NoBroadcastMatrix, v, i::Int, j::Int) = (A.data[i, j] = v)
Base.copyto!(dest::_NoBroadcastMatrix, src::AbstractArray) = (copyto!(dest.data, src); dest)
function Base.copyto!(dest::_NoBroadcastMatrix, ::Base.Broadcast.Broadcasted)
    error("broadcast assignment is disallowed on _NoBroadcastMatrix")
end

const T = Float64
const N, n = 6, 3

@testset "copyto! moves a structured matrix onto a foreign-storage destination" begin
    @testset "SkewSymMatrix" begin
        host = SkewSymMatrix(rand(T, n, n))
        dev = SkewSymMatrix(_NoBroadcastVector(zeros(T, n * (n - 1) ÷ 2)), n)
        copyto!(dev, host)
        @test dev.S.data ≈ host.S
    end

    @testset "SymmetricMatrix" begin
        host = SymmetricMatrix(rand(T, n, n))
        dev = SymmetricMatrix(_NoBroadcastVector(zeros(T, n * (n + 1) ÷ 2)), n)
        copyto!(dev, host)
        @test dev.S.data ≈ host.S
    end

    @testset "LowerTriangular" begin
        host = LowerTriangular(rand(T, n, n))
        dev = LowerTriangular(_NoBroadcastVector(zeros(T, n * (n - 1) ÷ 2)), n)
        copyto!(dev, host)
        @test dev.S.data ≈ host.S
    end

    @testset "UpperTriangular" begin
        host = UpperTriangular(rand(T, n, n))
        dev = UpperTriangular(_NoBroadcastVector(zeros(T, n * (n - 1) ÷ 2)), n)
        copyto!(dev, host)
        @test dev.S.data ≈ host.S
    end

    @testset "StiefelManifold" begin
        host = rand(StiefelManifold{T}, N, n)
        dev = StiefelManifold(_NoBroadcastMatrix(zeros(T, N, n)))
        copyto!(dev, host)
        @test dev.A.data ≈ host.A
    end

    @testset "GrassmannManifold" begin
        host = rand(GrassmannManifold{T}, N, n)
        dev = GrassmannManifold(_NoBroadcastMatrix(zeros(T, N, n)))
        copyto!(dev, host)
        @test dev.A.data ≈ host.A
    end

    @testset "StiefelLieAlgHorMatrix, transitively through its SkewSymMatrix block" begin
        # the lift needs no write of its own: it forwards to `copyto!` on its two components, so it
        # transfers exactly when its `SkewSymMatrix` block and its bare `B` block both do
        host = StiefelLieAlgHorMatrix(SkewSymMatrix(rand(T, n, n)), rand(T, N - n, n), N, n)
        dev = StiefelLieAlgHorMatrix(
            SkewSymMatrix(_NoBroadcastVector(zeros(T, n * (n - 1) ÷ 2)), n),
            _NoBroadcastMatrix(zeros(T, N - n, n)), N, n)
        copyto!(dev, host)
        @test dev.A.S.data ≈ host.A.S
        @test dev.B.data ≈ host.B
    end
end

@testset "assign! is the same contract and moves with copyto!" begin
    for (dev, host) in (
        (SkewSymMatrix(_NoBroadcastVector(zeros(T, n * (n - 1) ÷ 2)), n),
        SkewSymMatrix(rand(T, n, n))),
        (SymmetricMatrix(_NoBroadcastVector(zeros(T, n * (n + 1) ÷ 2)), n),
        SymmetricMatrix(rand(T, n, n))),
        (LowerTriangular(_NoBroadcastVector(zeros(T, n * (n - 1) ÷ 2)), n),
        LowerTriangular(rand(T, n, n))),
        (UpperTriangular(_NoBroadcastVector(zeros(T, n * (n - 1) ÷ 2)), n),
        UpperTriangular(rand(T, n, n))))
        assign!(dev, host)
        @test dev.S.data ≈ host.S
    end

    dev_lift = StiefelLieAlgHorMatrix(
        SkewSymMatrix(_NoBroadcastVector(zeros(T, n * (n - 1) ÷ 2)), n),
        _NoBroadcastMatrix(zeros(T, N - n, n)), N, n)
    host_lift = StiefelLieAlgHorMatrix(SkewSymMatrix(rand(T, n, n)), rand(T, N - n, n), N, n)
    assign!(dev_lift, host_lift)
    @test dev_lift.A.S.data ≈ host_lift.A.S
    @test dev_lift.B.data ≈ host_lift.B
end

# The "must keep throwing" half of §20 -- `hostA + devA` and `add!(devA, devA, hostA)` are
# computations on mismatched memory, not transfers, and must not be widened into working. The
# stand-ins here carry only half of what a real device needs for that. They have no
# `copyto!(dest, ::Broadcasted)`, so `add!` throws on them; they have no `BroadcastStyle` of their
# own, so `+` allocates its result through `similar`, lands on a plain `Array` and returns quietly.
# The property therefore belongs to the Metal verification, which is where it is pinned.
