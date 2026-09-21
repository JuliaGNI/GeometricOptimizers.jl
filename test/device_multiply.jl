# Products involving this package's own wrapper matrices, on a device backend.
#
# Three families reach the generic `AbstractMatrix` product unless a method stops them, and the
# generic product asks its argument for one entry at a time. That is scalar indexing, which no
# device array serves.
#
#   * the two `AbstractTriangular`s, which hold a packed storage vector and manufacture the rest of
#     the matrix in `getindex`;
#   * `StiefelProjection`, which holds an ordinary array and only has to be unwrapped;
#   * the two `AbstractLieAlgHorMatrix`es, which hold ordinary blocks and assemble the ambient
#     `N × N` matrix in `getindex`.
#
# The second is what stops a retraction: `geodesic(Y, Δ)` and `cayley(Y, Δ)` each take one product
# against the projection — `expB * E` and `cayleyB * E` — with `E` built from the horizontal lift
# and so carrying the point's own backend. Both operands are already there, and only the wrapper
# puts the product on the host path.
#
# `allowscalar(false)` is what makes this a test rather than a description: without it a scalar
# index on a `JLArray` merely warns. `JLArrays` stands in for the device, as it does in
# `similar_backend.jl` and `gradient_backend.jl`.
#
# The file holds two kinds of assertion, and they are not interchangeable. The seven "runs on the
# device" testsets each assert a product that the methods under test make reachable: strip those
# methods and every one of them raises `Scalar indexing is disallowed`. The one `@test_throws`
# testset asserts the opposite — it pins where the device path stops, and the limit it pins belongs
# to `JLArrays` rather than to the products here.

using GeometricOptimizers
using GeometricOptimizers: LowerTriangular, StiefelProjection, UpperTriangular, check,
                           global_rep
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using Random
using Test

Random.seed!(2718)

const T = Float32
const device = KernelAbstractions.get_backend(JLArray(zeros(T, 1)))
const N, n = 6, 3

allowscalar(false)

# A triangular whose storage is on the device, built by moving the packed vector rather than through
# the matrix constructor: `LowerTriangular(::AbstractMatrix)` runs a kernel per row and is a
# different thing to test.
device_triangular(MT, m) = MT(JLArray(rand(T, m * (m - 1) ÷ 2)), m)

@testset "a triangular times a matrix runs on the device" begin
    for MT in (LowerTriangular, UpperTriangular), m in (3, 6)

        A = device_triangular(MT, m)
        B = JLArray(rand(T, m, m))

        C = A * B
        @test C isa JLArray{T, 2}
        @test size(C) == (m, m)

        # the kernel reads the packed vector; the dense product reads the same matrix through
        # `getindex`. They have to agree, and the host copy is what says so.
        host = MT(Array(parent(A)), m)
        @test Array(C) ≈ Matrix{T}(host) * Array(B)
    end
end

@testset "a matrix times a triangular runs on the device" begin
    # `*(::AbstractMatrix, ::AbstractTriangular)` is `(A' * B')'`, and `adjoint` on one of these is a
    # type swap onto the same storage — so this reaches the *other* subtype's kernel and is not a
    # restatement of the testset above.
    #
    # The result is an `Adjoint` around a `JLArray` rather than a `JLArray`, because the outer
    # `adjoint` in `(A' * B')'` is lazy. It is still on the device, which is what this file is about,
    # and it is the shape `*(::AbstractMatrix, ::SkewSymMatrix)` returns as well —
    # `parent` is therefore what to assert on.
    for MT in (LowerTriangular, UpperTriangular), m in (3, 6)

        A = device_triangular(MT, m)
        B = JLArray(rand(T, m, m))

        C = B * A
        @test parent(C) isa JLArray{T, 2}

        host = MT(Array(parent(A)), m)
        @test Array(C) ≈ Array(B) * Matrix{T}(host)
    end
end

@testset "a triangular times a triangular runs on the device" begin
    # this materializes its right operand through `one`, which is `unit_matrix` and kernel-backed
    for MT₁ in (LowerTriangular, UpperTriangular), MT₂ in (LowerTriangular, UpperTriangular)

        A = device_triangular(MT₁, N)
        B = device_triangular(MT₂, N)

        C = A * B
        @test C isa JLArray{T, 2}
        @test Array(C) ≈
              Matrix{T}(MT₁(Array(parent(A)), N)) * Matrix{T}(MT₂(Array(parent(B)), N))
    end
end

@testset "a StiefelProjection times a matrix runs on the device" begin
    E = StiefelProjection(device, T, N, n)
    B = JLArray(rand(T, n, n))
    A = JLArray(rand(T, n, N))

    @test Array(E * B) ≈ Matrix{T}(StiefelProjection(N, n, T)) * Array(B)
    @test Array(A * E) ≈ Array(A) * Matrix{T}(StiefelProjection(N, n, T))
    @test E * B isa JLArray{T, 2}
    @test A * E isa JLArray{T, 2}

    b = JLArray(rand(T, n))
    @test Array(E * b) ≈ Matrix{T}(StiefelProjection(N, n, T)) * Array(b)
end

@testset "a geodesic retraction of a device-backed point runs end to end" begin
    # the payoff, and the assertion this file exists for. `geodesic` takes `expB * E` with
    # `E::StiefelProjection`, and that is the one product on its path that the generic
    # `AbstractMatrix` method cannot serve on a device.
    Y = rand(device, StiefelManifold, N, n)
    Δ = rgrad(Y, JLArray(rand(T, N, n)))

    Y₂ = geodesic(Y, Δ / 100)

    @test Y₂ isa StiefelManifold
    @test Y₂.A isa JLArray{T, 2}
    @test check(Y₂) < 1000 * eps(T)
end

@testset "cayley stops one step further on, and that is JLArrays' gap and not this package's" begin
    # `cayley` inverts a `2n × 2n` matrix with `LinearAlgebra.inv`. `JLArrays` supplies no `lu`, so
    # that falls through to the generic one, which scalar-indexes — and the retraction stops here
    # rather than on a product. `geodesic` above needs no inverse and completes on either backend.
    #
    # **`Metal` does supply it, and `cayley` completes there**: measured on an M4 Max under
    # `Metal.allowscalar(false)`, `check(cayley(Y, Δ/100)) = 2.3e-7`. So this is the reference
    # backend's limitation, not a statement about devices — which is exactly why it is pinned here
    # instead of described in a comment somewhere. Should `JLArrays` gain an `lu`, this assertion
    # fails and says so.
    Y = rand(device, StiefelManifold, N, n)
    Δ = rgrad(Y, JLArray(rand(T, N, n)))

    # matched on the message, not on `ErrorException`: the point of the assertion is *where* the
    # path stops, and every `error()` anywhere in `cayley` is an `ErrorException` too
    @test_throws "Scalar indexing is disallowed" cayley(Y, Δ / 100)
end

# A lift on a device, built the way the retractions build one: `global_rep` of a Riemannian gradient
# at a device-backed point. The blocks then carry the point's backend, which is what the products
# below run on.
device_lift(Y) = global_rep(GlobalSection(Y), rgrad(Y, JLArray(rand(T, N, n))))

# The host twin of a device lift, block by block. `Matrix(B)` would scalar-index `B`, so the dense
# form each assertion below compares against has to be built on the host first.
function host_lift(B::StiefelLieAlgHorMatrix)
    Matrix{T}(StiefelLieAlgHorMatrix(
        SkewSymMatrix(Array(B.A.S), B.A.n), Array(B.B), B.N, B.n))
end
function host_lift(B::GrassmannLieAlgHorMatrix)
    Matrix{T}(GrassmannLieAlgHorMatrix(
        Array(B.B), B.N, B.n))
end

@testset "a horizontal lift times a matrix runs on the device" begin
    # This is the third wrapper meeting the same gap the two above meet: an
    # `AbstractLieAlgHorMatrix` assembles its ambient `N × N` matrix in `getindex` too. It needs no
    # kernel, unlike the triangulars — it holds ordinary blocks, and the `A` block of a Stiefel lift
    # is a `SkewSymMatrix`, which carries a kernel-backed product of its own.
    for B in (device_lift(rand(device, StiefelManifold, N, n)),
        GrassmannLieAlgHorMatrix(JLArray(rand(T, N - n, n)), N, n))
        D = host_lift(B)
        C = JLArray(rand(T, N, n))

        @test B * C isa JLArray{T, 2}
        @test size(B * C) == (N, n)
        @test Array(B * C) ≈ D * Array(C)

        # The other order, which is `-transpose(B * transpose(C))` on `Bᵀ = -B`. It comes back a
        # `Transpose` around a `JLArray` and not a `JLArray`: `LinearAlgebra` pushes a unary minus
        # through the wrapper rather than materializing, so the outer `transpose` stays lazy. That
        # is on the device, which is what this file is about, and it is the shape
        # `*(::AbstractMatrix, ::SkewSymMatrix)` returns as well — `parent` is what to assert on,
        # exactly as in the triangular testset above.
        @test parent(C' * B) isa JLArray{T, 2}
        @test Array(C' * B) ≈ Array(C)' * D

        c = JLArray(rand(T, N))
        @test B * c isa JLArray{T, 1}
        @test Array(B * c) ≈ D * Array(c)
    end
end

@testset "a horizontal lift times a StiefelProjection runs on the device" begin
    # `B * E` unwraps the projection and hands the lift a bare array, so unwrapping alone moves the
    # scalar index one frame in rather than removing it. This is the assertion that says the gap is
    # shut and not moved.
    B = device_lift(rand(device, StiefelManifold, N, n))
    E = StiefelProjection(B)

    @test B * E isa JLArray{T, 2}
    @test Array(B * E) ≈ host_lift(B) * Matrix{T}(StiefelProjection(N, n, T))
end
