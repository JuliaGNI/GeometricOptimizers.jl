# Products involving this package's own wrapper matrices, on a device backend.
#
# Two families reach the generic `AbstractMatrix` product unless a method stops them, and the
# generic product asks its argument for one entry at a time. That is scalar indexing, which no
# device array serves.
#
#   * the two `AbstractTriangular`s, which hold a packed storage vector and manufacture the rest of
#     the matrix in `getindex`;
#   * `StiefelProjection`, which holds an ordinary array and only has to be unwrapped.
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
# The file holds two kinds of assertion, and they are not interchangeable. The five "runs on the
# device" testsets each assert a product that the methods under test make reachable: strip those
# methods and every one of them raises `Scalar indexing is disallowed`. The two `@test_throws`
# testsets assert the opposite — they pin where the device path stops, and each limit they pin
# belongs to `JLArrays` or to `AbstractLieAlgHorMatrix` rather than to the products here.

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

    @test_throws ErrorException cayley(Y, Δ / 100)
end

@testset "a lift times a StiefelProjection is still host-only, for a reason of its own" begin
    # `B * E` with `B` a horizontal lift is a *third* wrapper meeting the same class of gap: an
    # `AbstractLieAlgHorMatrix` has `getindex` and no kernel-backed `*` either, so unwrapping the
    # projection only moves the scalar index one frame in. The gap is left open rather than widened
    # into, because nothing under `src/` takes this product: the retractions form `expB * E` with a
    # dense `expB`, which is the testset above.
    Y = rand(device, StiefelManifold, N, n)
    B = global_rep(GlobalSection(Y), rgrad(Y, JLArray(rand(T, N, n))))
    E = StiefelProjection(B)

    @test_throws ErrorException B * E
end
