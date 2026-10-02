# `==` of two manifold points, and of two structured matrices of one family, compares the storage.
#
# The answer is `Base`'s for two arrays: equal entries. The generic method reads both one entry at a
# time, which a device refuses, and the optimizer compares iterates with `==` on every step
# (`latest_gradient_is_current`). A pair of two families keeps the generic method.

using GeometricOptimizers
using GeometricOptimizers: StrictlyLowerTriangular, StrictlyUpperTriangular
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: qr
using NeuralNetworkParameters: mapstorage
using Random
using Test

allowscalar(false)

const n = 4

function pairs_of(::Type{T}) where {T}
    rng = Random.Xoshiro(21)
    point = Matrix{T}(qr(randn(rng, T, 2n, n)).Q)[:, 1:n]
    (rand(rng, SymmetricMatrix{T}, n), rand(rng, SkewSymMatrix{T}, n),
        rand(rng, StrictlyLowerTriangular{T}, n), rand(rng, StrictlyUpperTriangular{T}, n),
        StiefelManifold(point), GrassmannManifold(copy(point)))
end

@testset "== compares the entries through the storage: $(nameof(typeof(A))), $T" for T in (Float32, Float64),
    A in pairs_of(T)

    B = mapstorage(copy, A)
    C = mapstorage(s -> s .+ one(T), A)
    @test A == B
    @test (A == B) == (Matrix(A) == Matrix(B))
    @test (A == C) == (Matrix(A) == Matrix(C))
    @test A != C

    # on a device, where the generic method raises
    dA, dB, dC = mapstorage(JLArray, A), mapstorage(JLArray, B), mapstorage(JLArray, C)
    @test dA == dB
    @test dA != dC
end

# `==` runs on every step, and so does `contains_nonfinite`, which walks the storage for the same
# reason: both infer and allocate nothing on the host
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
allocations(f::F, a::A, b::B) where {F, A, B} = (f(a, b); @allocated f(a, b))

@testset "== and contains_nonfinite infer and allocate nothing: $(nameof(typeof(A))), $T" for T in (
        Float32, Float64),
    A in pairs_of(T)

    B = mapstorage(copy, A)
    @test (@inferred A == B) === true
    @test allocations(==, A, B) == 0
    @test (@inferred GeometricOptimizers.contains_nonfinite(A)) === false
    @test allocations(GeometricOptimizers.contains_nonfinite, A) == 0
    @test allocations(collect, A) > 0
end

@testset "two families compare their entries, $T" for T in (Float32, Float64)
    S, K, L, U, Y, G = pairs_of(T)
    zero_sym, zero_skew = zero(S), zero(K)
    @test (zero_sym == zero_skew) == (Matrix(zero_sym) == Matrix(zero_skew))
    @test zero_sym == zero_skew
    @test (L == U) == (Matrix(L) == Matrix(U))
    # a Stiefel and a Grassmann point with one representative have equal entries
    @test Y == G
end
