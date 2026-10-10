# `==` of two manifold points, of two structured matrices of one family, and of two horizontal lifts
# of one type, compares the storage.
#
# The answer is `Base`'s for two arrays: equal entries. The generic method reads both one entry at a
# time, which a device refuses, and the optimizer compares iterates with `==` on every step
# (`latest_gradient_is_current`). A pair of two families keeps the generic method.

using GeometricOptimizers
using GeometricOptimizers: StrictlyLowerTriangular, StrictlyUpperTriangular
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: qr
using NeuralNetworkParameters: NetworkParameters, mapstorage
using Random
using Test
include("../helpers/eltypes.jl")

allowscalar(false)

const n = 4

function pairs_of(::Type{T}) where {T}
    rng = Random.Xoshiro(21)
    point = Matrix{T}(qr(randn(rng, T, 2n, n)).Q)[:, 1:n]
    (rand(rng, SymmetricMatrix{T}, n), rand(rng, SkewSymMatrix{T}, n),
        rand(rng, StrictlyLowerTriangular{T}, n), rand(rng, StrictlyUpperTriangular{T}, n),
        StiefelManifold(point), GrassmannManifold(copy(point)),
        rand(rng, StiefelLieAlgHorMatrix{T}, 2n, n), rand(rng, GrassmannLieAlgHorMatrix{T}, 2n, n))
end

# Two sizes can share a storage length: a `SkewSymMatrix` or a strictly triangular matrix of size
# 0 and of size 1 both store nothing, so the storage alone would call them equal
@testset "== tells two sizes of one family apart, $T" for T in REAL_ELTYPES
    for M in (SkewSymMatrix, StrictlyLowerTriangular, StrictlyUpperTriangular)
        @test M(T[], 0) != M(T[], 1)
        @test M(T[], 1) == M(T[], 1)
        @test eltype(M(T[], 1)) == T
    end
end

@testset "== compares the entries through the storage: $(nameof(typeof(A))), $T" for T in REAL_ELTYPES,
    A in pairs_of(T)

    B = mapstorage(copy, A)
    C = mapstorage(s -> s .+ one(T), A)
    @test eltype(A) == eltype(C) == T
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

@testset "== and contains_nonfinite infer and allocate nothing: $(nameof(typeof(A))), $T" for T in REAL_ELTYPES,
    A in pairs_of(T)

    B = mapstorage(copy, A)
    @test eltype(B) == T
    @test (@inferred A == B) === true
    @test allocations(==, A, B) == 0
    @test (@inferred GeometricOptimizers.contains_nonfinite(A)) === false
    @test allocations(GeometricOptimizers.contains_nonfinite, A) == 0
    @test allocations(collect, A) > 0
end

# one non-finite storage entry is found, in a leaf and in a set of leaves, on the host and on a
# device
@testset "contains_nonfinite finds a $bad entry: $(nameof(typeof(A))), $T" for T in REAL_ELTYPES,
    A in pairs_of(T), bad in (T(NaN), T(Inf), -T(Inf))

    poisoned = mapstorage(s -> (s = copy(s); s[1] = bad; s), A)
    @test eltype(poisoned) == T
    rest = (; W = randn(Random.Xoshiro(5), T, 2, 3))
    for todevice in (identity, JLArray)
        @test GeometricOptimizers.contains_nonfinite(mapstorage(todevice, poisoned)) ===
              true
        @test GeometricOptimizers.contains_nonfinite(mapstorage(todevice, A)) === false
        set(x) = NetworkParameters((; rest..., A = x))
        @test GeometricOptimizers.contains_nonfinite(mapstorage(todevice, set(poisoned))) ===
              true
        @test GeometricOptimizers.contains_nonfinite(mapstorage(todevice, set(A))) === false
    end
end

# every block of a lift decides `==`: a `C` that differs from `A` in one block only is not equal
# to it, whichever block that is
@testset "== of two lifts reads every block: $(nameof(typeof(A))), $T" for T in REAL_ELTYPES,
    A in pairs_of(T)[7:8]

    for k in eachindex(parent(A))
        C = mapstorage(copy, A)
        GeometricOptimizers.freeparameters(parent(C)[k]) .+= one(T)
        @test eltype(C) == T
        @test A != C
        @test (A == C) == (Matrix(A) == Matrix(C))
        @test mapstorage(JLArray, A) != mapstorage(JLArray, C)
    end
end

@testset "two families compare their entries, $T" for T in REAL_ELTYPES
    S, K, L, U, Y, G, H, R = pairs_of(T)
    zero_stiefel_lift, zero_grassmann_lift = mapstorage(zero, H), mapstorage(zero, R)
    @test eltype(zero_stiefel_lift) == eltype(zero_grassmann_lift) == T
    @test zero_stiefel_lift == zero_grassmann_lift
    @test (H == R) == (Matrix(H) == Matrix(R))
    zero_sym, zero_skew = zero(S), zero(K)
    @test (zero_sym == zero_skew) == (Matrix(zero_sym) == Matrix(zero_skew))
    @test zero_sym == zero_skew
    @test (L == U) == (Matrix(L) == Matrix(U))
    # a Stiefel and a Grassmann point with one representative have equal entries
    @test Y == G
end
