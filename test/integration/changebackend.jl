# `changebackend` for this package's structured matrices, from `ext/AbstractNeuralNetworksExt.jl`.
#
# There is no second device in CI, so what is pinned is the walk and the reconstruction rather than a
# transfer: `changebackend(CPU(), x)` allocates on the CPU backend and copies, so a leaf comes back
# equal, of the same type, and not the same array. That is exactly the property the five hand-written
# methods in `GeometricMachineLearning`'s HDF5 extension were there to provide, and it is what has to
# hold before they can be deleted from there.

using AbstractNeuralNetworks: changebackend, CPU, ZeroVector
using GeometricOptimizers
using LinearAlgebra: mul!
using NeuralNetworkParameters: NetworkParameters
using Random
using Test

include("../helpers/eltypes.jl")

Random.seed!(1234)

const N, n = 6, 3

# one leaf of every family, drawn in `T` from a seeded generator
function leaves(T)
    rng = Random.Xoshiro(1234)
    (
        stiefel = rand(rng, StiefelManifold{T}, N, n),
        grassmann = rand(rng, GrassmannManifold{T}, N, n),
        symmetric = SymmetricMatrix(randn(rng, T, n, n)),
        skew = SkewSymMatrix(randn(rng, T, n, n)),
        lower = StrictlyLowerTriangular(randn(rng, T, n, n)),
        upper = StrictlyUpperTriangular(randn(rng, T, n, n)),
        stiefhor = StiefelLieAlgHorMatrix(
            SkewSymMatrix(randn(rng, T, n, n)), randn(rng, T, N - n, n), N, n),
        grasshor = GrassmannLieAlgHorMatrix(randn(rng, T, N - n, n), N, n)
    )
end

@testset "the extension is loaded" begin
    @test Base.get_extension(GeometricOptimizers, :AbstractNeuralNetworksExt) !== nothing
end

@testset "every family keeps its type and its numbers, $T" for T in REAL_ELTYPES
    # one testset per family, so a failure names the leaf that failed
    for (k, x) in pairs(leaves(T))
        @testset "$k" begin
            y = changebackend(CPU(), x)
            @test typeof(y) == typeof(x)
            @test eltype(y) == T
            # a transfer copies the entries, so nothing rounds
            @test y == x
            # a transfer copies; it does not alias the source
            @test parent(y) !== parent(x)
        end
    end
end

@testset "the metadata a structured leaf carries survives, $T" for T in REAL_ELTYPES
    # `n` and `N` are not in the storage, so they can only come from the prototype
    ls = leaves(T)
    for k in (:symmetric, :skew, :lower, :upper)
        @test changebackend(CPU(), ls[k]).n == ls[k].n
    end
    for k in (:stiefhor, :grasshor)
        @test changebackend(CPU(), ls[k]).N == ls[k].N
        @test changebackend(CPU(), ls[k]).n == ls[k].n
    end
end

@testset "a horizontal lift keeps its structured block structured, $T" for T in REAL_ELTYPES
    # `StiefelLieAlgHorMatrix` holds a `SkewSymMatrix` as its first block, so the walk has to recurse
    # into it rather than densify it
    x = leaves(T).stiefhor
    y = changebackend(CPU(), x)
    @test y.A isa SkewSymMatrix
    @test eltype(y.A) == T
    @test y.A == x.A
end

@testset "a whole parameter set walks through the container methods, $T" for T in REAL_ELTYPES
    # `AbstractNeuralNetworks` supplies the `NamedTuple`/`NetworkParameters` methods; this is the check
    # that the leaf methods above meet them correctly
    ls = leaves(T)
    ps = NetworkParameters((L1 = (Y = ls.stiefel, b = randn(Random.Xoshiro(7), T, N)),
        L2 = (S = ls.symmetric, G = ls.stiefhor)))
    back = changebackend(CPU(), ps)

    @test back isa NetworkParameters
    @test keys(back) == keys(ps)
    @test back.L1.Y isa StiefelManifold
    @test back.L2.S isa SymmetricMatrix
    @test back.L2.G isa StiefelLieAlgHorMatrix
    @test back.L2.G.A isa SkewSymMatrix
    @test eltype(back.L1.b) == T
    @test back.L1.b == ps.L1.b
end

@testset "element type is preserved, $T" for T in REAL_ELTYPES
    x = SymmetricMatrix(randn(Random.Xoshiro(3), T, n, n))
    @test eltype(changebackend(CPU(), x)) === T
    Y = rand(Random.Xoshiro(4), StiefelManifold{T}, N, n)
    @test eltype(changebackend(CPU(), Y)) === T
end

@testset "a StiefelProjection keeps its type, $T" for T in REAL_ELTYPES
    E = GeometricOptimizers.StiefelProjection(T, N, n)
    F = changebackend(CPU(), E)
    @test F isa GeometricOptimizers.StiefelProjection{T}
    @test eltype(F) == T
    @test F == E
    @test F.A !== E.A
end

# The extension carries one more method, the tie-breaker between this package's vector `mul!` and
# `AbstractNeuralNetworks`' `mul!(out, A, ::ZeroVector)`. Without it every owned type is ambiguous.
@testset "a product with a ZeroVector is zero for every owned type, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(5)
    for A in (rand(rng, SkewSymMatrix{T}, N), rand(rng, SymmetricMatrix{T}, N),
        rand(rng, StiefelManifold{T}, N, n))
        out = randn(rng, T, N)
        @test mul!(out, A, ZeroVector(T, size(A, 2))) === out
        @test eltype(out) == T
        @test iszero(out)
    end
end
