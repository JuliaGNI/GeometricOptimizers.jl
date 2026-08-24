# `changebackend` for this package's structured matrices, from `ext/AbstractNeuralNetworksExt.jl`.
#
# There is no second device in CI, so what is pinned is the walk and the reconstruction rather than a
# transfer: `changebackend(CPU(), x)` allocates on the CPU backend and copies, so a leaf comes back
# equal, of the same type, and not the same array. That is exactly the property the five hand-written
# methods in `GeometricMachineLearning`'s HDF5 extension were there to provide, and it is what has to
# hold before they can be deleted from there.

using AbstractNeuralNetworks: changebackend, CPU
using GeometricOptimizers
using NeuralNetworkParameters: NetworkParameters
using Random
using Test

Random.seed!(1234)

const N, n = 6, 3

leaves = (
    stiefel   = rand(StiefelManifold{Float64}, N, n),
    grassmann = rand(GrassmannManifold{Float64}, N, n),
    symmetric = SymmetricMatrix(rand(n, n)),
    skew      = SkewSymMatrix(rand(n, n)),
    lower     = LowerTriangular(rand(n, n)),
    upper     = UpperTriangular(rand(n, n)),
    stiefhor  = StiefelLieAlgHorMatrix(SkewSymMatrix(rand(n, n)), rand(N - n, n), N, n),
    grasshor  = GrassmannLieAlgHorMatrix(rand(N - n, n), N, n),
)

@testset "the extension is loaded" begin
    @test Base.get_extension(GeometricOptimizers, :AbstractNeuralNetworksExt) !== nothing
end

@testset "every family keeps its type and its numbers" begin
    # one testset per family, so a failure names the leaf that failed
    for (k, x) in pairs(leaves)
        @testset "$k" begin
            y = changebackend(CPU(), x)
            @test typeof(y) == typeof(x)
            @test y ≈ x
            # a transfer copies; it does not alias the source
            @test parent(y) !== parent(x)
        end
    end
end

@testset "the metadata a structured leaf carries survives" begin
    # `n` and `N` are not in the storage, so they can only come from the prototype
    for k in (:symmetric, :skew, :lower, :upper)
        @test changebackend(CPU(), leaves[k]).n == leaves[k].n
    end
    for k in (:stiefhor, :grasshor)
        @test changebackend(CPU(), leaves[k]).N == leaves[k].N
        @test changebackend(CPU(), leaves[k]).n == leaves[k].n
    end
end

@testset "a horizontal lift keeps its structured block structured" begin
    # `StiefelLieAlgHorMatrix` holds a `SkewSymMatrix` as its first block, so the walk has to recurse
    # into it rather than densify it
    y = changebackend(CPU(), leaves.stiefhor)
    @test y.A isa SkewSymMatrix
    @test y.A ≈ leaves.stiefhor.A
end

@testset "a whole parameter set walks through the container methods" begin
    # `AbstractNeuralNetworks` supplies the `NamedTuple`/`NetworkParameters` methods; this is the check
    # that the leaf methods above meet them correctly
    ps = NetworkParameters((L1 = (Y = leaves.stiefel, b = rand(N)),
                            L2 = (S = leaves.symmetric, G = leaves.stiefhor)))
    back = changebackend(CPU(), ps)

    @test back isa NetworkParameters
    @test keys(back) == keys(ps)
    @test back.L1.Y isa StiefelManifold
    @test back.L2.S isa SymmetricMatrix
    @test back.L2.G isa StiefelLieAlgHorMatrix
    @test back.L2.G.A isa SkewSymMatrix
    @test back.L1.b ≈ ps.L1.b
end

@testset "element type is preserved" begin
    x = SymmetricMatrix(rand(Float32, n, n))
    @test eltype(changebackend(CPU(), x)) === Float32
    Y = rand(StiefelManifold{Float32}, N, n)
    @test eltype(changebackend(CPU(), Y)) === Float32
end
