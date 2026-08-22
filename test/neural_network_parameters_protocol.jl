# The `freeparameters`/`rebuild` protocol for this package's structured matrices, and the two things
# it buys: a flat form that has the right length, and an HDF5 round trip that gives the concrete type
# back rather than a densified copy of it.

using GeometricOptimizers
using LinearAlgebra: norm
using HDF5
using NeuralNetworkParameters
using NeuralNetworkParameters: freeparameters, rebuild, parameter_metadata, flatlength,
                               save, load
using Random
using Test

Random.seed!(1234)

# `freeparameters` of a horizontal lift is a tuple of blocks, one of which is itself structured, so
# comparing storage sizes means walking it.
_storage_lengths(x) = (s = freeparameters(x); s === x ? (length(x),) :
                       s isa Tuple ? mapreduce(_storage_lengths, (a, b) -> (a..., b...), s) :
                       _storage_lengths(s))

const N, n = 6, 3

# One of each family, plus a plain array to keep the tree honest. The horizontal lifts are built
# from their blocks rather than through the `(D, n)` constructors, which slice with `@views` — a lift
# holding `SubArray`s can never compare type-equal to one rebuilt from a flat vector, and that is a
# property of the constructor, not of the protocol under test.
leaves = (
    stiefel   = rand(StiefelManifold{Float64}, N, n),
    grassmann = rand(GrassmannManifold{Float64}, N, n),
    symmetric = SymmetricMatrix(rand(n, n)),
    skew      = SkewSymMatrix(rand(n, n)),
    lower     = LowerTriangular(rand(n, n)),
    upper     = UpperTriangular(rand(n, n)),
    stiefhor  = StiefelLieAlgHorMatrix(SkewSymMatrix(rand(n, n)), rand(N - n, n), N, n),
    grasshor  = GrassmannLieAlgHorMatrix(rand(N - n, n), N, n),
    plain     = rand(2, 2),
)

@testset "the extension is loaded" begin
    @test Base.get_extension(GeometricOptimizers, :NeuralNetworkParametersExt) !== nothing
end

@testset "freeparameters is Base.parent" begin
    for (k, x) in pairs(leaves)
        k === :plain && continue
        @test freeparameters(x) === parent(x)
        # and it is *not* the leaf itself, i.e. the leaf is not treated as terminal — which is what
        # would otherwise flatten a symmetric matrix as n² numbers, or fail outright for the types
        # with no `setindex!`
        @test freeparameters(x) !== x
    end
    @test freeparameters(leaves.plain) === leaves.plain
end

@testset "rebuild inverts freeparameters" begin
    for (k, x) in pairs(leaves)
        y = rebuild(x, freeparameters(x))
        @test typeof(y) == typeof(x)
        @test y == x
    end
end

@testset "rebuild carries the shape from the prototype, not the type" begin
    # this is what lets a flattened parameter set be differentiated: `data` comes back with a
    # different element type from the prototype
    A = leaves.symmetric
    dual = rebuild(A, Float32.(freeparameters(A)))
    @test dual isa SymmetricMatrix
    @test eltype(dual) == Float32
    @test size(dual) == size(A)
end

@testset "the flat form has the storage length, not the dense one" begin
    ps = NetworkParameters((L1 = leaves,))

    # n(n+1)/2 + n(n-1)/2 for symmetric + skew, n(n+1)/2 each for the two triangulars,
    # N*n for each manifold element, and the horizontal lifts' blocks
    expected = length(parent(leaves.stiefel)) + length(parent(leaves.grassmann)) +
               length(parent(leaves.symmetric)) + length(parent(leaves.skew)) +
               length(parent(leaves.lower)) + length(parent(leaves.upper)) +
               length(freeparameters(parent(leaves.stiefhor)[1])) +
               length(parent(leaves.stiefhor)[2]) +
               length(parent(leaves.grasshor)[1]) +
               length(leaves.plain)
    @test flatlength(ps) == expected

    v, layout = flatten(ps)
    @test length(v) == expected
    back = unflatten(layout, v)
    for k in keys(leaves)
        @test typeof(back.L1[k]) == typeof(leaves[k])
        @test back.L1[k] == leaves[k]
    end
end

@testset "parameter_metadata records what the storage does not determine" begin
    @test parameter_metadata(leaves.symmetric) == (n = n,)
    @test parameter_metadata(leaves.skew) == (n = n,)
    @test parameter_metadata(leaves.lower) == (n = n,)
    @test parameter_metadata(leaves.upper) == (n = n,)
    @test parameter_metadata(leaves.stiefhor) == (N = N, n = n)
    @test parameter_metadata(leaves.grasshor) == (N = N, n = n)
    # a manifold element's storage *is* the dense matrix, so there is nothing to record
    @test parameter_metadata(leaves.stiefel) == NamedTuple()
end

@testset "HDF5 round trip, with no prototype" begin
    # the registered form: `__init__` taught `load` how to rebuild each of these, so a file loads
    # without the caller having to supply a parameter set of the right shape
    ps = NetworkParameters((L1 = leaves,))
    file = tempname() * ".h5"
    try
        save(file, ps)
        read_back = load(NetworkParameters, file)

        @test keys(read_back) == keys(ps)
        for k in keys(leaves)
            @test typeof(read_back.L1[k]) == typeof(leaves[k])
            @test read_back.L1[k] ≈ leaves[k]
            # the structure survived: the storage is the storage, not a dense n×n
            @test _storage_lengths(read_back.L1[k]) == _storage_lengths(leaves[k])
        end
    finally
        isfile(file) && rm(file)
    end
end

@testset "HDF5 round trip against a prototype" begin
    # the form that bypasses the registry entirely
    ps = NetworkParameters((L1 = leaves,))
    file = tempname() * ".h5"
    try
        save(file, ps)
        read_back = load(NetworkParameters, file, ps)
        for k in keys(leaves)
            @test typeof(read_back.L1[k]) == typeof(leaves[k])
            @test read_back.L1[k] ≈ leaves[k]
        end
    finally
        isfile(file) && rm(file)
    end
end

@testset "Float32 parameters stay Float32" begin
    # `parameter_eltype` promotes over the leaves' storage, so a single-precision set must not widen
    ps = NetworkParameters((L1 = (W = SymmetricMatrix(rand(Float32, n, n)),
                                  Y = rand(StiefelManifold{Float32}, N, n)),))
    v, layout = flatten(ps)
    @test eltype(v) == Float32
    back = unflatten(layout, v)
    @test eltype(back.L1.W) == Float32
    @test eltype(back.L1.Y) == Float32
end

@testset "a file in GeometricMachineLearning's old layout still loads" begin
    # GML used to write these matrices itself: a group tagged `gml_type`, holding the fields under
    # their own names and no record of the key order. `NeuralNetworkParameters` recognises the tag
    # and rebuilds through the registry, passing the group's fields as both storage and metadata —
    # so the reconstructors registered here have to accept that shape too.
    file = tempname() * ".h5"
    try
        A = leaves.symmetric
        Y = leaves.stiefel
        h5open(file, "w") do h5
            g = HDF5.create_group(h5, "L1")

            gA = HDF5.create_group(g, "W")
            HDF5.attributes(gA)["gml_type"] = "SymmetricMatrix"
            gA["S"] = Array(A.S)
            gA["n"] = A.n

            gY = HDF5.create_group(g, "Y")
            HDF5.attributes(gY)["gml_type"] = "StiefelManifold"
            gY["A"] = Array(Y.A)

            g["b"] = [1.0, 2.0]
        end

        read_back = load(NetworkParameters, file)

        @test read_back.L1.W isa SymmetricMatrix
        @test read_back.L1.W ≈ A
        @test read_back.L1.Y isa StiefelManifold
        @test read_back.L1.Y ≈ Y
        @test read_back.L1.b == [1.0, 2.0]
    finally
        isfile(file) && rm(file)
    end
end
