# The `freeparameters`/`rebuild` protocol for this package's structured matrices, and the two things
# it buys: a flat form that has the right length, and an HDF5 round trip that gives the concrete type
# back rather than a densified copy of it.

using GeometricOptimizers
using HDF5
using NeuralNetworkParameters
using NeuralNetworkParameters: freeparameters, rebuild, parameter_metadata, flatlength,
                               parameterrange, save, load
using Random
using Test

Random.seed!(1234)

# Every HDF5 testset below writes a file, reads it back and wants it gone either way.
function withtempfile(f)
    file = tempname() * ".h5"
    try
        f(file)
    finally
        isfile(file) && rm(file)
    end
end

# Narrow a leaf to `Float32` the way the protocol itself would: down to the storage, convert, back up
# through `rebuild`. Broadcasting instead would densify the structured types, which is the very thing
# the protocol exists to avoid.
_narrow(x::Tuple) = map(_narrow, x)
_narrow(x) = (s = freeparameters(x); s === x ? Float32.(x) : rebuild(x, _narrow(s)))

# A `Manifold` the extension has `freeparameters` for — it is defined on the abstract type — but no
# `rebuild`, standing in for one added in a later release.
struct DummyManifold{T} <: Manifold{T} end

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
    for x in values(leaves)
        y = rebuild(x, freeparameters(x))
        @test typeof(y) == typeof(x)
        @test y == x
    end
end

@testset "rebuild carries the shape from the prototype, not the type" begin
    # this is what lets a flattened parameter set be differentiated: `data` comes back with a
    # different element type from the prototype. One of each family, because the shape each one takes
    # from the prototype differs — the `n` of a storage matrix, nothing at all for a manifold element,
    # and the `N`/`n` of a lift whose constructor additionally requires both blocks to share their
    # element type, which is exactly what a `Dual` flatten produces.
    for x in (leaves.symmetric, leaves.stiefel, leaves.stiefhor)
        narrowed = _narrow(x)
        @test nameof(typeof(narrowed)) === nameof(typeof(x))
        @test eltype(narrowed) == Float32
        @test size(narrowed) == size(x)
        @test Array(narrowed) ≈ Array(x)
    end
end

@testset "a family member with no rebuild says so" begin
    # `freeparameters` is defined on the abstract types, so a `Manifold` added in a later release gets
    # it for free. Without an erroring `rebuild`, `NeuralNetworkParameters`' `rebuild(::AbstractArray,
    # data) = data` would catch that type and quietly hand back the bare storage — a densified
    # parameter, no error anywhere.
    err = try
        rebuild(DummyManifold{Float64}(), rand(2, 2))
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("DummyManifold", err.msg)
end

@testset "the flat form has the storage length, not the dense one" begin
    ps = NetworkParameters((L1 = leaves,))

    # n(n+1)/2 for the symmetric matrix; n(n-1)/2 for the skew-symmetric one and for each of the two
    # triangulars, which keep the strict triangle; N*n for each manifold element; and the horizontal
    # lifts' blocks
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

# The whole `ParameterHandling` removal rests on this: the flat vector this package's protocol
# produces has to be the *same numbers in the same order* as the one `ParameterHandling.flatten`
# produced, because downstream code indexes it by hand. `test/named_tuple_parameters.jl` asserts
# literal ranges, and its `∇F!` slices the flat vector with them.
#
# It holds by construction rather than by luck — `Base.vec` and `Base.parent` return the same storage
# for every one of these types, and `NeuralNetworkParameters` copies a leaf in linear index order —
# but "by construction" is an argument, not a test. This is the test. It is written while both
# packages are still present precisely so that the equality is recorded before one of them goes.
@testset "the flat ordering agrees with the one downstream code indexes by" begin
    for (k, x) in pairs(leaves)
        v_ph, _ = GeometricOptimizers.ParameterHandling.flatten(Float64, x)
        v_nn, _ = NeuralNetworkParameters.flatten(Float64, x)
        @test v_nn == v_ph
    end

    # and through a container, which is where the per-leaf orders compose. Heterogeneous on purpose:
    # a manifold, a storage matrix, a lift and a plain array in one set.
    ps = (Y = leaves.stiefel, S = leaves.symmetric, G = leaves.stiefhor, W = leaves.plain)
    v_ph, _ = GeometricOptimizers.ParameterHandling.flatten(Float64, ps)
    v_nn, layout = NeuralNetworkParameters.flatten(Float64, ps)
    @test v_nn == v_ph

    # spelled out for the two that are not simply `vec` of the leaf, so a future reader can see
    # *which* numbers these are without running anything
    @test NeuralNetworkParameters.flatten(Float64, leaves.symmetric)[1] == parent(leaves.symmetric)
    @test NeuralNetworkParameters.flatten(Float64, leaves.stiefhor)[1] ==
          vcat(parent(parent(leaves.stiefhor)[1]), vec(parent(leaves.stiefhor)[2]))

    # the ranges, in declaration order, which is what a hand-written `∇F!` relies on
    @test parameterrange(layout.children.Y) == 1:(N * n)
    @test parameterrange(layout.children.W) ==
          (length(v_nn) - length(leaves.plain) + 1):length(v_nn)
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
    withtempfile() do file
        save(file, ps)
        read_back = load(NetworkParameters, file)

        @test keys(read_back) == keys(ps)
        for k in keys(leaves)
            @test typeof(read_back.L1[k]) == typeof(leaves[k])
            @test read_back.L1[k] ≈ leaves[k]
            # the structure survived: the storage is the storage, not a dense n×n
            @test _storage_lengths(read_back.L1[k]) == _storage_lengths(leaves[k])
        end
    end
end

@testset "HDF5 round trip against a prototype" begin
    # the form that bypasses the registry entirely
    ps = NetworkParameters((L1 = leaves,))
    withtempfile() do file
        save(file, ps)
        read_back = load(NetworkParameters, file, ps)
        for k in keys(leaves)
            @test typeof(read_back.L1[k]) == typeof(leaves[k])
            @test read_back.L1[k] ≈ leaves[k]
        end
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
    withtempfile() do file
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
    end
end
