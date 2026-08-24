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

# `NeuralNetworkParameters` is a hard dependency as of 0.5.0, so the protocol is simply there -- no
# extension to load and nothing to condition on. This asserts that rather than deleting the testset,
# because "the methods are present" is the precondition every testset below relies on.
@testset "the protocol is present unconditionally" begin
    @test freeparameters(leaves.symmetric) === parent(leaves.symmetric)
    @test rebuild(leaves.symmetric, parent(leaves.symmetric)) isa SymmetricMatrix
    @test isnothing(Base.get_extension(GeometricOptimizers, :NeuralNetworkParametersExt))
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

# The flat ordering, pinned absolutely.
#
# Downstream code indexes this vector by hand -- `test/named_tuple_parameters.jl` asserts literal
# ranges and its `∇F!` slices with them -- so the order is part of the contract, not an implementation
# detail. When this landed it was written as an elementwise comparison against
# `ParameterHandling.flatten`, which was still present, and the two agreed on every leaf family; see
# the commit that added it. With that package gone there is nothing left to compare against, so the
# expectations are spelled out instead. Which is the better test anyway: it says what the numbers
# *are* rather than that two implementations happen to concur.
@testset "the flat ordering is the one downstream code indexes by" begin
    # a manifold flattens as its dense storage, in linear index order
    @test flatten(Float64, leaves.stiefel)[1] == vec(parent(leaves.stiefel))
    @test flatten(Float64, leaves.grassmann)[1] == vec(parent(leaves.grassmann))

    # a storage matrix flattens as the vector it keeps, which is also what `vec` returns for it --
    # `n(n±1)/2` numbers, not `n²`
    for x in (leaves.symmetric, leaves.skew, leaves.lower, leaves.upper)
        @test flatten(Float64, x)[1] == parent(x)
        @test flatten(Float64, x)[1] == vec(x)
    end

    # a lift flattens block by block, in the order `parent` returns them, and the first block of a
    # `StiefelLieAlgHorMatrix` is itself structured so it contributes its own storage
    @test flatten(Float64, leaves.stiefhor)[1] ==
          vcat(parent(parent(leaves.stiefhor)[1]), vec(parent(leaves.stiefhor)[2]))
    @test flatten(Float64, leaves.grasshor)[1] == vec(parent(leaves.grasshor)[1])

    # a plain array is itself, in linear index order
    @test flatten(Float64, leaves.plain)[1] == vec(leaves.plain)

    # and through a container the per-leaf orders compose in declaration order. Heterogeneous on
    # purpose: a manifold, a storage matrix, a lift and a plain array in one set.
    ps = (Y = leaves.stiefel, S = leaves.symmetric, G = leaves.stiefhor, W = leaves.plain)
    v, layout = flatten(Float64, ps)
    @test v == vcat(flatten(Float64, leaves.stiefel)[1], flatten(Float64, leaves.symmetric)[1],
                    flatten(Float64, leaves.stiefhor)[1], flatten(Float64, leaves.plain)[1])
    @test parameterrange(layout.children.Y) == 1:(N * n)
    @test parameterrange(layout.children.W) == (length(v) - length(leaves.plain) + 1):length(v)
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

# `GlobalSection` of the container. This is one method, but it is the method that lets
# `GeometricMachineLearning` delete its own copy of it -- `GlobalSection` is this package's function
# and `NetworkParameters` is `NeuralNetworkParameters`', so a package that merely uses both owns
# neither name.
@testset "a GlobalSection can be taken of the container" begin
    nt = (L1 = (Y = leaves.stiefel, b = rand(N)), L2 = (Z = leaves.grassmann,))
    ps = NetworkParameters(nt)

    λ_container = GlobalSection(ps)
    λ_bare = GlobalSection(nt)

    # the result is the plain tree, not a `NetworkParameters` of sections: a section is not a
    # parameter, and everything downstream walks it as an ordinary container
    @test λ_container isa NamedTuple
    @test keys(λ_container) == keys(nt)
    @test typeof(λ_container) == typeof(λ_bare)

    # every leaf gets a section, a manifold one and an ordinary array alike
    @test λ_container.L1.Y isa GlobalSection
    @test λ_container.L1.b isa GlobalSection
    @test λ_container.L2.Z isa GlobalSection

    # and the section really is of these parameters -- `==`, not `===`, because the constructor
    # deliberately stores a copy of the anchor
    @test λ_container.L1.Y.Y == ps.L1.Y
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
