# The `freeparameters`/`rebuild` protocol for this package's structured matrices, and the two things
# it buys: a flat form that has the right length, and an HDF5 round trip that gives the concrete type
# back rather than a densified copy of it.

using GeometricOptimizers
using GeometricOptimizers: OptimizerSolution, _dot, _manifold_αmax
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
# Downstream code indexes this vector by hand -- `test/flat_parameters.jl` asserts literal
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

@testset "a parameter set is an OptimizerSolution and binds its element type" begin
    # this is the reason `NeuralNetworkParameters` 0.2 carries the element type on the type. Every
    # cache and state constructor, both `Optimizer` constructors and the `BFGSState` `update!` methods
    # take `T` from the *type* of the solution, in one of these two shapes, so a parameter set is only
    # usable as a solution if `T` binds for it.
    bound(x::VT) where {T, VT <: OptimizerSolution{T}} = T
    annotated(::OptimizerSolution{T}) where {T} = T

    ps = NetworkParameters((L1 = (W = [1.0 2.0; 3.0 4.0], b = [5.0, 6.0]),))
    @test bound(ps) === Float64
    @test annotated(ps) === Float64
    # `<:` and not `===`, and the difference is a Julia version rather than a weaker claim. Inference
    # spells "exactly `Float64`" as `Type{Float64}` up to 1.13 and as `Core.TypeEgal{Float64}` on
    # 1.14-DEV; both are `<: Type{Float64}`, and a failure to infer is not — giving up returns `Type`
    # or `DataType`, neither of which is a subtype of it. So this still catches what it exists to
    # catch, and stops asserting how the compiler happens to write the answer down.
    @test only(Base.return_types(bound, Tuple{typeof(ps)})) <: Type{Float64}

    ps32 = NetworkParameters((L1 = (W = Float32[1 2], b = Float32[3]),))
    @test bound(ps32) === Float32
    @test annotated(ps32) === Float32

    # a structured leaf reaches its element type through the protocol above
    @test bound(NetworkParameters((L1 = (W = SymmetricMatrix(rand(3, 3)),),))) === Float64

    # and the other members of the union still bind
    @test bound([1.0, 2.0]) === Float64
    @test bound(rand(StiefelManifold{Float64}, 4, 2)) === Float64

    # a *flat* set binds through the same type parameter as a nested one, because `T` comes off
    # `NetworkParameters{T}` directly rather than off a bound on the values -- see
    # `src/optimizer_solution.jl` for why that distinction is the whole design
    @test bound(NetworkParameters((a = [1.0], b = Float64[2 3]))) === Float64
    @test bound(NetworkParameters((a = Float32[1],))) === Float32
    @test bound(NetworkParameters((a = rand(StiefelManifold{Float64}, 4, 2), b = [1.0]))) === Float64
end

# The checks `_dot` and `_manifold_αmax` gained when they stopped writing their own recursion, which is
# the half of issue #70's fix that is not about the clock.
#
# Both used to pair *positionally* over `values` and check neither that the keys agreed nor that the two
# branches were the same width -- inherited from the `dot(flatten(a), flatten(b))` and the per-block
# ceiling they replaced rather than chosen, and `_dot`'s own comment said this was where such a check
# would go if one were ever wanted. Upstream's zipped fold pairs by key and checks the widths in its
# *generator*, so both are free at run time and a mismatch raises before the fold is specialised.
#
# The messages are upstream's, matched on substrings. `"and, in argument 2,"` is matched as well as
# `"different keys"` because the latter prefix is shared with the run-time `_check_keys` that
# `mapparameters` pays -- so this pins that the error came from the fold, which is the cheaper of the
# two.
@testset "the zipped folds check keys and widths" begin
    a = NetworkParameters((L1 = (W = rand(2, 2), b = rand(2)),))

    @test_throws "different keys" _dot(a, NetworkParameters((L2 = (W = rand(2, 2), b = rand(2)),)))
    @test_throws "and, in argument 2," _dot(a, NetworkParameters((L2 = (W = rand(2, 2), b = rand(2)),)))

    # key *order*, which a positional fold crossed over silently and is the worse of the two failures:
    # it produced a number rather than an error. Both sides need the same width here, or
    # `_children_arity` raises the width message first.
    @test_throws "different keys" _dot(a, NetworkParameters((L1 = (b = rand(2), W = rand(2, 2)),)))

    @test_throws "same number of children" _dot(a, NetworkParameters((L1 = (W = rand(2, 2),),)))

    # a whole set that is `nothing`. A fold reduces every leaf it is given, so a missing set would make
    # the answer a partial sum without saying so -- where `foreachparameters` skips one. The hole is at
    # a *branch*: against a single-block leaf a `nothing` reaches the operator by design, and what it
    # contributes is the caller's to decide, so that case is a `MethodError` from `dot` instead.
    holed = NetworkParameters((L1 = (W = rand(2, 2),), L2 = nothing))
    whole = NetworkParameters((L1 = (W = rand(2, 2),), L2 = (b = rand(2),)))
    @test_throws "partial sum" _dot(whole, holed)

    # `_manifold_αmax` is the same walk at the same arity, and gained the same checks
    @test_throws "different keys" _manifold_αmax(NetworkParameters((Y = rand(2),)),
                                                 NetworkParameters((Z = rand(2),)), 1.0)
    @test_throws "same number of children" _manifold_αmax(NetworkParameters((a = rand(2), b = rand(2))),
                                                          NetworkParameters((a = rand(2),)), 1.0)

    # and the check cannot be spelled around, which is the other half of it. Upstream pairs a `Tuple`
    # branch *positionally* -- a `Tuple`'s blocks have no keys to agree on -- so `_manifold_αmax` bounds
    # its first argument to a `NetworkParameters` rather than taking anything. Without that bound
    # `_manifold_αmax(values(sol), values(δ), c)`, which is how every call site here was spelled until
    # this release, would still compile and would still cross the keys silently.
    @test_throws MethodError _manifold_αmax(values((Y = rand(2), Z = rand(2))),
                                            values((Z = rand(2), Y = rand(2))), 1.0)
end
