# WARNING: do not use the aliases in this file as `struct` type-parameter bounds.
#
# Use them in method signatures, where they dispatch. As bounds on the type parameters of a `struct`
# they are ruinously expensive, because they *couple* the parameters.
#
# Inference cannot solve `NamedTuple{S,<:Tuple{Vararg{AbstractArray{T}}}}` down to a concrete
# `NamedTuple`, so `OptimizerCache(Adam(Float64), ps)` for a `NamedTuple` of parameters infers to a
# `UnionAll` whatever these structs look like — the outer constructors' own signatures are written in
# the same aliases and are enough to cause that on their own. Removing the struct bounds does not
# make the inferred type concrete and is not meant to.
#
# What the bounds add is the shared `T`. With
#
#     struct AdamCache{T,MT<:OptimizerSolution{T},VT<:GradientStorage{T},
#                        ST<:GlobalSectionSingleOrNamedTuple{T}} <: OptimizerCache{T}
#
# the inferred type is
#
#     AdamCache{T, NamedTuple{(:Y,:W,:b),s1}, NamedTuple{(:Y,:W,:b),s2}, NamedTuple{(:Y,:W,:b),s3}} where
#         {T, s1<:Tuple{Vararg{AbstractArray{T}}}, s2<:Tuple{Vararg{AbstractArray{T}}},
#             s3<:Tuple{Vararg{GlobalSection{T,AT,λT} where {AT<:AbstractArray{T},
#                                                            λT<:Union{Nothing,AbstractArray{T}}}}}}
#
# — one `T` tying all four parameters together underneath three nested `Vararg` unions, the last of
# them over a three-parameter `UnionAll`. Every method-table intersection involving such a type has
# to re-solve that constraint system in `subtype_unionall`. Unbounded, the same call infers to the
# same shape but with the parameters independent and `s3<:Tuple`, which costs nothing to intersect.
#
# The difference is not marginal: a nine-layer network whose optimizer was compiled through a
# function did not finish inferring in over an hour with the bounds, and takes ~14 s without them.
# See GeometricMachineLearning#230.
#
# Nothing is given up by dropping them. The invariant is enforced where it always really was, in the
# constructors, whose signatures take `x::OptimizerSolution{T}` and
# `g::AT where AT<:GradientStorage{T}` and build the `GlobalSection` themselves — the same
# guarantee, checked by dispatch, at no cost to inference. That holds for an *inner* constructor as
# readily as an outer one: what must not carry the aliases is the `struct` parameter list, not the
# methods.
#
# Every optimizer cache and state leaves its parameters unbounded, including the ones whose bounds
# were never the expensive kind (`NewtonOptimizerCache`, `NewtonOptimizerState`). The family stays
# uniform, and nobody has to work out per struct whether a given bound happens to be one that costs.
# `test/flat_parameters.jl` pins this.

"""
    OptimizerSolution

A type alias for the solution of an optimizer: an `AbstractVector`, a [`Manifold`](@ref), or a
[`NeuralNetworkParameters.NetworkParameters`](@extref) holding a whole set of network parameters.

A set of parameters enters this package as a container and never as a bare `NamedTuple`. That is the
constraint every other alias in this file is written against, so it is worth saying why the obvious
alternative is not available.

**A parameter set has to be a type somebody owns.**
`NamedTuple{S,<:Tuple{Vararg{AbstractArray{T}}}}` picks out the same values, but it is an *alias for
`Base.NamedTuple`* rather than a type of its own, and three things follow:

  - a method taking one is a method on `Base.NamedTuple`. Unless the function is also this package's,
    that is type piracy — and it is global, changing what the function means for every keyed
    `NamedTuple` in any session that loads this package, directly or through a dependency. Issue
    [#16](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/16) is the record of what that
    costs to undo;
  - it cannot be told apart from any *other* `NamedTuple` alias in the same method table. It would
    overlap `GlobalSectionNamedTuple` and the bare `NamedTuple` that a nested section tree is written
    on, and on the overlap neither method of such a pair is more specific — an ambiguity rather than a
    choice. `optimizers/named_tuple_wrapper.jl` is where that would bite;
  - its `T` binds through a `Vararg` bound on the *values*, which asserts "flat, and every leaf an
    `AbstractArray{T}`". `NetworkParameters{T}` asserts something else: a promotion over leaves at any
    depth. A union of the two would say one thing about one argument shape and another about the
    other.

`NetworkParameters{T}` has none of those properties. It also binds `T` from a direct type parameter
rather than through a `Vararg` bound on value types, which makes it the cheapest member of this union
to intersect, and it is the only member whose `T` is a *promotion* over the leaves rather than a
guarantee that every leaf is a `T`.

!!! note "A caller holding a bare `NamedTuple` wraps it"
    `NetworkParameters(ps)` shares the leaf arrays rather than copying them, so an in-place solve
    still writes through to the caller's own arrays and nothing has to be copied back. That is what
    `GeometricMachineLearning` and `GMLDatasets` do at the boundary.
"""
const OptimizerSolution{T} = Union{AbstractVector{T},Manifold{T},NetworkParameters{T}}

# A gradient is one leaf's worth of storage or a whole set of them. `AbstractArray` and not
# `AbstractVecOrMat`: a horizontal lift and a [`VectorStorageMatrix`](@ref) are both `AbstractArray`s
# that are neither.
const GradientStorage{T} = Union{AbstractArray{T},NetworkParameters{T}}

# note that this is *not* `Tuple{Vararg{ST}} where {ST<:GlobalSection{T}}`, as Julia's diagonal rule
# would make that homogeneous, i.e. it would not allow a `NamedTuple` holding the section of a
# `StiefelManifold` beside the section of an ordinary `Matrix` at the same time.
const GlobalSectionTuple{T} = Tuple{Vararg{GlobalSection{T}}}

const GlobalSectionNamedTuple{T,X} = begin
    NamedTuple{X,<:GlobalSectionTuple{T}}
end

const GlobalSectionSingleOrNamedTuple{T} = Union{GlobalSection{T},GlobalSectionNamedTuple{T}}
