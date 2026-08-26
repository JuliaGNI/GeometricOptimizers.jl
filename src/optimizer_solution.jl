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
#     struct AdamCache{T,MT<:OptimizerSolution{T},VT<:GradientArrayOrNamedTuple{T},
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
# `g::AT where AT<:GradientArrayOrNamedTuple{T}` and build the `GlobalSection` themselves — the same
# guarantee, checked by dispatch, at no cost to inference. That holds for an *inner* constructor as
# readily as an outer one: what must not carry the aliases is the `struct` parameter list, not the
# methods.
#
# Every optimizer cache and state leaves its parameters unbounded, including the ones whose bounds
# were never the expensive kind (`NewtonOptimizerCache`, `NewtonOptimizerState`). The family stays
# uniform, and nobody has to work out per struct whether a given bound happens to be one that costs.
# `test/named_tuple_parameters.jl` pins this.

# note that this is *not* `Tuple{Vararg{AT}} where {AT<:AbstractArray{T}}`, as Julia's
# diagonal rule would make that homogeneous, i.e. it would not allow a `NamedTuple` that
# stores e.g. a `StiefelManifold` and an ordinary `Matrix` at the same time.
const ArrayTuple{T} = Tuple{Vararg{AbstractArray{T}}}

const ArrayNamedTuple{T,S} = begin
    NamedTuple{S,<:ArrayTuple{T}}
end

"""
    ParameterContainer

The two shapes a set of parameters arrives in: a bare `NamedTuple` of arrays, or a
`NeuralNetworkParameters.NetworkParameters` holding one.

This is what the elementwise primitives of `src/optimizers/named_tuple_wrapper.jl` dispatch on. They
took `ArrayNamedTuple` alone until 0.6.0, so a container reached them and raised a `MethodError`
several frames into a solve rather than being turned away at the door.

The two are *not* interchangeable in shape. An `ArrayNamedTuple` is flat by construction — its values
are bounded by `AbstractArray{T}`, so a nested `NamedTuple` is not one — while a container is a tree
of layers. That is why the bodies walk with `mapparameters` rather than with `Base.map`: `map`
visits the entries of one level, which is the whole of a flat set and the *layers* of a nested one.

!!! note "`ParameterSet` is the wider name, and the one to reach for first"
    `T` means two different things across this union: for `ArrayNamedTuple{T}` it is a guarantee that
    every leaf is an `AbstractArray{T}` *and* that the set is flat, while for `NetworkParameters{T}` it
    is a promotion over leaves that may nest to any depth. So this alias means "flat and homogeneous,
    or nested and promoted", and a method written on it accepts a nested container while rejecting the
    nested plain `NamedTuple` describing the same network.

    That is why it is only used where `T` genuinely has to bind — beside a `Matrix{T}`, an `f̄::T`, a
    `b::T`. Everything else in this package takes `NeuralNetworkParameters.ParameterSet`, which is the
    same union without either bound and is what the rest of the ecosystem dispatches on:
    `AbstractNeuralNetworks`, `SymbolicNeuralNetworks` and `GeometricMachineLearning` all name it.

!!! info "Widening this union did not close issue #16"
    A method on this alias is still type piracy, and for both members. `ArrayNamedTuple` is an alias
    for `Base.NamedTuple`, which is the original complaint; `NetworkParameters` belongs to
    `NeuralNetworkParameters`, so a method pairing it with a generic from a third package — `l2norm`,
    `outer!`, `copyto!` — owns neither side either. Closing #16 needs the `NamedTuple` methods
    *removed*, and `GeometricMachineLearning` hands this package one bare layer `NamedTuple` per
    layer, so they stay.
"""
const ParameterContainer{T} = Union{ArrayNamedTuple{T},NetworkParameters{T}}

"""
    OptimizerSolution

A type alias for the solution of an optimizer: an `AbstractVector`, a [`Manifold`](@ref), a
`NamedTuple` of arrays, or a `NetworkParameters` holding one.

`NetworkParameters{T}` binds `T` from a direct type parameter rather than through a `Vararg` bound on
value types, so it is the cheapest member of this union to intersect. Note that it is the only member
whose `T` is a *promotion* over the leaves rather than a guarantee that every leaf is a `T`.
"""
const OptimizerSolution{T} = Union{AbstractVector{T},Manifold{T},ParameterContainer{T}}

const GradientArrayOrNamedTuple{T} = Union{AbstractArray{T},ParameterContainer{T}}

# see the remark on the diagonal rule above
const GlobalSectionTuple{T} = Tuple{Vararg{GlobalSection{T}}}

const GlobalSectionNamedTuple{T,X} = begin
    NamedTuple{X,<:GlobalSectionTuple{T}}
end

const GlobalSectionSingleOrNamedTuple{T} = Union{GlobalSection{T},GlobalSectionNamedTuple{T}}
