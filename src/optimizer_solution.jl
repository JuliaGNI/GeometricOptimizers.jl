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
    OptimizerSolution

A type alias for the solution of an optimizer, which can be either an `AbstractVector` or a [`Manifold`](@ref).
"""
const OptimizerSolution{T} = Union{AbstractVector{T},Manifold{T},ArrayNamedTuple{T}}

const GradientArrayOrNamedTuple{T} = Union{AbstractArray{T},ArrayNamedTuple{T}}

# see the remark on the diagonal rule above
const GlobalSectionTuple{T} = Tuple{Vararg{GlobalSection{T}}}

const GlobalSectionNamedTuple{T,X} = begin
    NamedTuple{X,<:GlobalSectionTuple{T}}
end

const GlobalSectionSingleOrNamedTuple{T} = Union{GlobalSection{T},GlobalSectionNamedTuple{T}}
