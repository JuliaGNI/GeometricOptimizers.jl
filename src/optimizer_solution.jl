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
