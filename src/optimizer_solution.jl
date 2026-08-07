const ArrayTuple{T} = Tuple{Vararg{AT}} where {AT<:AbstractArray{T}}

const ArrayNamedTuple{T,S} = begin
    NamedTuple{S,<:ArrayTuple{T}}
end

"""
    OptimizerSolution

A type alias for the solution of an optimizer, which can be either an `AbstractVector` or a [`Manifold`](@ref).
"""
const OptimizerSolution{T} = Union{AbstractVector{T},Manifold{T},ArrayNamedTuple{T}}

const GradientArrayOrNamedTuple{T} = Union{AbstractArray{T},ArrayNamedTuple{T}}

const GlobalSectionTuple{T} = Tuple{Vararg{GT}} where {GT<:GlobalSection{T}}

const GlobalSectionNamedTuple{T,X} = begin
    NamedTuple{X,<:GlobalSectionTuple{T}}
end

const GlobalSectionSingleOrNamedTuple{T} = Union{GlobalSection{T},GlobalSectionNamedTuple{T}}
