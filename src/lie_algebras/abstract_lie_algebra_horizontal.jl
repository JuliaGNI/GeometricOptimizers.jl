@doc raw"""
    AbstractLieAlgHorMatrix <: AbstractMatrix

`AbstractLieAlgHorMatrix` is a supertype for various horizontal components of Lie algebras. We usually call this ``\mathfrak{g}^\mathrm{hor}``.

See [`StiefelLieAlgHorMatrix`](@ref) and [`GrassmannLieAlgHorMatrix`](@ref) for concrete examples.
"""
abstract type AbstractLieAlgHorMatrix{T} <: AbstractMatrix{T} end

@doc raw"""
    manifold_type(B::AbstractLieAlgHorMatrix)

The manifold a retraction of `B` lands on.

``\mathfrak{g}^\mathrm{hor}`` is the horizontal component of the Lie algebra *of a specific
homogeneous space*, so the lift already determines where the retraction maps to. This is what lets
[`geodesic`](@ref) and [`cayley`](@ref) be written once for both manifolds rather than twice each.
"""
function manifold_type end
