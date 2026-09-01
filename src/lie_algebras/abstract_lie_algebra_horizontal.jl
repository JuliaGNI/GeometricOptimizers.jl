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

@doc raw"""
    parent(B::AbstractLieAlgHorMatrix)

The tuple of blocks `B`'s free parameters are stored in — `(A, B)` for a
[`StiefelLieAlgHorMatrix`](@ref), `(B,)` for a [`GrassmannLieAlgHorMatrix`](@ref) — and *not* the
single array every other `parent` this package defines returns.

Every operation on a lift that is elementwise *in the free parameters* — as opposed to in the ambient
``N\times{}N`` matrix, which has no `setindex!` and counts each off-diagonal block twice — is written
once over this tuple rather than once per lift type. That is the four methods below, and `l2norm`,
and the `_difference!` / `_add!` / `_rac!` / `_div!` / `_square!` family in `named_tuple_wrapper.jl`.
They used to exist for the Stiefel lift alone, which is half of why a [`GrassmannManifold`](@ref)
could not be driven through an [`Optimizer`](@ref) at all (issue A11).

The docstring is attached to the *signature* and not to the bare `Base.parent`: the package also
defines `parent` for [`Manifold`](@ref), `SkewSymMatrix`, `SymmetricMatrix` and `AbstractTriangular`,
each of which returns the single array it wraps, and a signature-less docstring would be shown as the
general meaning of `parent` for all of them.
"""
Base.parent(::AbstractLieAlgHorMatrix)

function _add!(A::AbstractLieAlgHorMatrix{T}, B::AbstractLieAlgHorMatrix{T}) where {T}
    (foreach(_add!, parent(A), parent(B)); A)
end

function assign!(B::AbstractLieAlgHorMatrix{T}, C::AbstractLieAlgHorMatrix{T}) where {T}
    (foreach(assign!, parent(B), parent(C)); nothing)
end

@doc raw"""
    vec(B::AbstractLieAlgHorMatrix)

The free parameters of `B`, laid out end to end and lazily — *not* the ``N^2`` entries of the matrix
`B` presents itself as.

# Examples

```jldoctest
using GeometricOptimizers

A = SkewSymMatrix([1, ], 2)
B = [2 3; ]
B̄ = StiefelLieAlgHorMatrix(A, B, 3, 2)
B̄ |> vec

# output

vcat(1-element Vector{Int64}, 2-element Vector{Int64}):
 1
 2
 3
```

# Implementation

This is using `Vcat` from the package `LazyArrays`, so nothing is copied.
"""
Base.vec(B::AbstractLieAlgHorMatrix) = LazyArrays.Vcat(map(vec, parent(B))...)

@doc raw"""
    one(B::AbstractLieAlgHorMatrix)

The ``N\times{}N`` identity, built with a `KernelAbstractions` kernel.

`Base.one(::AbstractMatrix)` writes the diagonal in a scalar-indexed loop, which is what a GPU array
cannot serve; [`geodesic`](@ref) reaches this on every retraction. It existed for the Stiefel lift
only, so the Grassmann retraction was taking the scalar-indexed path — the same hazard issue A19
recorded for [`GeometricOptimizers.𝔄`](@ref), whose argument is a bare matrix and which reached
`Base.one` until the ``2n\times{}2n`` identities went through
[`GeometricOptimizers.unit_matrix`](@ref) as well.
"""
function Base.one(B::AbstractLieAlgHorMatrix{T}) where {T}
    unit_matrix(KernelAbstractions.get_backend(B), T, B.N)
end
