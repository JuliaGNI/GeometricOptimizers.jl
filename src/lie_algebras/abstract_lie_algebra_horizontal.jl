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

@doc raw"""
    *(B::AbstractLieAlgHorMatrix, C::AbstractMatrix)
    *(C::AbstractMatrix, B::AbstractLieAlgHorMatrix)
    *(B::AbstractLieAlgHorMatrix, c::AbstractVector)

The product, taken on the stored blocks rather than through `getindex`.

Without these the product falls through to the generic `AbstractMatrix` path, which asks the lift
for one entry at a time. That is scalar indexing, so it **cannot run on a device at all** — the
third wrapper in this package to meet that gap, after [`StiefelProjection`](@ref) and the two
[`GeometricOptimizers.AbstractTriangular`](@ref)s.

A lift is block ``\begin{pmatrix} A & -B^T \\ B & \mathbb{O} \end{pmatrix}``, with ``A`` absent for
a [`GrassmannLieAlgHorMatrix`](@ref), so the product against an ``N\times{}m`` matrix is three
block products — two for a Grassmann lift — and nothing else. **No kernel is needed and none is
written**, which is where this differs from the triangulars: those hold a packed vector their
`getindex` unpacks, while a lift holds ordinary blocks and a [`SkewSymMatrix`](@ref) that already
carries a kernel-backed product of its own. The per-type part is the first ``n`` rows,
`_hor_top_rows`, beside each concrete lift.

`*(C, B)` is written `-transpose(B * transpose(C))`, on the identity ``B^T = -B``; the vector form
goes through the matrix one as a single column. `transpose` and not `adjoint` wherever one appears,
for the reason the comment on `*(::AbstractMatrix, ::SkewSymMatrix)` in
`special_matrices/skew_symmetric.jl` gives at length: the identity is a statement about the
transpose, and `getindex` builds the off-diagonal blocks entrywise without conjugating. The two
agree on a real element type and disagree on a complex one.
"""
function Base.:*(B::AbstractLieAlgHorMatrix{T}, C::AbstractMatrix{T}) where {T}
    @assert B.N == size(C, 1)
    _check_same_backend(B, C)
    backend = KernelAbstractions.get_backend(B)

    D = KernelAbstractions.allocate(backend, T, B.N, size(C, 2))
    C₁ = @view C[1:(B.n), :]
    C₂ = @view C[(B.n + 1):(B.N), :]
    @views D[1:(B.n), :] .= _hor_top_rows(B, C₁, C₂)
    @views D[(B.n + 1):(B.N), :] .= B.B * C₁
    D
end

function Base.:*(C::AbstractMatrix{T}, B::AbstractLieAlgHorMatrix{T}) where {T}
    -transpose(B * transpose(C))
end

# A row vector on the left is the one shape the method above leaves unsettled: it stands off against
# `LinearAlgebra`'s own row-vector product, and neither wins. *A row vector meets an owned matrix* in
# `src/ambiguities.jl` gives the mechanism and lists every site. The body is the one above, so a row
# vector gets the answer that method gives every other matrix, and gets it the same cheap way:
# `transpose(x)` is one column, which reaches the block products as a single column instead of
# materializing the lift. One pair covers both lifts, because the method above is written on
# `AbstractLieAlgHorMatrix` too.
function Base.:*(x::Adjoint{T, <:AbstractVector}, B::AbstractLieAlgHorMatrix{T}) where {T}
    -transpose(B * transpose(x))
end
function Base.:*(x::Transpose{T, <:AbstractVector}, B::AbstractLieAlgHorMatrix{T}) where {T}
    -transpose(B * transpose(x))
end

# see the comment on `*(::SkewSymMatrix, ::AbstractVector)`: the vector goes through the
# matrix--matrix path as a single column, and the `N × 1` result is reshaped back to a vector
function Base.:*(B::AbstractLieAlgHorMatrix, c::AbstractVector{T}) where {T}
    vec(B * reshape(c, length(c), 1))
end

# Two lifts are the standoff the two methods above create between themselves, and rule 2 in
# `src/ambiguities.jl` decides it: the right-hand operand is materialized, spelled `B₂ * one(B₂)` as
# the triangulars and the symmetric matrices already spell it. The result is dense, as every
# tie-breaker's is.
function Base.:*(B₁::AbstractLieAlgHorMatrix{T}, B₂::AbstractLieAlgHorMatrix{T}) where {T}
    B₁ * (B₂ * one(B₂))
end
