@doc raw"""
    StrictlyUpperTriangular(S::AbstractVector, n::Int)

Build an upper-triangular matrix from a vector.

An upper-triangular matrix is an ``n\times{}n`` matrix that has zeros on the diagonal and on the lower triangular.

The data are stored in a vector ``S`` similarly to other matrices. See [`StrictlyLowerTriangular`](@ref), [`SkewSymMatrix`](@ref) and [`SymmetricMatrix`](@ref).

The struct has two fields: `S` and `n`. The first stores all the entries of the matrix in a sparse fashion (in a vector) and the second is the dimension ``n`` for ``A\in\mathbb{R}^{n\times{}n}``.

`adjoint` (`U'`) returns a [`StrictlyLowerTriangular`](@ref) built around the *same* storage vector, not a copy: `parent(U') === parent(U)` holds, so writing into the adjoint also writes into `U`. Reusing the storage transposes without conjugating, so this method is defined for a real element type only; a complex one falls through to `LinearAlgebra`'s lazy `Adjoint`, which conjugates and does not alias.

# Examples
```jldoctest
using GeometricOptimizers
S = [1, 2, 3, 4, 5, 6]
StrictlyUpperTriangular(S, 4)

# output

4×4 StrictlyUpperTriangular{Int64, Vector{Int64}}:
 0  1  2  4
 0  0  3  5
 0  0  0  6
 0  0  0  0
```
"""
mutable struct StrictlyUpperTriangular{T, AT <: AbstractVector{T}} <: AbstractTriangular{T}
    S::AT
    n::Int
end

@doc raw"""
    StrictlyUpperTriangular(A::AbstractMatrix)

Build an upper-triangular matrix from a matrix.

This is done by taking the upper right of that matrix.

# Examples 
```jldoctest
using GeometricOptimizers
M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
StrictlyUpperTriangular(M)

# output

4×4 StrictlyUpperTriangular{Int64, Vector{Int64}}:
 0  2  3   4
 0  0  7   8
 0  0  0  12
 0  0  0   0
```
"""
function StrictlyUpperTriangular(S::AbstractMatrix{T}) where {T}
    n = size(S, 1)
    @assert size(S, 2) == n
    S_vec = map_to_up(S)
    StrictlyUpperTriangular(S_vec, n)
end

# see the allocators in `lower_triangular.jl`
function Base.zeros(backend::KernelAbstractions.Backend,
        ::Type{StrictlyUpperTriangular{T}}, n::Integer) where {T}
    StrictlyUpperTriangular(_zeros(backend, T, n * (n - 1) ÷ 2), Int(n))
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{StrictlyUpperTriangular{T}}, n::Integer) where {T}
    StrictlyUpperTriangular(_rand(rng, backend, T, n * (n - 1) ÷ 2), Int(n))
end

function Base.getindex(A::StrictlyUpperTriangular{T}, i::Int, j::Int) where {T}
    if j == i
        return zero(T)
    end
    if j > i
        return A.S[(j - 2) * (j - 1) ÷ 2 + i]
    end
    return zero(T)
end

# Row `i` of a strictly upper-triangular matrix has its entries in columns `(i+1):n`, and the entry
# at `(i, k)` is packed at `(k-2)(k-1)/2 + i` -- the column indexes the block, which is what makes
# this kernel different from the lower-triangular one rather than a mirror of it.
@kernel function up_mat_mul_kernel!(
        C::AbstractMatrix{T}, S::AbstractVector{T}, B::AbstractMatrix{T}, n) where {T}
    i, j = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k in (i + 1):n
        tmp_sum += S[(k - 2) * (k - 1) ÷ 2 + i] * B[k, j]
    end
    C[i, j] = tmp_sum
end

mat_mul_kernel(::StrictlyUpperTriangular, backend) = up_mat_mul_kernel!(backend)

# The projection that builds the packed vector from a dense matrix, in one kernel over the whole
# `n × n` range: it reads `A` entrywise and writes only the strict upper triangle, so a call
# allocates the packed result and nothing else.
#
# Reading `A[i, j]` under the guard `j > i` and not `A[j, i]` through `transpose(A)`: the kernel is
# indexed by the entry it writes, and the guard is the transpose that the old form expressed by
# handing the kernel a lazy `Transpose` wrapper. `transpose` and not `adjoint`, and this constructor
# performs no projection, it reads the entries that are there — the same convention the old body
# documented, one kernel launch instead of `n - 1` of them.
@kernel function up_projection_kernel(S::AbstractVector{T}, A::AbstractMatrix) where {T}
    i, j = @index(Global, NTuple)
    if j > i
        S[(j - 2) * (j - 1) ÷ 2 + i] = A[i, j]
    end
end

function map_to_up(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n
    backend = KernelAbstractions.get_backend(A)
    S = KernelAbstractions.zeros(backend, T, n * (n - 1) ÷ 2)
    up_projection_kernel(backend)(S, A; ndrange = (n, n))
    S
end

# define routines for generalizing ChainRulesCore to StrictlyUpperTriangular 
function ChainRulesCore.ProjectTo(A::AT) where {AT <: StrictlyUpperTriangular}
    ProjectTo{AT}(; triang = ProjectTo(A.S))
end
function (project::ProjectTo{<:StrictlyUpperTriangular})(dA::AbstractMatrix)
    StrictlyUpperTriangular(project.triang(map_to_up(dA)), size(dA, 2))
end
function (project::ProjectTo{<:StrictlyUpperTriangular})(dA::StrictlyUpperTriangular)
    StrictlyUpperTriangular(project.triang(dA.S), dA.n)
end

# A type swap, not a wrapper: the result is this package's own `StrictlyUpperTriangular`, built around the
# *same* storage vector `A.S` rather than a copy, so `parent(A') === parent(A)` holds and a write
# through the adjoint writes `A` too.
#
# Bound to a real element type, because reusing the storage transposes without conjugating. On a
# complex element type that is `transpose`, not `adjoint`, and `*(B, A::AbstractTriangular)` is
# written as `(A' * B')'` — so an unbound method returned a silently wrong product. The bound does
# not reject a complex argument: it falls through to `LinearAlgebra`'s lazy `Adjoint`, which
# conjugates and is correct. Real element types keep the storage-sharing swap below.
function Base.adjoint(A::StrictlyLowerTriangular{<:Real})
    StrictlyUpperTriangular(A.S, A.n)
end

# As above, and bound to a real element type for the same reason.
function Base.adjoint(A::StrictlyUpperTriangular{<:Real})
    StrictlyLowerTriangular(A.S, A.n)
end
