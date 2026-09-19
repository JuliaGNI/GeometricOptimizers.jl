@doc raw"""
    UpperTriangular(S::AbstractVector, n::Int)

Build an upper-triangular matrix from a vector.

An upper-triangular matrix is an ``n\times{}n`` matrix that has zeros on the diagonal and on the lower triangular.

The data are stored in a vector ``S`` similarly to other matrices. See [`LowerTriangular`](@ref), [`SkewSymMatrix`](@ref) and [`SymmetricMatrix`](@ref).

The struct has two fields: `S` and `n`. The first stores all the entries of the matrix in a sparse fashion (in a vector) and the second is the dimension ``n`` for ``A\in\mathbb{R}^{n\times{}n}``.

`adjoint` (`U'`) returns a [`LowerTriangular`](@ref) built around the *same* storage vector, not a copy: `parent(U') === parent(U)` holds, so writing into the adjoint also writes into `U`. Reusing the storage transposes without conjugating, so this method is defined for a real element type only; a complex one falls through to `LinearAlgebra`'s lazy `Adjoint`, which conjugates and does not alias.

# Examples
```jldoctest
using GeometricOptimizers
S = [1, 2, 3, 4, 5, 6]
UpperTriangular(S, 4)

# output

4×4 UpperTriangular{Int64, Vector{Int64}}:
 0  1  2  4
 0  0  3  5
 0  0  0  6
 0  0  0  0
```
"""
mutable struct UpperTriangular{T, AT <: AbstractVector{T}} <: AbstractTriangular{T}
    S::AT
    n::Int
end

@doc raw"""
    UpperTriangular(A::AbstractMatrix)

Build an upper-triangular matrix from a matrix.

This is done by taking the upper right of that matrix.

# Examples 
```jldoctest
using GeometricOptimizers
M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
UpperTriangular(M)

# output

4×4 UpperTriangular{Int64, Vector{Int64}}:
 0  2  3   4
 0  0  7   8
 0  0  0  12
 0  0  0   0
```
"""
function UpperTriangular(S::AbstractMatrix{T}) where {T}
    n = size(S, 1)
    @assert size(S, 2) == n
    S_vec = map_to_up(S)
    UpperTriangular(S_vec, n)
end

function Base.getindex(A::UpperTriangular{T}, i::Int, j::Int) where {T}
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

mat_mul_kernel(::UpperTriangular, backend) = up_mat_mul_kernel!(backend)

function map_to_up(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n
    backend = KernelAbstractions.get_backend(A)
    S = KernelAbstractions.zeros(backend, T, n * (n - 1) ÷ 2)
    assign_Skew_val! = assign_Skew_val_kernel!(backend)
    # `transpose` and not `adjoint`. The kernel is the one `map_to_low` uses, which reads the strict
    # *lower* triangle, so the upper triangle is reached by swapping the indices — a transpose, and
    # nothing more. `adjoint` conjugates on the way, which stores a triangle the docstring does not
    # promise: this constructor performs no projection, it reads the entries that are there. The two
    # are the same operation on a real element type.
    for i in 2:n
        assign_Skew_val!(S, transpose(A), i, ndrange = (i - 1))
    end
    S
end

# define routines for generalizing ChainRulesCore to UpperTriangular 
function ChainRulesCore.ProjectTo(A::AT) where {AT <: UpperTriangular}
    ProjectTo{AT}(; triang = ProjectTo(A.S))
end
function (project::ProjectTo{<:UpperTriangular})(dA::AbstractMatrix)
    UpperTriangular(project.triang(map_to_up(dA)), size(dA, 2))
end
function (project::ProjectTo{<:UpperTriangular})(dA::UpperTriangular)
    UpperTriangular(project.triang(dA.S), dA.n)
end

# A type swap, not a wrapper: the result is this package's own `UpperTriangular`, built around the
# *same* storage vector `A.S` rather than a copy, so `parent(A') === parent(A)` holds and a write
# through the adjoint writes `A` too.
#
# Bound to a real element type, because reusing the storage transposes without conjugating. On a
# complex element type that is `transpose`, not `adjoint`, and `*(B, A::AbstractTriangular)` is
# written as `(A' * B')'` — so an unbound method returned a silently wrong product. The bound does
# not reject a complex argument: it falls through to `LinearAlgebra`'s lazy `Adjoint`, which
# conjugates and is correct. Real element types keep the storage-sharing swap below.
function Base.adjoint(A::LowerTriangular{<:Real})
    UpperTriangular(A.S, A.n)
end

# As above, and bound to a real element type for the same reason.
function Base.adjoint(A::UpperTriangular{<:Real})
    LowerTriangular(A.S, A.n)
end
