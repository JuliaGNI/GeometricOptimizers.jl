@doc raw"""
    StrictlyLowerTriangular(S::AbstractVector, n::Int)

Build a lower-triangular matrix from a vector.

A lower-triangular matrix is an ``n\times{}n`` matrix that has zeros on the diagonal and on the upper triangular.

The data are stored in a vector ``S`` similarly to other matrices. See [`StrictlyUpperTriangular`](@ref), [`SkewSymMatrix`](@ref) and [`SymmetricMatrix`](@ref).

The struct has two fields: `S` and `n`. The first stores all the entries of the matrix in a sparse fashion (in a vector) and the second is the dimension ``n`` for ``A\in\mathbb{R}^{n\times{}n}``.

`adjoint` (`L'`) returns a [`StrictlyUpperTriangular`](@ref) built around the *same* storage vector, not a copy: `parent(L') === parent(L)` holds, so writing into the adjoint also writes into `L`. Reusing the storage transposes without conjugating, so this method is defined for a real element type only; a complex one falls through to `LinearAlgebra`'s lazy `Adjoint`, which conjugates and does not alias.

# Examples
```jldoctest
using GeometricOptimizers
S = [1, 2, 3, 4, 5, 6]
StrictlyLowerTriangular(S, 4)

# output

4×4 StrictlyLowerTriangular{Int64, Vector{Int64}}:
 0  0  0  0
 1  0  0  0
 2  3  0  0
 4  5  6  0
```
"""
mutable struct StrictlyLowerTriangular{T, AT <: AbstractVector{T}} <: AbstractTriangular{T}
    S::AT
    n::Int
end

@doc raw"""
    StrictlyLowerTriangular(A::AbstractMatrix)

Build a lower-triangular matrix from a matrix.

This is done by taking the lower left of that matrix.

# Examples 
```jldoctest
using GeometricOptimizers
M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
StrictlyLowerTriangular(M)

# output

4×4 StrictlyLowerTriangular{Int64, Vector{Int64}}:
  0   0   0  0
  5   0   0  0
  9  10   0  0
 13  14  15  0
```
"""
function StrictlyLowerTriangular(S::AbstractMatrix{T}) where {T}
    n = size(S, 1)
    @assert size(S, 2) == n
    S_vec = map_to_lo(S)
    StrictlyLowerTriangular(S_vec, n)
end

# The two allocators; `src/allocators.jl` has the chain. `Type{StrictlyLowerTriangular{T}}` and not
# `Type{<:…}`, so that a storage type the backend does not give is a `MethodError`.
function Base.zeros(backend::KernelAbstractions.Backend,
        ::Type{StrictlyLowerTriangular{T}}, n::Integer) where {T}
    StrictlyLowerTriangular(_zeros(backend, T, n * (n - 1) ÷ 2), Int(n))
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{StrictlyLowerTriangular{T}}, n::Integer) where {T}
    StrictlyLowerTriangular(_rand(rng, backend, T, n * (n - 1) ÷ 2), Int(n))
end

function Base.getindex(A::StrictlyLowerTriangular{T}, i::Int, j::Int) where {T}
    if j == i
        return zero(T)
    end
    if i > j
        return A.S[(i - 2) * (i - 1) ÷ 2 + j]
    end
    return zero(T)
end

# Row `i` of a strictly lower-triangular matrix has its entries in columns `1:(i-1)`, packed at
# `(i-2)(i-1)/2 + k`, which is the same arithmetic `getindex` above uses. See the docstring on
# `*(::AbstractTriangular, ::AbstractMatrix)` for why the product needs a kernel at all.
#
# `n` goes unused here. It is taken so that this kernel and `up_mat_mul_kernel!`, which does need it,
# share one launch signature — that is what lets `mat_mul_kernel` pick between them.
@kernel function lo_mat_mul_kernel!(
        C::AbstractMatrix{T}, S::AbstractVector{T}, B::AbstractMatrix{T}, n) where {T}
    i, j = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k in 1:(i - 1)
        tmp_sum += S[(i - 2) * (i - 1) ÷ 2 + k] * B[k, j]
    end
    C[i, j] = tmp_sum
end

mat_mul_kernel(::StrictlyLowerTriangular, backend) = lo_mat_mul_kernel!(backend)

function map_to_lo(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n
    backend = KernelAbstractions.get_backend(A)
    S = KernelAbstractions.zeros(backend, T, n * (n - 1) ÷ 2)
    assign_Skew_val! = assign_Skew_val_kernel!(backend)
    for i in 2:n
        assign_Skew_val!(S, A, i, ndrange = (i - 1))
    end
    S
end

# define routines for generalizing ChainRulesCore to StrictlyLowerTriangular 
function ChainRulesCore.ProjectTo(A::AT) where {AT <: StrictlyLowerTriangular}
    ProjectTo{AT}(; triang = ProjectTo(A.S))
end
function (project::ProjectTo{<:StrictlyLowerTriangular})(dA::AbstractMatrix)
    StrictlyLowerTriangular(project.triang(map_to_lo(dA)), size(dA, 2))
end
function (project::ProjectTo{<:StrictlyLowerTriangular})(dA::StrictlyLowerTriangular)
    StrictlyLowerTriangular(project.triang(dA.S), dA.n)
end
