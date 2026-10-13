@doc raw"""
    SkewSymMatrix(S::AbstractVector, n::Integer)

Instantiate a skew-symmetric matrix with information stored in vector `S`.

A skew-symmetric matrix ``A`` is a matrix ``A^T = -A``.

Internally the `struct` saves a vector ``S`` of size ``n(n-1)\div2``. The conversion is done the following way:
```math
[A]_{ij} = \begin{cases} 0                             & \text{if $i=j$} \\
                         S[( (i-2) (i-1) ) \div 2 + j] & \text{if $i>j$}\\
                         S[( (j-2) (j-1) ) \div 2 + i] & \text{else}. \end{cases}
```

So ``S`` stores a string of vectors taken from ``A``: ``S = [\tilde{a}_1, \tilde{a}_2, \ldots, \tilde{a}_n]`` with ``\tilde{a}_i = [[A]_{i1},[A]_{i2},\ldots,[A]_{i(i-1)}]``.

Also see [`SymmetricMatrix`](@ref), [`StrictlyLowerTriangular`](@ref) and [`StrictlyUpperTriangular`](@ref).

# Examples
```jldoctest
using GeometricOptimizers
S = [1, 2, 3, 4, 5, 6]
SkewSymMatrix(S, 4)

# output

4×4 SkewSymMatrix{Int64, Vector{Int64}}:
 0  -1  -2  -4
 1   0  -3  -5
 2   3   0  -6
 4   5   6   0
```
"""
mutable struct SkewSymMatrix{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Int

    function SkewSymMatrix(S::AbstractVector{T}, n::Integer) where {T}
        @assert length(S) == n * (n - 1) ÷ 2
        new{T, typeof(S)}(S, n)
    end
end

@doc raw"""
    SkewSymMatrix(A::AbstractMatrix)

Perform `0.5 * (A - transpose(A))` and store the matrix in an efficient way (as a vector with ``n(n-1)/2`` entries).

If the constructor is called with a matrix as input it returns a skew-symmetric matrix via the projection:
```math
A \mapsto \frac{1}{2}(A - A^T).
```

# Examples
```jldoctest
using GeometricOptimizers
M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
SkewSymMatrix(M)

# output

4×4 SkewSymMatrix{Float64, Vector{Float64}}:
 0.0  -1.5  -3.0  -4.5
 1.5   0.0  -1.5  -3.0
 3.0   1.5   0.0  -1.5
 4.5   3.0   1.5   0.0
```

# Extended help

Note that the constructor is designed in such a way that it always returns matrices of type `SkewSymMatrix{<:AbstractFloat}` when called with a matrix, even if this matrix is of type `AbstractMatrix{<:Integer}`.

If the user wishes to allocate a matrix `SkewSymMatrix{<:Integer}` then call:

```julia
SkewSymMatrix(::AbstractVector, n::Integer)
```

Note that this is different from [`StrictlyLowerTriangular`](@ref) and [`StrictlyUpperTriangular`](@ref) as no projection takes place there.
"""
function SkewSymMatrix(S::AbstractMatrix{T}) where {T}
    n = size(S, 1)
    @assert size(S, 2) == n
    S_vec = map_to_Skew(S)
    SkewSymMatrix(S_vec, n)
end

function return_element(S::AbstractVector{T}, i::Int, j::Int) where {T}
    if j == i
        zero(T)
    elseif i > j
        S[(i - 2) * (i - 1) ÷ 2 + j]
    else
        -S[(j - 2) * (j - 1) ÷ 2 + i]
    end
end

function Base.getindex(A::SkewSymMatrix, i::Int, j::Int)
    return_element(A.S, i, j)
end

Base.parent(A::SkewSymMatrix) = A.S
Base.size(A::SkewSymMatrix) = (A.n, A.n)

@kernel function addition_kernel!(C::AbstractMatrix, S::AbstractVector, B::AbstractMatrix)
    i, j = @index(Global, NTuple)
    C[i, j] = return_element(S, i, j) + B[i, j]
    nothing
end

# The sum kernels. `src/ambiguities.jl` has the `+` and `-` methods that reach them.
function _ladd(A::SkewSymMatrix{T}, B::AbstractMatrix{T}) where {T}
    @assert size(A) == size(B)
    backend = KernelAbstractions.get_backend(B)
    addition! = addition_kernel!(backend)
    C = KernelAbstractions.allocate(backend, T, size(A)...)
    addition!(C, A.S, B; ndrange = size(A))

    C
end

function _owned_add(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n
    SkewSymMatrix(A.S + B.S, A.n)
end

function add!(C::SkewSymMatrix, A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

# `_add!` and the other optimizer primitives act on `parent` of this type through `freeparameters`;
# they are in `optimizers/named_tuple_wrapper.jl`.

function _owned_sub(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n
    SkewSymMatrix(A.S - B.S, A.n)
end

function Base.:-(A::SkewSymMatrix)
    SkewSymMatrix(-A.S, A.n)
end

function Base.:*(A::SkewSymMatrix, α::Real)
    SkewSymMatrix(α * A.S, A.n)
end

Base.:*(α::Real, A::SkewSymMatrix) = A * α

# The two allocators every owned type has; `src/allocators.jl` has the chain that supplies `rng`,
# `backend` and `T` when a call leaves them out.
function Base.zeros(backend::KernelAbstractions.Backend,
        ::Type{SkewSymMatrix{T}}, n::Integer) where {T}
    SkewSymMatrix(_zeros(backend, T, n * (n - 1) ÷ 2), n)
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{SkewSymMatrix{T}}, n::Integer) where {T}
    SkewSymMatrix(_rand(rng, backend, T, n * (n - 1) ÷ 2), n)
end

function LinearAlgebra.mul!(C::SkewSymMatrix, A::SkewSymMatrix, α::Real)
    mul!(C.S, A.S, α)
    C
end
LinearAlgebra.mul!(C::SkewSymMatrix, α::Real, A::SkewSymMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::SkewSymMatrix, α::Real) = mul!(C, C, α)

# The product kernels. `src/ambiguities.jl` has the `*` and `mul!` methods that reach them. The
# in-place form is the one the others are written on: the retraction workspace writes the dense form
# of a lift's `A` block into a buffer it owns.
function _lmul_into!(C::AbstractMatrix{T}, A::SkewSymMatrix{T}, B::AbstractMatrix{T}) where {T}
    @assert A.n == size(B, 1)
    @assert size(B, 2) == size(C, 2)
    @assert A.n == size(C, 1)
    backend = KernelAbstractions.get_backend(A.S)

    skew_mat_mul! = skew_mat_mul_kernel!(backend)
    skew_mat_mul!(C, A.S, B, A.n, ndrange = size(C))
    C
end

function _lmul(A::SkewSymMatrix{T}, B::AbstractMatrix{T}) where {T}
    backend = KernelAbstractions.get_backend(A)
    _lmul_into!(KernelAbstractions.allocate(backend, T, A.n, size(B, 2)), A, B)
end

@kernel function skew_mat_mul_kernel!(
        C::AbstractMatrix{T}, S::AbstractVector{T}, B::AbstractMatrix{T}, n) where {T}
    i, j = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k in 1:(i - 1)
        tmp_sum += S[(i - 2) * (i - 1) ÷ 2 + k] * B[k, j]
    end
    for k in (i + 1):n
        tmp_sum += -S[(k - 2) * (k - 1) ÷ 2 + i] * B[k, j]
    end
    C[i, j] = tmp_sum
end

# `transpose` and not `adjoint`, on both operands. What makes this identity work is `Aᵀ = -A`,
# which is a statement about the transpose, so the two wrappers that undo each other around it have
# to be transposes as well: `(-A·Bᵀ)ᵀ = B·(-A)ᵀ = B·A`. Written `(-A * B')'` it reads
# `B·(-A)ᴴ = B·conj(A)`, which agrees only where `A` is real. The two are the same expression for a
# real element type, so nothing on the real path can tell them apart.
#
# The triangulars settle the same question the other way round, by binding `adjoint` to `Real` so
# that a complex argument falls through to `LinearAlgebra`. That works because their `adjoint`
# shares storage; this type has no such method, so the transpose belongs in the product.
#
# The minus sits outside the product rather than on `A`, which is arithmetically the same and much
# cheaper: `-A` builds a whole second packed vector before the kernel runs, of the order of `n²`
# entries against one column.
#
# A row vector reaches this method too, and gets it the same cheap way: `transpose(x)` is one
# column, which reaches the kernel as a single column instead of materializing `A`. It is a `Vector`
# for a real element type and an `n×1` wrapper for a complex one — either way one column, so the two
# return the same values on different backings.
function _rmul(B::AbstractMatrix{T}, A::SkewSymMatrix{T}) where {T}
    -transpose(A * transpose(B))
end

# The kernel this reaches is a matrix--matrix one, so the vector goes through it as a single column
# -- and the `n × 1` result is reshaped back, because a matrix times a vector is a vector. `vec`
# reshapes rather than copies, so the second step shares the kernel's buffer and copies no data.
function _lmul(A::SkewSymMatrix, b::AbstractVector{T}) where {T}
    vec(A * reshape(b, length(b), 1))
end

function Base.one(A::SkewSymMatrix{T}) where {T}
    unit_matrix(KernelAbstractions.get_backend(A.S), T, A.n)
end

# The dense form of a skew block, allocated. `_dense_skew!` in
# `retractions/retraction_workspace.jl` writes the same matrix into a buffer the caller owns, and it
# is the one a workspace reuses; this one allocates the result.
function _dense_skew(B::SkewSymMatrix{T}) where {T}
    backend = KernelAbstractions.get_backend(B.S)
    C = KernelAbstractions.allocate(backend, T, B.n, B.n)
    _dense_skew!(C, B, unit_matrix(B))

    C
end

# `*(::SkewSymMatrix, ::SkewSymMatrix)` is a product of two skew matrices, which is a general
# matrix, so the right operand has to be materialised densely first. `one(A2) * A2` was the obvious
# spelling and builds a second `N × N` identity to do it; `_dense_skew` writes the same matrix in
# one pass over the packed vector, so the product costs the destination and nothing else.
function Base.:*(A1::SkewSymMatrix{T}, A2::SkewSymMatrix{T}) where {T}
    A1 * _dense_skew(A2)
end

function Base.zero(A::SkewSymMatrix)
    SkewSymMatrix(zero(A.S), A.n)
end

function KernelAbstractions.get_backend(A::SkewSymMatrix)
    KernelAbstractions.get_backend(A.S)
end

function assign!(B::SkewSymMatrix{T}, C::SkewSymMatrix{T}) where {T}
    @assert B.n == C.n
    copyto!(B.S, C.S)
end

function Base.copy(A::SkewSymMatrix)
    SkewSymMatrix(copy(A.S), A.n)
end

# see the comment on `similar(::SymmetricMatrix)`
Base.similar(A::SkewSymMatrix) = SkewSymMatrix(similar(A.S), A.n)

@kernel function skew_projection_kernel(S::AbstractVector{T}, A::AbstractMatrix) where {T}
    i, j = @index(Global, NTuple)
    if i > j
        S[(i - 2) * (i - 1) ÷ 2 + j] = T(0.5) * (A[i, j] - A[j, i])
    end
end

# `transpose` and not `adjoint`, because the set being projected onto is `{M : Mᵀ = -M}` — which is
# what `getindex` reconstructs, and what the docstring above states. `(A - Aᴴ)/2` is the projection
# onto the skew-*Hermitian* matrices, and its strict lower triangle stored here gives a matrix that
# is neither that projection nor `(A - Aᵀ)/2`. The two agree for a real element type.
#
# One kernel over the whole `n × n` range, reading `A` entrywise and writing the packed vector, so a
# call allocates the packed result and nothing else. The dense symmetrisation that a broadcast would
# build is a full `n × n` matrix, and this projection reads every entry twice and writes half of
# them, so materialising it costs a temporary the size of the argument and a pass over it for a
# result half that size. `map_to_Skew!` is the same kernel into a `SkewSymMatrix` that already
# exists, which is what a `global_rep!` into a cache buffer uses.
function map_to_Skew(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n
    backend = KernelAbstractions.get_backend(A)
    S = _zeros(backend, T, n * (n - 1) ÷ 2)
    skew_projection_kernel(backend)(S, A; ndrange = (n, n))
    S
end

function map_to_Skew!(S::SkewSymMatrix{T}, A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n
    @assert S.n == n
    skew_projection_kernel(KernelAbstractions.get_backend(A))(S.S, A; ndrange = (n, n))
    S
end

# The projection halves a difference, so an integer matrix has to become a float one first. `float`
# and not a width chosen here: it is the function that answers "the float type this integer widens
# to", it gives `Float64` for every fixed-width integer and `BigFloat` for a `BigInt`, and it is
# what `zeros` and `rand` already follow.
function map_to_Skew(A::AbstractMatrix{T}) where {T <: Integer}
    map_to_Skew(float.(A))
end

function Base.copyto!(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n
    copyto!(A.S, B.S)
    A
end

# this fills the *storage*: `fill!(A, val)` gives a matrix whose strict lower triangle is `val`, whose
# strict upper triangle is `-val` and whose diagonal stays zero. A skew-symmetric matrix cannot hold a
# constant, and this is the only sensible reading of `fill!` for it. The optimizer caches use it to
# poison scratch arrays with `NaN`, where the sign does not matter.
Base.fill!(A::SkewSymMatrix, val) = (fill!(A.S, val); A)

function _round(A::SkewSymMatrix; kwargs...)
    SkewSymMatrix(_round(A.S; kwargs...), A.n)
end

function _round(A::AbstractArray; kwargs...)
    round.(A; kwargs...)
end

# define routines for generalizing ChainRulesCore to SkewSymMatrix
function ChainRulesCore.ProjectTo(A::SkewSymMatrix)
    ProjectTo{SkewSymMatrix}(; skew_sym = ProjectTo(A.S))
end
function (project::ProjectTo{SkewSymMatrix})(dA::AbstractMatrix)
    SkewSymMatrix(project.skew_sym(map_to_Skew(dA)), size(dA, 2))
end
function (project::ProjectTo{SkewSymMatrix})(dA::SkewSymMatrix)
    SkewSymMatrix(project.skew_sym(dA.S), dA.n)
end
