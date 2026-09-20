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

Also see [`SymmetricMatrix`](@ref), [`LowerTriangular`](@ref) and [`UpperTriangular`](@ref).

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

Note that this is different from [`LowerTriangular`](@ref) and [`UpperTriangular`](@ref) as no projection takes place there.
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

function Base.:+(A::SkewSymMatrix{T}, B::AbstractMatrix{T}) where {T}
    @assert size(A) == size(B)
    backend = KernelAbstractions.get_backend(B)
    addition! = addition_kernel!(backend)
    C = KernelAbstractions.allocate(backend, T, size(A)...)
    addition!(C, A.S, B; ndrange = size(A))

    C
end

Base.:+(B::AbstractMatrix, A::SkewSymMatrix) = A + B

function Base.:+(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n
    SkewSymMatrix(A.S + B.S, A.n)
end

function add!(C::SkewSymMatrix, A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

# `_add!` and the other optimizer primitives for this type are generic over `VectorStorageMatrix`,
# next to the rest of them in `optimizers/named_tuple_wrapper.jl`.

function Base.:-(A::SkewSymMatrix, B::SkewSymMatrix)
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

# The `n == 1` arm below -- `allocate` where every other length takes `zeros` -- is kept although no
# backend reachable here needs it, and it is deliberately not deleted. A `1x1` skew-symmetric matrix
# stores nothing, so what the arm avoids is `KernelAbstractions.zeros` at length zero. Measured:
# `CPU()` returns a `(0,)` array from both
# `zeros` and `allocate`, and `MetalBackend()` does too. Neither is therefore the case it guards.
# Nobody here has a CUDA device, and an older `KernelAbstractions` is the likely reason it exists.
# Two backends' worth of evidence is not enough to remove a guard -- find the backend or the
# version that failed first. `map_to_Skew` carries the same branch for the same reason.
function Base.zeros(backend::KernelAbstractions.Backend, ::Type{SkewSymMatrix{T}}, n::Int) where {T}
    _check_supported_eltype(backend, T)
    zero_vec = if n != 1
        KernelAbstractions.zeros(backend, T, n * (n - 1) ÷ 2)
    else
        KernelAbstractions.allocate(backend, T, n * (n - 1) ÷ 2)
    end
    SkewSymMatrix(zero_vec, n)
end

function Base.zeros(::Type{SkewSymMatrix{T}}, n::Int) where {T}
    SkewSymMatrix(zeros(T, n * (n - 1) ÷ 2), n)
end

# `SkewSymMatrix` is exported, so `zeros(SkewSymMatrix, n)` is public API and defaults to
# `Float64` just like `zeros(n)` does. It has to be kept alongside the parametric method above
# and not replaced by it: without it `zeros(SkewSymMatrix, n)` falls through to
# `Base.zeros(::Type, ::Int)`, which throws `MethodError: no method matching
# zero(::Type{SkewSymMatrix})`. `zeros(::Type{StiefelLieAlgHorMatrix}, N, n)` calls it, too.
Base.zeros(::Type{SkewSymMatrix}, n::Int) = zeros(SkewSymMatrix{Float64}, n)

function Base.rand(rng::Random.AbstractRNG, ::Type{SkewSymMatrix{T}}, n::Int) where {T}
    SkewSymMatrix(rand(rng, T, n * (n - 1) ÷ 2), n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SkewSymMatrix}, n::Int)
    SkewSymMatrix(rand(rng, n * (n - 1) ÷ 2), n)
end

function Base.rand(type::Type{SkewSymMatrix{T}}, n::Integer) where {T}
    rand(Random.default_rng(), type, n)
end

function Base.rand(type::Type{SkewSymMatrix}, n::Integer)
    rand(Random.default_rng(), type, n)
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        type::Type{SkewSymMatrix{T}}, n::Integer) where {T}
    _check_supported_eltype(backend, T)
    S = KernelAbstractions.allocate(backend, T, n * (n - 1) ÷ 2)
    Random.rand!(rng, S)
    SkewSymMatrix(S, n)
end

function Base.rand(backend::KernelAbstractions.Backend, type::Type{SkewSymMatrix{T}}, n::Integer) where {T}
    rand(Random.default_rng(), backend, type, n)
end

#these are Adam operations:
function scalar_add(A::SkewSymMatrix, δ::Real)
    SkewSymMatrix(A.S .+ δ, A.n)
end

#element-wise squares and square root (for Adam)
function ⊙²(A::SkewSymMatrix)
    SkewSymMatrix(A.S .^ 2, A.n)
end
function racᵉˡᵉ(A::SkewSymMatrix)
    SkewSymMatrix(sqrt.(A.S), A.n)
end
function /ᵉˡᵉ(A::SkewSymMatrix, B::SkewSymMatrix)
    @assert A.n == B.n
    SkewSymMatrix(A.S ./ B.S, A.n)
end

function LinearAlgebra.mul!(C::SkewSymMatrix, A::SkewSymMatrix, α::Real)
    mul!(C.S, A.S, α)
    C
end
LinearAlgebra.mul!(C::SkewSymMatrix, α::Real, A::SkewSymMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::SkewSymMatrix, α::Real) = mul!(C, C, α)

# The in-place form, and the one `*` below is written on — the shape `mul!(C, ::SymmetricMatrix,
# ::AbstractMatrix)` already has in `symmetric.jl`. It exists because the retraction workspace needs
# the dense form of a lift's `A` block written into a buffer it owns rather than returned in a fresh
# one, and because a structured matrix that has a `*` and no `mul!` makes a caller who has a
# destination allocate anyway.
function LinearAlgebra.mul!(C::AbstractMatrix, A::SkewSymMatrix, B::AbstractMatrix)
    @assert A.n == size(B, 1)
    @assert size(B, 2) == size(C, 2)
    @assert A.n == size(C, 1)
    backend = KernelAbstractions.get_backend(A.S)

    skew_mat_mul! = skew_mat_mul_kernel!(backend)
    skew_mat_mul!(C, A.S, B, A.n, ndrange = size(C))
    C
end

function Base.:*(A::SkewSymMatrix{T}, B::AbstractMatrix{T}) where {T}
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.allocate(backend, T, A.n, size(B, 2))
    LinearAlgebra.mul!(C, A, B)
    C
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
# cheaper: `-A` builds a whole second packed vector before the kernel runs. Measured at `n = 400`
# against one column, `-transpose(A * transpose(x))` allocates 7 520 B where
# `transpose(-A * transpose(x))` allocates 642 944 B, for the same values.
function Base.:*(B::AbstractMatrix{T}, A::SkewSymMatrix{T}) where {T}
    -transpose(A * transpose(B))
end

# A row vector on the left is the one shape the method above leaves unsettled: it stands off against
# `LinearAlgebra`'s own row-vector product, and neither wins. *A row vector meets an owned matrix* in
# `src/ambiguities.jl` gives the mechanism and lists every site. The body is the one above, so a row
# vector gets the answer that method gives every other matrix, including its `transpose`, and gets
# it the same cheap way: `transpose(x)` is one column, which reaches the kernel as a single column
# instead of materializing `A`. It is a `Vector` for a real element type and an `n×1` wrapper for a
# complex one -- either way one column, so the two return the same values on different backings.
# `T` is bound in both slots because the method above binds it there; free, these would not be
# contained in it and would separate nothing.
function Base.:*(x::Adjoint{T, <:AbstractVector}, A::SkewSymMatrix{T}) where {T}
    -transpose(A * transpose(x))
end
function Base.:*(x::Transpose{T, <:AbstractVector}, A::SkewSymMatrix{T}) where {T}
    -transpose(A * transpose(x))
end

# The kernel this reaches is a matrix--matrix one, so the vector goes through it as a single column
# -- and the `n × 1` result is reshaped back, because a matrix times a vector is a vector. `vec`
# reshapes rather than copies, so the second step shares the kernel's buffer and copies no data.
function Base.:*(A::SkewSymMatrix, b::AbstractVector{T}) where {T}
    vec(A * reshape(b, length(b), 1))
end

function Base.one(A::SkewSymMatrix{T}) where {T}
    unit_matrix(KernelAbstractions.get_backend(A.S), T, A.n)
end

# the first matrix is multiplied onto A2 in order for it to not be SkewSymMatrix!
function Base.:*(A1::SkewSymMatrix{T}, A2::SkewSymMatrix{T}) where {T}
    A1 * (one(A2) * A2)
end

@doc raw"""
    vec(A)

Output the associated vector of `A`.

# Examples

```jldoctest
using GeometricOptimizers

M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
SkewSymMatrix(M) |> vec

# output

6-element Vector{Float64}:
 1.5
 3.0
 1.5
 4.5
 3.0
 1.5
```
"""
function Base.vec(A::SkewSymMatrix)
    A.S
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

@kernel function assign_Skew_val_kernel!(S, A_skew, i)
    j = @index(Global)
    S[((i - 2) * (i - 1) ÷ 2 + j)] = A_skew[i, j]
end

# `transpose` and not `adjoint`, because the set being projected onto is `{M : Mᵀ = -M}` — which is
# what `getindex` reconstructs, and what the docstring above states. `(A - Aᴴ)/2` is the projection
# onto the skew-*Hermitian* matrices, and its strict lower triangle stored here gives a matrix that
# is neither that projection nor `(A - Aᵀ)/2`. The two agree for a real element type.
function map_to_Skew(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n
    A_skew = T(0.5) * (A - transpose(A))
    backend = KernelAbstractions.get_backend(A)
    # the `n != 1` branch of `zeros(::Backend, ::Type{SkewSymMatrix{T}}, n)` above, and the comment
    # there says why it stays
    S = if n != 1
        KernelAbstractions.zeros(backend, T, n * (n - 1) ÷ 2)
    else
        KernelAbstractions.allocate(backend, T, n * (n - 1) ÷ 2)
    end
    assign_Skew_val! = assign_Skew_val_kernel!(backend)
    for i in 2:n
        assign_Skew_val!(S, A_skew, i, ndrange = (i - 1))
    end
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
