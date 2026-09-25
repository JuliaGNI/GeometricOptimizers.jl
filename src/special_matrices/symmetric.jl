@doc raw"""
    SymmetricMatrix(S::AbstractVector, n::Integer)

Instantiate a symmetric matrix with information stored in vector `S`.

A `SymmetricMatrix` ``A`` is a matrix ``A^T = A``.

Internally the `struct` saves a vector ``S`` of size ``n(n+1)\div2``. The conversion is done the following way: 
```math
[A]_{ij} = \begin{cases} S[( (i-1) i ) \div 2 + j] & \text{if $i\geq{}j$}\\ 
                         S[( (j-1) j ) \div 2 + i] & \text{else}. \end{cases}
```

So ``S`` stores a string of vectors taken from ``A``: ``S = [\tilde{a}_1, \tilde{a}_2, \ldots, \tilde{a}_n]`` with ``\tilde{a}_i = [[A]_{i1},[A]_{i2},\ldots,[A]_{ii}]``.

Also see [`SkewSymMatrix`](@ref), [`LowerTriangular`](@ref) and [`UpperTriangular`](@ref).

# Examples 
```jldoctest
using GeometricOptimizers
S = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SymmetricMatrix(S, 4)

# output

4×4 SymmetricMatrix{Int64, Vector{Int64}}:
 1  2  4   7
 2  3  5   8
 4  5  6   9
 7  8  9  10
```
"""
mutable struct SymmetricMatrix{T, AT <: AbstractVector{T}} <: AbstractMatrix{T}
    S::AT
    n::Int

    function SymmetricMatrix(S::AbstractVector, n::Integer)
        @assert length(S) == n*(n+1)÷2
        new{eltype(S), typeof(S)}(S, n)
    end
end

@doc raw"""
    SymmetricMatrix(A::AbstractMatrix)

Perform a projection and store the matrix in an efficient way (as a vector with ``n(n+1)/2`` entries).

If the constructor is called with a matrix as input it returns a symmetric matrix via the *projection*:
```math
A \mapsto \frac{1}{2}(A + A^T).
```

# Examples
```jldoctest
using GeometricOptimizers
M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
SymmetricMatrix(M)

# output

4×4 SymmetricMatrix{Float64, Vector{Float64}}:
 1.0   3.5   6.0   8.5
 3.5   6.0   8.5  11.0
 6.0   8.5  11.0  13.5
 8.5  11.0  13.5  16.0
```

# Extended help

Note that the constructor is designed in such a way that it always returns matrices of type `SymmetricMatrix{<:AbstractFloat}` when called with a matrix, even if this matrix is of type `AbstractMatrix{<:Integer}`.

If the user wishes to allocate a matrix `SymmetricMatrix{<:Integer}` then call

```julia
SymmetricMatrix(::AbstractVector, n::Integer)
```

Note that this is different from [`LowerTriangular`](@ref) and [`UpperTriangular`](@ref) as no projection takes place there.
"""
function SymmetricMatrix(A::AbstractMatrix{T}) where {T}
    S = map_to_S(A)
    SymmetricMatrix(S, size(A, 1))
end

# I'm not 100% sure this is the best solution (needed for broadcasting operations ...)
function Base.setindex!(A::SymmetricMatrix{T}, val::T, i::Int, j::Int) where {T}
    if i ≥ j
        A.S[i * (i - 1) ÷ 2 + j] = val
    else
        A.S[j * (j - 1) ÷ 2 + i] = val
    end
end

@kernel function assign_S_val_kernel!(S, A_sym, i)
    j = @index(Global)
    S[i * (i - 1) ÷ 2 + j] = A_sym[i, j]
end

# `transpose` and not `adjoint`, for the reason `map_to_Skew` in `skew_symmetric.jl` spells out: the
# set being projected onto is `{M : Mᵀ = M}`, which is what `getindex` reconstructs and what the
# docstring above states. The two agree for a real element type.
function map_to_S(A::AbstractMatrix{T}) where {T <: Number}
    n = size(A, 1)
    @assert size(A, 2) == n
    A_sym = T(0.5)*(A + transpose(A))
    backend = KernelAbstractions.get_backend(A)
    S = KernelAbstractions.zeros(backend, T, n*(n+1)÷2)
    assign_S_val! = assign_S_val_kernel!(backend)
    for i in 1:n
        assign_S_val!(S, A_sym, i, ndrange = i)
    end
    S
end

# see the comment on `map_to_Skew(::AbstractMatrix{<:Integer})`
function map_to_S(A::AbstractMatrix{T}) where {T <: Integer}
    map_to_S(float.(A))
end

# A symmetric matrix is its own *transpose*, and only a real one is its own adjoint. Bound to `Real`
# for the same reason `adjoint(::LowerTriangular{<:Real})` in `upper_triangular.jl` is: the bound
# does not reject a complex argument, it hands it to `LinearAlgebra`'s lazy `Adjoint`, which
# conjugates and is correct. Returning `A` unconditionally answered `A' == A` for a complex `A`,
# which is false.
function LinearAlgebra.Adjoint(A::SymmetricMatrix{<:Real})
    A
end

function Base.zero(A::SymmetricMatrix)
    SymmetricMatrix(zero(A.S), A.n)
end

# `similar` has to preserve the type: the optimizer caches allocate their scratch arrays with it and
# then require every one of them to have the same type as the parameter. The generic `AbstractArray`
# fallback returns a dense `Matrix` and makes the cache constructors inapplicable.
Base.similar(A::SymmetricMatrix) = SymmetricMatrix(similar(A.S), A.n)

# This fills the *storage*, so `fill!(A, val)` gives a matrix whose entries are all `val` — the
# diagonal included, since it is part of `S` for a symmetric matrix. The optimizer caches use it to
# poison scratch arrays with `NaN`.
Base.fill!(A::SymmetricMatrix, val) = (fill!(A.S, val); A)

function Base.getindex(A::SymmetricMatrix, i::Int, j::Int)
    if i ≥ j
        A.S[((i - 1) * i) ÷ 2 + j]
    else
        A.S[(j - 1) * j ÷ 2 + i]
    end
end

Base.parent(A::SymmetricMatrix) = A.S
Base.size(A::SymmetricMatrix) = (A.n, A.n)

# `src/ambiguities.jl` has the `+` and `-` methods that reach these two.
function _owned_add(A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n
    SymmetricMatrix(A.S + B.S, A.n)
end

function add!(C::SymmetricMatrix, A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function _owned_sub(A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n
    SymmetricMatrix(A.S - B.S, A.n)
end

function Base.:-(A::SymmetricMatrix)
    SymmetricMatrix(-A.S, A.n)
end

function Base.:*(A::SymmetricMatrix, α::Real)
    SymmetricMatrix(α*A.S, A.n)
end

Base.:*(α::Real, A::SymmetricMatrix) = A*α

# The backend-taking allocators mirror `SkewSymMatrix`'s method for method, as the rest of the two
# types do: a symmetric matrix is an optimizer parameter in exactly the same way, and
# `GeometricMachineLearning`'s SympNet and symplectic-attention layers are parametrized by both.
#
# No `n == 1` branch here, unlike `SkewSymMatrix`'s: this storage is `n(n+1)/2`, which is `1` at
# `n = 1` rather than `0`, so the length-zero case that guard is about does not arise.
function Base.zeros(backend::KernelAbstractions.Backend,
        ::Type{SymmetricMatrix{T}}, n::Int) where {T}
    _check_supported_eltype(backend, T)
    SymmetricMatrix(KernelAbstractions.zeros(backend, T, n*(n+1)÷2), n)
end

function Base.rand(rng::Random.AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{SymmetricMatrix{T}}, n::Integer) where {T}
    _check_supported_eltype(backend, T)
    S = KernelAbstractions.allocate(backend, T, n*(n+1)÷2)
    Random.rand!(rng, S)
    SymmetricMatrix(S, n)
end

function Base.rand(backend::KernelAbstractions.Backend,
        type::Type{SymmetricMatrix{T}}, n::Integer) where {T}
    rand(Random.default_rng(), backend, type, n)
end

function Base.zeros(::Type{SymmetricMatrix{T}}, n::Int) where {T}
    SymmetricMatrix(zeros(T, n*(n+1)÷2), n)
end

function Base.zeros(::Type{SymmetricMatrix}, n::Int)
    SymmetricMatrix(zeros(n*(n+1)÷2), n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SymmetricMatrix{T}}, n::Int) where {T}
    SymmetricMatrix(rand(rng, T, n*(n+1)÷2), n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SymmetricMatrix}, n::Int)
    SymmetricMatrix(rand(rng, n*(n+1)÷2), n)
end

function Base.rand(type::Type{SymmetricMatrix{T}}, n::Integer) where {T}
    rand(Random.default_rng(), type, n)
end

function Base.rand(type::Type{SymmetricMatrix}, n::Integer)
    rand(Random.default_rng(), type, n)
end

#these are Adam operations:
function scalar_add(A::SymmetricMatrix, δ::Real)
    SymmetricMatrix(A.S .+ δ, A.n)
end

#element-wise squares and square root (for Adam)
function ⊙²(A::SymmetricMatrix)
    SymmetricMatrix(A.S .^ 2, A.n)
end
function racᵉˡᵉ(A::SymmetricMatrix)
    SymmetricMatrix(sqrt.(A.S), A.n)
end
function /ᵉˡᵉ(A::SymmetricMatrix, B::SymmetricMatrix)
    @assert A.n == B.n
    SymmetricMatrix(A.S ./ B.S, A.n)
end

function LinearAlgebra.mul!(C::SymmetricMatrix, A::SymmetricMatrix, α::Real)
    mul!(C.S, A.S, α)
    C
end
LinearAlgebra.mul!(C::SymmetricMatrix, α::Real, A::SymmetricMatrix) = mul!(C, A, α)
LinearAlgebra.rmul!(C::SymmetricMatrix, α::Real) = mul!(C, C, α)

@kernel function symmetric_mat_mul_kernel!(
        C::AbstractMatrix{T}, S::AbstractVector{T}, B::AbstractMatrix{T}, n) where {T}
    i, j = @index(Global, NTuple)

    tmp_sum = zero(T)
    for k in 1:i
        tmp_sum += S[((i - 1) * i) ÷ 2 + k] * B[k, j]
    end
    for k in (i + 1):n
        tmp_sum += S[((k - 1) * k) ÷ 2 + i] * B[k, j]
    end
    C[i, j] = tmp_sum
end

# The product kernels. `src/ambiguities.jl` has the `*` and `mul!` methods that reach them.
function _lmul_into!(C::AbstractMatrix{T}, A::SymmetricMatrix{T}, B::AbstractMatrix{T}) where {T}
    @assert A.n == size(B, 1)
    @assert size(B, 2) == size(C, 2)
    @assert A.n == size(C, 1)
    backend = KernelAbstractions.get_backend(A.S)
    symmetric_mat_mul! = symmetric_mat_mul_kernel!(backend)
    symmetric_mat_mul!(C, A.S, B, A.n, ndrange = size(C))
    C
end

function _lmul(A::SymmetricMatrix{T}, B::AbstractMatrix{T}) where {T}
    backend = KernelAbstractions.get_backend(A.S)
    _lmul_into!(KernelAbstractions.allocate(backend, T, A.n, size(B, 2)), A, B)
end

# `transpose` and not `adjoint`, for the reason the counterpart in `skew_symmetric.jl` spells out:
# the identity rests on `Aᵀ = A`, so `(A·Bᵀ)ᵀ = B·Aᵀ = B·A`. Written `(A * B')'` it reads
# `B·Aᴴ = B·conj(A)`, which agrees only where `A` is real. A row vector reaches this method too, as
# the one column `transpose(x)`, for the reason the counterpart gives.
_rmul(B::AbstractMatrix{T}, A::SymmetricMatrix{T}) where {T} = transpose(A * transpose(B))

# see the comment on `_lmul(::SkewSymMatrix, ::AbstractVector)`: the vector goes through the
# matrix--matrix kernel as a single column, and the `n × 1` result is reshaped back to a vector
function _lmul(A::SymmetricMatrix{T}, b::AbstractVector{T}) where {T}
    vec(A * reshape(b, length(b), 1))
end

function Base.one(A::SymmetricMatrix{T}) where {T}
    unit_matrix(KernelAbstractions.get_backend(A.S), T, A.n)
end

function assign!(B::SymmetricMatrix{T}, C::SymmetricMatrix{T}) where {T}
    @assert B.n == C.n
    copyto!(B.S, C.S)

    nothing
end

function Base.copy(A::SymmetricMatrix)
    SymmetricMatrix(copy(A.S), A.n)
end

function Base.copyto!(A::SymmetricMatrix{T}, B::SymmetricMatrix{T}) where {T}
    @assert A.n == B.n
    copyto!(A.S, B.S)

    A
end

# define routines for generalizing ChainRulesCore to SymmetricMatrix 
function ChainRulesCore.ProjectTo(A::SymmetricMatrix)
    ProjectTo{SymmetricMatrix}(; symmetric = ProjectTo(A.S))
end
function (project::ProjectTo{SymmetricMatrix})(dA::AbstractMatrix)
    SymmetricMatrix(project.symmetric(map_to_S(dA)), size(dA, 2))
end
function (project::ProjectTo{SymmetricMatrix})(dA::SymmetricMatrix)
    SymmetricMatrix(project.symmetric(dA.S), dA.n)
end
