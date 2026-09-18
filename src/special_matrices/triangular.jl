@doc raw"""
    AbstractTriangular

See [`UpperTriangular`](@ref) and [`LowerTriangular`](@ref).
"""
abstract type AbstractTriangular{T} <: AbstractMatrix{T} end

Base.parent(A::AbstractTriangular) = A.S
Base.size(A::AbstractTriangular) = (A.n, A.n)

function Base.:+(A::AT, B::AT) where {AT <: AbstractTriangular}
    @assert A.n == B.n
    AT(A.S + B.S, A.n)
end

function add!(C::AT, A::AT, B::AT) where {AT <: AbstractTriangular}
    @assert A.n == B.n == C.n
    add!(C.S, A.S, B.S)
end

function Base.:-(A::AT, B::AT) where {AT <: AbstractTriangular}
    @assert A.n == B.n
    AT(A.S - B.S, A.n)
end

function Base.:-(A::AT) where {AT <: AbstractTriangular}
    AT(-A.S, A.n)
end

function Base.:*(A::AT, α::Real) where {AT <: AbstractTriangular}
    AT(α * A.S, A.n)
end

Base.:*(α::Real, A::AT) where {AT <: AbstractTriangular} = A * α

function Base.zeros(backend::KernelAbstractions.Backend, ::Type{AT},
        n::Int) where {T, AT <: AbstractTriangular{T}}
    # Base.typename(AT).wrapper strips the type parameters, giving the bare constructor
    # (UpperTriangular or LowerTriangular) without a call into the evaluator.
    Base.typename(AT).wrapper(KernelAbstractions.zeros(backend, T, n*(n-1)÷2), n)
end

function Base.zeros(::Type{AT}, n::Int) where {T, AT <: AbstractTriangular{T}}
    zeros(CPU(), AT, n)
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{AT}, n::Integer) where {T, AT <: AbstractTriangular{T}}
    S = KernelAbstractions.allocate(backend, T, n*(n-1)÷2)
    Random.rand!(rng, S)
    Base.typename(AT).wrapper(S, n)
end

function Base.rand(rng::Random.AbstractRNG, type::Type{AT}, n::Int) where {
        T, AT <: AbstractTriangular{T}}
    rand(rng, CPU(), type, n)
end

function Base.rand(type::Type{AT}, n::Integer) where {T, AT <: AbstractTriangular{T}}
    rand(Random.default_rng(), type, n)
end

function Base.rand(::Type{AT}, n::Integer) where {AT <: AbstractTriangular}
    rand(AT{Float64}, n)
end

function Base.rand(backend::KernelAbstractions.Backend, type::Type{AT},
        n::Integer) where {T, AT <: AbstractTriangular{T}}
    rand(Random.default_rng(), backend, type, n)
end

# these are Adam operations:
function scalar_add(A::AT, δ::Real) where {T, AT <: AbstractTriangular{T}}
    AT(A.S .+ δ, A.n)
end

#element-wise squares and square root (for Adam)
function ⊙²(A::AT) where {AT <: AbstractTriangular}
    AT(A.S .^ 2, A.n)
end
function racᵉˡᵉ(A::AT) where {AT <: AbstractTriangular}
    AT(sqrt.(A.S), A.n)
end
function /ᵉˡᵉ(A::AT, B::AT) where {AT <: AbstractTriangular}
    @assert A.n == B.n
    AT(A.S ./ B.S, A.n)
end

function LinearAlgebra.mul!(C::AT, A::AT, α::Real) where {AT <: AbstractTriangular}
    mul!(C.S, A.S, α)
    C
end
LinearAlgebra.mul!(C::AT, α::Real, A::AT) where {AT <: AbstractTriangular} = mul!(C, A, α)
LinearAlgebra.rmul!(C::AT, α::Real) where {AT <: AbstractTriangular} = mul!(C, C, α)

function Base.one(A::AbstractTriangular{T}) where {T}
    unit_matrix(KernelAbstractions.get_backend(A.S), T, A.n)
end

# the first matrix is multiplied onto A2 in order for it to not be SkewSymMatrix!
function Base.:*(A1::AbstractTriangular{T}, A2::AbstractTriangular{T}) where {T}
    A1 * (A2 * one(A2))
end

@doc raw"""
    vec(A::AbstractTriangular)

Return the associated vector to ``A``.

# Examples

```jldoctest
using GeometricOptimizers

M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
LowerTriangular(M) |> vec

# output

6-element Vector{Int64}:
  5
  9
 10
 13
 14
 15
```
"""
function Base.vec(A::AbstractTriangular)
    A.S
end

function Base.zero(A::AT) where {AT <: AbstractTriangular}
    AT(zero(A.S), A.n)
end

# see the comment on `similar(::SymmetricMatrix)`
function Base.similar(A::AT) where {AT <: AbstractTriangular}
    AT(similar(A.S), A.n)
end

# this fills the *storage*, i.e. the strict triangle; the rest of the matrix stays zero, because a
# triangular matrix cannot hold a constant everywhere. See the comment on `fill!(::SkewSymMatrix, …)`.
Base.fill!(A::AbstractTriangular, val) = (fill!(A.S, val); A)

function KernelAbstractions.get_backend(A::AbstractTriangular)
    KernelAbstractions.get_backend(A.S)
end

function assign!(B::AbstractTriangular, C::AbstractTriangular)
    copyto!(B, C)

    nothing
end

function Base.copy(A::AT) where {AT <: AbstractTriangular}
    AT(copy(A.S), A.n)
end

# The species check is a runtime one for the reason given on `copyto!(::Manifold, ::Manifold)`: a
# type parameter shared by both arguments binds the whole type, storage array included, and so
# excludes exactly the host-to-device transfer this method exists for. Both arguments are only
# constrained to `AbstractTriangular`, so without the check a `LowerTriangular` destination accepts
# an `UpperTriangular` source and takes its storage into the opposite triangle.
function Base.copyto!(A::AbstractTriangular, B::AbstractTriangular)
    AT, BT = Base.typename(typeof(A)).wrapper, Base.typename(typeof(B)).wrapper
    AT === BT || throw(ArgumentError("cannot copyto! a $BT into a $AT"))
    @assert A.n == B.n
    copyto!(A.S, B.S)
    A
end

# see the comment on `*(::SkewSymMatrix, ::AbstractVector)`: the vector goes through the
# matrix--matrix path as a single column, and the `n × 1` result is reshaped back to a vector
function Base.:*(A::AbstractTriangular, b::AbstractVector{T}) where {T}
    vec(A * reshape(b, length(b), 1))
end

function Base.:*(B::AbstractMatrix{T}, A::AbstractTriangular{T}) where {T}
    (A' * B')'
end
