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
    # (UpperTriangular or LowerTriangular) as a constant the compiler can see, which is what makes
    # the return type inferrable: `Base.return_types` gives the concrete triangular type here and
    # `Any` for a name resolved through the evaluator at run time.
    Base.typename(AT).wrapper(KernelAbstractions.zeros(backend, T, n*(n-1)÷2), n)
end

# The host spelling is `zeros(T, m)` and not `zeros(CPU(), AT, n)`: `KernelAbstractions.zeros` on a
# `CPU` returns the same `Vector{T}` with the same values and charges for it. It fills rather than
# reaching `calloc`, so it loses the zero page, and the gap therefore grows with the length rather
# than being a constant. Measured by `scripts/host_allocation_cost.jl` on Julia 1.13 at
# `--check-bounds=auto`, one cold process per run: 128 B of overhead up to 500 elements, 144 B at
# 1024 and 12 384 B at 2^18.
#
# The time ratio is not monotone, and the worst case is the *small* matrix rather than the large
# one. `KernelAbstractions.zeros` has a floor of about 120 ns whatever the length, so the ratio
# starts near 25x at one element, falls to about 1.9x at 1024 to 2048 elements as the host path
# grows into that floor, then rises again to about 5x at 2^18 as the fill outgrows `calloc`.
#
# Treat the figures as this machine's -- what does not move is that an overhead is paid at every
# length, that the bytes grow with the length, and that the device spelling was never the faster of
# the two at any length measured. The
# host path is the common one here and must not pay for the device machinery. Every other
# host-placing allocator in this package -- `SkewSymMatrix`'s, `SymmetricMatrix`'s and both lie
# algebras' -- already spells it this way.
function Base.zeros(::Type{AT}, n::Int) where {T, AT <: AbstractTriangular{T}}
    Base.typename(AT).wrapper(zeros(T, n*(n-1)÷2), n)
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{AT}, n::Integer) where {T, AT <: AbstractTriangular{T}}
    S = KernelAbstractions.allocate(backend, T, n*(n-1)÷2)
    Random.rand!(rng, S)
    Base.typename(AT).wrapper(S, n)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{AT}, n::Int) where {
        T, AT <: AbstractTriangular{T}}
    Base.typename(AT).wrapper(rand(rng, T, n*(n-1)÷2), n)
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
