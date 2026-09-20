@doc raw"""
    AbstractTriangular

See [`UpperTriangular`](@ref) and [`LowerTriangular`](@ref).
"""
abstract type AbstractTriangular{T} <: AbstractMatrix{T} end

Base.parent(A::AbstractTriangular) = A.S
Base.size(A::AbstractTriangular) = (A.n, A.n)

# Each argument carries its own type, and the species is compared at run time. Bound as
# `(A::AT, B::AT) where {AT <: AbstractTriangular}` these three do not dispatch for a pair whose
# storage arrays differ -- a host `LowerTriangular{T, Vector{T}}` and a device
# `LowerTriangular{T, JLArray{T, 1}}` are different concrete types, so `AT` cannot bind both and the
# call reaches `Base`'s generic array `+` at `arraymath.jl:8` instead.
# `copyto!(::AbstractTriangular, …)` below carries this same idiom against the same whole-type
# binding, as does `Manifold`'s `copyto!`.
#
# `Base.typename(…).wrapper` and not `typeof`: the question is whether both are lower or both upper,
# not whether their storage agrees. With two independent arguments and no species check at all, an
# `UpperTriangular` added to a `LowerTriangular` reads the upper storage into the lower triangle and
# returns a `LowerTriangular`, which is not the sum of the two.
#
# The sum of a lower and an upper triangular *is* well defined — it is a general matrix, which is
# what `Base`'s generic `+` returns for the pair and what `*` between the two species already
# returns. So `+` and `-` hand a mixed species to that path rather than refusing it. Only the
# packed-storage shortcut needs the two to agree.
#
# That path broadcasts through `getindex`, so it is host-only: a mixed-species pair whose storage is
# on a device raises `Scalar indexing is disallowed` rather than returning the dense sum. The `*`
# above is not the same in that respect -- it runs a kernel and answers on a device. A device
# mixed-species sum therefore needs a kernel of its own, which nothing asks for yet.
_triangular_species(A::AbstractTriangular) = Base.typename(typeof(A)).wrapper

function Base.:+(A::AbstractTriangular, B::AbstractTriangular)
    @assert A.n == B.n
    _check_same_backend(A, B)
    AT = _triangular_species(A)
    AT === _triangular_species(B) ||
        return invoke(+, Tuple{AbstractArray, AbstractArray}, A, B)
    AT(A.S + B.S, A.n)
end

# `add!` is the one that refuses, and not by choice: it writes the sum into a triangular destination,
# and a destination of one species cannot hold the sum of the two. There is no dense path to fall
# back to, because the caller owns the destination.
function add!(C::AbstractTriangular, A::AbstractTriangular, B::AbstractTriangular)
    @assert A.n == B.n == C.n
    AT = _triangular_species(A)
    AT === _triangular_species(B) === _triangular_species(C) ||
        throw(ArgumentError("add! needs all three arguments to be the same triangular species"))
    add!(C.S, A.S, B.S)
end

function Base.:-(A::AbstractTriangular, B::AbstractTriangular)
    @assert A.n == B.n
    _check_same_backend(A, B)
    AT = _triangular_species(A)
    AT === _triangular_species(B) ||
        return invoke(-, Tuple{AbstractArray, AbstractArray}, A, B)
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
    _check_supported_eltype(backend, T)
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
    _check_supported_eltype(backend, T)
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

# Two independent arguments, for the reason `+` above gives: bound as `(C::AT, A::AT)` a destination
# and a source whose storage arrays differ are already different concrete types, so `AT` cannot bind
# both and the call reaches `LinearAlgebra`'s generic `mul!`, which reaches `setindex!` on a type
# that has none.
function LinearAlgebra.mul!(C::AbstractTriangular, A::AbstractTriangular, α::Real)
    _triangular_species(C) === _triangular_species(A) ||
        throw(ArgumentError("mul! needs the destination and the source to be the same triangular species"))
    _check_same_backend(C, A)
    mul!(C.S, A.S, α)
    C
end
function LinearAlgebra.mul!(C::AbstractTriangular, α::Real, A::AbstractTriangular)
    mul!(C, A, α)
end
LinearAlgebra.rmul!(C::AT, α::Real) where {AT <: AbstractTriangular} = mul!(C, C, α)

function Base.one(A::AbstractTriangular{T}) where {T}
    unit_matrix(KernelAbstractions.get_backend(A.S), T, A.n)
end

@doc raw"""
    *(A::AbstractTriangular, B::AbstractMatrix)

The product, read off the packed storage vector by a kernel rather than through `getindex`.

Without this method the product falls through to the generic `AbstractMatrix` path, which asks `A`
for one entry at a time. That is scalar indexing, so it **cannot run on a device at all**. The
packed vector holds ``n(n-1)/2`` entries and the generic path reads ``n^2`` of them, the other
``n(n+1)/2`` being zeros that [`LowerTriangular`](@ref) and [`UpperTriangular`](@ref) manufacture —
but **do not read a host speed-up into that**, which is what the arithmetic invites. Measured
against the very path this method shadows, the ratio runs from 0.76x to 1.72x and is not monotone
in `n`. At `n = 6` — the size the retraction tests use — both products are *slower* than the path
they shadow, because the kernel launch has a fixed cost that a small product cannot amortize.
`scripts/triangular_multiply_cost.jl` is the check and `CHANGELOG.md` carries its table. The device
is what this buys.

The kernel is the one each subtype supplies, and it is where the two differ: a lower-triangular row
`i` runs over `1:(i-1)` and an upper-triangular one over `(i+1):n`, reading the same packed vector
through different index arithmetic.

`*(::AbstractMatrix, ::AbstractTriangular)` is written as `(A' * B')'`. For a **real** element type
that goes through here as well, because `adjoint` on one of these is then a type swap onto the same
storage. A complex one does not: the swap is bound to `Real`, for the reason the comment on
`adjoint(::LowerTriangular)` in `upper_triangular.jl` gives, and the lazy `Adjoint` it falls through
to is not an `AbstractTriangular`. So that product stays on the generic path and stays host-only.
"""
function Base.:*(A::AbstractTriangular{T}, B::AbstractMatrix{T}) where {T}
    m1, m2 = size(B)
    @assert m1 == A.n
    _check_same_backend(A, B)
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.allocate(backend, T, A.n, m2)

    triangular_mat_mul! = mat_mul_kernel(A, backend)
    triangular_mat_mul!(C, A.S, B, A.n, ndrange = size(C))
    C
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

# A row vector on the left is the one shape the method above leaves unsettled: it stands off against
# `LinearAlgebra`'s own row-vector product, and neither wins. *A row vector meets an owned matrix* in
# `src/ambiguities.jl` gives the mechanism and lists every site. The body is the one above, so a row
# vector gets the answer that method gives every other matrix, and gets it the same cheap way: `x'`
# is one column, which reaches the kernel as a single column instead of materializing `A`. It is a
# `Vector` for a real element type and an `n×1` wrapper for a complex one -- either way one column,
# so the two return the same values on different backings. `T` is bound in both slots because the
# method above binds it there; free, these would not be contained in it and would separate nothing.
#
# One pair covers both triangulars, because the method above is written on `AbstractTriangular` too.
Base.:*(x::Adjoint{T, <:AbstractVector}, A::AbstractTriangular{T}) where {T} = (A' * x')'
Base.:*(x::Transpose{T, <:AbstractVector}, A::AbstractTriangular{T}) where {T} = (A' * x')'
