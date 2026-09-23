# The binary arithmetic on this package's own matrix types, written once over a `Union`.
#
# Every owned type computes a product or a sum against a plain matrix on either side. Written as a
# `*(::X, ::AbstractMatrix)` and a `*(::AbstractMatrix, ::X)` per type, two owned operands `X * Y`
# are ambiguous: each method is narrower in one argument and wider in the other. So `*`, `+` and `-`
# each have three entry methods here, on `(Owned, AbstractMatrix)`, `(AbstractMatrix, Owned)` and
# `(Owned, Owned)`, and nothing per type. The third is contained in the other two and separates
# them. A per-type method that stays on `*` or `+` is a same-type (or same-pair) specialisation,
# which is narrower than `(Owned, Owned)` and so wins over it.
#
# The per-type code is on internal functions that `Base` and `LinearAlgebra` never see: `_lmul` for
# an owned left operand, `_rmul` for an owned right operand, `_ladd` for an owned operand in a sum.
# Each has one untyped fallback, which is the method the call reaches when no per-type method
# applies — an element type that does not match a kernel's, or a type with no kernel at all. The
# fallback of `_lmul` and `_rmul` `invoke`s the method `LinearAlgebra` or `Base` has for the plain
# operand's type, so such a call gets the answer it gets without this package. The fallback of
# `_ladd` checks the backends and `invoke`s `Base`'s generic `+`.
#
# Two rules decide a product of two owned operands, in `_owned_mul`:
#
#  1. `StiefelManifold`, `SymplecticStiefelManifold`, `StiefelProjection` and the adjoint of a point
#     hold their entries in an ordinary array. One of them unwraps — the left one if both are
#     wrappers — and the other operand's method answers for that array.
#  2. Where both operands compute and share an element type, the right-hand one is materialized:
#     `B * one(B)`, or `Matrix(S)` for an `Sfac`. The result is dense.
#
# `+` follows the same shape, in `_owned_add`: a `StiefelProjection` unwraps, and otherwise the
# operand that owns an addition kernel answers. Addition commutes, so `(AbstractMatrix, Owned)`
# hands the pair to the owned operand's method with the operands swapped.

# The types that take part in a sum or a difference. `Manifold` covers the Stiefel, Grassmann and
# symplectic Stiefel points; `AbstractLieAlgHorMatrix` covers both horizontal lifts.
const OwnedMatrix = Union{SkewSymMatrix, SymmetricMatrix, AbstractTriangular,
    AbstractLieAlgHorMatrix, StiefelProjection, Manifold}

# The wrappers of rule 1.
const OwnedWrapper = Union{StiefelManifold, SymplecticStiefelManifold, StiefelProjection,
    Adjoint{<:Any, <:StiefelManifold}, Adjoint{<:Any, <:SymplecticStiefelManifold}}

# The types that take part in a product: each has a product kernel of its own. `GrassmannManifold`
# and its adjoint have none, so a product with one of them reaches `LinearAlgebra` as the plain
# matrix it is.
const OwnedFactor = Union{SkewSymMatrix, SymmetricMatrix, AbstractTriangular,
    AbstractLieAlgHorMatrix, Sfac, OwnedWrapper}

Base.:*(A::OwnedFactor, B::AbstractMatrix) = _lmul(A, B)
Base.:*(A::OwnedFactor, b::AbstractVector) = _lmul(A, b)
Base.:*(A::AbstractMatrix, B::OwnedFactor) = _rmul(A, B)
Base.:*(A::OwnedFactor, B::OwnedFactor) = _owned_mul(A, B)

# `LinearAlgebra` has a `*(::Adjoint{<:Any, <:AbstractVector}, ::AbstractMatrix)` and a `Transpose`
# counterpart, each narrower on the left than `(AbstractMatrix, OwnedFactor)` and wider on the right.
# These two separate that pair. One method on `AdjOrTransAbsVec` does not: it is ambiguous with the
# `Adjoint` one in `LinearAlgebra`.
Base.:*(x::Adjoint{<:Any, <:AbstractVector}, B::OwnedFactor) = _rmul(x, B)
Base.:*(x::Transpose{<:Any, <:AbstractVector}, B::OwnedFactor) = _rmul(x, B)

_lmul(A, B) = invoke(*, Tuple{AbstractMatrix, _invoke_type(B)}, A, B)
_rmul(A, B) = invoke(*, Tuple{_invoke_type(A), AbstractMatrix}, A, B)

# The other operand keeps its own type for the `invoke`, so a method `LinearAlgebra` has for it still
# answers. An owned one, which `_owned_mul` passes on as a plain matrix, takes `AbstractMatrix`
# instead: its own type would find the methods above again.
_invoke_type(X) = typeof(X)
_invoke_type(::OwnedFactor) = AbstractMatrix

_unwrap(A::Union{StiefelManifold, SymplecticStiefelManifold, StiefelProjection}) = A.A
_unwrap(A::Adjoint) = parent(A).A'

_materialize(B) = B * one(B)
_materialize(S::Sfac) = Matrix(S)

# `Adjoint{<:StiefelManifold}` has a kernel on the left only, so on the right it is a plain matrix
# to the left operand's kernel. Two computing operands whose element types differ meet no kernel
# either way, so they go to the generic product directly rather than through a materialized copy;
# an `Sfac`'s kernels take any element type, so a pair with one keeps rule 2.
function _owned_mul(A, B)
    A isa OwnedWrapper && return _unwrap(A) * B
    B isa Adjoint{<:Any, <:StiefelManifold} && return _lmul(A, B)
    B isa OwnedWrapper && return A * _unwrap(B)
    if eltype(A) === eltype(B) || A isa Sfac || B isa Sfac
        A * _materialize(B)
    else
        invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
    end
end

Base.:+(A::OwnedMatrix, B::AbstractMatrix) = _ladd(A, B)
Base.:+(A::AbstractMatrix, B::OwnedMatrix) = _ladd(B, A)
Base.:+(A::OwnedMatrix, B::OwnedMatrix) = _owned_add(A, B)

_ladd(A, B) = _guarded_dense(+, A, B)

function _owned_add(A, B)
    A isa StiefelProjection && return A.A + B
    B isa StiefelProjection && return A + B.A
    B isa Union{SkewSymMatrix, StiefelLieAlgHorMatrix} && return _ladd(B, A)
    _ladd(A, B)
end

# Checks the backends and hands the pair to `Base`'s generic `+` or `-`: `invoke` and not a plain
# operator, which would re-enter the methods here. It is the fallback of `_ladd`, and `-` has no
# per-type method against a plain matrix, so all three `-` entries call it directly.
function _guarded_dense(op, A, B)
    _check_same_backend(A, B)
    invoke(op, Tuple{AbstractArray, AbstractArray}, A, B)
end

Base.:-(A::OwnedMatrix, B::AbstractMatrix) = _guarded_dense(-, A, B)
Base.:-(A::AbstractMatrix, B::OwnedMatrix) = _guarded_dense(-, A, B)
Base.:-(A::OwnedMatrix, B::OwnedMatrix) = _guarded_dense(-, A, B)

function Base.vcat(E::StiefelProjection{T}, F::StiefelProjection{T}) where {T <: Number}
    vcat(E.A, F.A)
end
function Base.hcat(E::StiefelProjection{T}, F::StiefelProjection{T}) where {T <: Number}
    hcat(E.A, F.A)
end
