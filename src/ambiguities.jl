# The binary arithmetic on this package's own matrix types, written once over a `Union`.
#
# Every owned type computes a product or a sum against a plain matrix on either side. Written as a
# `*(::X, ::AbstractMatrix)` and a `*(::AbstractMatrix, ::X)` per type, two owned operands `X * Y`
# are ambiguous: each method is narrower in one argument and wider in the other. So `*`, `+`, `-`
# and `mul!` each have entry methods here, on `(Owned, AbstractMatrix)`, `(AbstractMatrix, Owned)`
# and `(Owned, Owned)`, and nothing per type. The third is contained in the other two and separates
# them. `*(::SkewSymMatrix, ::SkewSymMatrix)` and `*(::Sfac, ::Sfac)` are the two same-type
# specialisations that stay on `*`; they are narrower than `(Owned, Owned)`, and each reaches the
# entries again.
#
# The entry methods are where the backends are checked, and the only place: a pair on two backends
# raises an `ArgumentError` naming both, before any kernel runs. Pure dispatch cannot do this,
# because one backend legitimately holds a `Vector` and a `Matrix`, or a view and its parent.
#
# The per-type code is on internal functions that `Base` and `LinearAlgebra` never see: `_lmul` and
# `_lmul_into!` for an owned left operand, `_rmul` and `_rmul_into!` for an owned right operand,
# `_ladd` for an owned operand in a sum, and `_owned_add` and `_owned_sub` for two owned operands.
# Each has one untyped fallback, which is the method the call reaches when no per-type method
# applies — an element type that does not match a kernel's, or a type with no kernel at all. The
# fallbacks give the answer the call gets without this package: `_lmul` and `_rmul` `invoke` the
# method `LinearAlgebra` or `Base` has for the plain operand's type, `_lmul_into!` and
# `_rmul_into!` call the five-argument `mul!`, and the sums go to `_dense`.
#
# Two rules decide a product of two owned operands, in `_owned_mul` and `_owned_mul!`:
#
#  1. A point, a `StiefelProjection` and the adjoint of either hold their entries in an ordinary
#     array. One of them unwraps — the left one if both are wrappers — and the other operand's
#     method answers for that array.
#  2. Where both operands compute and share an element type, the right-hand one is materialized:
#     `B * one(B)`, or `Matrix(S)` for an `Sfac`. The result is dense.
#
# `+` follows the same shape, in `_owned_add`: a wrapper unwraps, a pair of one structure keeps it,
# and otherwise the sum is dense. Against a plain matrix the owned operand's addition kernel answers;
# addition commutes, so `(AbstractMatrix, Owned)` hands the pair to it with the operands swapped.

# The wrappers of rule 1.
const ArrayWrapper = Union{StiefelManifold, GrassmannManifold, SymplecticStiefelManifold,
    StiefelProjection}
const OwnedWrapper = Union{ArrayWrapper, Adjoint{<:Any, <:ArrayWrapper}}

# The adjoint of a real skew-symmetric matrix or horizontal lift is the matrix negated. `adjoint`
# keeps it lazy, and these methods compute with the parent and the sign. A complex one is not owned,
# and reaches `LinearAlgebra` as the plain matrix it is.
const SkewAdjoint = Adjoint{<:Real, <:Union{SkewSymMatrix, AbstractLieAlgHorMatrix}}

# The types that take part in a product, a sum or a difference.
const OwnedMatrix = Union{SkewSymMatrix, SymmetricMatrix, AbstractTriangular,
    AbstractLieAlgHorMatrix, Sfac, SkewAdjoint, OwnedWrapper}

function Base.:*(A::OwnedMatrix, B::AbstractMatrix)
    _check_same_backend(A, B)
    _lmul(A, B)
end
function Base.:*(A::OwnedMatrix, b::AbstractVector)
    _check_same_backend(A, b)
    _lmul(A, b)
end
function Base.:*(A::AbstractMatrix, B::OwnedMatrix)
    _check_same_backend(A, B)
    _rmul(A, B)
end
function Base.:*(A::OwnedMatrix, B::OwnedMatrix)
    _check_same_backend(A, B)
    _owned_mul(A, B)
end

# `LinearAlgebra` has a `*(::Adjoint{<:Any, <:AbstractVector}, ::AbstractMatrix)` and a `Transpose`
# counterpart, each narrower on the left than `(AbstractMatrix, OwnedMatrix)` and wider on the right.
# These two separate that pair. One method on `AdjOrTransAbsVec` does not: it is ambiguous with the
# `Adjoint` one in `LinearAlgebra`.
function Base.:*(x::Adjoint{<:Any, <:AbstractVector}, B::OwnedMatrix)
    _check_same_backend(x, B)
    _rmul(x, B)
end
function Base.:*(x::Transpose{<:Any, <:AbstractVector}, B::OwnedMatrix)
    _check_same_backend(x, B)
    _rmul(x, B)
end

_lmul(A, B) = invoke(*, Tuple{AbstractMatrix, _invoke_type(B)}, A, B)
_rmul(A, B) = invoke(*, Tuple{_invoke_type(A), AbstractMatrix}, A, B)

# The other operand keeps its own type for the `invoke`, so a method `LinearAlgebra` has for it still
# answers. An owned one, which `_owned_mul` passes on as a plain matrix, takes `AbstractMatrix`
# instead: its own type would find the methods above again.
_invoke_type(X) = typeof(X)
_invoke_type(::OwnedMatrix) = AbstractMatrix

_unwrap(A::ArrayWrapper) = A.A
_unwrap(A::Adjoint) = _unwrap(parent(A))'

# The wrappers' kernels: the product of the array they hold.
_lmul(A::OwnedWrapper, B::AbstractVecOrMat) = _unwrap(A) * B
_rmul(B::AbstractMatrix, A::OwnedWrapper) = B * _unwrap(A)
_lmul_into!(C, A::OwnedWrapper, B) = mul!(C, _unwrap(A), B)
_rmul_into!(C, B, A::OwnedWrapper) = mul!(C, B, _unwrap(A))

# The minus goes on the result, where it costs one pass over the product; on the parent it would
# build a second packed matrix first.
_lmul(A::SkewAdjoint, B::AbstractVecOrMat) = -(parent(A) * B)
_rmul(B::AbstractMatrix, A::SkewAdjoint) = -(B * parent(A))
_lmul_into!(C, A::SkewAdjoint, B) = (mul!(C, parent(A), B); C .= .-C)
_rmul_into!(C, B, A::SkewAdjoint) = (mul!(C, B, parent(A)); C .= .-C)

# `adjoint` again, and not the dense matrix `Base` gives: this is what `LinearAlgebra` does for `-`
# on an `Adjoint`, and it keeps the scalar product on the packed storage.
Base.:*(α::Real, A::SkewAdjoint) = adjoint(α * parent(A))
Base.:*(A::SkewAdjoint, α::Real) = adjoint(parent(A) * α)

# A scalar product or a negation of a wrapper is one of the array it holds. `Base` gives the same
# dense result through `getindex`, one entry at a time.
Base.:*(α::Real, A::OwnedWrapper) = α * _unwrap(A)
Base.:*(A::OwnedWrapper, α::Real) = _unwrap(A) * α
Base.:-(A::OwnedWrapper) = -_unwrap(A)

_materialize(B) = B * one(B)
_materialize(S::Sfac) = Matrix(S)
_materialize(A::SkewAdjoint) = -_materialize(parent(A))

# Two computing operands whose element types differ meet no kernel either way, so they go to the
# generic product directly rather than through a materialized copy; an `Sfac`'s kernels take any
# element type, so a pair with one keeps rule 2.
function _owned_mul(A, B)
    A isa OwnedWrapper && return _unwrap(A) * B
    B isa OwnedWrapper && return A * _unwrap(B)
    if eltype(A) === eltype(B) || A isa Sfac || B isa Sfac
        A * _materialize(B)
    else
        invoke(*, Tuple{AbstractMatrix, AbstractMatrix}, A, B)
    end
end

# `mul!` into a destination the caller owns, on the same three entries and the same two rules.
# `LinearAlgebra`'s own three-argument `mul!` is untyped, so each of these is narrower than it.
function LinearAlgebra.mul!(C::AbstractMatrix, A::OwnedMatrix, B::AbstractMatrix)
    _check_same_backend(A, B, C)
    _lmul_into!(C, A, B)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::OwnedMatrix)
    _check_same_backend(A, B, C)
    _rmul_into!(C, A, B)
end
function LinearAlgebra.mul!(C::AbstractMatrix, A::OwnedMatrix, B::OwnedMatrix)
    _check_same_backend(A, B, C)
    _owned_mul!(C, A, B)
end

# The kernels are matrix--matrix ones, so the vector goes through them as a single column. `reshape`
# shares the buffer, so the kernel writes straight into `c`.
function LinearAlgebra.mul!(c::AbstractVector, A::OwnedMatrix, b::AbstractVector)
    _check_same_backend(A, b, c)
    _lmul_into!(reshape(c, length(c), 1), A, reshape(b, length(b), 1))
    c
end

_lmul_into!(C, A, B) = mul!(C, A, B, true, false)
_rmul_into!(C, A, B) = mul!(C, A, B, true, false)

# A computing type on the right has no in-place kernel: its `_rmul` is the transpose of a left
# product, so on a device the product is taken there and copied into `C`. The host keeps the generic
# five-argument `mul!`, which reads `A` through `getindex` and allocates nothing, as `_dense` does.
function _rmul_into!(C::AbstractMatrix{T}, B::AbstractMatrix{T},
        A::Union{SkewSymMatrix{T}, SymmetricMatrix{T}, AbstractTriangular{T},
            AbstractLieAlgHorMatrix{T}}) where {T}
    KernelAbstractions.get_backend(A) isa CPU && return mul!(C, B, A, true, false)
    C .= _rmul(B, A)
end

function _owned_mul!(C, A, B)
    A isa OwnedWrapper && return mul!(C, _unwrap(A), B)
    B isa OwnedWrapper && return mul!(C, A, _unwrap(B))
    if eltype(A) === eltype(B) || A isa Sfac || B isa Sfac
        mul!(C, A, _materialize(B))
    else
        mul!(C, A, B, true, false)
    end
end

function Base.:+(A::OwnedMatrix, B::AbstractMatrix)
    _check_same_backend(A, B)
    _ladd(A, B)
end
function Base.:+(A::AbstractMatrix, B::OwnedMatrix)
    _check_same_backend(A, B)
    _ladd(B, A)
end
function Base.:+(A::OwnedMatrix, B::OwnedMatrix)
    _check_same_backend(A, B)
    _owned_add(A, B)
end

_ladd(A, B) = _dense(+, A, B)
_ladd(A::OwnedWrapper, B) = _unwrap(A) + B

# The pairs that keep a structure have a method of `_owned_add` beside their type. Any other pair of
# two computing operands is dense: an addition kernel reads its second operand as a plain array.
function _owned_add(A, B)
    A isa OwnedWrapper && return _unwrap(A) + B
    B isa OwnedWrapper && return A + _unwrap(B)
    _dense(+, A, B)
end

# `-` has no per-type method against a plain matrix. The pairs that keep a structure have a method of
# `_owned_sub` beside their type.
function Base.:-(A::OwnedMatrix, B::AbstractMatrix)
    _check_same_backend(A, B)
    _dense(-, A, B)
end
function Base.:-(A::AbstractMatrix, B::OwnedMatrix)
    _check_same_backend(A, B)
    _dense(-, A, B)
end
function Base.:-(A::OwnedMatrix, B::OwnedMatrix)
    _check_same_backend(A, B)
    _owned_sub(A, B)
end

_owned_sub(A, B) = _dense(-, A, B)

# The dense sum or difference. On the host it is `Base`'s generic `+` or `-`, which reads an owned
# operand through `getindex` in one pass: `invoke` and not a plain operator, which would re-enter the
# methods here. A device serves no `getindex`, so there an owned operand becomes the plain array it
# holds, or the dense matrix it computes, and the backend's own broadcast takes the rest. That costs
# a product for a computing operand, which is why the host keeps the one pass.
function _dense(op, A, B)
    KernelAbstractions.get_backend(A) isa CPU &&
        return invoke(op, Tuple{AbstractArray, AbstractArray}, A, B)
    op(_plain(A), _plain(B))
end

_plain(A) = A
_plain(A::OwnedMatrix) = _materialize(A)
_plain(A::OwnedWrapper) = _unwrap(A)

function Base.vcat(E::StiefelProjection{T}, F::StiefelProjection{T}) where {T <: Number}
    vcat(E.A, F.A)
end
function Base.hcat(E::StiefelProjection{T}, F::StiefelProjection{T}) where {T <: Number}
    hcat(E.A, F.A)
end
