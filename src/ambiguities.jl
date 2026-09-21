# Tie-breakers for the pairs where one of this package's own matrix types meets another one.
#
# Every pair below is a standoff between an `Owned ∘ AbstractMatrix` method and an
# `AbstractMatrix ∘ Owned` one. For two owned operands neither of the two is more specific, so
# without a method here the call would be ambiguous and an ordinary product or sum would raise a
# `MethodError`.
# The comment above the four `Sfac`-`Sfac` methods in `decompositions/symplectic_sr.jl` gives the
# mechanism for the case it is written about, and it generalizes: a signature that is narrower in
# one argument and wider in the other does not win. That is also why a single method taking a
# `Union` of the owned types is no help — it is wider in the slot the standoff is about. Separating
# a pair needs a signature contained in both, and that is one method per pair.
#
# Two rules fix what each of them returns.
#
#  1. `StiefelManifold`, `SymplecticStiefelManifold` and `StiefelProjection` hold their entries in
#     an ordinary array, and the method being bypassed on that side would have done nothing but
#     unwrap it. So unwrap it here and hand the result to the other operand's method: the answer is
#     then whatever that method gives for a plain array, which is what the wrapper stood for.
#  2. Where both operands compute — the two kernel-backed matrices, the triangulars and the `Sfac`
#     operator — one of them has to be materialized. It is the right-hand one, written
#     `B * one(B)` for a kernel-backed matrix or a triangular and `Matrix(B)` for an `Sfac`. That
#     is how `*(::SymmetricMatrix, ::SymmetricMatrix)` and `*(::AbstractTriangular,
#     ::AbstractTriangular)` already materialize their right operand;
#     `*(::SkewSymMatrix, ::SkewSymMatrix)` spells it `one(B) * B` and `*(::Sfac, ::Sfac)`
#     materializes both. All four of those return dense, and so does every method here.
#
# Three of the pairs are not in this file. `+` on a `SkewSymMatrix` and a `StiefelLieAlgHorMatrix`
# is the one case where both operands are skew-symmetric, so the sum is too and keeps that
# structure; its two methods are in `lie_algebras/stiefel_lie_algebra_horizontal.jl`, next to the
# packing they need. `*` on an adjoint `StiefelManifold` and a `StiefelManifold` is separated in
# `manifolds/stiefel_manifold.jl`, by the method for that pair which lives beside the rest of the
# manifold's arithmetic.
#
# Each signature binds the element type wherever one of the two methods it separates binds it, so
# that it is contained in both. Without that it separates only the part of the overlap where the
# element types agree.
#
# ## A row vector meets an owned matrix
#
# A second class of standoff. It is not in this file because the other method is not this
# package's: `LinearAlgebra` carries
# `*(::Adjoint{T, <:AbstractVector} where T, ::AbstractMatrix)` and a `Transpose` counterpart, each
# narrower in the left argument than an `AbstractMatrix ∘ Owned` method and wider in the right. So
# `v' * X` and `transpose(v) * X` are ambiguous for every owned type that has such a method, and
# `Test.detect_ambiguities` in `test/ambiguities.jl` cannot report them: that sweep keeps only pairs
# whose two methods both belong here. Each type's own testset covers its pair instead.
#
# Two methods per `AbstractMatrix ∘ Owned` method, beside the method they separate:
#
#   | type                                    | file                                       |
#   |:----------------------------------------|:-------------------------------------------|
#   | `StiefelManifold`                       | `manifolds/stiefel_manifold.jl`            |
#   | `SymplecticStiefelManifold`             | `manifolds/symplectic_stiefel_manifold.jl` |
#   | `Adjoint{<:SymplecticStiefelManifold}`  | `manifolds/symplectic_stiefel_manifold.jl` |
#   | `Sfac{false}`, `Sfac{true}`             | `decompositions/symplectic_sr.jl`          |
#   | `StiefelProjection`                     | `special_matrices/stiefel_projection.jl`   |
#   | `SkewSymMatrix`                         | `special_matrices/skew_symmetric.jl`       |
#   | `SymmetricMatrix`                       | `special_matrices/symmetric.jl`            |
#   | `AbstractTriangular`                    | `special_matrices/triangular.jl`           |
#
# Everything else here is absent because it defines no `*(::AbstractMatrix, ::Owned)`, so a row
# vector against it reaches `LinearAlgebra` unopposed: `GrassmannManifold` and the two horizontal
# lifts define no `*` against a matrix at all, and `Adjoint{<:StiefelManifold}` has only the method
# that takes it on the *left*. `Adjoint{<:SymplecticStiefelManifold}` is in the table because
# [`metric`](@ref) needs the mirror as well.
#
# The element-type rule is the same: bound where the method being separated binds it, free where it
# does not. What each returns is decided once rather than by the two rules above, because here the
# bypassed method already takes an arbitrary matrix on the left. Every one of them repeats that
# method's body verbatim, so a row vector gets exactly the answer any other matrix gets -- and gets
# it the cheap way, since none of those bodies materializes its owned operand.

# The matrix types this package owns and defines arithmetic for. It is written out rather than taken
# from `subtypes`, because it is the list the backend guard at the foot of this file is measured
# against and a silent change to it would change what that guard covers. `Manifold` covers the
# Stiefel, Grassmann and symplectic Stiefel points at once; `AbstractLieAlgHorMatrix` covers both
# horizontal lifts.
const OwnedMatrix = Union{SkewSymMatrix, SymmetricMatrix, AbstractTriangular,
    AbstractLieAlgHorMatrix, StiefelProjection, Manifold}

function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}},
        B::SymplecticStiefelManifold) where {
        T, AT <: AbstractMatrix{T}}
    Y.parent.A' * B
end
function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, B::Sfac{false}) where {
        T, AT <: AbstractMatrix{T}}
    Y.parent.A' * B
end
function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, B::Sfac{true}) where {
        T, AT <: AbstractMatrix{T}}
    Y.parent.A' * B
end
function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}},
        B::AbstractTriangular{T}) where {
        T, AT <: AbstractMatrix{T}}
    Y.parent.A' * B
end
function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, B::SkewSymMatrix{T}) where {
        T, AT <: AbstractMatrix{T}}
    Y.parent.A' * B
end
function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}},
        B::SymmetricMatrix{T}) where {
        T, AT <: AbstractMatrix{T}}
    Y.parent.A' * B
end

# `Adjoint{SymplecticStiefelManifold}` multiplies on both sides of a bare `AbstractMatrix`, so it
# meets every other owned type twice over. Rule 1 throughout: the adjoint of a point is an ordinary
# array transposed, so it unwraps and the other operand's method answers.
# Both element types are free on the first method below, for the reason spelled out on
# `*(::Adjoint{<:SymplecticStiefelManifold}, ::Adjoint{<:SymplecticStiefelManifold})`: the two
# methods this separates bind `T` from different arguments, so their overlap does not require the
# two to agree.
function Base.:*(Y::Adjoint{T₁, StiefelManifold{T₁, AT₁}},
        U::Adjoint{T₂, SymplecticStiefelManifold{T₂, AT₂}}) where {
        T₁, AT₁ <: AbstractMatrix{T₁}, T₂, AT₂ <: AbstractMatrix{T₂}}
    Y.parent.A' * U
end
function Base.:*(Y::StiefelManifold,
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    Y.A * U
end
function Base.:*(V::SymplecticStiefelManifold,
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    V.A * U
end
function Base.:*(E::StiefelProjection,
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    E.A * U
end
function Base.:*(S::Sfac{false},
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    S * U.parent.A'
end
function Base.:*(S::Sfac{true},
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    S * U.parent.A'
end
function Base.:*(A::AbstractTriangular{T},
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    A * U.parent.A'
end
function Base.:*(A::SkewSymMatrix{T},
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    A * U.parent.A'
end
function Base.:*(A::SymmetricMatrix{T},
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    A * U.parent.A'
end

function Base.:*(U::Adjoint{T, SymplecticStiefelManifold{T, AT}},
        Y::StiefelManifold) where {T, AT <: AbstractMatrix{T}}
    U.parent.A' * Y
end
function Base.:*(U::Adjoint{T, SymplecticStiefelManifold{T, AT}},
        E::StiefelProjection) where {T, AT <: AbstractMatrix{T}}
    U.parent.A' * E
end
function Base.:*(U::Adjoint{T, SymplecticStiefelManifold{T, AT}},
        B::Sfac{false}) where {
        T, AT <: AbstractMatrix{T}}
    U.parent.A' * B
end
function Base.:*(U::Adjoint{T, SymplecticStiefelManifold{T, AT}},
        B::Sfac{true}) where {
        T, AT <: AbstractMatrix{T}}
    U.parent.A' * B
end
function Base.:*(U::Adjoint{T, SymplecticStiefelManifold{T, AT}},
        B::AbstractTriangular{T}) where {T, AT <: AbstractMatrix{T}}
    U.parent.A' * B
end
function Base.:*(U::Adjoint{T, SymplecticStiefelManifold{T, AT}},
        B::SkewSymMatrix{T}) where {T, AT <: AbstractMatrix{T}}
    U.parent.A' * B
end
function Base.:*(U::Adjoint{T, SymplecticStiefelManifold{T, AT}},
        B::SymmetricMatrix{T}) where {T, AT <: AbstractMatrix{T}}
    U.parent.A' * B
end

Base.:*(Y::StiefelManifold, B::StiefelManifold) = Y.A * B
Base.:*(Y::StiefelManifold, B::SymplecticStiefelManifold) = Y.A * B
Base.:*(Y::StiefelManifold, B::Sfac{false}) = Y.A * B
Base.:*(Y::StiefelManifold, B::Sfac{true}) = Y.A * B
Base.:*(Y::StiefelManifold{T}, B::AbstractTriangular{T}) where {T} = Y.A * B
Base.:*(Y::StiefelManifold{T}, B::SkewSymMatrix{T}) where {T} = Y.A * B
Base.:*(Y::StiefelManifold{T}, B::SymmetricMatrix{T}) where {T} = Y.A * B

Base.:*(U::SymplecticStiefelManifold, B::StiefelManifold) = U.A * B
Base.:*(U::SymplecticStiefelManifold, B::SymplecticStiefelManifold) = U.A * B
Base.:*(U::SymplecticStiefelManifold, B::Sfac{false}) = U.A * B
Base.:*(U::SymplecticStiefelManifold, B::Sfac{true}) = U.A * B
Base.:*(U::SymplecticStiefelManifold{T}, B::AbstractTriangular{T}) where {T} = U.A * B
Base.:*(U::SymplecticStiefelManifold{T}, B::SkewSymMatrix{T}) where {T} = U.A * B
Base.:*(U::SymplecticStiefelManifold{T}, B::SymmetricMatrix{T}) where {T} = U.A * B

Base.:*(S::Sfac{false}, Y::StiefelManifold) = S * Y.A
Base.:*(S::Sfac{false}, U::SymplecticStiefelManifold) = S * U.A
Base.:*(S::Sfac{false, T}, B::AbstractTriangular{T}) where {T} = S * (B * one(B))
Base.:*(S::Sfac{false, T}, B::SkewSymMatrix{T}) where {T} = S * (B * one(B))
Base.:*(S::Sfac{false, T}, B::SymmetricMatrix{T}) where {T} = S * (B * one(B))

Base.:*(S::Sfac{true}, Y::StiefelManifold) = S * Y.A
Base.:*(S::Sfac{true}, U::SymplecticStiefelManifold) = S * U.A
Base.:*(S::Sfac{true, T}, B::AbstractTriangular{T}) where {T} = S * (B * one(B))
Base.:*(S::Sfac{true, T}, B::SkewSymMatrix{T}) where {T} = S * (B * one(B))
Base.:*(S::Sfac{true, T}, B::SymmetricMatrix{T}) where {T} = S * (B * one(B))

Base.:*(A::SkewSymMatrix{T}, Y::StiefelManifold{T}) where {T} = A * Y.A
Base.:*(A::SkewSymMatrix{T}, U::SymplecticStiefelManifold{T}) where {T} = A * U.A
Base.:*(A::SkewSymMatrix{T}, S::Sfac{false, T}) where {T} = A * Matrix(S)
Base.:*(A::SkewSymMatrix{T}, S::Sfac{true, T}) where {T} = A * Matrix(S)
Base.:*(A::SkewSymMatrix{T}, B::AbstractTriangular{T}) where {T} = A * (B * one(B))
Base.:*(A::SkewSymMatrix{T}, B::SymmetricMatrix{T}) where {T} = A * (B * one(B))

Base.:*(A::SymmetricMatrix{T}, Y::StiefelManifold{T}) where {T} = A * Y.A
Base.:*(A::SymmetricMatrix{T}, U::SymplecticStiefelManifold{T}) where {T} = A * U.A
Base.:*(A::SymmetricMatrix{T}, S::Sfac{false, T}) where {T} = A * Matrix(S)
Base.:*(A::SymmetricMatrix{T}, S::Sfac{true, T}) where {T} = A * Matrix(S)
Base.:*(A::SymmetricMatrix{T}, B::AbstractTriangular{T}) where {T} = A * (B * one(B))
Base.:*(A::SymmetricMatrix{T}, B::SkewSymMatrix{T}) where {T} = A * (B * one(B))

# `StiefelProjection` and `AbstractTriangular` each have a `*` against a bare `AbstractMatrix`, so
# each of them meets every other owned type in the same standoff as the types above. The two rules
# at the head of this file decide all of it: the projection unwraps under rule 1, since it holds its
# entries in an ordinary array; a triangular computes, so under rule 2 it materializes whatever is
# to its right unless that operand is itself a wrapper.
Base.:*(E::StiefelProjection, Y::StiefelManifold) = E.A * Y
Base.:*(E::StiefelProjection, U::SymplecticStiefelManifold) = E.A * U
Base.:*(E::StiefelProjection, F::StiefelProjection) = E.A * F
Base.:*(E::StiefelProjection, S::Sfac{false}) = E.A * S
Base.:*(E::StiefelProjection, S::Sfac{true}) = E.A * S
Base.:*(E::StiefelProjection{T}, B::AbstractTriangular{T}) where {T} = E.A * B
Base.:*(E::StiefelProjection{T}, A::SkewSymMatrix{T}) where {T} = E.A * A
Base.:*(E::StiefelProjection{T}, A::SymmetricMatrix{T}) where {T} = E.A * A

function Base.:*(Y::Adjoint{T, StiefelManifold{T, AT}}, E::StiefelProjection) where {
        T, AT <: AbstractMatrix{T}}
    Y.parent.A' * E
end
Base.:*(Y::StiefelManifold, E::StiefelProjection) = Y.A * E
Base.:*(U::SymplecticStiefelManifold, E::StiefelProjection) = U.A * E
Base.:*(S::Sfac{false}, E::StiefelProjection) = S * E.A
Base.:*(S::Sfac{true}, E::StiefelProjection) = S * E.A
Base.:*(A::SkewSymMatrix{T}, E::StiefelProjection{T}) where {T} = A * E.A
Base.:*(A::SymmetricMatrix{T}, E::StiefelProjection{T}) where {T} = A * E.A

Base.:*(A::AbstractTriangular{T}, Y::StiefelManifold{T}) where {T} = A * Y.A
Base.:*(A::AbstractTriangular{T}, U::SymplecticStiefelManifold{T}) where {T} = A * U.A
Base.:*(A::AbstractTriangular{T}, E::StiefelProjection{T}) where {T} = A * E.A
Base.:*(A::AbstractTriangular{T}, S::Sfac{false, T}) where {T} = A * Matrix(S)
Base.:*(A::AbstractTriangular{T}, S::Sfac{true, T}) where {T} = A * Matrix(S)
Base.:*(A::AbstractTriangular{T}, B::SkewSymMatrix{T}) where {T} = A * (B * one(B))
Base.:*(A::AbstractTriangular{T}, B::SymmetricMatrix{T}) where {T} = A * (B * one(B))

Base.:+(E::StiefelProjection, A::SkewSymMatrix) = E.A + A
Base.:+(E::StiefelProjection, C::StiefelLieAlgHorMatrix) = E.A + C
Base.:+(E::StiefelProjection, F::StiefelProjection) = E.A + F
Base.:+(A::SkewSymMatrix{T}, E::StiefelProjection{T}) where {T} = A + E.A
Base.:+(C::StiefelLieAlgHorMatrix, E::StiefelProjection) = C + E.A

# ## A difference against a plain `AbstractMatrix`
#
# `_check_same_backend` guards a pair only where this package owns the method. Without the three
# below, **no owned matrix type has a `-` against a plain `AbstractMatrix` at all**, so every such
# pair falls to `Base`'s generic `-` at `arraymath.jl:6`, which broadcasts and takes its backend from
# the argument order: `SkewSymMatrix(host) - JLArray` answers on the device, the reversed order
# answers on the device too, and with the structured operand on the device the pair raises
# `Scalar indexing is disallowed`, which names neither operand.
#
# The three methods below close that. Each checks, then hands the pair to the same `Base` method it
# would otherwise reach, so no same-backend call changes its value, its type or its backend. `invoke`
# and not a plain `-`, which would re-enter them.
#
# Three methods and not two. `(Owned, AbstractMatrix)` and `(AbstractMatrix, Owned)` are each
# narrower in one argument and wider in the other, so for two owned operands neither wins — the
# standoff the head of this file describes. `(Owned, Owned)` is contained in both and separates them.
# It is wider than every concrete same-type method above it, so `SkewSymMatrix - SkewSymMatrix`,
# `AbstractTriangular - AbstractTriangular` and the rest keep their structure-preserving results.
#
# **`+` is not done the same way and is still open.** `SkewSymMatrix`, `StiefelLieAlgHorMatrix` and
# `StiefelProjection` each already own a kernel-backed `+(X, ::AbstractMatrix)`, which is narrower in
# the left argument than an `(AbstractMatrix, Owned)` method and wider in the right. A `Union` method
# is wider in the slot that standoff is about, exactly as the head of this file says, so adding the
# `+` triple makes sixteen owned pairs ambiguous — measured. Closing `+` needs a tie-breaker per pair
# and a decision per pair about which of the two kernels answers. That is open issue A25 in
# `CHANGELOG.md`.
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
