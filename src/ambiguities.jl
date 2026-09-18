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
function Base.:*(S::Sfac{false},
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    S * U.parent.A'
end
function Base.:*(S::Sfac{true},
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    S * U.parent.A'
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

function Base.vcat(E::StiefelProjection{T}, F::StiefelProjection{T}) where {T <: Number}
    vcat(E.A, F.A)
end
function Base.hcat(E::StiefelProjection{T}, F::StiefelProjection{T}) where {T <: Number}
    hcat(E.A, F.A)
end
