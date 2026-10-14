# The method families both horizontal lifts share, written once over the block tuple `parent`
# returns. `parent`'s docstring in `abstract_lie_algebra_horizontal.jl` says why an elementwise
# operation on a lift is written over the tuple rather than over the ambient `N × N` matrix.
#
# Each method binds both lifts to the same concrete type, so dispatch refuses a mixed pair: two lifts
# of different species have different blocks, and their sum falls through to the dense
# `_owned_add(A, B)` of `ambiguities.jl`. `zeros` and `rand` stay in each type's own file, because a
# caller may name the storage arrays in the type it passes.
for L in (:StiefelLieAlgHorMatrix, :GrassmannLieAlgHorMatrix)
    @eval begin
        _owned_add(A::$L, B::$L) = $L(map(+, parent(A), parent(B))..., A.N, A.n)
        _owned_sub(A::$L, B::$L) = $L(map(-, parent(A), parent(B))..., A.N, A.n)
        add!(C::$L, A::$L, B::$L) = (foreach(add!, parent(C), parent(A), parent(B)); C)

        Base.:-(A::$L) = $L(map(-, parent(A))..., A.N, A.n)
        Base.:*(A::$L, α::Real) = $L(map(b -> b * α, parent(A))..., A.N, A.n)
        function LinearAlgebra.mul!(C::$L, A::$L, α::Real)
            foreach((c, a) -> mul!(c, a, α), parent(C), parent(A))
            C
        end

        Base.zero(B::$L) = $L(map(zero, parent(B))..., B.N, B.n)
        Base.copy(B::$L) = $L(map(copy, parent(B))..., B.N, B.n)
        Base.copyto!(A::$L, B::$L) = (foreach(copyto!, parent(A), parent(B)); A)
        Base.fill!(B::$L, val) = (foreach(b -> fill!(b, val), parent(B)); B)
        _round(B::$L; kwargs...) = $L(map(b -> _round(b; kwargs...), parent(B))..., B.N, B.n)

        # on the backend `A` is already on, which is what makes `similar` usable as the
        # like-for-like allocation of an optimizer cache, whose gradient blocks share one storage type
        function Base.similar(A::$L, N::Integer, n::Integer)
            zeros(KernelAbstractions.get_backend(A), $L{eltype(A)}, N, n)
        end
        Base.similar(A::$L) = similar(A, A.N, A.n)
    end
end

Base.:*(α::Real, A::AbstractLieAlgHorMatrix) = A * α
function LinearAlgebra.mul!(C::AbstractLieAlgHorMatrix, α::Real, A::AbstractLieAlgHorMatrix)
    mul!(C, A, α)
end
LinearAlgebra.rmul!(C::AbstractLieAlgHorMatrix, α::Real) = mul!(C, C, α)
