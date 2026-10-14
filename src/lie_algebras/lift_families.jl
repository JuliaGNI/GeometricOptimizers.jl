# The method families both horizontal lifts share, written once over the block tuple.
#
# Every body below was, before this file, the same character-for-character in
# `stiefel_lie_algebra_horizontal.jl` and `grassmann_lie_algebra_horizontal.jl` apart from which
# constructor the result takes. `parent`'s docstring in `abstract_lie_algebra_horizontal.jl` already
# records why an elementwise operation on a lift is written over the tuple rather than per type: the
# tuple is the free parameters, and the ambient `N × N` matrix counts each off-diagonal block twice.
#
# The file is included after both concrete types, because the bodies below name them. The one hook a
# concrete lift supplies is `lift_type`, which is its one-argument constructor; everything else
# follows from `parent`, `N` and `n`, which are this file's own signature.
#
# A lift's blocks are its `N × N` ambient shape over its free parameters, and the mixed pair with a
# `SkewSymMatrix` is the one sum that keeps a structure besides two of a kind — it lives beside the
# Stiefel lift, whose `A` block makes it possible at all.

# The one-argument constructor of a lift, so that a family can rebuild one from its blocks without
# naming either type. `Base.typename(…).wrapper` is the type as a value: `StiefelLieAlgHorMatrix` for
# a `StiefelLieAlgHorMatrix{Float64, …}`, with no storage array to carry.
lift_type(B::AbstractLieAlgHorMatrix) = Base.typename(typeof(B)).wrapper

# The blocks a family works on. `parent` of both lifts, which is what `named_tuple_wrapper.jl`'s
# primitives already use.
_blocks(B::AbstractLieAlgHorMatrix) = parent(B)

# `_owned_add` and `_owned_sub` between two lifts of *the same* species: the sum of two lifts is a
# lift, so the packed-storage shortcut works and the block-wise sum keeps it. Two lifts of different
# species — a `StiefelLieAlgHorMatrix` and a `GrassmannLieAlgHorMatrix` — have different numbers of
# blocks, and the sum of the two is a general matrix. The species check is a runtime one, exactly as
# `_owned_add(::AbstractTriangular, ::AbstractTriangular)` does it, for the same reason: a shared type
# parameter cannot bind both because each lift's storage array is part of its type.
function _owned_add(A::AbstractLieAlgHorMatrix, B::AbstractLieAlgHorMatrix)
    lift_type(A) === lift_type(B) || return _dense(+, A, B)
    lift_type(A)(map(+, parent(A), parent(B))..., A.N, A.n)
end

function _owned_sub(A::AbstractLieAlgHorMatrix, B::AbstractLieAlgHorMatrix)
    lift_type(A) === lift_type(B) || return _dense(-, A, B)
    lift_type(A)(map(-, parent(A), parent(B))..., A.N, A.n)
end

function add!(C::AbstractLieAlgHorMatrix, A::AbstractLieAlgHorMatrix,
        B::AbstractLieAlgHorMatrix)
    foreach(add!, parent(C), parent(A), parent(B))

    C
end

Base.:-(A::AbstractLieAlgHorMatrix) = lift_type(A)(map(-, parent(A))..., A.N, A.n)

function Base.:*(A::AbstractLieAlgHorMatrix, α::Real)
    lift_type(A)(map(b -> b * α, parent(A))..., A.N, A.n)
end

Base.:*(α::Real, A::AbstractLieAlgHorMatrix) = A * α

function LinearAlgebra.mul!(C::AbstractLieAlgHorMatrix, A::AbstractLieAlgHorMatrix, α::Real)
    foreach((c, a) -> mul!(c, a, α), parent(C), parent(A))

    C
end
function LinearAlgebra.mul!(C::AbstractLieAlgHorMatrix, α::Real, A::AbstractLieAlgHorMatrix)
    mul!(C, A, α)
end
LinearAlgebra.rmul!(C::AbstractLieAlgHorMatrix, α::Real) = mul!(C, C, α)

Base.zero(B::AbstractLieAlgHorMatrix) = lift_type(B)(map(zero, parent(B))..., B.N, B.n)
Base.copy(B::AbstractLieAlgHorMatrix) = lift_type(B)(map(copy, parent(B))..., B.N, B.n)
Base.fill!(B::AbstractLieAlgHorMatrix, val) = (foreach(b -> fill!(b, val), parent(B)); B)

function Base.copyto!(A::AbstractLieAlgHorMatrix, B::AbstractLieAlgHorMatrix)
    foreach(copyto!, parent(A), parent(B))

    A
end

# `_round` rounds the blocks, so a truncated display keeps its type. `_round` itself takes `kwargs`
# and is defined once for a `Manifold` and each structured matrix already.
function _round(B::AbstractLieAlgHorMatrix; kwargs...)
    lift_type(B)(map(b -> _round(b; kwargs...), parent(B))..., B.N, B.n)
end

# `zeros` and `rand` are the two methods that legitimately stay per type. A caller may name the
# storage arrays as well as the element type — `StiefelLieAlgHorMatrix{Float64, Vector{Float64},
# Matrix{Float64}}` — and a three-parameter `Type` is not callable, so the body has to spell the
# constructor out. They are one line per type, in each type's own file; every other family in this
# file is written once for both.

# `similar` reads the shape from an instance, so the two arms differ only in how many blocks the lift
# holds. This is what makes `similar` usable as the like-for-like allocation of an optimizer cache:
# the cache constructors bind their gradient blocks to a single `AT <: GradientStorage{T}`, so a host
# block beside a device one does not dispatch.
function Base.similar(A::StiefelLieAlgHorMatrix, N::Integer, n::Integer)
    StiefelLieAlgHorMatrix(similar(A.A), similar(A.B, N - n, n), N, n)
end
function Base.similar(A::GrassmannLieAlgHorMatrix, N::Integer, n::Integer)
    GrassmannLieAlgHorMatrix(similar(A.B, N - n, n), N, n)
end
function Base.similar(A::AbstractLieAlgHorMatrix, dims::Union{
        Integer, AbstractUnitRange}...)
    similar(A, dims...)
end
Base.similar(A::AbstractLieAlgHorMatrix) = similar(A, A.N, A.n)
