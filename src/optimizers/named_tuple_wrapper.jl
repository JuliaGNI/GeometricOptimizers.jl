function GradientAutodiff(F, nt::NamedTuple)
    v, unflatten = ParameterHandling.flatten(nt)
    GradientAutodiff(_x -> F(unflatten(_x)), v)
end

function ParameterHandling.flatten(::Type{T}, x::Manifold{R}) where {T<:AbstractFloat,R<:Real}
    v, unflatten = ParameterHandling.flatten(T, x.A)
    v, _v -> StiefelManifold(unflatten(_v))
end

function ParameterHandling.flatten(::Type{T}, x::NamedTuple) where {T<:AbstractFloat}
    x_vec, unflatten = ParameterHandling.flatten(T, values(x))
    function unflatten_to_NamedTuple(v::Vector{R}) where {R<:Real}
        v_vec_vec = unflatten(v)
        return NamedTuple{keys(x),typeof(v_vec_vec)}(v_vec_vec)
    end
    return x_vec, unflatten_to_NamedTuple
end

function ParameterHandling.flatten(::Type{T}, x::Tuple) where {T<:AbstractFloat}
    vec1, back1 = ParameterHandling.flatten(T, first(x))
    vec2, back2 = ParameterHandling.flatten(T, Base.tail(x))
    l1 = length(vec1)
    l2 = length(vec2)
    function unflatten_to_Tuple(v::Vector{R}) where {R<:Real}
        return (back1(v[1:l1]), back2(v[(l1+1):(l1+l2)])...)
    end
    return vcat(vec1, vec2), unflatten_to_Tuple
end

function ParameterHandling.flatten(::Type{T}, x::Tuple{}) where {T<:AbstractFloat}
    v = T[]
    unflatten_to_empty_Tuple(::Vector{R}) where {R<:Real} = x
    return v, unflatten_to_empty_Tuple
end

function ParameterHandling.flatten(::Type{T}, x::Vector{R}) where {T<:AbstractFloat,R<:Real}
    unflatten_to_Vector(v::Vector{T}) = convert(Vector{R}, v)
    unflatten_to_Vector(v::Vector{<:ForwardDiff.Dual}) = v
    return Vector{T}(x), unflatten_to_Vector
end

function ParameterHandling.flatten(::Type{T}, x::Union{AbstractMatrix{R},AbstractArray{R,3}}) where {T<:AbstractFloat,R<:Real}
    x_vec, from_vec = ParameterHandling.flatten(T, vec(x))
    Array_from_vec(x_vec) = reshape(from_vec(x_vec), size(x))
    return x_vec, Array_from_vec
end

function ParameterHandling.flatten(::Type{T}, s::SkewSymMatrix{R}) where {T<:AbstractFloat,R<:Real}
    x_vec, from_vec = ParameterHandling.flatten(T, vec(s))
    Array_from_vec(x_vec) = SkewSymMatrix(from_vec(x_vec), s.n)
    return x_vec, Array_from_vec
end

function ParameterHandling.flatten(::Type{T}, g::StiefelLieAlgHorMatrix{R}) where {T<:AbstractFloat,R<:Real}
    x_vec, from_vec = ParameterHandling.flatten(T, (g.A, g.B))
    Array_from_vec(x_vec) = StiefelLieAlgHorMatrix(from_vec(x_vec)..., g.N, g.n)
    return x_vec, Array_from_vec
end

# note the type piracy here!
function (grad::Gradient{T})(nt::ArrayNamedTuple{T}) where {T}
    # unflatten not needed here
    v, unflatten = ParameterHandling.flatten(nt)
    grads = (unflatten ∘ grad)(v)
    vals = ()
    for _key in keys(nt)
        vals = (vals..., rgrad(nt[_key], grads[_key]))
    end
    NamedTuple{keys(nt)}(vals)
end

# technically constitutes type piracy
function (grad::Gradient{T})(g::ArrayNamedTuple{T}, x::ArrayNamedTuple{T}, state::OptimizerState{T}) where {T}
    _copyto!(g, global_rep(section(state), grad(x)))
end

_zero(a::AbstractArray) = zero(a)
_zero(a::ArrayNamedTuple) = apply_toNT(_zero, a)

_copy(a::AbstractArray) = copy(a)
_copy(a::ArrayNamedTuple) = apply_toNT(_copy, a)

_similar(a::StiefelManifold{T}) where {T} = rand(StiefelManifold{T}, size(a)...)
_similar(a::AbstractArray) = similar(a)
_similar(a::ArrayNamedTuple) = apply_toNT(_similar, a)

_fill!(a::AbstractArray{T}, b::T) where {T} = fill!(a, b)

_fill!(a::Manifold{T}, ::T) where {T} = a

_copyto!(a::AbstractArray{T}, b::AbstractArray{T}) where {T} = copyto!(a, b)
function _copyto!(a::ArrayNamedTuple{T}, b::ArrayNamedTuple{T}) where {T}
    apply_toNT(_copyto!, a, b)
end

function Base.copyto!(Λ::GlobalSectionNamedTuple{T}, x::ArrayNamedTuple{T}) where {T}
    apply_toNT(copyto!, Λ, x)
    Λ
end

function Base.copyto!(Λ::GlobalSection{T,MT}, x::MT) where {T,MT<:StiefelManifold}
    copyto!(Λ.Y, x)
    Λ
end

_copyto!(Λ::GlobalSectionNamedTuple, x::ArrayNamedTuple) = copyto!(Λ, x)

function _copyto!(x::ArrayNamedTuple, Λ::GlobalSectionNamedTuple)
    apply_toNT(copyto!, x, Λ)
    x
end

function _copyto!(Λ₁::GlobalSectionNamedTuple, Λ₂::GlobalSectionNamedTuple)
    apply_toNT(_copyto!, Λ₁, Λ₂)
    Λ₁
end

function _copyto!(Λ₁::GlobalSection{T,MT}, Λ₂::GlobalSection{T,MT}) where {T,MT<:StiefelManifold{T}}
    _copyto!(Λ₁.Y, Λ₂.Y)
    _copyto!(Λ₁.λ, Λ₂.λ)
    Λ₁
end

function _fill!(a::ArrayNamedTuple{T}, b::T) where {T}
    fill_closure!(_a) = _fill!(_a, b)
    apply_toNT(fill_closure!, a)
    a
end

function _difference!(c::AbstractArray{T}, a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    @assert axes(a) == axes(b) == axes(c)
    c .= a .- b
end

function _difference!(c::SkewSymMatrix, a::SkewSymMatrix, b::SkewSymMatrix)
    _difference!(c.S, a.S, b.S)
    c
end

function _difference!(c::StiefelLieAlgHorMatrix, a::StiefelLieAlgHorMatrix, b::StiefelLieAlgHorMatrix)
    _difference!(c.A, a.A, b.A)
    _difference!(c.B, a.B, b.B)
    c
end

_difference!(c::ArrayNamedTuple{T}, a::ArrayNamedTuple{T}, b::ArrayNamedTuple{T}) where {T} = apply_toNT(_difference!, c, a, b)

_rmul!(a::AbstractArray, b) = rmul!(a, b)

function _rmul!(a::ArrayNamedTuple, b)
    rmul_closure!(a) = _rmul!(a, b)
    apply_toNT(rmul_closure!, a)
    a
end

function _mul!(c::AbstractVecOrMat, a::AbstractMatrix, b::AbstractVecOrMat)
    mul!(c, a, b)
end

function _mul!(c::ArrayNamedTuple, a::ArrayNamedTuple, b::ArrayNamedTuple)
    apply_toNT(_mul!, c, a, b)
    c
end

function _mul!(c::ArrayNamedTuple, a::AbstractMatrix, b::ArrayNamedTuple)
    v_c, c_unflatten = ParameterHandling.flatten(c)
    v_b, b_unflatten = ParameterHandling.flatten(b)

    _mul!(v_c, a, v_b)
    _copyto!(c, c_unflatten(v_c))
end

function _mul(α::T, a::GradientArrayOrNamedTuple{T}) where {T}
    b = _copy(a)
    _rmul!(b, α)
end

_add!(a::AbstractArray{T}, b::AbstractArray{T}) where {T} = a .+= b

function _add!(a::ArrayNamedTuple{T}, b::ArrayNamedTuple{T}) where {T}
    apply_toNT(_add!, a, b)
    a
end

_add!(a::AbstractArray{T}, b::T) where {T} = a .+= b

function _add!(a::SkewSymMatrix{T}, b::T) where {T}
    _add!(a.S, b)
    a
end

function _add!(a::StiefelLieAlgHorMatrix{T}, b::T) where {T}
    _add!(a.A, b)
    _add!(a.B, b)
    a
end

function _add!(a::ArrayNamedTuple{T}, b::T) where {T}
    closure(a) = _add!(a, b)
    apply_toNT(closure, a)
    a
end

"""
    _rac!(B, A)

Compute the element-wise square-root of `A`.
"""
_rac!(B::AbstractArray, A::AbstractArray) = B .= sqrt.(A)

function _rac!(B::SkewSymMatrix, A::SkewSymMatrix)
    _rac!(B.S, A.S)
    B
end

function _rac!(B::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix)
    _rac!(B.A, A.A)
    _rac!(B.B, A.B)
    B
end

_rac!(b::ArrayNamedTuple, a::ArrayNamedTuple) = apply_toNT(_rac!, b, a)

_rac!(a) = _rac!(a, a)

"""
    _div!(C, A, B)

Divide `A` by `B` (elment-wise)
"""
function _div!(C::AbstractArray, A::AbstractArray, B::AbstractArray)
    @assert axes(A) == axes(B) == axes(C)
    C .= A ./ B
end

function _div!(C::SkewSymMatrix, A::SkewSymMatrix, B::SkewSymMatrix)
    _div!(C.S, A.S, B.S)
    C
end

function _div!(C::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix, B::StiefelLieAlgHorMatrix)
    _div!(C.A, A.A, B.A)
    _div!(C.B, A.B, B.B)
    C
end

function _div!(C::ArrayNamedTuple, A::ArrayNamedTuple, B::ArrayNamedTuple)
    apply_toNT(_div!, C, A, B)
    C
end

_div!(a, b) = _div!(a, a, b)

"""
    _square!(B, A)

"""
_square!(B::AbstractArray, A::AbstractArray) = B .= A .^ 2

function _square!(B::SkewSymMatrix, A::SkewSymMatrix)
    _square!(B.S, A.S)
    B
end

function _square!(B::StiefelLieAlgHorMatrix, A::StiefelLieAlgHorMatrix)
    _square!(B.A, A.A)
    _square!(B.B, A.B)
    B
end

_square!(b::ArrayNamedTuple, a::ArrayNamedTuple) = apply_toNT(_square!, b, a)

function _square(a)
    b = _copy(a)
    _square!(b, a)
    b
end


Base.copyto!(dest::AT, src::GlobalSection{T,AT}) where {T,AT<:AbstractArray{T}} = copyto!(dest, src.Y)
_copyto!(dest, src::GlobalSection) = copyto!(dest, src)
rgrad(ps::ArrayNamedTuple, dx::ArrayNamedTuple) = apply_toNT(rgrad, ps, dx)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end
