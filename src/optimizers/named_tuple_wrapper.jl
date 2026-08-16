function GradientAutodiff(F, nt::NamedTuple)
    v, unflatten = ParameterHandling.flatten(nt)
    GradientAutodiff(_x -> F(unflatten(_x)), v)
end

# `∇F!` is called on the flattened parameters, i.e. on `ParameterHandling.flatten(nt)[1]`.
function GradientFunction(F, ∇F!, nt::NamedTuple)
    v, unflatten = ParameterHandling.flatten(nt)
    GradientFunction(_x -> F(unflatten(_x)), ∇F!, v)
end

# `ParameterHandling.flatten` defaults to `Float64`, which would silently promote e.g.
# `Float32` parameters, so we flatten to the element type of the parameters instead.
ParameterHandling.flatten(x::ArrayNamedTuple{T}) where {T<:AbstractFloat} = ParameterHandling.flatten(T, x)

function ParameterHandling.flatten(::Type{T}, x::Manifold{R}) where {T<:AbstractFloat,R<:Real}
    v, unflatten = ParameterHandling.flatten(T, x.A)
    # The manifold has to come back as the *same* kind of manifold — hardcoding
    # `StiefelManifold` here silently turned a `GrassmannManifold` into a `StiefelManifold` on
    # every round trip. See `manifold_constructor` for why it is the type *name*.
    MT = manifold_constructor(x)
    v, _v -> MT(unflatten(_v))
end

# The `flatten` methods below that dispatch on Base types (`NamedTuple`, `Tuple`, `Vector`,
# `AbstractMatrix`/`AbstractArray{,3}`) are type piracy, and they shadow methods that
# ParameterHandling already defines: `T<:AbstractFloat` is narrower than its `T<:Real`. The
# methods for `Manifold`, `SkewSymMatrix` and `StiefelLieAlgHorMatrix` are fine, those types
# are ours. See issue #16.
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

function ParameterHandling.flatten(::Type{T}, g::GrassmannLieAlgHorMatrix{R}) where {T<:AbstractFloat,R<:Real}
    x_vec, from_vec = ParameterHandling.flatten(T, g.B)
    Array_from_vec(x_vec) = GrassmannLieAlgHorMatrix(from_vec(x_vec), g.N, g.n)
    return x_vec, Array_from_vec
end

# Type piracy: `Gradient` is SimpleSolvers' and `ArrayNamedTuple` is an alias for Base's
# `NamedTuple`. A wrapper `struct` would fix this locally. See issue #16.
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

# This is *not* type piracy, unlike the method above: it dispatches on `OptimizerState`,
# which is defined in this package (see `optimizers/optimizer_state.jl`), and one owned
# argument type is enough.
function (grad::Gradient{T})(g::ArrayNamedTuple{T}, x::ArrayNamedTuple{T}, state::OptimizerState{T}) where {T}
    _copyto!(g, global_rep(section(state), grad(x)))
end

_zero(a::AbstractArray) = zero(a)
_zero(a::ArrayNamedTuple) = apply_toNT(_zero, a)

_copy(a::AbstractArray) = copy(a)
_copy(a::ArrayNamedTuple) = apply_toNT(_copy, a)

# `Base.similar` is deliberately an error on a `Manifold` — an arbitrary array of that shape is not a
# point of it — so a fresh *random* point stands in for it. `Manifold` and not `StiefelManifold`, and
# built with `manifold_constructor` for the same reason `flatten` above is: a `NamedTuple` holding a
# `GrassmannManifold` used to reach the `AbstractArray` method below and raise that error while
# building an `AdamState` or a `MomentumState`. See issue A11.
_similar(a::Manifold{T}) where {T} = rand(manifold_constructor(a){T}, size(a)...)
_similar(a::AbstractArray) = similar(a)
_similar(a::ArrayNamedTuple) = apply_toNT(_similar, a)

_fill!(a::AbstractArray{T}, b::T) where {T} = fill!(a, b)

_fill!(a::Manifold{T}, ::T) where {T} = a

_copyto!(a::AbstractArray{T}, b::AbstractArray{T}) where {T} = copyto!(a, b)
function _copyto!(a::ArrayNamedTuple{T}, b::ArrayNamedTuple{T}) where {T}
    apply_toNT(_copyto!, a, b)
end

# Type piracy again by way of the aliases: both `GlobalSectionNamedTuple` and
# `ArrayNamedTuple` are `NamedTuple`. See issue #16.
function Base.copyto!(Λ::GlobalSectionNamedTuple{T}, x::ArrayNamedTuple{T}) where {T}
    apply_toNT(copyto!, Λ, x)
    Λ
end

function Base.copyto!(Λ::GlobalSection{T,MT}, x::MT) where {T,MT<:Manifold}
    # only the anchor moves; `Λ.λ` is deliberately left alone, since recomputing the lift would move
    # the frame the secant pair of a quasi-Newton method is expressed in
    copyto!(Λ.Y, x)
    Λ
end

_copyto!(Λ::GlobalSectionNamedTuple, x::ArrayNamedTuple) = copyto!(Λ, x)

# the bare-`Manifold` counterpart of the line above
_copyto!(Λ::GlobalSection{T,MT}, x::MT) where {T,MT<:Manifold} = copyto!(Λ, x)

function _copyto!(x::ArrayNamedTuple, Λ::GlobalSectionNamedTuple)
    apply_toNT(copyto!, x, Λ)
    x
end

function _copyto!(Λ₁::GlobalSectionNamedTuple, Λ₂::GlobalSectionNamedTuple)
    apply_toNT(_copyto!, Λ₁, Λ₂)
    Λ₁
end

function _copyto!(Λ₁::GlobalSection{T,MT}, Λ₂::GlobalSection{T,MT}) where {T,MT<:Manifold{T}}
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

# The elementwise helpers on a horizontal lift act on its *free parameters* and not on the ambient
# `N × N` matrix, which has no `setindex!` and would count each off-diagonal block twice besides.
# `Base.parent` is the tuple of those blocks — `(A, B)` for `StiefelLieAlgHorMatrix`, `(B,)` for
# `GrassmannLieAlgHorMatrix` — so one method over it covers both lifts and whatever is added next.
# These used to be written out for the Stiefel lift only, which is half of why a `GrassmannManifold`
# could not be optimized over (issue A11); the Stiefel bodies were exactly this `foreach`, so nothing
# about that path changes. The Stiefel `A` block still routes through the `SkewSymMatrix` method.
function _difference!(c::AbstractLieAlgHorMatrix, a::AbstractLieAlgHorMatrix, b::AbstractLieAlgHorMatrix)
    foreach(_difference!, parent(c), parent(a), parent(b))
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

# The same ambient/intrinsic boundary as the method above, for a *bare* `Manifold`: the quasi-Newton
# `Q` is sized by the length of the flattening, while the direction and the gradient are horizontal
# lifts of the ambient shape (`3 × 3` against an intrinsic 2, for `St(3, 1)`). Multiplying them
# directly reaches `setindex!`, which the lift types do not define.
function _mul!(c::AbstractLieAlgHorMatrix, a::AbstractMatrix, b::AbstractLieAlgHorMatrix)
    v_c, c_unflatten = ParameterHandling.flatten(c)
    v_b, _ = ParameterHandling.flatten(b)

    _mul!(v_c, a, v_b)
    _copyto!(c, c_unflatten(v_c))
end

function _mul(α::T, a::GradientArrayOrNamedTuple{T}) where {T}
    b = _copy(a)
    _rmul!(b, α)
end

@doc raw"""
    _dot(a, b)

The inner product of two gradients or directions, taken in the *flattened* coordinates.

# Implementation

For an `AbstractVecOrMat` this is `LinearAlgebra.dot`. For a horizontal lift — or a `NamedTuple` of
them — it is emphatically not: `dot` on an [`AbstractLieAlgHorMatrix`](@ref) is the *ambient*
Frobenius product, which counts each of the off-diagonal blocks of the lift twice and so comes out
exactly twice the product of the free parameters. The intrinsic coordinates are the ones every other
quantity in this package is expressed in — `Q` is sized by the flattening, [`outer!`](@ref) flattens
before it forms its outer product, and the `α` of a line search parameterizes a curve in them — so
pairing a gradient with a direction has to happen there too.

Used by [`trial_slope`](@ref) for ``\varphi'(\alpha)``, and by the quasi-Newton caches for
``\delta^T\gamma``, whose value has to be consistent with the flattened `T₁`, `T₂` and `γ^TQγ` it
divides.
"""
_dot(a::AbstractVecOrMat, b::AbstractVecOrMat) = dot(a, b)

const LiftOrNamedTuple{T} = Union{AbstractLieAlgHorMatrix{T},ArrayNamedTuple{T}}

# `flatten` is given `T` explicitly: the one-argument `ParameterHandling.flatten` defaults to
# `Float64`, which would make this return a `Float64` for `Float32` parameters and break every
# element type the result is combined with downstream.
_dot(a::LiftOrNamedTuple{T}, b::LiftOrNamedTuple{T}) where {T} =
    dot(ParameterHandling.flatten(T, a)[1], ParameterHandling.flatten(T, b)[1])

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

function _add!(a::AbstractLieAlgHorMatrix{T}, b::T) where {T}
    foreach(aᵢ -> _add!(aᵢ, b), parent(a))
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

function _rac!(B::AbstractLieAlgHorMatrix, A::AbstractLieAlgHorMatrix)
    foreach(_rac!, parent(B), parent(A))
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

function _div!(C::AbstractLieAlgHorMatrix, A::AbstractLieAlgHorMatrix, B::AbstractLieAlgHorMatrix)
    foreach(_div!, parent(C), parent(A), parent(B))
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

function _square!(B::AbstractLieAlgHorMatrix, A::AbstractLieAlgHorMatrix)
    foreach(_square!, parent(B), parent(A))
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
