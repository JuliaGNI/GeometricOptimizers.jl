# The two `Gradient` constructors that take a parameter set rather than a vector.
#
# `SimpleSolvers` wants a flat vector, so the set is flattened once here and the layout is captured in
# the closure. A `ParameterLayout` is a *value*, which is the difference that matters: the
# `ParameterHandling` version this replaces returned a chain of nested closures, one per level of the
# tree, and that chain was not type stable. The element type comes from the parameters themselves
# rather than defaulting to `Float64`, which used to promote a `Float32` network silently.
function GradientAutodiff(F, nt::NamedTuple)
    v, layout = flatten(nt)
    GradientAutodiff(_x -> F(unflatten(layout, _x)), v)
end

# `∇F!` is called on the flattened parameters, i.e. on `flatten(nt)[1]`.
function GradientFunction(F, ∇F!, nt::NamedTuple)
    v, layout = flatten(nt)
    GradientFunction(_x -> F(unflatten(layout, _x)), ∇F!, v)
end


# Type piracy: `Gradient` is SimpleSolvers' and `ArrayNamedTuple` is an alias for Base's
# `NamedTuple`. A wrapper `struct` would fix this locally. See issue #16.
function (grad::Gradient{T})(nt::ArrayNamedTuple{T}) where {T}
    v, layout = flatten(nt)
    # `rgrad` takes the *whole* leaf, not its storage: it is the Riemannian projection and needs the
    # point it projects at, so this is `mapparameters` and not `mapstorage`.
    mapparameters(rgrad, nt, unflatten(layout, grad(v)))
end

# This is *not* type piracy, unlike the method above: it dispatches on `OptimizerState`,
# which is defined in this package (see `optimizers/optimizer_state.jl`), and one owned
# argument type is enough.
function (grad::Gradient{T})(g::ArrayNamedTuple{T}, x::ArrayNamedTuple{T}, state::OptimizerState{T}) where {T}
    _copyto!(g, global_rep(section(state), grad(x)))
end

_zero(a::AbstractArray) = zero(a)
_zero(a::ArrayNamedTuple) = map(_zero, a)

_copy(a::AbstractArray) = copy(a)
_copy(a::ArrayNamedTuple) = map(_copy, a)

# `Base.similar` is deliberately an error on a `Manifold` — an arbitrary array of that shape is not a
# point of it — so a fresh *random* point stands in for it. `Manifold` and not `StiefelManifold`, and
# built with `manifold_constructor` for the same reason `flatten` above is: a `NamedTuple` holding a
# `GrassmannManifold` used to reach the `AbstractArray` method below and raise that error while
# building an `AdamState` or a `MomentumState`. See issue A11.
_similar(a::Manifold{T}) where {T} = rand(manifold_constructor(a){T}, size(a)...)
_similar(a::AbstractArray) = similar(a)
_similar(a::ArrayNamedTuple) = map(_similar, a)

_fill!(a::AbstractArray{T}, b::T) where {T} = fill!(a, b)

_fill!(a::Manifold{T}, ::T) where {T} = a

_copyto!(a::AbstractArray{T}, b::AbstractArray{T}) where {T} = copyto!(a, b)
function _copyto!(a::ArrayNamedTuple{T}, b::ArrayNamedTuple{T}) where {T}
    map(_copyto!, a, b)
end

# Type piracy again by way of the aliases: both `GlobalSectionNamedTuple` and
# `ArrayNamedTuple` are `NamedTuple`. See issue #16.
function Base.copyto!(Λ::GlobalSectionNamedTuple{T}, x::ArrayNamedTuple{T}) where {T}
    map(copyto!, Λ, x)
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
    map(copyto!, x, Λ)
    x
end

function _copyto!(Λ₁::GlobalSectionNamedTuple, Λ₂::GlobalSectionNamedTuple)
    map(_copyto!, Λ₁, Λ₂)
    Λ₁
end

function _copyto!(Λ₁::GlobalSection{T,MT}, Λ₂::GlobalSection{T,MT}) where {T,MT<:Manifold{T}}
    _copyto!(Λ₁.Y, Λ₂.Y)
    _copyto!(Λ₁.λ, Λ₂.λ)
    Λ₁
end

function _fill!(a::ArrayNamedTuple{T}, b::T) where {T}
    fill_closure!(_a) = _fill!(_a, b)
    map(fill_closure!, a)
    a
end

function _difference!(c::AbstractArray{T}, a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    @assert axes(a) == axes(b) == axes(c)
    c .= a .- b
end

function _difference!(c::MT, a::MT, b::MT) where {MT<:VectorStorageMatrix}
    _difference!(parent(c), parent(a), parent(b))
    c
end

# The elementwise helpers on a horizontal lift act on its *free parameters* and not on the ambient
# `N × N` matrix, which has no `setindex!` and would count each off-diagonal block twice besides.
# `Base.parent` is the tuple of those blocks — `(A, B)` for `StiefelLieAlgHorMatrix`, `(B,)` for
# `GrassmannLieAlgHorMatrix` — so one method over it covers both lifts and whatever is added next.
# These used to be written out for the Stiefel lift only, which is half of why a `GrassmannManifold`
# could not be optimized over (issue A11); the Stiefel bodies were exactly this `foreach`, so nothing
# about that path changes. The Stiefel `A` block still routes through the method above.
function _difference!(c::AbstractLieAlgHorMatrix, a::AbstractLieAlgHorMatrix, b::AbstractLieAlgHorMatrix)
    foreach(_difference!, parent(c), parent(a), parent(b))
    c
end

_difference!(c::ArrayNamedTuple{T}, a::ArrayNamedTuple{T}, b::ArrayNamedTuple{T}) where {T} = map(_difference!, c, a, b)

_rmul!(a::AbstractArray, b) = rmul!(a, b)

# `LinearAlgebra.rmul!` writes back through `setindex!`, which three of these four do not have;
# scaling the free parameters is the same operation and is what the matrix they represent scales by.
function _rmul!(a::VectorStorageMatrix, b)
    rmul!(parent(a), b)
    a
end

function _rmul!(a::ArrayNamedTuple, b)
    rmul_closure!(a) = _rmul!(a, b)
    map(rmul_closure!, a)
    a
end

function _mul!(c::AbstractVecOrMat, a::AbstractMatrix, b::AbstractVecOrMat)
    mul!(c, a, b)
end

function _mul!(c::ArrayNamedTuple, a::ArrayNamedTuple, b::ArrayNamedTuple)
    map(_mul!, c, a, b)
    c
end

# `c` supplies its layout but not its numbers: it is the destination, so only its shape matters.
# That is one flatten fewer than this used to do, and `unflatten!` writes the result back through
# `copyto!` instead of building a fresh parameter set for `_copyto!` to copy out of.
function _mul!(c::ArrayNamedTuple, a::AbstractMatrix, b::ArrayNamedTuple)
    layout = parameterlayout(c)
    v_b, _ = flatten(b)
    v_c = similar(v_b)

    _mul!(v_c, a, v_b)
    unflatten!(c, layout, v_c)
end

# The same ambient/intrinsic boundary as the method above, for a *bare* `Manifold`: the quasi-Newton
# `Q` is sized by the length of the flattening, while the direction and the gradient are horizontal
# lifts of the ambient shape (`3 × 3` against an intrinsic 2, for `St(3, 1)`). Multiplying them
# directly reaches `setindex!`, which the lift types do not define.
function _mul!(c::AbstractLieAlgHorMatrix, a::AbstractMatrix, b::AbstractLieAlgHorMatrix)
    layout = parameterlayout(c)
    v_b, _ = flatten(b)
    v_c = similar(v_b)

    _mul!(v_c, a, v_b)
    unflatten!(c, layout, v_c)
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

# `flatten` is given `T` explicitly so that the result is a `T` even when a set happens to promote
# to something wider; every quantity this is combined with downstream is a `T`.
_dot(a::LiftOrNamedTuple{T}, b::LiftOrNamedTuple{T}) where {T} =
    dot(flatten(T, a)[1], flatten(T, b)[1])

_add!(a::AbstractArray{T}, b::AbstractArray{T}) where {T} = a .+= b

function _add!(a::MT, b::MT) where {MT<:VectorStorageMatrix}
    _add!(parent(a), parent(b))
    a
end

function _add!(a::ArrayNamedTuple{T}, b::ArrayNamedTuple{T}) where {T}
    map(_add!, a, b)
    a
end

_add!(a::AbstractArray{T}, b::T) where {T} = a .+= b

function _add!(a::VectorStorageMatrix{T}, b::T) where {T}
    _add!(parent(a), b)
    a
end

function _add!(a::AbstractLieAlgHorMatrix{T}, b::T) where {T}
    foreach(aᵢ -> _add!(aᵢ, b), parent(a))
    a
end

function _add!(a::ArrayNamedTuple{T}, b::T) where {T}
    closure(a) = _add!(a, b)
    map(closure, a)
    a
end

"""
    _rac!(B, A)

Compute the element-wise square-root of `A`.
"""
_rac!(B::AbstractArray, A::AbstractArray) = B .= sqrt.(A)

function _rac!(B::MT, A::MT) where {MT<:VectorStorageMatrix}
    _rac!(parent(B), parent(A))
    B
end

function _rac!(B::AbstractLieAlgHorMatrix, A::AbstractLieAlgHorMatrix)
    foreach(_rac!, parent(B), parent(A))
    B
end

_rac!(b::ArrayNamedTuple, a::ArrayNamedTuple) = map(_rac!, b, a)

_rac!(a) = _rac!(a, a)

"""
    _div!(C, A, B)

Divide `A` by `B` (elment-wise)
"""
function _div!(C::AbstractArray, A::AbstractArray, B::AbstractArray)
    @assert axes(A) == axes(B) == axes(C)
    C .= A ./ B
end

function _div!(C::MT, A::MT, B::MT) where {MT<:VectorStorageMatrix}
    _div!(parent(C), parent(A), parent(B))
    C
end

function _div!(C::AbstractLieAlgHorMatrix, A::AbstractLieAlgHorMatrix, B::AbstractLieAlgHorMatrix)
    foreach(_div!, parent(C), parent(A), parent(B))
    C
end

function _div!(C::ArrayNamedTuple, A::ArrayNamedTuple, B::ArrayNamedTuple)
    map(_div!, C, A, B)
    C
end

_div!(a, b) = _div!(a, a, b)

"""
    _square!(B, A)

"""
_square!(B::AbstractArray, A::AbstractArray) = B .= A .^ 2

function _square!(B::MT, A::MT) where {MT<:VectorStorageMatrix}
    _square!(parent(B), parent(A))
    B
end

function _square!(B::AbstractLieAlgHorMatrix, A::AbstractLieAlgHorMatrix)
    foreach(_square!, parent(B), parent(A))
    B
end

_square!(b::ArrayNamedTuple, a::ArrayNamedTuple) = map(_square!, b, a)

function _square(a)
    b = _copy(a)
    _square!(b, a)
    b
end


Base.copyto!(dest::AT, src::GlobalSection{T,AT}) where {T,AT<:AbstractArray{T}} = copyto!(dest, src.Y)
_copyto!(dest, src::GlobalSection) = copyto!(dest, src)
rgrad(ps::ArrayNamedTuple, dx::ArrayNamedTuple) = map(rgrad, ps, dx)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end
