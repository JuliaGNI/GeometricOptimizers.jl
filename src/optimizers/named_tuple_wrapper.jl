# The two `Gradient` constructors that take a parameter set rather than a vector.
#
# `SimpleSolvers` wants a flat vector, so the set is flattened once here and the layout is captured in
# the closure. A `ParameterLayout` is a *value*, which is the difference that matters: the
# `ParameterHandling` version this replaces returned a chain of nested closures, one per level of the
# tree, and that chain was not type stable. The element type comes from the parameters themselves
# rather than defaulting to `Float64`, which used to promote a `Float32` network silently.
#
# `ParameterSet` and not `ParameterContainer`: `F` is the caller's objective and
# these have to accept whatever it was written against, including a nested `NamedTuple` that no
# `ArrayNamedTuple` bound admits. `unflatten` returns the shape the layout was built from, so a
# container in gives a container back and `F` sees the type it was written for.
function GradientAutodiff(F, nt::ParameterSet)
    v, layout = flatten(nt)
    GradientAutodiff(_x -> F(unflatten(layout, _x)), v)
end

# `∇F!` is called on the flattened parameters, i.e. on `flatten(nt)[1]`.
function GradientFunction(F, ∇F!, nt::ParameterSet)
    v, layout = flatten(nt)
    GradientFunction(_x -> F(unflatten(layout, _x)), ∇F!, v)
end

# Type piracy: `Gradient` is SimpleSolvers' and this package owns neither member of
# `ParameterContainer` -- `ArrayNamedTuple` is an alias for Base's `NamedTuple`, and the container
# belongs to `NeuralNetworkParameters`. Taking the container did *not* fix that, and the note on
# `ParameterContainer` says why the `NamedTuple` half cannot simply be dropped. See issue #16.
function (grad::Gradient{T})(nt::ParameterContainer{T}) where {T}
    v, layout = flatten(nt)
    # `rgrad` takes the *whole* leaf, not its storage: it is the Riemannian projection and needs the
    # point it projects at, so this walks whole leaves rather than their storage.
    mapparameters(rgrad, nt, unflatten(layout, grad(v)))
end

# This is *not* type piracy, unlike the method above: it dispatches on `OptimizerState`,
# which is defined in this package (see `optimizers/optimizer_state.jl`), and one owned
# argument type is enough.
function (grad::Gradient{T})(g::ParameterContainer{T}, x::ParameterContainer{T}, state::OptimizerState{T}) where {T}
    _copyto!(g, global_rep(section(state), grad(x)))
end

# `NeuralNetworkParameters.mapparameters` and not `map`, here and in every primitive below. `map`
# visits the entries of one level, which is the whole of a flat `ArrayNamedTuple` but only the
# *layers* of a container, whose leaves are one level further down. `mapparameters` recurses on the
# branches, so it reaches leaves at any depth and rebuilds the shape it was given -- a container comes
# back a container, a `NamedTuple` a `NamedTuple`. See [`ParameterContainer`](@ref).
#
# The in-place primitives take `mapparameters!`, which is `foreachparameters` returning its
# destination: the tree of results a `map`-shaped walk builds is allocated and immediately discarded
# on every call, which was 992 bytes per `update!` on the flat problem of
# `scripts/optimizer_allocations.jl`.
#
# Both check that the keys agree at every level, which is the property this file depends on and which
# `Base.foreach` over `NamedTuple`s does *not* have -- it goes through `zip`, iterates values, and so
# neither compares the keys nor notices that one tree is shorter. `mapparameters` normalises its
# trailing arguments through an exhaustive three-method `_as_namedtuple`, so a container may be walked
# in lockstep with the plain `NamedTuple` tree `GlobalSection(::NetworkParameters)` deliberately
# returns, and pairing a *leaf* with a branch raises a `MethodError` naming the type.
#
# `_mapleaves`/`_mapleaves!` in `src/parameter_walks.jl` were a local copy of all of this until
# 0.6.0. They existed only because `mapparameters` could not be compiled on a wide-flat set -- see
# the 0.6.0 entry in the changelog, and `NeuralNetworkParameters` 0.2.2, which fixed that from this
# package's report. The local copy also normalised its trailing arguments with a *catch-all*, so a
# leaf paired with a branch fell through to the generic iterator `map`, which zipped the branch's
# entries against the leaf's elements and returned a truncated `Array` instead of raising.
_zero(a::AbstractArray) = zero(a)
_zero(a::ParameterContainer) = mapparameters(_zero, a)

_copy(a::AbstractArray) = copy(a)
_copy(a::ParameterContainer) = mapparameters(_copy, a)

# `Base.similar` is deliberately an error on a `Manifold` — an arbitrary array of that shape is not a
# point of it — so a fresh *random* point stands in for it. `Manifold` and not `StiefelManifold`, and
# built with `manifold_constructor` for the same reason `flatten` above is: a `NamedTuple` holding a
# `GrassmannManifold` used to reach the `AbstractArray` method below and raise that error while
# building an `AdamState` or a `MomentumState`. See issue A11.
_similar(a::Manifold{T}) where {T} = rand(manifold_constructor(a){T}, size(a)...)
_similar(a::AbstractArray) = similar(a)
_similar(a::ParameterContainer) = mapparameters(_similar, a)

_fill!(a::AbstractArray{T}, b::T) where {T} = fill!(a, b)

_fill!(a::Manifold{T}, ::T) where {T} = a

_copyto!(a::AbstractArray{T}, b::AbstractArray{T}) where {T} = copyto!(a, b)
function _copyto!(a::ParameterContainer{T}, b::ParameterContainer{T}) where {T}
    mapparameters!(_copyto!, a, b)
end

# Type piracy again by way of the aliases: both `GlobalSectionNamedTuple` and
# `ArrayNamedTuple` are `NamedTuple`. See issue #16.
#
# These come in pairs, and the second of each pair is what a *container* solution needs.
# `GlobalSectionNamedTuple` is flat by construction — a `NamedTuple` whose values are `GlobalSection`s
# — and the section tree of a container is nested, its values being layers. There is no widening of
# the alias that would cover both: a "`NamedTuple` of `GlobalSection`s to any depth" is a recursive
# type, which Julia cannot express. So the *other* side carries the dispatch, and it can, because a
# container is a type with a name rather than an alias for `NamedTuple`. That is the one place where
# taking the container bought this file something beyond a wider signature.
#
# `mapparameters!` walks whichever shape it is given first and normalises the rest, so the bodies are
# identical either way.
function Base.copyto!(Λ::GlobalSectionNamedTuple{T}, x::ParameterContainer{T}) where {T}
    mapparameters!(copyto!, Λ, x)
    Λ
end

function Base.copyto!(Λ::NamedTuple, x::NetworkParameters)
    mapparameters!(copyto!, Λ, x)
    Λ
end

function Base.copyto!(Λ::GlobalSection{T,MT}, x::MT) where {T,MT<:Manifold}
    # only the anchor moves; `Λ.λ` is deliberately left alone, since recomputing the lift would move
    # the frame the secant pair of a quasi-Newton method is expressed in
    copyto!(Λ.Y, x)
    Λ
end

_copyto!(Λ::GlobalSectionNamedTuple, x::ParameterContainer) = copyto!(Λ, x)
_copyto!(Λ::NamedTuple, x::NetworkParameters) = copyto!(Λ, x)

# the bare-`Manifold` counterpart of the line above
_copyto!(Λ::GlobalSection{T,MT}, x::MT) where {T,MT<:Manifold} = copyto!(Λ, x)

function _copyto!(x::ParameterContainer, Λ::GlobalSectionNamedTuple)
    mapparameters!(copyto!, x, Λ)
    x
end

function _copyto!(x::NetworkParameters, Λ::NamedTuple)
    mapparameters!(copyto!, x, Λ)
    x
end

function _copyto!(Λ₁::GlobalSectionNamedTuple, Λ₂::GlobalSectionNamedTuple)
    mapparameters!(_copyto!, Λ₁, Λ₂)
    Λ₁
end

# Two *nested* section trees, which is the shape a container's section takes and which
# `GlobalSectionNamedTuple` cannot describe. Written on the bare `NamedTuple` because neither argument
# is a container to dispatch on; the flat method above is strictly more specific, so it still wins
# where it applies, and the leaves settle the rest -- a pair that is not two sections has no
# `_copyto!` at the bottom of this walk either way.
function _copyto!(Λ₁::NamedTuple, Λ₂::NamedTuple)
    mapparameters!(_copyto!, Λ₁, Λ₂)
    Λ₁
end

function _copyto!(Λ₁::GlobalSection{T,MT}, Λ₂::GlobalSection{T,MT}) where {T,MT<:Manifold{T}}
    _copyto!(Λ₁.Y, Λ₂.Y)
    _copyto!(Λ₁.λ, Λ₂.λ)
    Λ₁
end

function _fill!(a::ParameterContainer{T}, b::T) where {T}
    fill_closure!(_a) = _fill!(_a, b)
    mapparameters!(fill_closure!, a)
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

_difference!(c::ParameterContainer{T}, a::ParameterContainer{T}, b::ParameterContainer{T}) where {T} =
    mapparameters!(_difference!, c, a, b)

_rmul!(a::AbstractArray, b) = rmul!(a, b)

# `LinearAlgebra.rmul!` writes back through `setindex!`, which three of these four do not have;
# scaling the free parameters is the same operation and is what the matrix they represent scales by.
function _rmul!(a::VectorStorageMatrix, b)
    rmul!(parent(a), b)
    a
end

function _rmul!(a::ParameterContainer, b)
    rmul_closure!(a) = _rmul!(a, b)
    mapparameters!(rmul_closure!, a)
    a
end

function _mul!(c::AbstractVecOrMat, a::AbstractMatrix, b::AbstractVecOrMat)
    mul!(c, a, b)
end

# Two more `_mul!` methods stood here until 0.6.0, one for a container destination and one for a bare
# lift, each flattening `b`, allocating a result vector and unflattening it back. `_flat_mul!` does that
# through the cache's buffers now, so neither had a caller left. See [`_flat_scratch`](@ref).
#
# `_mul!(c::ParameterContainer, a::ParameterContainer, b::ParameterContainer)` -- the *elementwise*
# product, three parameter sets -- is gone with them, and had no caller before this release either.

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
quantity in this package is expressed in — `Q` is sized by the flattening, its outer products are
formed there (see [`_flat_scratch`](@ref)), and the `α` of a line search parameterizes a curve in them —
so
pairing a gradient with a direction has to happen there too.

Used by [`trial_slope`](@ref) for ``\varphi'(\alpha)``, and by the quasi-Newton caches for
``\delta^T\gamma``, whose value has to be consistent with the flattened `T₁`, `T₂` and `γ^TQγ` it
divides.

!!! info "No flat vector is built"
    This is the *value* the flattened inner product has, not the flattening. `flatten` writes the
    leaves one after another into one vector, so ``\langle\mathrm{flatten}(a),
    \mathrm{flatten}(b)\rangle`` is the sum of the per-leaf inner products, and the sum can be taken
    without the vectors. Until 0.6.0 this allocated two of them per call — once per line-search trial
    slope, which is the hottest site there is, once per `OptimizerStatus`, and twice per quasi-Newton
    `update!`.

    The summation order changes with it: per leaf and then across, rather than one `dot` over the
    concatenation. Both are ``\sum_i a_ib_i``; they differ at round-off, and
    `test/flat_buffer_allocations.jl` pins the two against each other.
"""
_dot(a::AbstractVecOrMat, b::AbstractVecOrMat) = dot(a, b)

const LiftOrNamedTuple{T} = Union{AbstractLieAlgHorMatrix{T},ParameterContainer{T}}

# `T` is named on the result rather than on a `flatten`, and for the same reason the `flatten(T, ·)`
# this replaced named it there: every quantity this is combined with downstream is a `T`, and a
# container's `T` is a *promotion* over its leaves rather than a guarantee about each of them, so a
# mixed-precision set must not decide the type of the pairing. It is the one place where the two forms
# differ by more than round-off — the old one converted the leaves and then paired them, this one
# pairs them and then converts.
_dot(a::LiftOrNamedTuple{T}, b::LiftOrNamedTuple{T}) where {T} = T(_dot_leaves(a, b))

# Down to the free parameters and no further, exactly as `flatten` goes: `dot` of a lift is the
# *ambient* Frobenius product and `dot` of a [`VectorStorageMatrix`](@ref) reads a dense interface that
# has neither the right length nor, for three of the four, any way to be read at all. `freeparameters`
# is the same protocol `flatten` walks, so the two agree leaf for leaf by construction rather than by
# two implementations happening to concur.
#
# Positional, over `values`, and so it checks neither that the keys agree nor that the two branches are
# the same width -- where `mapparameters` gets the first from `_check_keys` and the second from
# `_children_arity`. That is not a regression: the
# `dot(flatten(a), flatten(b))` this replaced was positional over the flattening in exactly the same
# way. It is named here because this is where such a check would go if one is ever wanted, and because
# the arity case is the worse of the two: a width mismatch falls through to the generic method below
# with a `Tuple` in hand and raises `freeparameters`' "no protocol" error, which names neither `_dot`
# nor the shapes.
_dot_leaves(a::ParameterSet, b) = _dot_leaves(values(a), values(b))

# `false` is the strong zero: it takes its type from whatever it is added to, and there is no `T` in
# scope to write `zero(T)` with. A one-leaf set adds it to that leaf's pairing and stays a `T`.
_dot_leaves(::Tuple{}, ::Tuple{}) = false
_dot_leaves(a::Tuple, b::Tuple) =
    _dot_leaves(first(a), first(b)) + _dot_leaves(Base.tail(a), Base.tail(b))

# `s === a` is `NeuralNetworkParameters.isterminal(a)`, which exists for exactly this question. Written
# out because the storage is wanted either way, so asking the predicate would call `freeparameters`
# twice -- but named, so that the two cannot drift apart unnoticed.
function _dot_leaves(a, b)
    s = freeparameters(a)
    s === a ? dot(a, b) : _dot_leaves(s, freeparameters(b))
end

_add!(a::AbstractArray{T}, b::AbstractArray{T}) where {T} = a .+= b

function _add!(a::MT, b::MT) where {MT<:VectorStorageMatrix}
    _add!(parent(a), parent(b))
    a
end

function _add!(a::ParameterContainer{T}, b::ParameterContainer{T}) where {T}
    mapparameters!(_add!, a, b)
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

function _add!(a::ParameterContainer{T}, b::T) where {T}
    closure(a) = _add!(a, b)
    mapparameters!(closure, a)
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

_rac!(b::ParameterContainer, a::ParameterContainer) = mapparameters!(_rac!, b, a)

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

function _div!(C::ParameterContainer, A::ParameterContainer, B::ParameterContainer)
    mapparameters!(_div!, C, A, B)
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

_square!(b::ParameterContainer, a::ParameterContainer) = mapparameters!(_square!, b, a)

function _square(a)
    b = _copy(a)
    _square!(b, a)
    b
end


Base.copyto!(dest::AT, src::GlobalSection{T,AT}) where {T,AT<:AbstractArray{T}} = copyto!(dest, src.Y)
_copyto!(dest, src::GlobalSection) = copyto!(dest, src)
rgrad(ps::ParameterContainer, dx::ParameterContainer) = mapparameters(rgrad, ps, dx)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end
