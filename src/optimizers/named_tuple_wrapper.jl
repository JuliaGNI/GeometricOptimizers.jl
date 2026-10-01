# `GradientAutodiff(F, ps)` and `GradientFunction(F, ∇F!, ps)` for a parameter set are not here. Both are `SimpleSolvers` functions taking a `NeuralNetworkParameters` type, so this package
# owned neither side of either signature and every package that loaded this one -- directly or through
# a dependency -- got the new meaning for the rest of its session. They are `SimpleSolvers`' own
# methods as of 0.13.2, in `ext/SimpleSolversNeuralNetworkParametersExt.jl`, and reach this package
# unchanged because both packages are hard dependencies here. `GradientFiniteDifferences` never had a
# parameter-set method and still does not. See issue #16.
#
# What the extension could not take with it is the *functor*, whose body is `rgrad`, this package's
# Riemannian projection. That one is de-pirated by wrapping instead: see
# [`RiemannianGradient`](@ref) in `utils.jl`.

# This pairs `ps` with the unflattened gradient leaf by leaf, so both trees have to hold arrays of one
# element type for `rgrad` to have a method at every position -- which a container guarantees and a
# loose pairing would not.
function (grad::RiemannianGradient{T})(ps::NetworkParameters{T}) where {T}
    v, layout = flatten(ps)
    # `rgrad` takes the *whole* leaf, not its storage: it is the Riemannian projection and needs the
    # point it projects at, so this walks whole leaves rather than their storage.
    mapparameters(rgrad, ps, unflatten(layout, grad.gradient(v)))
end

# `Gradient` and not `RiemannianGradient`: this one needs no wrapper to be owned, because it
# dispatches on `OptimizerState`, which is defined in this package (see
# `optimizers/optimizer_state.jl`), and one owned argument type is enough. It stays on the abstract
# type so that a caller with a gradient of its own that knows how to project onto a parameter set
# reaches it too; `grad(x)` below is what has to have such a method, and for anything
# [`Optimizer`](@ref) builds that is [`RiemannianGradient`](@ref).
function (grad::Gradient{T})(g::NetworkParameters{T}, x::NetworkParameters{T},
        state::OptimizerState{T}) where {T}
    _copyto!(g, global_rep(section(state), grad(x)))
end

# `NeuralNetworkParameters.mapparameters` and not `map`, here and in every primitive below. `map`
# visits the entries of one level, which for a container is its *layers* -- its leaves are one level
# further down, and further still for a deeper network. `mapparameters` recurses on the branches, so it
# reaches leaves at any depth and rebuilds the shape it was given: a container comes back a container
# and the plain `NamedTuple` of a section tree comes back a plain `NamedTuple`.
#
# The in-place primitives take `mapparameters!`, which is `foreachparameters` returning its
# destination: the tree of results a `map`-shaped walk builds is allocated and immediately discarded
# on every call, which was 992 bytes per `update!` on the flat problem of
# `scripts/optimizer_allocations.jl`.
#
# Both check that the keys agree at every level, which is the property this file depends on and which
# `Base.foreach` over `NamedTuple`s does *not* have -- it goes through `zip`, iterates values, and so
# neither compares the keys nor notices that one tree is shorter. [`_dot`](@ref) joins them in this
# release, and more cheaply than either: `foldstorage` checks the keys and the widths in its
# *generator*, so both cost nothing at run time and a mismatch raises before the fold is specialised at
# all. Until now that one paired positionally over `values` and checked neither — inherited from the
# `dot(flatten(a), flatten(b))` it replaced rather than chosen, and its own comment said this was where
# such a check would go if one were ever wanted. `mapparameters` normalises its
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
# the zero tangent vector at a point, which is a horizontal lift and not the point's own shape
_zero(a::Union{StiefelManifold, GrassmannManifold}) = zero_tangent(a)
_zero(a::NetworkParameters) = mapparameters(_zero, a)

_copy(a::AbstractArray) = copy(a)
_copy(a::NetworkParameters) = mapparameters(_copy, a)

# `Base.similar` is deliberately an error on a `Manifold` — an arbitrary array of that shape is not a
# point of it — so a fresh *random* point stands in for it. `Manifold` and not `StiefelManifold`, and
# built with `manifold_constructor` for the same reason `flatten` above is: a `NamedTuple` holding a
# `GrassmannManifold` used to reach the `AbstractArray` method below and raise that error while
# building an `AdamState` or a `MomentumState`. See issue A11.
#
# The point is drawn on `a`'s own backend, as `_zero` and `_copy` allocate by construction:
# `AdamState` and `MomentumState` declare a single `OT` for `x` and `x̄`, and `x̄` is `_similar(x)`.
function _similar(a::Manifold{T}) where {T}
    rand(KernelAbstractions.get_backend(a), manifold_constructor(a){T}, size(a)...)
end
_similar(a::AbstractArray) = similar(a)
_similar(a::NetworkParameters) = mapparameters(_similar, a)

_fill!(a::AbstractArray{T}, b::T) where {T} = fill!(a, b)

# The elementwise primitives. Each walks a leaf, or a whole parameter set, to its leaves, checks that
# a structured leaf meets a leaf of its own kind and size, and applies one elementwise operation to
# the free parameters with `mapstorage!`. So a structured matrix and a horizontal lift need no method
# of their own: `freeparameters` is the storage vector of a `VectorStorageMatrix` and the tuple of
# blocks of a lift. Both walks skip a `nothing` in a source set and allocate nothing. The signatures
# bind one element type where the operation needs it. Only `_copyto!` writes a manifold point, and it
# calls each leaf's own `copyto!`, with its checks; the others do not guard a point.
function _copyto!(a::GradientStorage{T}, b::GradientStorage{T}) where {T}
    mapparameters!(copyto!, a, b)
end

const _StructuredLeaf = Union{Manifold, VectorStorageMatrix, AbstractLieAlgHorMatrix}

_check_leaves(x, ys::Vararg{Any, N}) where {N} = nothing
function _check_leaves(x::_StructuredLeaf, ys::Vararg{Any, N}) where {N}
    foreach(y -> _check_leaf_pair(x, y), ys)
end
# a horizontal lift reports `(N, N)` whatever its `n`, so `n` is part of its shape
_leaf_shape(x) = size(x)
_leaf_shape(x::AbstractLieAlgHorMatrix) = (size(x)..., x.n)

function _check_leaf_pair(x, y)
    Base.typename(typeof(x)) === Base.typename(typeof(y)) &&
    _leaf_shape(x) == _leaf_shape(y) ||
        throw(ArgumentError(string("an elementwise primitive pairs a `", nameof(typeof(x)),
            "` of shape ", _leaf_shape(x), " with a `", nameof(typeof(y)), "` of shape ",
            _leaf_shape(y))))
    nothing
end

struct _StorageStep{F} <: Function
    f::F
end
function (step::_StorageStep)(dest, srcs::Vararg{Any, N}) where {N}
    _check_leaves(dest, srcs...)
    mapstorage!(step.f, dest, srcs...)
end
function _storagewise!(f::F, dest, srcs::Vararg{Any, N}) where {F, N}
    mapparameters!(
        _StorageStep(f), dest, srcs...)
end

# These come in pairs, and the second of each pair is what a *nested* container needs.
# `GlobalSectionNamedTuple` is flat by construction — a `NamedTuple` whose values are `GlobalSection`s
# — and the section tree of a container is nested, its values being layers. There is no widening of
# the alias that would cover both: a "`NamedTuple` of `GlobalSection`s to any depth" is a recursive
# type, which Julia cannot express. So the *other* side carries the dispatch, and it can, because a
# container is a type with a name rather than an alias for `NamedTuple`. That asymmetry is what makes
# each pair *order* itself: `(::GlobalSectionNamedTuple{T}, ::NetworkParameters{T})` is strictly more
# specific than `(::NamedTuple, ::NetworkParameters)`, and likewise the other way round, so dispatch
# picks the section pairing on the overlap without being told to. Were a parameter set allowed to be a
# bare `NamedTuple` as well, neither method of a pair would be more specific and every one of these
# calls would be a run-time `MethodError: … is ambiguous`; see [`OptimizerSolution`](@ref).
#
# `mapparameters!` walks whichever shape it is given first and normalises the rest, so the bodies are
# identical either way.
#
# `_copyto!` and not `Base.copyto!`, and that is about ownership rather than taste: `copyto!` is
# `Base`'s, `NamedTuple` is `Base`'s and `NetworkParameters` is `NeuralNetworkParameters`', so a
# `Base.copyto!` method pairing them would own neither side. `_copyto!` is this package's own function,
# which is enough. Every caller here and in `GeometricMachineLearning` goes through it.
#
# The `copyto!` passed to `mapparameters!` is the *leaf* operation and stays `Base`'s: at the bottom of
# this walk a pair is two arrays or a `GlobalSection` and its anchor, and the method for the latter
# dispatches on a type of this package's own.
function _copyto!(Λ::GlobalSectionNamedTuple{T}, x::NetworkParameters{T}) where {T}
    mapparameters!(copyto!, Λ, x)
    Λ
end

function _copyto!(Λ::NamedTuple, x::NetworkParameters)
    mapparameters!(copyto!, Λ, x)
    Λ
end

# the storage type is free in both arguments, for the reason given on the section-to-section
# `copyto!` methods in `global_sections.jl`, and the lift is bound to an array for the reason given
# there too
function Base.copyto!(Λ::GlobalSection{T, <:Manifold, <:AbstractArray}, x::Manifold) where {T}
    # only the anchor moves; `Λ.λ` is deliberately left alone, since recomputing the lift would move
    # the frame the secant pair of a quasi-Newton method is expressed in
    copyto!(Λ.Y, x)
    Λ
end

# the bare-`Manifold` counterpart of the line above
function _copyto!(Λ::GlobalSection{T, <:Manifold, <:AbstractArray}, x::Manifold) where {T}
    copyto!(Λ, x)
end

# The section of a state that starts a solve at `x`; see `initialize_state!`. Unlike `copyto!` above,
# this gives a manifold section the frame of its new anchor, because the frame of another point does
# not complete `x`. A section already anchored at `x` keeps its frame, so the solve of a fresh state
# draws no random number.
function _start_section!(Λ::GlobalSection{T, <:AbstractVecOrMat{T}, Nothing}, x) where {T}
    copyto!(Λ, x)
end

function _start_section!(Λ::GlobalSection{T, <:Manifold, <:AbstractArray}, x::Manifold) where {T}
    if parent(Λ.Y) != parent(x)
        copyto!(Λ.Y, x)
        copyto!(Λ.λ, global_section(x))
    end
    Λ
end

function _start_section!(Λ::NamedTuple, x::NetworkParameters)
    mapparameters!(_start_section!, Λ, x)
    Λ
end

function _copyto!(x::NetworkParameters, Λ::GlobalSectionNamedTuple)
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

function _copyto!(Λ₁::GlobalSection{T, MT}, Λ₂::GlobalSection{
        T, MT}) where {T, MT <: Manifold{T}}
    _copyto!(Λ₁.Y, Λ₂.Y)
    _copyto!(Λ₁.λ, Λ₂.λ)
    Λ₁
end

function _fill!(a::NetworkParameters{T}, b::T) where {T}
    fill_closure!(_a) = _fill!(_a, b)
    mapparameters!(fill_closure!, a)
    a
end

function _difference!(c::GradientStorage{T}, a::GradientStorage{T},
        b::GradientStorage{T}) where {T}
    _storagewise!(c, a, b) do c, a, b
        @assert axes(a) == axes(b) == axes(c)
        c .= a .- b
    end
end

_rmul!(a::GradientStorage, b) = _storagewise!(a -> rmul!(a, b), a)

function _mul(α::T, a::GradientStorage{T}) where {T}
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

Used by [`trial_slope`](@ref) for ``\varphi'(\alpha)``, by the quasi-Newton caches for
``\delta^T\gamma``, whose value has to be consistent with the flattened `T₁`, `T₂` and `γ^TQγ` it
divides, for the predicted decrease ``\widetilde{\Delta f}``, so that it is comparable with the
measured ``\Delta f``, and by [`ensure_descent!`](@ref)'s descent test, for the same reason.

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

    What the *grouping* of the leaves does to that sum is nothing, and as of this release that holds by
    construction rather than by luck. `foldstorage` threads its accumulator through the nested branches,
    so a left fold over a tree is the left fold over the flat leaf list whatever shape the tree has —
    where the `Base.tail` recursion this replaced was a right fold that happened to align. So the same
    numbers written flat, written nested, and wrapped in a container all pair to the same `Float64`,
    exactly, and the test asserts `==` for the three of them.
"""
_dot(a::AbstractVecOrMat, b::AbstractVecOrMat) = dot(a, b)

# `foldstorage` walks down to the free parameters, as `flatten` does, so the two agree leaf for
# leaf. A named function and not a closure, so that the fold takes it as a constant.
_dot_leaf(acc, x, y) = acc + dot(x, y)

const LiftOrParameters{T} = Union{AbstractLieAlgHorMatrix{T}, NetworkParameters{T}}

# Everything `_dot` accepts, with the element type left off: the pair whose element types differ. A
# pair of lifts of two element types reaches this and not the `AbstractVecOrMat` method above, whose
# ambient Frobenius product is twice the intrinsic one.
const DottableSet = Union{AbstractLieAlgHorMatrix, NetworkParameters}

# The accumulator starts at `zero(T)`, the promotion over the leaves, and not at the strong zero
# `false`, which would take the type of the first leaf of the left fold.
function _dot(a::LiftOrParameters{T}, b::LiftOrParameters{T}) where {T}
    foldstorage(_dot_leaf, zero(T), a, b)
end

# The pair whose element types differ, which binds no `T`; `parameter_eltype` is the promotion over
# the leaves. The method above is strictly more specific, so it takes every pair of one element
# type.
function _dot(a::DottableSet, b::DottableSet)
    foldstorage(
        _dot_leaf, zero(promote_type(parameter_eltype(a), parameter_eltype(b))), a, b)
end

function _add!(a::GradientStorage{T}, b::GradientStorage{T}) where {T}
    _storagewise!((a, b) -> a .+= b, a, b)
end

_add!(a::GradientStorage{T}, b::T) where {T} = _storagewise!(a -> a .+= b, a)

"""
    _rac!(B, A)

Compute the element-wise square-root of `A`.
"""
_rac!(B::GradientStorage, A::GradientStorage) = _storagewise!((B, A) -> B .= sqrt.(A), B, A)

"""
    _div!(C, A, B)

Divide `A` by `B` (elment-wise)
"""
function _div!(C::GradientStorage, A::GradientStorage, B::GradientStorage)
    _storagewise!(C, A, B) do C, A, B
        @assert axes(A) == axes(B) == axes(C)
        C .= A ./ B
    end
end

"""
    _square!(B, A)

"""
function _square!(B::GradientStorage, A::GradientStorage)
    _storagewise!((B, A) -> B .= A .^ 2, B, A)
end

function _square(a)
    b = _copy(a)
    _square!(b, a)
    b
end

function Base.copyto!(dest::AbstractArray{T}, src::GlobalSection{T}) where {T}
    copyto!(dest, src.Y)
    dest
end
_copyto!(dest, src::GlobalSection) = copyto!(dest, src)
rgrad(ps::NetworkParameters, dx::NetworkParameters) = mapparameters(rgrad, ps, dx)

function rgrad(Y::AbstractVecOrMat, dx::AbstractVecOrMat)
    @assert size(Y) == size(dx)
    dx
end
