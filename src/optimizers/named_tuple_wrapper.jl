# `GradientAutodiff(F, ::ParameterSet)` and `GradientFunction(F, ∇F!, ::ParameterSet)` used to stand
# here. Both are `SimpleSolvers` functions taking a `NeuralNetworkParameters` type, so this package
# owned neither side of either signature and every package that loaded this one -- directly or through
# a dependency -- got the new meaning for the rest of its session. They are `SimpleSolvers`' own
# methods as of 0.13.2, in `ext/SimpleSolversNeuralNetworkParametersExt.jl`, and reach this package
# unchanged because both packages are hard dependencies here. `GradientFiniteDifferences` never had a
# parameter-set method and still does not. See issue #16.
#
# What the extension could not take with it is the *functor*, whose body is `rgrad`, this package's
# Riemannian projection. That one is de-pirated by wrapping instead: see
# [`RiemannianGradient`](@ref) in `utils.jl`.

# `ParameterContainer` and not `ParameterSet`: this pairs `ps` with the unflattened gradient leaf by
# leaf, so both trees have to hold arrays of one element type for `rgrad` to have a method at every
# position. The two constructors upstream take the wider `ParameterSet`, because there `F` is the
# caller's objective and has to accept whatever it was written against.
function (grad::RiemannianGradient{T})(ps::ParameterContainer{T}) where {T}
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
#
# `_copyto!` and not `Base.copyto!`, which is what these two were until this release: `copyto!` is
# `Base`'s, `NamedTuple` is `Base`'s and `NetworkParameters` is `NeuralNetworkParameters`', so a method
# pairing them owned neither side. Those were two of the three surviving sites of issue [#16] group 3,
# and unlike the third they needed no coordination to remove — every caller in this package and every
# caller in `GeometricMachineLearning` (`src/optimizers/optimizer.jl:233-245`, six sites) already went
# through `_copyto!`, and the two `Base.copyto!` methods existed only to be forwarded to. The
# signatures are unchanged, so dispatch resolves exactly as it did. The `copyto!` passed to
# `mapparameters!` is the *leaf* operation and stays `Base`'s: at the bottom of this walk a pair is two
# arrays or a `GlobalSection` and its anchor, and the method for the latter dispatches on a type of
# this package's own.
function _copyto!(Λ::GlobalSectionNamedTuple{T}, x::ParameterContainer{T}) where {T}
    mapparameters!(copyto!, Λ, x)
    Λ
end

function _copyto!(Λ::NamedTuple, x::NetworkParameters)
    mapparameters!(copyto!, Λ, x)
    Λ
end

function Base.copyto!(Λ::GlobalSection{T,MT}, x::MT) where {T,MT<:Manifold}
    # only the anchor moves; `Λ.λ` is deliberately left alone, since recomputing the lift would move
    # the frame the secant pair of a quasi-Newton method is expressed in
    copyto!(Λ.Y, x)
    Λ
end

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

# The four disambiguations the pairs above need, and the reason they are needed rather than a matter
# of taste.
#
# `ArrayNamedTuple` and `GlobalSectionNamedTuple` are *aliases for `NamedTuple`*, which is the same
# property that made half of issue #16 piracy, showing up here as ambiguity instead. Each pair above
# is written flat-then-nested — `GlobalSectionNamedTuple` for the flat section tree, a bare
# `NamedTuple` for the nested one, because "a `NamedTuple` of `GlobalSection`s to any depth" is a
# recursive type Julia cannot express — and a bare `NamedTuple` also matches a flat
# `GlobalSectionNamedTuple`, while `NetworkParameters` is also a `ParameterContainer`. So on the
# overlap neither method of a pair is more specific and dispatch has nowhere to go:
#
# | call | ambiguous between |
# |---|---|
# | `NetworkParameters ← GlobalSectionNamedTuple` | `(::ParameterContainer, ::GlobalSectionNamedTuple)` and `(::NetworkParameters, ::NamedTuple)` |
# | `GlobalSectionNamedTuple ← NetworkParameters` | `(::GlobalSectionNamedTuple, ::ParameterContainer)` and `(::NamedTuple, ::NetworkParameters)` |
# | `ArrayNamedTuple ← NetworkParameters` | `(::ParameterContainer{T}, ::ParameterContainer{T})` and `(::NamedTuple, ::NetworkParameters)` |
# | `NetworkParameters ← ArrayNamedTuple` | `(::ParameterContainer{T}, ::ParameterContainer{T})` and `(::NetworkParameters, ::NamedTuple)` |
#
# All four raise `MethodError: … is ambiguous` at run time, and the first is what a *flat*
# `NetworkParameters` with a manifold leaf hits on the first `solver_step!`. The suite misses every
# one of them: `test/network_parameters_optimizer.jl` drives a **nested** container, whose section tree
# is a `NamedTuple` of `NamedTuple`s and therefore not a `GlobalSectionNamedTuple`, and
# `test/named_tuple_parameters.jl` drives a **bare flat** `NamedTuple`, which is not a
# `NetworkParameters`. Flat-and-wrapped is the corner between them, and it is the shape a caller gets
# by wrapping a flat parameter `NamedTuple` — which is what `GMLDatasets`' MNIST scripts would produce
# if they moved to the container. `test/container_section_copy.jl` covers it now.
#
# Each body is the one the *semantics* pick, not an arbitrary tie-break, and on the overlap the two
# candidates agree anyway. The first two are section copies, so they take the section pairing. The
# last two are parameter copies — a nested section tree is a `NamedTuple` of `NamedTuple`s and so is
# never an `ArrayNamedTuple`, which makes an `ArrayNamedTuple` argument unambiguously parameters — so
# they take `_copyto!` at the leaves. That is the same function as the `copyto!` the section pairing
# uses once a leaf is two arrays (`_copyto!(a::AbstractArray{T}, b::AbstractArray{T})` forwards to it),
# so no shape that reached one of these by accident changes its answer.
# `{T}` coupled across the arguments, as everywhere else in this file. Dropping it to cover the
# mixed-element-type overlap as well does not work and is worth recording, because it looks like the
# more thorough choice: an uncoupled `(::GlobalSectionNamedTuple, ::NetworkParameters)` is *itself*
# ambiguous against the coupled `(::GlobalSectionNamedTuple{T}, ::ParameterContainer{T})` above --
# more specific in the second argument, less in the first -- so it trades one ambiguity for another.
# Closing the mixed case as well takes both a coupled and an uncoupled method per direction, and it is
# not worth four methods: a section is built from the iterate it is a frame for
# (`GlobalSection(_copy(x))`), so its element type is the iterate's by construction and a mismatched
# pair cannot arise from this package's own API. See the residue note below.
_copyto!(x::NetworkParameters{T}, Λ::GlobalSectionNamedTuple{T}) where {T} =
    (mapparameters!(copyto!, x, Λ); x)

_copyto!(Λ::GlobalSectionNamedTuple{T}, x::NetworkParameters{T}) where {T} =
    (mapparameters!(copyto!, Λ, x); Λ)

_copyto!(a::ArrayNamedTuple{T}, b::NetworkParameters{T}) where {T} = mapparameters!(_copyto!, a, b)

_copyto!(a::NetworkParameters{T}, b::ArrayNamedTuple{T}) where {T} = mapparameters!(_copyto!, a, b)

# What is left, and why it is left. `Test.detect_ambiguities` still reports pairs in this family, and
# every one of them is a type intersection with no inhabitant this package's API can build:
#
#  - **the empty `NamedTuple`.** `(;)` is vacuously a `NamedTuple` of arrays, of `GlobalSection`s and
#    of anything else, so it sits on every overlap here at once. It is unreachable: `parameter_eltype`
#    of an empty set is `Union{}`, and `Optimizer` on one raises `ArgumentError: cannot construct a
#    value of type Union{}` long before a copy happens.
#  - **the mixed-element-type overlap**, per the note above.
#  - **`copyto!(::GlobalSection{T,MT,Nothing}, ::GlobalSection{T,MT,Nothing}) where {MT<:Manifold}`**
#    (`global_sections/global_sections.jl:376` against `:382`) — a section anchored on a `Manifold`
#    whose lift is `nothing`. `GlobalSection(::Manifold)` always builds the lift; `λ === nothing` is
#    the *plain array* case, where `MT<:Manifold` does not hold.
#
# They are listed rather than closed because a method that exists only to satisfy a static checker is
# a method somebody later has to reason about. The four above were closed because each has a witness
# and one of them is reached by an ordinary `solver_step!`.


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

    What the *grouping* of the leaves does to that sum is nothing, and as of this release that holds by
    construction rather than by luck. `foldstorage` threads its accumulator through the nested branches,
    so a left fold over a tree is the left fold over the flat leaf list whatever shape the tree has —
    where the `Base.tail` recursion this replaced was a right fold that happened to align. So the same
    numbers written flat, written nested, and wrapped in a container all pair to the same `Float64`,
    exactly, and the test asserts `==` for the three of them.
"""
_dot(a::AbstractVecOrMat, b::AbstractVecOrMat) = dot(a, b)

# `foldstorage` and not `foldparameters`: down to the free parameters and no further, exactly as
# `flatten` goes. `dot` of a lift is the *ambient* Frobenius product, and `dot` of a
# [`VectorStorageMatrix`](@ref) reads a dense interface that has neither the right length nor, for
# three of the four, any way to be read at all. `foldstorage` descends through `freeparameters` until a
# leaf is terminal, which is the same protocol `flatten` walks — so the two agree leaf for leaf by
# construction rather than by two implementations happening to concur.
#
# A named method and not a closure, so that nothing here depends on how `op` is specialised. Upstream
# does not annotate its `op` and says why: the obligation is the caller's, discharged either by a
# closure (which is its own type) or by a literal like this one. What must not come between the two is a
# function boundary that only *passes* `op` along — `foldstorage` is `@inline` and folds into the body
# below, where `_dot_leaf` is a constant, but behind a `@noinline` the same fold costs 6 160 bytes at
# arity two on a 369-leaf set.
_dot_leaf(acc, x, y) = acc + dot(x, y)

const LiftOrNamedTuple{T} = Union{AbstractLieAlgHorMatrix{T},ParameterContainer{T}}

# Everything `_dot` accepts, with the element type left off. The widening is that a nested plain
# `NamedTuple` is covered now — its values are branches rather than arrays, so it is not an
# `ArrayNamedTuple` and the alias above does not reach it. `l2norm` and `solution_scale` took it in
# 0.6.0 and this was deliberately left out, for the reason the accumulator comment below gives; the
# compile sweep reported the gap as a `MethodError` on one of its four cells.
#
# **The lift is in this union to fix a wrong number, not to widen anything**, and it is the one member
# whose old behaviour was silent. An [`AbstractLieAlgHorMatrix`](@ref) is an `AbstractMatrix`, so a pair
# of lifts whose element types *differ* did not miss the alias above and raise — it fell through to
# `_dot(::AbstractVecOrMat, ::AbstractVecOrMat)` and came back with the *ambient* Frobenius product,
# which is twice the pairing of the free parameters. Measured on `St(6,3)`, `Float32` against `Float64`:
# 5.504356027190567 before, 2.7521780135952834 here, and the second is `dot(flatten(a), flatten(b))`.
# That is the factor of two `docs/src/linesearch_on_manifolds.md` gives a section to — it reaches
# [`trial_slope`](@ref), the quasi-Newton denominator and the predicted decrease. Same-eltype pairs
# always took the method above, which is why nothing caught it.
#
# It works because `parameter_eltype` recurses: its `AbstractArray` method asks `freeparameters` first
# and only falls back to `eltype` for a terminal leaf, so a lift answers with the promotion over its
# blocks rather than with the union's `Union{}` catch-all. Nothing had to be added upstream for that.
const DottableSet = Union{AbstractLieAlgHorMatrix,ParameterSet}

# `zero(T)` and not the strong zero `false`. Upstream's fold is a **left** fold where the recursion this
# replaced was a right one, so `false` would take its type from the *first* leaf in `flatten` order:
# a mixed-precision set would accumulate its narrow prefix in the narrow type, losing significant
# figures of the small terms and depending on where in the set the widest leaf happens to sit. `T` here
# is a *promotion* over the leaves rather than a guarantee about each of them, which is exactly what an
# accumulator wants -- and it is why the old form named `T` on the result, `T(_dot_leaves(a, b))`,
# converting after pairing. Accumulating in it subsumes that conversion.
_dot(a::LiftOrNamedTuple{T}, b::LiftOrNamedTuple{T}) where {T} =
    foldstorage(_dot_leaf, zero(T), a, b)

# The widened shape, and the pair whose element types differ, neither of which binds a `T` on the
# signature. `parameter_eltype` is upstream's promotion over the leaves — the same quantity
# `NetworkParameters{T}` derives at construction — so this agrees with the method above wherever both
# would apply, and that one is strictly more specific, so it wins whenever it does.
#
# It is a separate method and not one widened signature because **`parameter_eltype` is not free on a
# wide bare `NamedTuple`**: upstream's `_promote_eltypes` is a `@generated` `promote_type` chain, and at
# 369 children in one branch it costs 6 144 bytes a call even though it infers to `Type{Float32}`. That
# is the flat MNIST shape and `trial_slope` is the hottest caller there is, so it must not land on that
# path. Written as one method it did. Above, `T` comes off the signature and costs nothing; here the
# shapes that reach it are the nested ones, whose branches are narrow enough for the chain to be
# cheap. Measured in `test/flat_buffer_allocations.jl`, which pins both paths at zero.
_dot(a::DottableSet, b::DottableSet) = foldstorage(
    _dot_leaf, zero(promote_type(parameter_eltype(a), parameter_eltype(b))), a, b)

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
