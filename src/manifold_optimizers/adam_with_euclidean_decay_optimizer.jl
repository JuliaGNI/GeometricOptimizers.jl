# `AdamWithEuclideanDecay` differs from `Adam` in the direction only — the moments, and hence
# the cache and the state, are the same objects. It therefore reuses [`AdamCache`](@ref) and
# [`AdamState`](@ref) rather than duplicating them, and
# `OptimizerState(AdamWithEuclideanDecay(), x)` returns an `AdamState`.
Hessian(::AdamWithEuclideanDecay, ::OptimizerProblem, ::OptimizerSolution{T}) where {T} = NoHessian{T}()
OptimizerState(::AdamWithEuclideanDecay, x...) = AdamState(x...)

@doc raw"""
    _is_decayable(x)

Whether Euclidean weight decay can move `x` at all.

An ordinary array is decayable and a `NamedTuple` is decayable as soon as one of its entries is,
which is the case [`AdamWithEuclideanDecay`](@ref) exists for. The
[`StiefelManifold`](@ref) and the [`GrassmannManifold`](@ref) are not: ``\lambda{}x`` is the
gradient of ``\frac{\lambda}{2}||x||^2``, which is constant on them, so the decay vanishes there
identically (see the [Weight Decay on Manifolds](@ref) page).

# Implementation

This is the *only* place that geometric fact is recorded — [`_weight_decay!`](@ref) consults it
rather than restating it, so the no-op and the warning in `OptimizerCache` cannot drift apart.

It is declared on the two concrete manifolds and not on [`Manifold`](@ref) on purpose. What makes
the decay vanish is that these two are *compact*, with ``||Y||_F^2 = \mathrm{tr}(Y^TY) = n``; it
is not a consequence of being a manifold, and on a noncompact one Euclidean decay may well do
something. A manifold added later therefore has to decide for itself, and gets an error saying so
rather than inheriting a silent no-op that may be wrong for it.
"""
_is_decayable(::StiefelManifold) = false
_is_decayable(::GrassmannManifold) = false
_is_decayable(::AbstractArray) = true
# `any` over the values and not over the leaves: a block that is itself a branch is asked recursively,
# which is what a container needs -- its top-level values are layers, never parameters. Written on
# `ParameterSet` rather than on [`ParameterContainer`](@ref) because a layer
# whose weights do not share one element type is not an `ArrayNamedTuple`, and the recursion has to
# reach it all the same.
_is_decayable(ps::ParameterSet) = any(_is_decayable, values(ps))

# no `Manifold` fallback: see the docstring above. Without this method the `AbstractArray` one
# would catch a new manifold (`Manifold <: AbstractMatrix`) and claim it *is* decayable, which is
# the one answer that is certainly wrong — `δ` is a horizontal lift and cannot be added to `x`.
function _is_decayable(Y::Manifold)
    error("`AdamWithEuclideanDecay` does not know whether Euclidean weight decay acts on a " *
          "$(nameof(typeof(Y))). It vanishes on `StiefelManifold` and `GrassmannManifold` " *
          "because both are compact, so `λ‖Y‖²/2` is constant there; that argument does not " *
          "carry over on its own. Add an `_is_decayable(::$(nameof(typeof(Y))))` method — and, " *
          "if it is `true`, a `_weight_decay!` method that says what the decay does in the " *
          "horizontal representation.")
end

function OptimizerCache(method::AdamWithEuclideanDecay{T}, x::OptimizerSolution{T}) where {T}
    # A `λ` that cannot reach a single weight is almost certainly not what was meant, and it is
    # invisible otherwise: the run is `Adam`, it converges, and nothing anywhere says that the
    # decay was dropped. Warning here rather than at every step costs one check per optimizer.
    if !iszero(method.λ) && !_is_decayable(x)
        @warn "`AdamWithEuclideanDecay` was given λ = $(method.λ), but none of the parameters " *
              "can be decayed by it: Euclidean weight decay is identically zero on a manifold " *
              "(`rgrad(Y, λY) = 𝕆`), so this run is exactly `Adam`. Use `Adam` if that is what " *
              "is wanted; see https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28."
    end
    AdamCache(_copy(x), _zero(x), _zero(x))
end

@doc raw"""
    _weight_decay!(δ, x, λ)

Subtract the decoupled weight decay ``\lambda{}x`` from the direction `δ`, in place.

This is called after the [`Adam`](@ref) direction has been formed, i.e. `δ` is
``-m_1/(\sqrt{m_2} + \delta)`` on entry and ``-m_1/(\sqrt{m_2} + \delta) - \lambda{}x`` on
exit. It has to happen *before* the line search scales the direction, so that the decay is
multiplied by the learning rate as [loshchilov2019decoupled](@cite) prescribes, and it must not
touch the moments — that is the whole point of the *decoupling*, see
[`AdamWithEuclideanDecay`](@ref).

# Implementation

The [`Manifold`](@ref) method is a genuine no-op and not an omission: `δ` is then an element of
``\mathfrak{g}^\mathrm{hor}`` rather than something that could be added to `x` at all, and the
quantity it would have to be corrected by — the Riemannian gradient of
``\frac{\lambda}{2}||x||^2`` — is zero. It asserts [`_is_decayable`](@ref) rather than restating
why, so that the no-op holds only for the manifolds on which that has been established; see
[issue #28](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28) for what a decay that
did something here would have to look like. `test/adam_with_euclidean_decay.jl` checks that
`rgrad(Y, λY)` really does vanish rather than taking it on trust.
"""
function _weight_decay!(δ::AbstractArray{T}, x::AbstractArray{T}, λ::T) where {T}
    @assert axes(δ) == axes(x)
    δ .-= λ .* x
    δ
end

function _weight_decay!(δ::AbstractLieAlgHorMatrix{T}, x::Manifold{T}, ::T) where {T}
    # constant-folded to nothing for the manifolds that define `_is_decayable`, and the error of
    # `_is_decayable(::Manifold)` for one that does not
    @assert !_is_decayable(x) "`_is_decayable($(nameof(typeof(x)))) == true` contradicts the no-op here"
    δ
end

function _weight_decay!(δ::ParameterContainer{T}, x::ParameterContainer{T}, λ::T) where {T}
    weight_decay_closure!(δᵢ, xᵢ) = _weight_decay!(δᵢ, xᵢ, λ)
    mapparameters!(weight_decay_closure!, δ, x)
    δ
end

function update!(cache::AdamCache{T}, state::AdamState{T}, gradient::Gradient{T}, method::AdamWithEuclideanDecay{T}, x::OptimizerSolution{T}) where {T}
    update!(cache, state, gradient, method.β₁, method.β₂, method.δ, state.iterations, x)
    _weight_decay!(direction(cache), x, method.λ)

    cache
end
