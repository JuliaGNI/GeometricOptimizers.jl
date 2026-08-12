# `AdamWithEuclideanDecay` differs from `Adam` in the direction only — the moments, and hence
# the cache and the state, are the same objects. It therefore reuses [`AdamCache`](@ref) and
# [`AdamState`](@ref) rather than duplicating them, and
# `OptimizerState(AdamWithEuclideanDecay(), x)` returns an `AdamState`.
Hessian(::AdamWithEuclideanDecay, ::OptimizerProblem, ::OptimizerSolution{T}) where {T} = NoHessian{T}()
OptimizerState(::AdamWithEuclideanDecay, x...) = AdamState(x...)

"""
    _is_decayable(x)

Whether Euclidean weight decay can move `x` at all.

A [`Manifold`](@ref) is not decayable: `λx` is the gradient of a function that is *constant* on
it, so the decay vanishes there identically (see [`AdamWithEuclideanDecay`](@ref)). A
`NamedTuple` is decayable as soon as one of its entries is, which is the case
[`AdamWithEuclideanDecay`](@ref) exists for.
"""
_is_decayable(::Manifold) = false
_is_decayable(::AbstractArray) = true
_is_decayable(ps::ArrayNamedTuple) = any(_is_decayable, values(ps))

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
quantity it would have to be corrected by, the Riemannian gradient of
``\frac{\lambda}{2}||x||^2``, is zero on both manifolds of this package. See
[`AdamWithEuclideanDecay`](@ref) for the computation and
[issue #28](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28) for what a decay that
did something here would have to look like. `test/adam_with_euclidean_decay.jl` checks that
`rgrad(Y, λY)` really does vanish rather than taking it on trust.
"""
function _weight_decay!(δ::AbstractArray{T}, x::AbstractArray{T}, λ::T) where {T}
    @assert axes(δ) == axes(x)
    δ .-= λ .* x
    δ
end

_weight_decay!(δ::AbstractLieAlgHorMatrix{T}, ::Manifold{T}, ::T) where {T} = δ

function _weight_decay!(δ::ArrayNamedTuple{T}, x::ArrayNamedTuple{T}, λ::T) where {T}
    weight_decay_closure!(δ, x) = _weight_decay!(δ, x, λ)
    apply_toNT(weight_decay_closure!, δ, x)
    δ
end

function update!(cache::AdamCache{T}, state::AdamState{T}, gradient::Gradient{T}, method::AdamWithEuclideanDecay{T}, x::OptimizerSolution{T}) where {T}
    update!(cache, state, gradient, method.β₁, method.β₂, method.δ, state.iterations, x)
    _weight_decay!(direction(cache), x, method.λ)

    cache
end
