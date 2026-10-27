# The methods of `DirectStiefelMethod` step a `StiefelManifold` in its tangent space. Their caches
# and states hold `M × k`, `k × k` and length-`k` arrays and no `GlobalSection`, and their training
# step is in `optimizers/training_optimizer.jl`. This file holds what the three share, the cache of
# `DirectMomentum` and `DirectScalarMomentAdam`, and `DirectMomentum`'s state and step.

# `rgrad(Y, ∇L, metric)` written into `G_R`, with `ws.YᵀX` and `ws.A` for the `k × k` products. The
# Euclidean form is the tangent projection at `Y`, which is the kernel of `ProjectionTransport` with
# `Y` as the new point.
function _direct_rgrad!(G_R, Y::StiefelManifold{T}, ∇L, ::CanonicalMetric,
        ws::DirectCayleyWorkspace{T}) where {T}
    rgrad!(G_R, ws.YᵀX, Y, ∇L)
end

function _direct_rgrad!(G_R, Y::StiefelManifold{T}, ∇L, ::EuclideanMetric,
        ws::DirectCayleyWorkspace{T}) where {T}
    _check_gradient_backend(Y, ∇L)
    _transport!(G_R, Y, Y, ∇L, ∇L, ProjectionTransport(), ws)
end

# The weights of the bias-corrected moments of step `t`, as in `Adam`.
_bias_corrected_weights(β, t) = ((β - β^t) / (1 - β^t), (1 - β) / (1 - β^t))

# `D ← -min(α, α_max / (‖D‖_F + δ)) D`: the step size of the line search, capped so that the
# Frobenius norm of the step is at most `α_max`.
function _capped_step!(D, α, method::Union{DirectScalarMomentAdam, FactoredStiefelAdam})
    s = min(α, method.α_max / (norm(D) + method.δ))
    D .*= -s
end

"""
    DirectStiefelCache <: OptimizerCache

The cache of [`DirectMomentum`](@ref) and [`DirectScalarMomentAdam`](@ref):

- `x::`[`StiefelManifold`](@ref): the point the step is taken from,
- `g`: the Riemannian gradient at `x`, an `M × k` matrix,
- `δ`: the step ``\\Delta``, a tangent vector at `x`.
"""
struct DirectStiefelCache{T, XT, AT} <: OptimizerCache{T}
    x::XT
    g::AT
    δ::AT
end

function DirectStiefelCache(x::StiefelManifold{T}) where {T}
    DirectStiefelCache{T, typeof(x), typeof(x.A)}(_copy(x), zero(x.A), zero(x.A))
end

direction(cache::DirectStiefelCache) = cache.δ

OptimizerCache(::DirectMomentum, x::StiefelManifold) = DirectStiefelCache(x)

"""
    DirectMomentumState <: OptimizerState

The state of [`DirectMomentum`](@ref): the iteration number, the point `x` and the momentum `p`, a
tangent vector at `x`.
"""
mutable struct DirectMomentumState{T, XT, AT} <: OptimizerState{T}
    iterations::Int
    x::XT
    p::AT
end

function DirectMomentumState(x::StiefelManifold{T}) where {T}
    _x = _copy(x)
    DirectMomentumState{T, typeof(_x), typeof(x.A)}(0, _x, zero(x.A))
end

OptimizerState(::DirectMomentum, x::StiefelManifold) = DirectMomentumState(x)

# The tangent vector at the point of the state that the training step transports to the new point.
_transported_moment(state::DirectMomentumState) = state.p

# The transport of the first moment of the state from `Y` to `Y_new`, along the step of the cache.
function _transport_moment!(state, cache, method, Y_new, Y, ws)
    B = _transported_moment(state)
    transport!(B, Y_new, Y, direction(cache), B, method.transport, ws)
end

# `p ← αp + G_R` and `Δ = -ηp`, for the momentum `p` already transported to the point of the step.
function _direct_step!(cache::DirectStiefelCache{T}, state::DirectMomentumState{T},
        method::DirectMomentum{T}, η) where {T}
    state.p .= method.α .* state.p .+ cache.g
    cache.δ .= -η .* state.p
end
