OptimizerCache(::DirectScalarMomentAdam, x::StiefelManifold) = DirectStiefelCache(x)

"""
    DirectScalarMomentAdamState <: OptimizerState

The state of [`DirectScalarMomentAdam`](@ref): the iteration number, the point `x`, the first moment
`B₁`, a tangent vector at `x`, and the scalar second moment `v`. Both moments are stored
bias-corrected.
"""
mutable struct DirectScalarMomentAdamState{T, XT, AT} <: OptimizerState{T}
    iterations::Int
    x::XT
    B₁::AT
    v::T
end

function DirectScalarMomentAdamState(x::StiefelManifold{T}) where {T}
    _x = _copy(x)
    DirectScalarMomentAdamState{T, typeof(_x), typeof(x.A)}(0, _x, zero(x.A), zero(T))
end

function OptimizerState(::DirectScalarMomentAdam, x::StiefelManifold)
    DirectScalarMomentAdamState(x)
end

# The weights of the two moments of step `t` and the second moment of that step, from the state
# before it.
function _scalar_moment_update(state::DirectScalarMomentAdamState, cache, method, t)
    c₁₁, c₁₂ = _bias_corrected_weights(method.β₁, t)
    c₂₁, c₂₂ = _bias_corrected_weights(method.β₂, t)
    c₁₁, c₁₂, c₂₁ * state.v + c₂₂ * sum(abs2, cache.g)
end

# The moments of the step are not written here: the step reads them from the state and the gradient,
# and `_transport_moment!` writes them after the retraction, so that a step that `cayley_direct!`
# refuses leaves the state as it was.
function _direct_step!(cache::DirectStiefelCache{T}, state::DirectScalarMomentAdamState{T},
        method::DirectScalarMomentAdam{T}, α) where {T}
    c₁₁, c₁₂, v = _scalar_moment_update(state, cache, method, iteration_number(state) + 1)
    cache.δ .= (c₁₁ .* state.B₁ .+ c₁₂ .* cache.g) ./ (sqrt(v) + method.δ)
    _capped_step!(cache.δ, α, method)
end

# The moments of the step just taken, and the transport of the first moment from `Y` to `Y_new`,
# along the step of the cache.
function _transport_moment!(state::DirectScalarMomentAdamState, cache, method, Y_new, Y, ws)
    c₁₁, c₁₂, v = _scalar_moment_update(state, cache, method, iteration_number(state))
    state.B₁ .= c₁₁ .* state.B₁ .+ c₁₂ .* cache.g
    state.v = v
    transport!(state.B₁, Y_new, Y, direction(cache), state.B₁, method.transport, ws)
end
