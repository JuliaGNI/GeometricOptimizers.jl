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

_transported_moment(state::DirectScalarMomentAdamState) = state.B₁

function _direct_step!(cache::DirectStiefelCache{T}, state::DirectScalarMomentAdamState{T},
        method::DirectScalarMomentAdam{T}, α) where {T}
    t = iteration_number(state)
    c₁₁, c₁₂ = _bias_corrected_weights(method.β₁, t)
    c₂₁, c₂₂ = _bias_corrected_weights(method.β₂, t)
    state.B₁ .= c₁₁ .* state.B₁ .+ c₁₂ .* cache.g
    state.v = c₂₁ * state.v + c₂₂ * sum(abs2, cache.g)
    cache.δ .= state.B₁ ./ (sqrt(state.v) + method.δ)
    _capped_step!(cache.δ, α, method)
end
