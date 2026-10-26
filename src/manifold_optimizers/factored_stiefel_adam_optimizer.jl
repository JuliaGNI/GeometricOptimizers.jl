"""
    FactoredStiefelAdamCache <: OptimizerCache

The cache of [`FactoredStiefelAdam`](@ref):

- `x::`[`StiefelManifold`](@ref): the point the step is taken from,
- `g`: the Riemannian gradient at `x`, an `M × k` matrix,
- `δ`: the step ``\\Delta``, a tangent vector at `x`,
- `S`: a ``k\\times{}k`` product ``Y^TX``,
- `Ã`: the scaled vertical part ``\\tilde{A}``, a skew-symmetric ``k\\times{}k`` matrix,
- `K`: a horizontal part ``X - Y(Y^TX)``, an `M × k` matrix,
- `KᵀK`: the ``k\\times{}k`` product whose diagonal holds the squared column norms of `K`.
"""
struct FactoredStiefelAdamCache{T, XT, AT, ST} <: OptimizerCache{T}
    x::XT
    g::AT
    δ::AT
    S::ST
    Ã::ST
    K::AT
    KᵀK::ST
end

function FactoredStiefelAdamCache(x::StiefelManifold{T}) where {T}
    k = size(x, 2)
    small() = fill!(similar(x.A, k, k), zero(T))
    FactoredStiefelAdamCache{T, typeof(x), typeof(x.A), typeof(small())}(
        _copy(x), zero(x.A), zero(x.A), small(), small(), zero(x.A), small())
end

direction(cache::FactoredStiefelAdamCache) = cache.δ

OptimizerCache(::FactoredStiefelAdam, x::StiefelManifold) = FactoredStiefelAdamCache(x)

"""
    FactoredStiefelAdamState <: OptimizerState

The state of [`FactoredStiefelAdam`](@ref): the iteration number, the point `x`, the first moment
`B₁`, a tangent vector at `x`, the vertical second moment `V_W`, a symmetric ``k\\times{}k`` matrix,
and the horizontal second moment `v_K`, a vector of length ``k``. The moments are stored
bias-corrected.
"""
mutable struct FactoredStiefelAdamState{T, XT, AT, ST, VT} <: OptimizerState{T}
    iterations::Int
    x::XT
    B₁::AT
    V_W::ST
    v_K::VT
end

function FactoredStiefelAdamState(x::StiefelManifold{T}) where {T}
    _x = _copy(x)
    k = size(x, 2)
    V_W = fill!(similar(x.A, k, k), zero(T))
    v_K = fill!(similar(x.A, k), zero(T))
    FactoredStiefelAdamState{T, typeof(_x), typeof(x.A), typeof(V_W), typeof(v_K)}(
        0, _x, zero(x.A), V_W, v_K)
end

OptimizerState(::FactoredStiefelAdam, x::StiefelManifold) = FactoredStiefelAdamState(x)

_transported_moment(state::FactoredStiefelAdamState) = state.B₁

function _direct_step!(
        cache::FactoredStiefelAdamCache{T}, state::FactoredStiefelAdamState{T},
        method::FactoredStiefelAdam{T}, α) where {T}
    t = iteration_number(state)
    c₁₁, c₁₂ = _bias_corrected_weights(method.β₁, t)
    c₂₁, c₂₂ = _bias_corrected_weights(method.β₂, t)
    Y, G, S, K = cache.x.A, cache.g, cache.S, cache.K
    # The second moments: entrywise on the vertical part `A = (YᵀG - GᵀY)/2`, whose square is
    # symmetric, and columnwise on the horizontal part `K = G - Y(YᵀG)`.
    mul!(S, Y', G)
    state.V_W .= c₂₁ .* state.V_W .+ c₂₂ .* ((S .- S') ./ 2) .^ 2
    copyto!(K, G)
    mul!(K, Y, S, -one(T), one(T))
    mul!(cache.KᵀK, K', K)
    state.v_K .= c₂₁ .* state.v_K .+
                 c₂₂ .* view(cache.KᵀK, LinearAlgebra.diagind(cache.KᵀK))
    # The first moment, split into its vertical and its horizontal part at `Y`, each scaled by its
    # second moment. The split is formed from `B₁` itself, so `Ã` is skew-symmetric and `YᵀK = 0`
    # whatever drift `B₁` carries off the tangent space.
    state.B₁ .= c₁₁ .* state.B₁ .+ c₁₂ .* G
    mul!(S, Y', state.B₁)
    cache.Ã .= ((S .- S') ./ 2) ./ (sqrt.(state.V_W) .+ method.δ)
    copyto!(K, state.B₁)
    mul!(K, Y, S, -one(T), one(T))
    K .= K ./ (sqrt.(transpose(state.v_K)) .+ method.δ)
    # `D = YÃ + K̃`
    copyto!(cache.δ, K)
    mul!(cache.δ, Y, cache.Ã, one(T), one(T))
    _capped_step!(cache.δ, α, method)
end
