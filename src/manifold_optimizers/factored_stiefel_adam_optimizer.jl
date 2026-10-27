"""
    FactoredStiefelAdamCache <: OptimizerCache

The cache of [`FactoredStiefelAdam`](@ref):

- `x::`[`StiefelManifold`](@ref): the point the step is taken from,
- `g`: the Riemannian gradient at `x`, an `M × k` matrix,
- `δ`: the step ``\\Delta``, a tangent vector at `x`,
- `S` and `A`: a ``k\\times{}k`` product ``Y^TX`` and its skew-symmetric part,
- `Ã`: the scaled vertical part ``\\tilde{A}``, a skew-symmetric ``k\\times{}k`` matrix,
- `K`: a horizontal part ``X - Y(Y^TX)``, an `M × k` matrix,
- `B_V`: the vertical part of the first moment, an `M × k` matrix, while it is transported,
- `n²`: the squared column norms of `K`, a vector of length ``k``.
"""
struct FactoredStiefelAdamCache{T, XT, AT, ST, VT} <: OptimizerCache{T}
    x::XT
    g::AT
    δ::AT
    S::ST
    A::ST
    Ã::ST
    K::AT
    B_V::AT
    n²::VT
end

function FactoredStiefelAdamCache(x::StiefelManifold{T}) where {T}
    k = size(x, 2)
    small() = fill!(similar(x.A, k, k), zero(T))
    n² = fill!(similar(x.A, k), zero(T))
    FactoredStiefelAdamCache{T, typeof(x), typeof(x.A), typeof(small()), typeof(n²)}(
        _copy(x), zero(x.A), zero(x.A), small(), small(), small(), zero(x.A), zero(x.A), n²)
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

# `A ← ½(YᵀX - XᵀY)`, the vertical part of `X` at `Y` in `𝔰𝔬(k)`, with `S ← YᵀX`.
function _vertical_part!(A, S, Y, X)
    mul!(S, Y', X)
    A .= (S .- S') ./ 2
end

# `K ← X - Y(YᵀX)`, the horizontal part of `X` at `Y`, with `S` for `YᵀK`. One pass of the
# projection leaves `‖YᵀK‖ ≈ eps‖X‖`, which the columnwise scaling of `K` may divide by a small
# column norm; the second pass leaves `‖YᵀK‖ ≈ eps‖K‖`.
function _horizontal_part!(K, S, Y, X)
    T = eltype(K)
    copyto!(K, X)
    for _ in 1:2
        mul!(S, Y', K)
        mul!(K, Y, S, -one(T), one(T))
    end
    K
end

function _direct_step!(
        cache::FactoredStiefelAdamCache{T}, state::FactoredStiefelAdamState{T},
        method::FactoredStiefelAdam{T}, α) where {T}
    t = iteration_number(state)
    c₁₁, c₁₂ = _bias_corrected_weights(method.β₁, t)
    c₂₁, c₂₂ = _bias_corrected_weights(method.β₂, t)
    Y, G, S, A, K = cache.x.A, cache.g, cache.S, cache.A, cache.K
    # The second moments: entrywise on the vertical part `A`, whose square is symmetric, and
    # columnwise on the horizontal part `K`.
    _vertical_part!(A, S, Y, G)
    state.V_W .= c₂₁ .* state.V_W .+ c₂₂ .* A .^ 2
    _horizontal_part!(K, S, Y, G)
    sum!(abs2, transpose(cache.n²), K)
    state.v_K .= c₂₁ .* state.v_K .+ c₂₂ .* cache.n²
    # The floor `ε = √eps(T)·σ`, `σ² = ΣV_W + Σv_K`, in both denominators: a part of the first
    # moment whose own second moment is zero is round-off, and is not scaled up against the other.
    ε = sqrt(eps(T)) * sqrt(sum(state.V_W) + sum(state.v_K))
    # The first moment, split into its vertical and its horizontal part at `Y`, each scaled by its
    # second moment.
    state.B₁ .= c₁₁ .* state.B₁ .+ c₁₂ .* G
    _vertical_part!(cache.Ã, S, Y, state.B₁)
    cache.Ã ./= sqrt.(state.V_W) .+ method.δ .+ ε
    _horizontal_part!(K, S, Y, state.B₁)
    K ./= sqrt.(transpose(state.v_K)) .+ method.δ .+ ε
    # `D = YÃ + K̃`
    copyto!(cache.δ, K)
    mul!(cache.δ, Y, cache.Ã, one(T), one(T))
    _capped_step!(cache.δ, α, method)
end

# The vertical and the horizontal part of `B₁` are transported on their own, and each is put back
# into its own subspace at `Y_new`, so that the transport mixes no part of the one into the other.
function _transport_moment!(
        state::FactoredStiefelAdamState{T}, cache::FactoredStiefelAdamCache{T},
        method::FactoredStiefelAdam, Y_new, Y, ws) where {T}
    B, B_V, K, S, A, Δ = state.B₁, cache.B_V, cache.K, cache.S, cache.A, cache.δ
    _vertical_part!(A, S, Y.A, B)
    mul!(B_V, Y.A, A)
    _horizontal_part!(K, S, Y.A, B)
    transport!(B_V, Y_new, Y, Δ, B_V, method.transport, ws)
    transport!(K, Y_new, Y, Δ, K, method.transport, ws)
    _vertical_part!(A, S, Y_new.A, B_V)
    _horizontal_part!(B, S, Y_new.A, K)
    mul!(B, Y_new.A, A, one(T), one(T))
end
