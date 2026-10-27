"""
    FactoredStiefelAdamCache <: OptimizerCache

The cache of [`FactoredStiefelAdam`](@ref):

- `x::`[`StiefelManifold`](@ref): the point the step is taken from,
- `g`: the Riemannian gradient at `x`, an `M × k` matrix,
- `δ`: the step ``\\Delta``, a tangent vector at `x`,
- `S` and `A`: a ``k\\times{}k`` product ``Y^TX``, and the vertical part ``Y^TG_\\mathcal{R}``, a
  skew-symmetric ``k\\times{}k`` matrix, which the transport overwrites after the step,
- `Ã`: the scaled vertical part ``\\tilde{A}``, a skew-symmetric ``k\\times{}k`` matrix,
- `K`: the horizontal part ``G_\\mathcal{R} - Y(Y^TG_\\mathcal{R})``, an `M × k` matrix,
- `n²` and `d`: the squared column norms of `K`, and the factors that scale the columns of the
  horizontal part of the first moment, two vectors of length ``k``.
"""
struct FactoredStiefelAdamCache{T, XT, AT, ST, VT} <: OptimizerCache{T}
    x::XT
    g::AT
    δ::AT
    S::ST
    A::ST
    Ã::ST
    K::AT
    n²::VT
    d::VT
end

function FactoredStiefelAdamCache(x::StiefelManifold{T}) where {T}
    k = size(x, 2)
    small() = fill!(similar(x.A, k, k), zero(T))
    vector() = fill!(similar(x.A, k), zero(T))
    FactoredStiefelAdamCache{T, typeof(x), typeof(x.A), typeof(small()), typeof(vector())}(
        _copy(x), zero(x.A), zero(x.A), small(), small(), small(), zero(x.A), vector(),
        vector())
end

direction(cache::FactoredStiefelAdamCache) = cache.δ

OptimizerCache(::FactoredStiefelAdam, x::StiefelManifold) = FactoredStiefelAdamCache(x)

"""
    FactoredStiefelAdamState <: OptimizerState

The state of [`FactoredStiefelAdam`](@ref): the iteration number, the point `x`, the first moment
``B_1 = YA_1 + K_1``, a tangent vector at `x`, in its two parts, the vertical part
`A₁ = ½(YᵀB₁ - B₁ᵀY)`, a skew-symmetric ``k\\times{}k`` matrix, and the horizontal part
`K₁ = B₁ - Y(YᵀB₁)`, an `M × k` matrix, the vertical second moment `V_W`, a symmetric
``k\\times{}k`` matrix, and the horizontal second moment `v_K`, a vector of length ``k``. The
moments are stored bias-corrected.
"""
mutable struct FactoredStiefelAdamState{T, XT, AT, ST, VT} <: OptimizerState{T}
    iterations::Int
    x::XT
    A₁::ST
    K₁::AT
    V_W::ST
    v_K::VT
end

function FactoredStiefelAdamState(x::StiefelManifold{T}) where {T}
    _x = _copy(x)
    k = size(x, 2)
    small() = fill!(similar(x.A, k, k), zero(T))
    v_K = fill!(similar(x.A, k), zero(T))
    FactoredStiefelAdamState{T, typeof(_x), typeof(x.A), typeof(small()), typeof(v_K)}(
        0, _x, small(), zero(x.A), small(), v_K)
end

OptimizerState(::FactoredStiefelAdam, x::StiefelManifold) = FactoredStiefelAdamState(x)

# `K ← K - Y(YᵀK)` twice, with `S = YᵀK` on entry and `S` as scratch. For the `X` on entry, one
# pass leaves `‖YᵀK‖ ≈ eps‖X‖`, which the columnwise scaling may divide by a small column norm; the
# second pass leaves `‖YᵀK‖ ≈ eps‖K‖`.
function _horizontal_part!(K, S, Y)
    T = eltype(K)
    mul!(K, Y, S, -one(T), one(T))
    mul!(S, Y', K)
    mul!(K, Y, S, -one(T), one(T))
end

# The step reads `G_R = rgrad(Y, G, metric)` through its vertical part `A = YᵀG_R`, which is
# skew-symmetric, and its horizontal part `K = P_H(Y)G_R` only. `G_R` differs from `G` by `Y` times
# a `k × k` matrix, so `K = P_H(Y)G`, and with `S = YᵀG`, `A = S - Sᵀ` for the canonical metric and
# `A = (S - Sᵀ)/2` for the Euclidean one. All three come from the one product `S`, and the cache
# keeps `G_R = YA + K`.
function _direct_gradient!(cache::FactoredStiefelAdamCache{T}, Y::StiefelManifold{T}, G,
        metric, ws) where {T}
    _check_gradient_backend(Y, G)
    S, A, K = cache.S, cache.A, cache.K
    mul!(S, Y.A', G)
    A .= (S .- S') .* _vertical_weight(metric, T)
    copyto!(K, G)
    _horizontal_part!(K, S, Y.A)
    copyto!(cache.g, K)
    mul!(cache.g, Y.A, A, one(T), one(T))
end

_vertical_weight(::CanonicalMetric, T) = one(T)
_vertical_weight(::EuclideanMetric, T) = one(T) / 2

# The state holds the first moment in its two parts at `Y`. `P_V(Y)B₁ = YA₁` and `P_H(Y)B₁ = K₁`
# are linear in `B₁`, so the update `B₁ ← c₁₁B₁ + c₁₂G_R` is the same update of each part, with the
# parts `A` and `K` of `G_R`.
function _direct_step!(
        cache::FactoredStiefelAdamCache{T}, state::FactoredStiefelAdamState{T},
        method::FactoredStiefelAdam{T}, α) where {T}
    t = iteration_number(state)
    c₁₁, c₁₂ = _bias_corrected_weights(method.β₁, t)
    c₂₁, c₂₂ = _bias_corrected_weights(method.β₂, t)
    Y, A, K = cache.x.A, cache.A, cache.K
    # The second moments: entrywise on the vertical part `A`, whose square is symmetric, and
    # columnwise on the horizontal part `K`.
    state.V_W .= c₂₁ .* state.V_W .+ c₂₂ .* A .^ 2
    sum!(abs2, transpose(cache.n²), K)
    state.v_K .= c₂₁ .* state.v_K .+ c₂₂ .* cache.n²
    # The floor `ε = √eps(T)·σ`, `σ² = ΣV_W + Σv_K`, in both denominators: a part of the first
    # moment whose own second moment is zero is round-off, and is not scaled up against the other.
    ε = sqrt(eps(T)) * sqrt(sum(state.V_W) + sum(state.v_K))
    state.A₁ .= c₁₁ .* state.A₁ .+ c₁₂ .* A
    state.K₁ .= c₁₁ .* state.K₁ .+ c₁₂ .* K
    # each part scaled by its second moment, and `D = YÃ + K̃`
    cache.Ã .= state.A₁ ./ (sqrt.(state.V_W) .+ method.δ .+ ε)
    cache.d .= inv.(sqrt.(state.v_K) .+ method.δ .+ ε)
    cache.δ .= state.K₁ .* transpose(cache.d)
    mul!(cache.δ, Y, cache.Ã, one(T), one(T))
    _capped_step!(cache.δ, α, method)
end

# The split transport `P_V(Y_new)𝒯(YA₁) + P_H(Y_new)𝒯(K₁)`: each part of the first moment is
# transported on its own and put back into its own subspace at `Y_new`, so that the transport mixes
# no part of the one into the other.
function _transport_moment!(
        state::FactoredStiefelAdamState{T}, cache::FactoredStiefelAdamCache{T},
        method::FactoredStiefelAdam, Y_new, Y, ws) where {T}
    A₁, K₁, S, P = state.A₁, state.K₁, cache.S, cache.A
    mul!(P, Y_new.A', Y.A)
    _transport_vertical!(A₁, S, P, method.transport)
    _transport_horizontal!(K₁, Y_new, Y, cache.δ, method.transport, ws)
    mul!(S, Y_new.A', K₁)
    _horizontal_part!(K₁, S, Y_new.A)
end

# `A₁ ← ½(N - Nᵀ)` for `N = Y_newᵀ𝒯(YA₁)`, from `P = Y_newᵀY`, with `S` as scratch. The projection
# changes `YA₁` by `Y_new` times a symmetric matrix, which has no skew-symmetric part, so
# `N = PA₁`.
function _transport_vertical!(A₁, S, P, ::ProjectionTransport)
    mul!(S, P, A₁)
    A₁ .= (S .- S') ./ 2
end

# The generator of the Cayley transform along `YA₁` is `YA₁Yᵀ`. With `W` the generator along the
# step, `𝕀 + W/2 = 2𝕀 - (𝕀 - W/2)` makes `W₊ = (𝕀 - W/2)⁻¹Y = (Y + Y_new)/2`, so
# `𝒯(YA₁) = W₊A₁(YᵀW₊)` and `N = (𝕀 + P)A₁(𝕀 + P)ᵀ/4`.
function _transport_vertical!(A₁, S, P, ::DifferentiatedCayleyTransport)
    T = eltype(A₁)
    P ./= 2
    view(P, LinearAlgebra.diagind(P)) .+= one(T) / 2
    mul!(S, P, A₁)
    mul!(A₁, S, P')
    copyto!(S, A₁)
    A₁ .= (S .- S') ./ 2
end

# `𝒯(K₁)` up to a part that `P_H(Y_new)` removes. The projection changes `K₁` by `Y_new` times a
# matrix, so `P_H(Y_new)` of it is `P_H(Y_new)K₁`, and `K₁` stays.
_transport_horizontal!(K₁, Y_new, Y, Δ, ::ProjectionTransport, ws) = K₁
function _transport_horizontal!(K₁, Y_new, Y, Δ, tr::DifferentiatedCayleyTransport, ws)
    transport!(K₁, Y_new, Y, Δ, K₁, tr, ws)
end
