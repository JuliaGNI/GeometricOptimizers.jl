# The retractions of a `BiorthogonalLieAlgHorMatrix`.
#
# A retraction of a lift `X ∈ 𝔤ʰᵒʳ ⊂ 𝔤𝔩(N)` is a group element `g = R(X)`, and the biorthogonal pair
# moves by `g` in its first half and by `g⁻ᵀ` in its second. Both retractions this package ships have
# `R(X)⁻¹ = R(-X)`, `exp(X)⁻¹ = exp(-X)` and `Cayley(X)⁻¹ = Cayley(-X)`, so `g⁻ᵀ` is the transpose of
# the same retraction of `-X`, taken through the same factorisation and at the same `O(N²n)` cost, and
# no `N × N` inverse is formed.
#
# The retraction returns the pair `[g g⁻ᵀ]`, a point of `𝓑(N, N) ≅ GL(N)`, as the retraction of a
# Stiefel lift returns the `N × N` orthogonal matrix as a point of `St(N, N)`.

# The factorisation of `lift_factors`, with `C` in place of the `-Bᵀ` of a Stiefel lift:
#
#     [A C; B 𝕆] = [½A 𝕀; B 𝕆] [𝕀 𝕆; ½A C]
function lift_factors(B::BiorthogonalLieAlgHorMatrix)
    T = eltype(B)
    N, n = B.N, B.n
    backend = KernelAbstractions.get_backend(B)
    unit = unit_matrix(backend, T, n)

    B̂ = KernelAbstractions.zeros(backend, T, N, 2n)
    B̄ᵗ = KernelAbstractions.zeros(backend, T, 2n, N)

    @views begin
        B̂[1:n, 1:n] .= T(0.5) .* B.A
        B̂[(n + 1):N, 1:n] .= B.B
        B̂[1:n, (n + 1):(2n)] .= unit
        B̄ᵗ[1:n, 1:n] .= unit
        B̄ᵗ[(n + 1):(2n), 1:n] .= T(0.5) .* B.A
        B̄ᵗ[(n + 1):(2n), (n + 1):N] .= B.C
    end

    (B̂, B̄ᵗ')
end

function lift_factors!(ws::RetractionWorkspace{T}, B::BiorthogonalLieAlgHorMatrix{T}) where {T}
    N, n = B.N, B.n
    @assert (ws.N, ws.n) == (N, n)

    @views begin
        ws.B̂[1:n, 1:n] .= T(0.5) .* B.A
        ws.B̂[(n + 1):N, 1:n] .= B.B
        ws.B̄ᵗ[(n + 1):(2n), 1:n] .= T(0.5) .* B.A
        ws.B̄ᵗ[(n + 1):(2n), (n + 1):N] .= B.C
    end

    nothing
end

# A point of `𝓑(n, N)` is `N × 2n`, and the retraction works with the `N × N` matrices of the group.
function RetractionWorkspace(Y::BiorthogonalManifold{T}, retraction = nothing) where {T}
    N, n = _pair_size(Y)
    RetractionWorkspace(KernelAbstractions.get_backend(Y.A), T, N, n, retraction)
end

# The pair `[g g⁻ᵀ]` from `g` and `g⁻¹`.
function _group_pair(g::AbstractMatrix, g⁻¹::AbstractMatrix)
    BiorthogonalManifold(hcat(g, permutedims(g⁻¹)))
end

_group_matrix(Y::BiorthogonalManifold) = first(_biorthogonal_halves(Y))

@doc raw"""
    geodesic(B̄::BiorthogonalLieAlgHorMatrix, algorithm = ScaledSquaring())

The group exponential of a biorthogonal lift, as the pair ``[\exp(\bar{B})\ \ \exp(\bar{B})^{-T}]``,
i.e. as a [`BiorthogonalManifold`](@ref) of ``N\times2N`` matrices.

``\exp(\bar{B})^{-T} = \exp(-\bar{B})^T``, and both exponentials are taken by
[`geodesic(::AbstractLieAlgHorMatrix)`](@ref)'s method. The curve ``t\mapsto\lambda(Y)\exp(t\bar{B})\cdot(E, E)``
is the orbit of a one-parameter subgroup and not a geodesic of the
[`metric(::BiorthogonalManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref) of the manifold, which no
group element preserves; the name is kept because it is what [`Geodesic`](@ref) selects for every
manifold.
"""
function geodesic(B::BiorthogonalLieAlgHorMatrix,
        algorithm::AbstractExponentialAlgorithm = ScaledSquaring())
    _group_pair(_exp_matrix(B, algorithm), _exp_matrix(-B, algorithm))
end

function geodesic(::BiorthogonalLieAlgHorMatrix, ::ProjectedSkew)
    throw(ArgumentError("ProjectedSkew exponentiates a skew-symmetric lift, and a BiorthogonalLieAlgHorMatrix is not skew-symmetric; use Geodesic() with another exponential algorithm, or Cayley()"))
end

@doc raw"""
    cayley(B̄::BiorthogonalLieAlgHorMatrix)

The Cayley retraction of a biorthogonal lift, as the pair
``[\mathrm{Cayley}(\bar{B})\ \ \mathrm{Cayley}(\bar{B})^{-T}]``, i.e. as a [`BiorthogonalManifold`](@ref)
of ``N\times2N`` matrices.

Both factors are taken by the method of [`cayley(::StiefelLieAlgHorMatrix)`](@ref), the second
through ``\mathrm{Cayley}(\bar{B})^{-1} = \mathrm{Cayley}(-\bar{B})``. ``\mathrm{Cayley}(\bar{B})`` is
invertible wherever ``\mathbb{I} - \frac{1}{2}\bar{B}`` and ``\mathbb{I} + \frac{1}{2}\bar{B}`` are,
which for a lift that is not skew-symmetric is not every lift: an eigenvalue ``\pm2`` of ``\bar{B}``
makes the ``2n\times2n`` matrix the method inverts singular.
"""
function cayley(B::BiorthogonalLieAlgHorMatrix)
    _group_pair(_cayley_matrix(B), _cayley_matrix(-B))
end

# The point `λ(Y)R(B)⋅(E, E)`: the first `n` columns of each half of the pair, moved by the section.
function _retract_point(Y::BiorthogonalManifold, λY::GlobalSection, pair::BiorthogonalManifold)
    n = size(Y, 2) ÷ 2
    g, g⁻ᵀ = _biorthogonal_halves(pair)
    @views apply_section(λY, typeof(Y)(hcat(g[:, 1:n], g⁻ᵀ[:, 1:n])))
end

function geodesic(Y::BiorthogonalManifold{T}, Δ::AbstractMatrix{T},
        algorithm::AbstractExponentialAlgorithm = ScaledSquaring()) where {T}
    λY = GlobalSection(Y)
    _retract_point(Y, λY, geodesic(global_rep(λY, Δ), algorithm))
end

function cayley(Y::BiorthogonalManifold{T}, Δ::AbstractMatrix{T}) where {T}
    λY = GlobalSection(Y)
    _retract_point(Y, λY, cayley(global_rep(λY, Δ)))
end

# The inverse of the group element `retraction_matrix!` wrote, written into `ws.retracted` in its
# place. The shipped retractions take it as the retraction of `-B`; any other one is asked for its
# pair and the second half transposed.
function _inverse_retraction_matrix!(ws::RetractionWorkspace, R::Union{Cayley, Geodesic},
        B::BiorthogonalLieAlgHorMatrix)
    retraction_matrix!(ws, R, -B)
end

function _inverse_retraction_matrix!(ws::RetractionWorkspace, R::AbstractRetraction,
        B::BiorthogonalLieAlgHorMatrix)
    copyto!(ws.retracted, transpose(last(_biorthogonal_halves(retraction(R, B)))))
end

# The workspace arm of `update_section!` for a biorthogonal section: the frame `[Φ K]` moves by `g`
# and the dual frame `[Ψ K̃]` by `g⁻ᵀ`, the second as `[Ψ K̃](g⁻¹)ᵀ`, so that `ws.retracted` holds
# `g` and then `g⁻¹` and no transpose is materialised. The frame is written before `g⁻¹` is formed,
# and the dual frame is read only after; where `Λᵗ === Λ⁽ᵗ⁻¹⁾`, the two halves the first write
# touches are not the two the second product reads.
function _update_section!(ws::RetractionWorkspace{T}, Λᵗ::GlobalSection,
        Λ⁽ᵗ⁻¹⁾::GlobalSection, B⁽ᵗ⁻¹⁾::BiorthogonalLieAlgHorMatrix{T},
        retraction::AbstractRetraction) where {T}
    N, n = B⁽ᵗ⁻¹⁾.N, B⁽ᵗ⁻¹⁾.n
    Φ₀, Ψ₀ = _biorthogonal_halves(Λ⁽ᵗ⁻¹⁾.Y)
    K₀, K̃₀ = _biorthogonal_halves(Λ⁽ᵗ⁻¹⁾.λ)
    Φ, Ψ = _biorthogonal_halves(Λᵗ.Y)
    K, K̃ = _biorthogonal_halves(Λᵗ.λ)

    g = retraction_matrix!(ws, retraction, B⁽ᵗ⁻¹⁾)
    @views begin
        mul!(ws.product, Φ₀, g[1:n, :])
        mul!(ws.product, K₀, g[(n + 1):N, :], one(T), one(T))
        Φ .= ws.product[:, 1:n]
        K .= ws.product[:, (n + 1):N]
    end

    g⁻¹ = _inverse_retraction_matrix!(ws, retraction, B⁽ᵗ⁻¹⁾)
    @views begin
        mul!(ws.product, Ψ₀, transpose(g⁻¹[:, 1:n]))
        mul!(ws.product, K̃₀, transpose(g⁻¹[:, (n + 1):N]), one(T), one(T))
        Ψ .= ws.product[:, 1:n]
        K̃ .= ws.product[:, (n + 1):N]
    end

    nothing
end

# `(𝕀 + aX)⁻¹X(𝕀 - aX)⁻¹E` for `X = B̂B̄ᵗ`, through the Woodbury identity as in
# `retraction_differential(::Cayley, ::AbstractLieAlgHorMatrix, α)`.
function _cayley_differential_columns(B̂, B̄ᵗ, a, E)
    G = B̄ᵗ * B̂
    𝕀 = unit_matrix(G)
    w₁ = E + B̂ * ((𝕀 - a * G) \ (a * (B̄ᵗ * E)))      # (𝕀 - aX)⁻¹ E
    w₂ = B̂ * (B̄ᵗ * w₁)                                # X (𝕀 - aX)⁻¹ E
    w₂ - B̂ * ((𝕀 + a * G) \ (a * (B̄ᵗ * w₂)))
end

@doc raw"""
    retraction_differential(::Cayley, B̄::BiorthogonalLieAlgHorMatrix, α)

[`retraction_differential`](@ref) for a biorthogonal lift. ``D(\alpha) = (\mathbb{I} +
\frac{\alpha}{2}\bar{B})^{-1}\bar{B}(\mathbb{I} - \frac{\alpha}{2}\bar{B})^{-1}`` as for every lift,
but its horizontal part is not determined by its first ``n`` columns alone: the action of
``D\in\mathfrak{gl}(N)`` on ``(E, E)`` is ``(DE, -D^TE)``, and the vertical space is the lower right
block. So the first ``n`` columns give ``A`` and ``B``, and the first ``n`` rows, which are the first
``n`` columns of ``D^T = (\mathbb{I} - \frac{\alpha}{2}\bar{B}^T)^{-1}\bar{B}^T(\mathbb{I} +
\frac{\alpha}{2}\bar{B}^T)^{-1}``, give ``C``. Both are taken through [`lift_factors`](@ref), the second
with the two factors exchanged and ``\alpha`` negated.
"""
function retraction_differential(::Cayley, B::BiorthogonalLieAlgHorMatrix{T}, α) where {T}
    iszero(α) && return B

    N, n = B.N, B.n
    a = T(α) / 2
    B̂, B̄ = lift_factors(B)
    E = StiefelProjection(B)
    columns = _cayley_differential_columns(B̂, B̄', a, E)
    rows = _cayley_differential_columns(B̄, B̂', -a, E)
    @views BiorthogonalLieAlgHorMatrix(columns[1:n, :], columns[(n + 1):N, :],
        permutedims(rows[(n + 1):N, :]), N, n)
end

# The in-place `Cayley` arm of `retraction_differential!` for a host `Matrix` reads only the first
# `n` columns of `D(α)`, which is all a lift whose horizontal part they determine needs. This lift's
# is not, so it takes the allocating method and copies the answer in.
function retraction_differential!(
        D::BiorthogonalLieAlgHorMatrix{T}, ::RetractionWorkspace{T, Matrix{T}}, R::Cayley,
        B::BiorthogonalLieAlgHorMatrix{T}, α) where {T <: LinearAlgebra.BlasFloat}
    _copyto!(D, retraction_differential(R, B, α))
    D
end
