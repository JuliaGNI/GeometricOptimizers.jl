# The global section of the manifold of biorthogonal pairs, and the maps between its tangent spaces
# and the global tangent space `BiorthogonalLieAlgHorMatrix`.
#
# A section is stored as for the other manifolds, the point `Y = [Φ Ψ]` and the completion `λ`, but
# the completion is `[K K̃]`: the frame `λ(Y) = [Φ K]` and the dual frame `λ(Y)⁻ᵀ = [Ψ K̃]` side by
# side. The general linear group acts on the pair by `g⋅(Φ, Ψ) = (gΦ, g⁻ᵀΨ)`, so the second half of
# every point and of every section moves by the inverse transpose of what moves the first, and the
# inverse is not the transpose as it is for the orthogonal group. Keeping both frames is what lets
# `global_rep`, `apply_section` and `update_section!` stay products with `N × n` and `N × (N - n)`
# blocks, with no `N × N` inverse anywhere.

# The group element `λ(Y) = [Φ K]`, not the stored `[Φ Ψ K K̃]`.
Base.size(λY::GlobalSection{T, <:BiorthogonalManifold}) where {T} = (size(λY.Y, 1), size(λY.Y, 1))

@doc raw"""
    Matrix(λY::GlobalSection{T, <:BiorthogonalManifold})

The group element ``\lambda(Y) = [\Phi\ \ K]\in{}GL(N)``. Its inverse transpose is
``[\Psi\ \ \tilde{K}]``; see [`global_section(::BiorthogonalManifold)`](@ref).
"""
function Base.Matrix(λY::GlobalSection{T, <:BiorthogonalManifold}) where {T}
    Φ, _ = _biorthogonal_halves(λY.Y)
    K, _ = _biorthogonal_halves(λY.λ)
    hcat(Matrix(Φ), Matrix(K))
end

function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {
        T, AT <: BiorthogonalManifold{T}}
    Y = BiorthogonalManifold(zero(Y₂.A))
    apply_section!(Y, λY, Y₂)

    Y
end

# `λ(Y)⋅(Φ₂, Ψ₂) = (λ(Y)Φ₂, λ(Y)⁻ᵀΨ₂)`, for a pair `Y₂` of any number of columns: `Y₂` is a point of
# `𝓑(n, N)` in `geodesic(::BiorthogonalManifold, …)` and the pair `(g, g⁻ᵀ)` of `𝓑(N, N)` in the
# transport of a section. Both products are formed before `Y` is written, since `Y` may be `Y₂`.
function apply_section!(Y::AT, λY::GlobalSection{T, AT, <:AbstractArray},
        Y₂::MT) where {T, AT <: BiorthogonalManifold{T}, MT <: BiorthogonalManifold{T}}
    n = size(λY.Y, 2) ÷ 2
    N = size(λY.Y, 1)
    Φ, Ψ = _biorthogonal_halves(λY.Y)
    K, K̃ = _biorthogonal_halves(λY.λ)
    Φ₂, Ψ₂ = _biorthogonal_halves(Y₂)
    @views begin
        λΦ₂ = Φ * Φ₂[1:n, :] + K * Φ₂[(n + 1):N, :]
        λΨ₂ = Ψ * Ψ₂[1:n, :] + K̃ * Ψ₂[(n + 1):N, :]
    end
    Φ₁, Ψ₁ = _biorthogonal_halves(Y)
    Φ₁ .= λΦ₂
    Ψ₁ .= λΨ₂

    Y
end

@doc raw"""
    global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:BiorthogonalManifold{T}}

Express the tangent vector ``\Delta = [\Delta_\Phi\ \ \Delta_\Psi]`` at ``Y = (\Phi, \Psi)`` as a
[`BiorthogonalLieAlgHorMatrix`](@ref).

This is the map ``\Delta\mapsto\lambda(Y)^{-1}\Omega(Y, \Delta)\lambda(Y)``, with ``\Omega`` the
horizontal lift [`GeometricOptimizers.Ω`](@ref) of the biorthogonal manifold. A horizontal ``X`` acts
on ``(E, E)`` through its first ``n`` columns and its first ``n`` rows, and
``\lambda(Y)X\cdot(E, E) = (\lambda(Y)XE, -\lambda(Y)^{-T}X^TE)``, so these are
``\lambda(Y)^{-1}\Delta_\Phi`` and ``-\lambda(Y)^T\Delta_\Psi``. With ``\lambda(Y) = [\Phi\ \ K]`` and
``\lambda(Y)^{-T} = [\Psi\ \ \tilde{K}]`` the blocks are

```math
A = \Psi^T\Delta_\Phi = -\Delta_\Psi^T\Phi, \qquad B = \tilde{K}^T\Delta_\Phi, \qquad C = -\Delta_\Psi^TK .
```

The two expressions for ``A`` agree for a tangent `Δ`, and their mean is taken all the same, as
[`global_rep`](@ref) takes the skew-symmetric part for the [`StiefelManifold`](@ref).

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: apply_section
import Random

Random.seed!(123)

Y = rand(BiorthogonalManifold, 6, 4)
Δ = rgrad(Y, randn(6, 4))
λY = GlobalSection(Y)
X = global_rep(λY, Δ)

# the tangent vector `λ(Y)X⋅(E, E)` is `Δ`
Φ, Ψ, K, K̃ = Y[:, 1:2], Y[:, 3:4], λY.λ[:, 1:4], λY.λ[:, 5:8]
XE, XᵀE = Matrix(X)[:, 1:2], Matrix(X)'[:, 1:2]
hcat([Φ K] * XE, -[Ψ K̃] * XᵀE) ≈ Δ

# output

true
```
"""
function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {
        T, AT <: BiorthogonalManifold{T}}
    global_rep!(zero_tangent(λY.Y), λY, Δ)
end

# The blocks are zeroed before each `mul!` for the reason `global_rep!` of the Stiefel lift gives.
function global_rep!(D::BiorthogonalLieAlgHorMatrix{T},
        λY::GlobalSection{T, <:BiorthogonalManifold, <:AbstractArray},
        Δ::AbstractMatrix{T}) where {T}
    ∇L = _ambient(Δ)
    @assert size(λY.Y) == size(∇L) == (D.N, 2 * D.n)
    Φ, Ψ = _biorthogonal_halves(λY.Y)
    K, K̃ = _biorthogonal_halves(λY.λ)
    ΔΦ, ΔΨ = _biorthogonal_halves(∇L)
    mul!(fill!(D.A, zero(T)), Ψ', ΔΦ, T(0.5), zero(T))
    mul!(D.A, ΔΨ', Φ, -T(0.5), one(T))
    mul!(fill!(D.B, zero(T)), K̃', ΔΦ)
    mul!(fill!(D.C, zero(T)), ΔΨ', K, -one(T), zero(T))
    D
end

# The derivative of the loss with respect to the stored blocks of a lift at the section; see
# `gradient_rep!`. `Δ` is the Riemannian gradient of the metric the pair inherits from `ℝ^{N×2n}`,
# which is the orthogonal projection of the Euclidean gradient onto the tangent space, so it pairs
# with every tangent vector as the Euclidean gradient does, and with `λ(Y)X⋅(E, E)` that pairing is
#
#     ⟨ΔΦ, ΦA + KB⟩ - ⟨ΔΨ, ΨAᵀ + K̃Cᵀ⟩ = ⟨ΦᵀΔΦ - ΔΨᵀΨ, A⟩ + ⟨KᵀΔΦ, B⟩ + ⟨-ΔΨᵀK̃, C⟩.
#
# These are `global_rep!`'s blocks with the frame and the dual frame exchanged. Every block is stored
# once, so the flat inner product of `_dot` is `tr(XᵀY)` on the dense lifts and `_section_sharp!`
# has nothing to rescale.
function gradient_rep!(D::BiorthogonalLieAlgHorMatrix{T},
        λY::GlobalSection{T, <:BiorthogonalManifold, <:AbstractArray},
        Δ::AbstractMatrix{T}) where {T}
    ∇L = _ambient(Δ)
    @assert size(λY.Y) == size(∇L) == (D.N, 2 * D.n)
    Φ, Ψ = _biorthogonal_halves(λY.Y)
    K, K̃ = _biorthogonal_halves(λY.λ)
    ΔΦ, ΔΨ = _biorthogonal_halves(∇L)
    mul!(fill!(D.A, zero(T)), Φ', ΔΦ)
    mul!(D.A, ΔΨ', Ψ, -one(T), one(T))
    mul!(fill!(D.B, zero(T)), K', ΔΦ)
    mul!(fill!(D.C, zero(T)), ΔΨ', K̃, -one(T), zero(T))
    D
end

# The transport of the section along a step: `λ(Y) ← λ(Y)g` and `λ(Y)⁻ᵀ ← λ(Y)⁻ᵀg⁻ᵀ`, with `g` the
# retraction of the lift. The retraction of a `BiorthogonalLieAlgHorMatrix` is the pair `[g g⁻ᵀ]`, so
# this is `apply_section!` of that pair, read back into the frame and the dual frame. See
# `_update_section!(::Nothing, …)` in `global_sections.jl`, of which this is the biorthogonal case.
function _update_section!(::Nothing, Λᵗ::GlobalSection, Λ⁽ᵗ⁻¹⁾::GlobalSection,
        B⁽ᵗ⁻¹⁾::BiorthogonalLieAlgHorMatrix, retraction)
    N, n = B⁽ᵗ⁻¹⁾.N, B⁽ᵗ⁻¹⁾.n
    group = retraction(B⁽ᵗ⁻¹⁾)
    apply_section!(group, Λ⁽ᵗ⁻¹⁾, group)
    G, G⁻ᵀ = _biorthogonal_halves(group)
    Φ, Ψ = _biorthogonal_halves(Λᵗ.Y)
    K, K̃ = _biorthogonal_halves(Λᵗ.λ)
    @views begin
        Φ .= G[:, 1:n]
        K .= G[:, (n + 1):N]
        Ψ .= G⁻ᵀ[:, 1:n]
        K̃ .= G⁻ᵀ[:, (n + 1):N]
    end

    nothing
end
