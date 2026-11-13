@doc raw"""
    BiorthogonalManifold(A)

The manifold of biorthogonal pairs, stored as the ``N\times2n`` matrix `A` ``= [\Phi\ \ \Psi]``:

```math
\mathcal{B}(n, N) = \{(\Phi, \Psi)\in\mathbb{R}^{N\times{}n}\times\mathbb{R}^{N\times{}n} : \Psi^T\Phi = \mathbb{I}_n\}.
```

Compare [`StiefelManifold`](@ref), whose constraint is ``Y^TY = \mathbb{I}_n``: the Stiefel manifold
is the set of pairs with ``\Phi = \Psi``. ``\mathcal{B}(n, N)`` is the homogeneous space
``GL(N)/GL(N - n)`` of the action ``g\cdot(\Phi, \Psi) = (g\Phi, g^{-T}\Psi)``, of dimension
``2Nn - n^2``; the manual page [The Biorthogonal Manifold](@ref) derives this. For ``n = N`` it is
``GL(N)`` itself, as the pairs ``(g, g^{-T})``, which is what [`cayley`](@ref) and
[`geodesic`](@ref) of a [`BiorthogonalLieAlgHorMatrix`](@ref) return.

A pair ``(\Phi, \Psi)`` makes ``\mathrm{diag}(\Phi, \Psi)`` a point of the
[`SymplecticStiefelManifold`](@ref) of ``2N\times2n`` matrices. Biorthogonal weights of this kind
are used in [otto2023learning, friedl2026learning](@cite).

The metric is the one the pairs inherit from ``\mathbb{R}^{N\times{}2n}``, see
[`metric(::BiorthogonalManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref), and the optimizers step
a point through the global tangent space [`BiorthogonalLieAlgHorMatrix`](@ref).

`rand(BiorthogonalManifold, N, 2n)` draws a point, with the size of the stored matrix as for every
other manifold here. The manifold is not compact, so there is no uniform distribution, and the draw
is the one [`rand(::Type{<:BiorthogonalManifold}, ::Integer, ::Integer)`](@ref) describes.

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: check
import Random

Random.seed!(1234)

Y = rand(BiorthogonalManifold, 6, 4)
Φ, Ψ = Y[:, 1:2], Y[:, 3:4]
(check(Y) < 1e-12, Ψ' * Φ ≈ [1 0; 0 1])

# output

(true, true)
```
"""
mutable struct BiorthogonalManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
    function BiorthogonalManifold(A::AbstractMatrix)
        @assert iseven(size(A, 2))
        @assert size(A, 1) ≥ size(A, 2) ÷ 2
        _check_real_pair(eltype(A))
        new{eltype(A), typeof(A)}(A)
    end
end

# The dual frame moves by `g⁻ᵀ` and is formed with `transpose`, which is the right map for the
# constraint `ΨᵀΦ = 𝕀` of a real pair. A complex pair would need the adjoint in every one of those
# places, and `check` and the metric would have to agree on which constraint is meant, so a complex
# element type is refused rather than given a wrong answer.
function _check_real_pair(::Type{T}) where {T}
    T <: Real || throw(ArgumentError(
        "a BiorthogonalManifold has a real element type, and $(T) is not real"))
    nothing
end

# The inner constructor suppresses the default `BiorthogonalManifold{T, AT}(A)`, which the generic
# `Base.copy(::Manifold)` calls; see the same method of `SymplecticStiefelManifold`.
BiorthogonalManifold{T, AT}(A::AT) where {T, AT <: AbstractMatrix{T}} = BiorthogonalManifold(A)

# The two halves of a point, as views of its storage.
_biorthogonal_halves(A::AbstractMatrix) = (n = size(A, 2) ÷ 2; @views (A[:, 1:n], A[:, (n + 1):(2n)]))
_biorthogonal_halves(Y::BiorthogonalManifold) = _biorthogonal_halves(Y.A)

# The sizes `(N, n)` of the pair, which are not `size(Y)`: the stored matrix has `2n` columns.
_pair_size(Y::BiorthogonalManifold) = (size(Y, 1), size(Y, 2) ÷ 2)

@doc raw"""
    rand(::Type{BiorthogonalManifold{T}}, N, n2)

Draw a point of the manifold of biorthogonal pairs ``(\Phi, \Psi)`` with ``N`` rows and
``n = n_2/2`` columns each.

``\Phi`` and ``W`` are Gaussian, and

```math
\Psi = \Phi(\Phi^T\Phi)^{-1} + (\mathbb{I} - \Phi(\Phi^T\Phi)^{-1}\Phi^T)W,
```

the point of the fiber over ``\Phi`` that ``W`` selects; ``\Psi^T\Phi = \mathbb{I}_n`` for every
``W``. ``(\Phi^T\Phi)^{-1}`` is applied through a Cholesky factor, so the draw takes the products, the
`cholesky` and the triangular solves that [`rand`](@ref) of a [`StiefelManifold`](@ref) takes, and
runs on the same backends.
"""
Base.rand(::Type{<:BiorthogonalManifold}, ::Integer, ::Integer)

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend, ::Type{MT},
        N::Integer, n2::Integer) where {T, MT <: BiorthogonalManifold{T}}
    @assert iseven(n2) && N ≥ n2 ÷ 2
    _check_real_pair(T)
    n = n2 ÷ 2
    Φ = KernelAbstractions.allocate(backend, T, N, n)
    W = KernelAbstractions.allocate(backend, T, N, n)
    randn!(rng, Φ)
    randn!(rng, W)
    # `Φ(ΦᵀΦ)⁻¹`, the dual basis of `Φ` inside its own span
    Φ⁺ᵀ = Φ / cholesky(Symmetric(Φ' * Φ))
    BiorthogonalManifold(hcat(Φ, Φ⁺ᵀ + W - Φ⁺ᵀ * (Φ' * W)))
end

# The host draw names no backend for its arrays, as the other manifolds' host draws do.
function Base.rand(rng::AbstractRNG, ::CPU, ::Type{MT},
        N::Integer, n2::Integer) where {T, MT <: BiorthogonalManifold{T}}
    @assert iseven(n2) && N ≥ n2 ÷ 2
    _check_real_pair(T)
    n = n2 ÷ 2
    Φ = randn(rng, T, N, n)
    W = randn(rng, T, N, n)
    Φ⁺ᵀ = Φ / cholesky(Symmetric(Φ' * Φ))
    BiorthogonalManifold(hcat(Φ, Φ⁺ᵀ + W - Φ⁺ᵀ * (Φ' * W)))
end

@doc raw"""
    check(Y::BiorthogonalManifold)

How far `Y` is from the manifold, as ``\|\Psi^T\Phi - \mathbb{I}_n\|``.

This replaces the generic [`check(::Manifold)`](@ref), whose residual is the orthonormality one.
"""
function check(Y::BiorthogonalManifold)
    Φ, Ψ = _biorthogonal_halves(Y)
    norm(Ψ' * Φ - I)
end

# The size guard of `rgrad`, `metric` and `Ω`: every matrix argument has the shape of the point.
function _check_point_size(Y::BiorthogonalManifold, X::AbstractMatrix)
    size(X) == size(Y.A) || throw(DimensionMismatch(
        "a matrix of size $(size(X)) at a point of size $(size(Y.A))"))
    nothing
end

@doc raw"""
    metric(Y::BiorthogonalManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)

The metric the manifold of biorthogonal pairs inherits from ``\mathbb{R}^{N\times{}2n}``:

```math
g_{(\Phi, \Psi)}(\Delta_1, \Delta_2) = \mathrm{tr}(\Delta_{1,\Phi}^T\Delta_{2,\Phi}) + \mathrm{tr}(\Delta_{1,\Psi}^T\Delta_{2,\Psi}),
```

with ``\Delta = [\Delta_\Phi\ \ \Delta_\Psi]`` stored as the point is.

The general linear group does not act by isometries, and no metric on ``\mathcal{B}(n, N)`` is
invariant under it, because the isotropy group ``GL(N - n)`` is not compact. This one is the
simplest; at the [`StiefelManifold`](@ref) it restricts to twice the Euclidean metric of the Stiefel
manifold, because a pair ``(\Delta, \Delta)`` counts ``\Delta`` twice.
"""
function metric(Y::BiorthogonalManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)
    _check_point_size(Y, Δ₁)
    _check_point_size(Y, Δ₂)
    dot(Δ₁, Δ₂)
end

@doc raw"""
    rgrad(Y::BiorthogonalManifold, ∇L::AbstractMatrix)

The Riemannian gradient of the metric
[`metric(::BiorthogonalManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref), i.e. the orthogonal
projection of the Euclidean gradient ``\nabla{}L = [\nabla_\Phi L\ \ \nabla_\Psi L]`` onto the
tangent space

```math
T_{(\Phi, \Psi)}\mathcal{B}(n, N) = \{(\Delta_\Phi, \Delta_\Psi) : \Psi^T\Delta_\Phi + \Delta_\Psi^T\Phi = \mathbb{O}\}.
```

The normal space is ``\{(\Psi M, \Phi M^T) : M\in\mathbb{R}^{n\times{}n}\}``, the image of the adjoint
of the differential of ``(\Phi, \Psi)\mapsto\Psi^T\Phi``, so

```math
\mathrm{grad}\,L = (\nabla_\Phi L - \Psi M,\ \nabla_\Psi L - \Phi M^T), \qquad
\Psi^T\Psi M + M\Phi^T\Phi = \Psi^T\nabla_\Phi L + (\nabla_\Psi L)^T\Phi .
```

# Implementation

The Sylvester equation for ``M`` is ``n\times{}n`` and its two coefficients are symmetric positive
definite. It is solved in the eigenbases of ``\Psi^T\Psi = V_1\Lambda_1V_1^T`` and
``\Phi^T\Phi = V_2\Lambda_2V_2^T``, where it is diagonal:
``M = V_1\left[(V_1^TRV_2)_{ij}/(\lambda_{1,i} + \lambda_{2,j})\right]V_2^T``. The backend has to supply
the eigendecomposition of a symmetric matrix, which the host does and `Metal` does not.
"""
function rgrad(Y::BiorthogonalManifold, ∇L::AbstractMatrix)
    _check_point_size(Y, ∇L)
    _check_gradient_backend(Y, ∇L)
    Φ, Ψ = _biorthogonal_halves(Y)
    ∇Φ, ∇Ψ = _biorthogonal_halves(∇L)
    M = _sylvester_spd(Ψ' * Ψ, Φ' * Φ, Ψ' * ∇Φ + ∇Ψ' * Φ)
    hcat(∇Φ - Ψ * M, ∇Ψ - Φ * M')
end

# The solution `M` of `P M + M Q = R` for symmetric positive definite `P` and `Q`.
function _sylvester_spd(P::AbstractMatrix, Q::AbstractMatrix, R::AbstractMatrix)
    λ₁, V₁ = eigen(Symmetric(P))
    λ₂, V₂ = eigen(Symmetric(Q))
    V₁ * ((V₁' * R * V₂) ./ (λ₁ .+ transpose(λ₂))) * V₂'
end

@doc raw"""
    global_section(Y::BiorthogonalManifold)

The completion of a biorthogonal pair to an element of the general linear group and its dual: the
``N\times2(N - n)`` matrix ``[K\ \ \tilde{K}]`` with

```math
\lambda(Y) = [\Phi\ \ K] \in GL(N), \qquad \lambda(Y)^{-T} = [\Psi\ \ \tilde{K}],
```

so that ``\lambda(Y)\cdot(E, E) = (\Phi, \Psi)`` for the action ``g\cdot(\Phi, \Psi) = (g\Phi, g^{-T}\Psi)``
and ``E`` the first ``n`` columns of the identity. ``K`` is an orthonormal basis of the kernel of
``\Psi^T``, which makes ``[\Phi\ \ K]`` invertible (the manual page
[The Biorthogonal Manifold](@ref) shows this), and ``\tilde{K} = (\mathbb{I} - \Psi\Phi^T)K``.
Then ``\tilde{K}^T\Phi = \mathbb{O}`` and ``\tilde{K}^TK = \mathbb{I}_{N-n}``, which is
``\lambda(Y)^{-1}\lambda(Y) = \mathbb{I}_N``.

A [`GlobalSection`](@ref) stores the dual frame next to the frame because the group does not act by
orthogonal matrices here: ``\lambda(Y)^{-1}`` is not ``\lambda(Y)^T``, and the optimizers transport
both, as [`update_section!`](@ref) describes. At a point ``\Phi = \Psi`` of the
[`StiefelManifold`](@ref), ``\tilde{K} = K`` and the section is the one
[`global_section(::StiefelManifold)`](@ref) draws.

``K`` is drawn as that function draws it, from the orthonormal basis of the span of ``\Psi`` that
[`_cholesky_qr2`](@ref GeometricOptimizers._cholesky_qr2) gives, so it runs on the same backends.
CholeskyQR2 squares the condition number of ``\Psi``, and a valid pair can have a ``\Psi`` whose
squared condition number exceeds the precision of its element type. For such a ``\Psi`` the basis
comes from a Householder QR instead, which is a host factorization.
"""
function global_section(Y::BiorthogonalManifold)
    Φ, Ψ = _biorthogonal_halves(Y)
    Q = _cholesky_qr2(Ψ)
    K = _complement_columns(Q === nothing ? Matrix(qr(Ψ).Q) : Q)
    K̃ = K - Ψ * (Φ' * K)
    λ = hcat(K, K̃)
    # the storage type of the point, for the reason `global_section(::StiefelManifold)` gives
    λ isa typeof(Y.A) ? λ : typeof(Y.A)(λ)
end

# The zero horizontal lift at `Y`, as `zero_tangent(::StiefelManifold)`.
function zero_tangent(Y::BiorthogonalManifold{T}) where {T}
    N, n = _pair_size(Y)
    zeros(KernelAbstractions.get_backend(Y.A), BiorthogonalLieAlgHorMatrix{T}, N, n)
end

# The zero matrix of the point's own shape, on the point's backend.
Base.zero(Y::BiorthogonalManifold) = zero(Y.A)
