@doc raw"""
    Ω(Y::StiefelManifold{T}, Δ::AbstractMatrix{T}) where T

Perform *canonical horizontal lift* for the Stiefel manifold:

```math
    \Delta \mapsto (\mathbb{I} - \frac{1}{2}YY^T)\Delta{}Y^T - Y\Delta^T(\mathbb{I} - \frac{1}{2}YY^T).
```

Internally this performs 

```julia
SkewSymMatrix(2 * (I(n) - .5 * Y * Y') * Δ * Y')
```

It uses [`SkewSymMatrix`](@ref) to save memory. 

# Examples 

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: StiefelProjection

E = StiefelManifold(StiefelProjection(5, 2))
Δ = [0. -1.; 1. 0.; 2. 3.; 4. 5.; 6. 7.]
GeometricOptimizers.Ω(E, Δ)

# output

5×5 SkewSymMatrix{Float64, Vector{Float64}}:
 0.0  -1.0  -2.0  -4.0  -6.0
 1.0   0.0  -3.0  -5.0  -7.0
 2.0   3.0   0.0  -0.0  -0.0
 4.0   5.0   0.0   0.0  -0.0
 6.0   7.0   0.0   0.0   0.0
```

Note that the output of `Ω` is a skew-symmetric matrix, i.e. an element of ``\mathfrak{g}``.
"""
function Ω(Y::StiefelManifold{T}, Δ::AbstractMatrix{T}) where {T}
    # `2(𝕀 - ½YYᵀ)ΔYᵀ`, multiplied out so that it forms no identity and one `N × N` product
    SkewSymMatrix((2 * Δ - Y.A * (Y.A' * Δ)) * Y.A')
end

@doc raw"""
    Ω(Y::GrassmannManifold{T}, Δ::AbstractMatrix{T}) where T

Perform the *canonical horizontal lift* for the Grassmann manifold:

```math
    \Delta \mapsto \Omega^{St}(\Delta),
```

where ``\Omega^{St}`` is the canonical horizontal lift for the Stiefel manifold.

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: StiefelProjection

E = GrassmannManifold(StiefelProjection(5, 2))
Δ = [0. 0.; 0. 0.; 2. 3.; 4. 5.; 6. 7.]
GeometricOptimizers.Ω(E, Δ)

# output

5×5 SkewSymMatrix{Float64, Vector{Float64}}:
 0.0  -0.0  -2.0  -4.0  -6.0
 0.0   0.0  -3.0  -5.0  -7.0
 2.0   3.0   0.0  -0.0  -0.0
 4.0   5.0   0.0   0.0  -0.0
 6.0   7.0   0.0   0.0   0.0
```
"""
function Ω(Y::GrassmannManifold{T}, Δ::AbstractMatrix{T}) where {T}
    # multiplied out as in the `StiefelManifold` method
    SkewSymMatrix((2 * Δ - Y.A * (Y.A' * Δ)) * Y.A')
end

@doc raw"""
    Ω(U::SymplecticStiefelManifold{T}, Δ::AbstractMatrix{T}) where T

Perform the horizontal lift for the symplectic Stiefel manifold, (3.7) of [bendokat2021real](@cite):

```math
    \Delta \mapsto \left(\mathbb{I} - \frac{1}{2}UU^+\right)\Delta{}U^+ - U\Delta^+\left(\mathbb{I} - \frac{1}{2}UU^+\right),
```

with ``U^+ = \mathbb{J}_{2n}^TU^T\mathbb{J}_{2N}`` the symplectic inverse. The result is a Hamiltonian
``2N\times2N`` matrix, i.e. an element of ``\mathfrak{g}``, horizontal with respect to the
pseudo-Riemannian metric ``h`` of [bendokat2021real](@cite), and is returned dense: the symplectic
Lie algebra has no structured type in this package. ``\Omega(U, \Delta)U = \Delta``.

[`global_rep`](@ref) computes ``\lambda(U)^{-1}\Omega(U, \Delta)\lambda(U)`` without forming this
matrix; see there.
"""
function Ω(U::SymplecticStiefelManifold{T}, Δ::AbstractMatrix{T}) where {T}
    N = size(U, 1)
    # `U⁺` and `Δ⁺` as `n × N` matrices, and `𝕀 - ½UU⁺` applied as a product rather than formed
    U⁺ = _poisson_tmul(U.A' * _poisson_tensor(U, N))
    Δ⁺ = _poisson_tmul(Δ' * _poisson_tensor(U, N))
    ΔU⁺ = Δ * U⁺
    UΔ⁺ = U.A * Δ⁺
    (ΔU⁺ - U.A * (U⁺ * ΔU⁺) / 2) - (UΔ⁺ - (UΔ⁺ * U.A) * U⁺ / 2)
end
