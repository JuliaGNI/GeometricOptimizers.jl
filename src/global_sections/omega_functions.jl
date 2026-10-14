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
    _omega_lift(Y, Δ)
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
    _omega_lift(Y, Δ)
end

# The body both `Ω` methods share: `2(𝕀 - ½YYᵀ)ΔYᵀ`, multiplied out as `(2Δ - Y(YᵀΔ))Yᵀ`, which
# takes one `N × N` product and no identity. It reads the storage of `Y`, which a device multiplies.
function _omega_lift(Y::Manifold{T}, Δ::AbstractMatrix{T}) where {T}
    SkewSymMatrix((2 * Δ - Y.A * (Y.A' * Δ)) * Y.A')
end
