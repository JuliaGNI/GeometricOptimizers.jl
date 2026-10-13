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

# The one body both `Ω` methods share. The Grassmann lift drops the `A` block and keeps the `B`
# block, and the skew-symmetrised `n × n` block is the same expression for both, so the two
# manifolds take one function and differ only in what `getindex` makes of the result.
#
# Three products, two of them the same `Y * Y'`: the identity is built once with `unit_matrix`
# rather than `Base.one`, whose diagonal loop is scalar-indexed and so unavailable on a
# `KernelAbstractions` backend, and the doubled product is written once. `Y * Y'` was computed a
# third time in a variable nothing read, at the cost of an `N × N` matrix and an `N × N` product.
function _omega_lift(Y::Manifold{T}, Δ::AbstractMatrix{T}) where {T}
    skew_matrix = 2 * (unit_matrix(KernelAbstractions.get_backend(Y), T, size(Y, 1)) -
                   T(0.5) * Y * Y')
    SkewSymMatrix(skew_matrix * Δ * Y')
end
