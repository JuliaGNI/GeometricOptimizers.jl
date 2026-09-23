@doc raw"""
    StiefelManifold <: Manifold

An implementation of the Stiefel manifold [hairer2006geometric](@cite). The Stiefel manifold is the collection of all matrices ``Y\in\mathbb{R}^{N\times{}n}`` whose columns are orthonormal, i.e.

```math
    St(n, N) = \{Y: Y^TY = \mathbb{I}_n \}.
```

The Stiefel manifold can be shown to have manifold structure (as the name suggests) and this is heavily used in `GeometricOptimizers`. It is further a compact space.
More information can be found in the docstrings for [`rgrad(::StiefelManifold, ::AbstractMatrix)`](@ref) and [`metric(::StiefelManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref).
"""
mutable struct StiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
end

@doc raw"""
    rgrad(Y::StiefelManifold, ∇L::AbstractMatrix)

Compute the Riemannian gradient for the Stiefel manifold at `Y` based on `∇L`.

Here ``Y\in{}St(N,n)`` and ``\nabla{}L\in\mathbb{R}^{N\times{}n}`` is the Euclidean gradient.

The function computes the Riemannian gradient with respect to the canonical metric:
[`metric(::StiefelManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref).

The precise form of the mapping is:
```math
\mathtt{rgrad}(Y, \nabla{}L) \mapsto \nabla{}L - Y(\nabla{}L)^TY
```

Note the property ``Y^T\mathtt{rgrad}(Y, \nabla{}L)\in\mathcal{S}_\mathrm{skew}(n).``

# Examples

```jldoctest
using GeometricOptimizers

Y = StiefelManifold([1 0 ; 0 1 ; 0 0; 0 0])
Δ = [1 2; 3 4; 5 6; 7 8]
rgrad(Y, Δ)

# output

4×2 Matrix{Int64}:
 0  -1
 1   0
 5   6
 7   8
```
"""
function rgrad(Y::StiefelManifold, ∇L::AbstractMatrix)
    ∇L = _match_backend(Y, ∇L) # TEMPORARY, see `_match_backend`
    ∇L - Y.A * (∇L' * Y.A)
end

@doc raw"""
    metric(Y::StiefelManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)

Compute the dot product for `Δ₁` and `Δ₂` at `Y`.

This uses the canonical Riemannian metric for the Stiefel manifold:
```math
g_Y: (\Delta_1, \Delta_2) \mapsto \mathrm{Tr}(\Delta_1^T(\mathbb{I} - \frac{1}{2}YY^T)\Delta_2).
```
"""
function metric(Y::StiefelManifold{T}, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix) where {T}
    LinearAlgebra.tr(Δ₁' * Δ₂) - (T(1) / 2) * LinearAlgebra.tr((Δ₁' * Y.A) * (Y.A' * Δ₂))
end

@doc raw"""
    global_section(Y::StiefelManifold)

Compute a matrix of size ``N\times(N-n)`` whose columns are orthogonal to the columns in `Y`.

This matrix is also called ``Y_\perp`` [absil2004riemannian, absil2008optimization, bendokat2020grassmann](@cite).

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: global_section
import Random

Random.seed!(123)

Y = StiefelManifold([1. 0.; 0. 1.; 0. 0.; 0. 0.])

round.(global_section(Y); digits = 3)

# output

4×2 Matrix{Float64}:
  0.0     0.0
  0.0     0.0
 -0.936   0.353
 -0.353  -0.936
```

# Implementation

Internally we do:

```julia
orthonormal_columns() do
    A = randn(N, N - n) # or the gpu equivalent
    A - Y.A * (Y.A' * A)
end
```

The orthonormalization is **CholeskyQR2 and not `LinearAlgebra.qr!`**, on every backend — see
[`_cholesky_qr2`](@ref GeometricOptimizers._cholesky_qr2). `qr!` is a host factorization here:
`Metal` implements no `qr` for its array type at all, so a `qr!` in this function puts
`GlobalSection(Y)` and `Optimizer(Y, F)` out of reach on a device — and this function is on the path
[`geodesic`](@ref) and [`cayley`](@ref) take on every step.

`A` is square inside the complement of `Y`, so its condition number has a square Gaussian's heavy
tail and CholeskyQR2 breaks down on it about once in a hundred and twenty in `Float32`. A draw it
cannot orthonormalize is *replaced* — see
[`orthonormal_columns`](@ref GeometricOptimizers.orthonormal_columns) for the measurement and for
why a redraw rather than a repair is the honest answer.
"""
function global_section(Y::StiefelManifold{T}) where {T}
    N, n = size(Y)
    backend = KernelAbstractions.get_backend(Y)
    λ = orthonormal_columns() do
        A = KernelAbstractions.allocate(backend, T, N, N - n)
        randn!(A)
        A - Y.A * (Y.A' * A)
    end

    # The section's storage array has to be the *point's* array type, which `test/device_copyto.jl`
    # relies on to move a section between two of them. It already is for every `KernelAbstractions`
    # backend, and the branch folds away there; the projection above only loses the point's type for
    # a wrapper `allocate` does not know how to produce. An unconditional `typeof(Y.A)(…)` would copy
    # even where the two already agree, which is why the branch is here. `convert` is not the
    # spelling: an `AbstractMatrix` outside `Base`'s hierarchy need define no method for it.
    λ isa typeof(Y.A) ? λ : typeof(Y.A)(λ)
end

function Base.zero(Y::StiefelManifold{T}) where {T}
    N, n = size(Y)
    backend = KernelAbstractions.get_backend(Y.A)
    zeros(backend, StiefelLieAlgHorMatrix{T}, N, n)
end
