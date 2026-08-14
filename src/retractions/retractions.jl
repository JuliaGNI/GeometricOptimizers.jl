geodesic(A::AbstractVecOrMat) = A
cayley(A::AbstractVecOrMat) = A

# On a vector space the retraction is addition, so the exponential never enters and the choice of
# algorithm is irrelevant — but the methods have to exist, because `retraction(::Geodesic, x)` now
# passes one along on every parameter type.
geodesic(A::AbstractVecOrMat, ::AbstractExponentialAlgorithm) = A

geodesic(B::NamedTuple) = apply_toNT(geodesic, B)
geodesic(B::NamedTuple, algorithm::AbstractExponentialAlgorithm) =
    apply_toNT(Bᵢ -> geodesic(Bᵢ, algorithm), B)

@doc raw"""
    lift_factors(B::AbstractLieAlgHorMatrix)

Factor a horizontal lift into the two ``N\times{}2n`` matrices ``B'`` and ``B''`` with
``\bar{B} = B'(B'')^T``:

```math
\bar{B} = \begin{bmatrix}
    A & -B^T \\
    B & \mathbb{O}
\end{bmatrix} = \begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix} =: B'(B'')^T.
```

Every retraction in this package is built on this factorisation — it is what makes their cost scale
with ``n`` rather than with ``N``, since the only matrix function either of them evaluates is on the
``2n\times{}2n`` product ``(B'')^TB'``. Both [`geodesic`](@ref) and [`cayley`](@ref) use it, for both
manifolds.

For a [`GrassmannLieAlgHorMatrix`](@ref) this is the same expression with ``A \equiv \mathbb{O}``.
"""
function lift_factors(B::StiefelLieAlgHorMatrix)
    T = eltype(B)
    E = StiefelProjection(B)
    unit = one(B.A)
    A_mat = B.A * unit

    B̂ = hcat(vcat(T(0.5) * A_mat, B.B), E)
    B̄ = hcat(vcat(unit, T(0.5) * A_mat), vcat(zero(B.B'), -B.B'))'

    (B̂, B̄)
end

function lift_factors(B::GrassmannLieAlgHorMatrix)
    T = eltype(B)
    E = StiefelProjection(B)
    backend = KernelAbstractions.get_backend(B)
    zero_mat = KernelAbstractions.zeros(backend, T, B.n, B.n)

    B̂ = hcat(vcat(zero_mat, B.B), E)
    B̄ = hcat(vcat(one(zero_mat), zero_mat), vcat(zero(B.B'), -B.B'))'

    (B̂, B̄)
end

@doc raw"""
    geodesic(Y::Manifold, Δ, algorithm = ScaledSquaring())

Take as input an element of a manifold `Y` and a tangent vector in `Δ` in the corresponding tangent space and compute the geodesic (exponential map).

In different notation: take as input an element ``x`` of ``\mathcal{M}`` and an element of ``T_x\mathcal{M}`` and return ``\mathtt{geodesic}(x, v_x) = \exp(v_x).``

`algorithm` selects how the exponential is evaluated and is passed straight through to
[`geodesic(::AbstractLieAlgHorMatrix)`](@ref); see [`AbstractExponentialAlgorithm`](@ref) for the
choice. It matters only for a large ``\Delta`` — the default [`ScaledSquaring`](@ref) is accurate at
every step size.

# Examples

```jldoctest
using GeometricOptimizers

Y = StiefelManifold([1. 0. 0.;]' |> Matrix)
Δ = [0. .5 0.;]' |> Matrix
Y₂ = GeometricOptimizers.geodesic(Y, Δ)

Y₂' * Y₂ ≈ [1.;]

# output

true
```

# Implementation

Internally this `geodesic` method calls [`geodesic(::AbstractLieAlgHorMatrix)`](@ref).
"""
function geodesic(Y::Manifold{T}, Δ::AbstractMatrix{T},
    algorithm::AbstractExponentialAlgorithm=ScaledSquaring()) where {T}
    λY = GlobalSection(Y)
    B = global_rep(λY, Δ)
    E = StiefelProjection(B)
    expB = geodesic(B, algorithm)
    λY * typeof(Y)(expB * E)
end

@doc raw"""
    geodesic(B̄::AbstractLieAlgHorMatrix, algorithm = ScaledSquaring())

Compute the geodesic of a horizontal lift, i.e. ``\exp(\bar{B})``.

Works for both [`StiefelLieAlgHorMatrix`](@ref) and [`GrassmannLieAlgHorMatrix`](@ref) —
[`manifold_type`](@ref) supplies the manifold the result belongs to.

# Implementation

Internally this is using

```math
\exp(\bar{B}) = \mathbb{I} + B'\mathfrak{A}(B', B'')(B'')^T,
```

with ``\bar{B} = B'(B'')^T`` the factorisation of [`lift_factors`](@ref). The only matrix function
this evaluates is ``\mathfrak{A}`` on the ``2n\times{}2n`` product ``(B'')^TB'``, so the cost is set
by ``n`` and not by ``N``, and so is the accuracy — see [`AbstractExponentialAlgorithm`](@ref) for
the choice of `algorithm` and [`GeometricOptimizers.𝔄`](@ref) for the implementations.

!!! warning "The default changed in 0.2.0"
    The unscaled series was the only algorithm before, and it silently leaves the manifold for
    ``\|\bar{B}\| \gtrsim 50``. It is still reachable as [`TaylorSeries`](@ref), and its docstring
    carries the measurements. The default is now [`ScaledSquaring`](@ref), which is both accurate at
    every lift norm and faster.
"""
function geodesic(B::AbstractLieAlgHorMatrix, algorithm::AbstractExponentialAlgorithm = ScaledSquaring())
    B̂, B̄ = lift_factors(B)

    manifold_type(B)(one(B) + B̂ * 𝔄(B̂, B̄, algorithm) * B̄')
end

function geodesic(B::AbstractLieAlgHorMatrix, ::ProjectedSkew)
    B̂, B̄ = lift_factors(B)

    # `B̄` is skew-symmetric of rank ≤ 2n, so its range and its row space coincide and both sit
    # inside the range of `B̂`. In an orthonormal basis `Q` of that range it *is* a 2n × 2n
    # skew-symmetric matrix, and the exponential of one of those can be formed from an
    # eigendecomposition — which makes the result orthogonal by construction rather than by
    # cancellation, at every lift norm.
    Q = Matrix(qr(B̂).Q)
    M = (Q' * B̂) * (B̄' * Q)
    M = (M - M') / 2                              # `M` is skew up to round-off; make it exactly so

    # `im * M` is Hermitian for real skew `M`, so `M = -i·V·Λ·V*` and `exp(M) = ℜ(V·exp(-iΛ)·V*)`.
    Λ, V = eigen(Hermitian(im * M))
    expM = real(V * Diagonal(cis.(-Λ)) * V')

    manifold_type(B)(one(B) + Q * (expM - I) * Q')
end

cayley(B::NamedTuple) = apply_toNT(cayley, B)

@doc raw"""
    cayley(Y::Manifold, Δ)

Take as input an element of a manifold `Y` and a tangent vector in `Δ` in the corresponding tangent space and compute the Cayley retraction.

In different notation: take as input an element ``x`` of ``\mathcal{M}`` and an element of ``T_x\mathcal{M}`` and return ``\mathrm{Cayley}(v_x).``

# Examples

```jldoctest
using GeometricOptimizers

Y = StiefelManifold([1. 0. 0.;]' |> Matrix)
Δ = [0. .5 0.;]' |> Matrix
Y₂ = GeometricOptimizers.cayley(Y, Δ)

Y₂' * Y₂ ≈ [1.;]

# output

true
```

See the example in [`geodesic(::Manifold, ::AbstractMatrix)`](@ref).
"""
function cayley(Y::Manifold{T}, Δ::AbstractMatrix{T}) where {T}
    λY = GlobalSection(Y)
    B = global_rep(λY, Δ)
    E = StiefelProjection(B)
    cayleyB = cayley(B)
    λY * typeof(Y)(cayleyB * E)
end

@doc raw"""
    cayley(B̄::StiefelLieAlgHorMatrix)

Compute the Cayley retraction of `B`.

# Implementation

Internally this is using

```math
\mathrm{Cayley}(\bar{B}) = \mathbb{I} + \frac{1}{2} B' (\mathbb{I}_{2n} - \frac{1}{2} (B'')^T B')^{-1} (B'')^T (\mathbb{I} + \frac{1}{2} B),
```
with
```math
\bar{B} = \begin{bmatrix}
    A & -B^T \\
    B & \mathbb{O}
\end{bmatrix} = \begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix} =: B'(B'')^T,
```
i.e. ``\bar{B}`` is expressed as a product of two ``N\times{}2n`` matrices.
"""
function cayley(B::StiefelLieAlgHorMatrix)
    T = eltype(B)
    𝕀_small = one(B.A)
    𝕆 = zero(𝕀_small)
    𝕀_small2 = hcat(vcat(𝕀_small, 𝕆), vcat(𝕆, 𝕀_small))
    𝕀_big = one(B)
    B̂, B̄ = lift_factors(B)

    StiefelManifold((𝕀_big + T(0.5) * B̂ * inv(𝕀_small2 - T(0.5) * B̄' * B̂) * B̄') * (𝕀_big + T(0.5) * B))
end

@doc raw"""
    cayley(B̄::GrassmannLieAlgHorMatrix)

Compute the Cayley retraction of `B`.

This is equivalent to the method of [`cayley`](@ref) for [StiefelLieAlgHorMatrix](@ref).

See [`cayley(::StiefelLieAlgHorMatrix)`](@ref).
"""
function cayley(B::GrassmannLieAlgHorMatrix)
    T = eltype(B)
    backend = KernelAbstractions.get_backend(B)
    𝕆 = KernelAbstractions.zeros(backend, T, B.n, B.n)
    𝕀_small = one(𝕆)
    𝕀_small2 = hcat(vcat(𝕀_small, 𝕆), vcat(𝕆, 𝕀_small))
    𝕀_big = one(B)
    B̂, B̄ = lift_factors(B)

    GrassmannManifold((𝕀_big + T(0.5) * B̂ * inv(𝕀_small2 - T(0.5) * B̄' * B̂) * B̄') * (𝕀_big + T(0.5) * B))
end

@doc raw"""
    retraction_differential(retraction, B, α)

The horizontal lift ``D(\alpha)`` that generates the velocity of the curve a line search walks along.

[`trial_iterate!`](@ref) builds its trial point as ``x(\alpha) = \Lambda_0\mathrm{retract}(\alpha\bar{B})E``
with ``\Lambda_0`` the section the step starts from, so with ``W(\alpha) = \Lambda_0\mathrm{retract}(\alpha\bar{B})``
the frame the cache holds afterwards, this returns the ``D(\alpha)`` for which

```math
\frac{dx}{d\alpha} = W(\alpha)D(\alpha)E .
```

[`trial_slope`](@ref) pairs the gradient against it, which is what makes ``\varphi'(\alpha)`` the
derivative of ``\varphi``.

# Implementation

For [`Geodesic`](@ref) this is ``\bar{B}`` itself. The exponential is a one-parameter subgroup, so
``\frac{d}{d\alpha}\exp(\alpha\bar{B}) = \exp(\alpha\bar{B})\bar{B}`` and the frame factors out
whole. The same holds for an ordinary array under either retraction, where the retraction is addition
and ``\varphi`` is affine in ``\alpha``.

[`Cayley`](@ref) is *not* a one-parameter subgroup, and this is what that costs. With
``M = (\mathbb{I} - \frac{\alpha}{2}\bar{B})^{-1}``,

```math
\frac{d}{d\alpha}\mathrm{Cayley}(\alpha\bar{B}) = M\bar{B}M ,
\qquad
D(\alpha) = \mathrm{Cayley}(\alpha\bar{B})^{-1}M\bar{B}M = M^T\bar{B}M ,
```

using ``M^T = (\mathbb{I} + \frac{\alpha}{2}\bar{B})^{-1}`` for skew ``\bar{B}``. ``D`` is skew but
not horizontal; only its first ``n`` columns ever enter — ``D(\alpha)E`` — and the horizontal
projection has the same ones, so what is returned is that projection. For a
[`GrassmannLieAlgHorMatrix`](@ref) the top block is dropped rather than kept, which is not an
approximation either: [`rgrad`](@ref) there is ``\nabla{}L - Y(Y^T\nabla{}L)`` and so satisfies
``Y^T\mathrm{rgrad} = 0``, which is exactly the pairing that block would contribute to.

Both inverses are taken through [`lift_factors`](@ref) and the Woodbury identity, exactly as
[`cayley`](@ref) does, so the cost is ``O(Nn^2 + n^3)`` and no ``N\times{}N`` matrix is formed.

`α = 0` returns `B` unchanged, and that is the case that matters for cost:
[`SimpleSolvers.Backtracking`](@extref) — the default of [`default_linesearch`](@ref) — evaluates
``\varphi'`` only at ``\alpha = 0``, so it never reaches the general branch.
"""
retraction_differential(::Geodesic, B, α) = B

retraction_differential(::Cayley, B::AbstractVecOrMat, α) = B

retraction_differential(R::Cayley, B::NamedTuple, α) =
    apply_toNT(Bᵢ -> retraction_differential(R, Bᵢ, α), B)

function retraction_differential(::Cayley, B::AbstractLieAlgHorMatrix{T}, α) where {T}
    iszero(α) && return B

    a = T(α) / 2
    B̂, B̄ = lift_factors(B)
    E = StiefelProjection(B)
    G = B̄' * B̂
    𝕀 = one(G)

    w₁ = E + B̂ * ((𝕀 - a * G) \ (a * (B̄' * E)))      # M E
    w₂ = B̂ * (B̄' * w₁)                                # B̄ M E
    V = w₂ - B̂ * ((𝕀 + a * G) \ (a * (B̄' * w₂)))     # Mᵀ B̄ M E

    horizontal_lift(B, V)
end

@doc raw"""
    horizontal_lift(B, V)

Rebuild a lift of the same type as `B` from the ``N\times{}n`` block `V`, i.e. from the first ``n``
columns of a skew matrix. Used by [`retraction_differential`](@ref), which is the only place a lift
has to be assembled from its action on ``E`` rather than from its own blocks.
"""
horizontal_lift(B::StiefelLieAlgHorMatrix, V::AbstractMatrix) =
    StiefelLieAlgHorMatrix(SkewSymMatrix(V[1:(B.n), :]), V[(B.n+1):(B.N), :], B.N, B.n)

horizontal_lift(B::GrassmannLieAlgHorMatrix, V::AbstractMatrix) =
    GrassmannLieAlgHorMatrix(V[(B.n+1):(B.N), :], B.N, B.n)

# This used to be an empty method body, i.e. every combination that is not covered below
# returned `nothing` and failed somewhere downstream with an unrelated message.
function retraction(R::AbstractRetraction, x::AbstractArray)
    error("retraction is not implemented for $(typeof(R)) and $(typeof(x)).")
end

retraction(::Cayley, x::AbstractArray) = cayley(x)
retraction(R::Geodesic, x::AbstractArray) = geodesic(x, R.algorithm)

(R::AbstractRetraction)(x::AbstractArray) = retraction(R, x)
