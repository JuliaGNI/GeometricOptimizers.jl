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
    geodesic(Y::Manifold, Δ)

Take as input an element of a manifold `Y` and a tangent vector in `Δ` in the corresponding tangent space and compute the geodesic (exponential map).

In different notation: take as input an element ``x`` of ``\mathcal{M}`` and an element of ``T_x\mathcal{M}`` and return ``\mathtt{geodesic}(x, v_x) = \exp(v_x).``


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

Internally this `geodesic` method calls [`geodesic(::StiefelLieAlgHorMatrix)`](@ref).
"""
function geodesic(Y::Manifold{T}, Δ::AbstractMatrix{T}) where {T}
    λY = GlobalSection(Y)
    B = global_rep(λY, Δ)
    E = StiefelProjection(B)
    expB = geodesic(B)
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

See the example in [`geodesic(::Manifold{T}, ::AbstractMatrix{T}) where T`].
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

# This used to be an empty method body, i.e. every combination that is not covered below
# returned `nothing` and failed somewhere downstream with an unrelated message.
function retraction(R::AbstractRetraction, x::AbstractArray)
    error("retraction is not implemented for $(typeof(R)) and $(typeof(x)).")
end

retraction(::Cayley, x::AbstractArray) = cayley(x)
retraction(R::Geodesic, x::AbstractArray) = geodesic(x, R.algorithm)

(R::AbstractRetraction)(x::AbstractArray) = retraction(R, x)
