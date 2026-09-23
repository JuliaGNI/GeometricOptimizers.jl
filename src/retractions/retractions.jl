geodesic(A::AbstractVecOrMat) = A
cayley(A::AbstractVecOrMat) = A

# On a vector space the retraction is addition, so the exponential never enters and the choice of
# algorithm is irrelevant — but the methods have to exist, because `retraction(::Geodesic, x)` now
# passes one along on every parameter type.
geodesic(A::AbstractVecOrMat, ::AbstractExponentialAlgorithm) = A

# `mapparameters` throughout: a direction is of the parameters' shape, so a container solution hands
# these a container, whose leaves are a level below what `map` would reach.
#
# A retraction is applied to a *direction*, which `mapparameters` rebuilds in the shape of the
# parameters it was derived from — so these take the container, as every direction in this package is
# one.
geodesic(B::NetworkParameters) = mapparameters(geodesic, B)
function geodesic(B::NetworkParameters, algorithm::AbstractExponentialAlgorithm)
    mapparameters(Bᵢ -> geodesic(Bᵢ, algorithm), B)
end

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

# Implementation

Both factors are allocated once, zeroed, and written block by block, rather than assembled out of
nested `vcat`s and `hcat`s. The blocks are the same ones the formula above names; what goes is the
chain of intermediate matrices each nesting level materialised — a `vcat` of two blocks, then an
`hcat` of two of those, per factor, per call. The two zero blocks are then never written at all.
This is on the per-iteration path: an optimizer step applies a retraction once per line-search trial
and three times besides, and each of those calls this.

``(B'')^T`` is the matrix that is built, and ``B''`` is returned as its adjoint, because every
caller — [`cayley`](@ref), [`geodesic`](@ref), [`retraction_differential`](@ref) — consumes it as
``(B'')^T`` and the two cancel exactly.
"""
function lift_factors(B::StiefelLieAlgHorMatrix)
    T = eltype(B)
    N, n = B.N, B.n
    backend = KernelAbstractions.get_backend(B)
    unit = one(B.A)
    A_mat = B.A * unit

    B̂ = KernelAbstractions.zeros(backend, T, N, 2n)
    B̄ᵗ = KernelAbstractions.zeros(backend, T, 2n, N)

    # `transpose` and not `adjoint`: this is the upper-right block of the lift, which `getindex`
    # builds as `-B.B[j, i]` — entrywise, without conjugating. `+(::StiefelLieAlgHorMatrix,
    # ::AbstractMatrix)` rebuilds the same block and spells it the same way.
    @views begin
        B̂[1:n, 1:n] .= T(0.5) .* A_mat
        B̂[(n + 1):N, 1:n] .= B.B
        B̂[1:n, (n + 1):(2n)] .= unit
        B̄ᵗ[1:n, 1:n] .= unit
        B̄ᵗ[(n + 1):(2n), 1:n] .= T(0.5) .* A_mat
        B̄ᵗ[(n + 1):(2n), (n + 1):N] .= .-transpose(B.B)
    end

    (B̂, B̄ᵗ')
end

function lift_factors(B::GrassmannLieAlgHorMatrix)
    T = eltype(B)
    N, n = B.N, B.n
    backend = KernelAbstractions.get_backend(B)
    unit = unit_matrix(backend, T, n)

    B̂ = KernelAbstractions.zeros(backend, T, N, 2n)
    B̄ᵗ = KernelAbstractions.zeros(backend, T, 2n, N)

    # `transpose` for the reason the Stiefel method above gives
    @views begin
        B̂[(n + 1):N, 1:n] .= B.B
        B̂[1:n, (n + 1):(2n)] .= unit
        B̄ᵗ[1:n, 1:n] .= unit
        B̄ᵗ[(n + 1):(2n), (n + 1):N] .= .-transpose(B.B)
    end

    (B̂, B̄ᵗ')
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
        algorithm::AbstractExponentialAlgorithm = ScaledSquaring()) where {T}
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
    T = eltype(B)
    B̂, B̄ = lift_factors(B)

    # `mul!` into the identity rather than `one(B) + …`, for the reason
    # [`cayley(::StiefelLieAlgHorMatrix)`](@ref) gives: the identity is built either way, and adding
    # into it makes the product's destination the answer instead of one more `N × N` temporary.
    retracted = unit_matrix(KernelAbstractions.get_backend(B), T, B.N)
    mul!(retracted, B̂ * 𝔄(B̂, B̄, algorithm), B̄', one(T), one(T))

    manifold_type(B)(retracted)
end

# Every step runs on the lift's backend: the `qr` and the `eigen` are the backend's own, and nothing
# is copied to the host. A backend that supplies neither for a dense array raises inside them.
function geodesic(B::AbstractLieAlgHorMatrix, ::ProjectedSkew)
    T = eltype(B)
    backend = KernelAbstractions.get_backend(B)
    B̂, B̄ = lift_factors(B)

    # `B̄` is skew-symmetric of rank ≤ 2n, so its range and its row space coincide and both sit
    # inside the range of `B̂`. In an orthonormal basis `Q` of that range it *is* a 2n × 2n
    # skew-symmetric matrix, and the exponential of one of those can be formed from an
    # eigendecomposition — which makes the result orthogonal by construction rather than by
    # cancellation, at every lift norm. The thin `Q` is the factor applied to the first `2n` columns
    # of the identity, which keeps it on `B̂`'s backend where `Matrix` would copy it to the host.
    Q = qr(B̂).Q * StiefelProjection(B̂).A
    M = (Q' * B̂) * (B̄' * Q)
    M = (M - M') / 2                              # `M` is skew up to round-off; make it exactly so

    # `im * M` is Hermitian for real skew `M`, so `M = -i·V·Λ·V*` and `exp(M) = ℜ(V·exp(-iΛ)·V*)`.
    # The broadcast scales the columns of `V`, which is the product with `Diagonal(exp(-iΛ))`.
    Λ, V = eigen(Hermitian(im * M))
    expM = real((V .* transpose(cis.(-Λ))) * V')

    retracted = unit_matrix(backend, T, B.N)
    mul!(retracted, Q * (expM - unit_matrix(backend, T, 2 * B.n)), Q', one(T), one(T))
    manifold_type(B)(retracted)
end

cayley(B::NetworkParameters) = mapparameters(cayley, B)

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
\mathrm{Cayley}(\bar{B}) = \mathbb{I} + B' \left(\mathbb{I}_{2n} - \frac{1}{2} (B'')^T B'\right)^{-1} (B'')^T,
```
with
```math
\bar{B} = \begin{bmatrix}
    A & -B^T \\
    B & \mathbb{O}
\end{bmatrix} = \begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix} =: B'(B'')^T,
```
i.e. ``\bar{B}`` is expressed as a product of two ``N\times{}2n`` matrices.

This is the same shape [`geodesic(::AbstractLieAlgHorMatrix)`](@ref) has, and for the same reason:
the only matrix function either evaluates is on the ``2n\times{}2n`` product ``(B'')^TB'``. Getting
there from the definition ``\mathrm{Cayley}(\bar{B}) = (\mathbb{I} - \frac{1}{2}\bar{B})^{-1}(\mathbb{I} + \frac{1}{2}\bar{B})``
takes two steps. Write ``C`` for the ``2n\times{}2n`` inverse above. Splitting
``\mathbb{I} + \frac{1}{2}\bar{B} = (\mathbb{I} - \frac{1}{2}\bar{B}) + \bar{B}`` gives
``\mathrm{Cayley}(\bar{B}) = \mathbb{I} + (\mathbb{I} - \frac{1}{2}\bar{B})^{-1}\bar{B}``, and the
Woodbury identity turns the remaining inverse into
``(\mathbb{I} - \frac{1}{2}\bar{B})^{-1} = \mathbb{I} + \frac{1}{2}B'C(B'')^T``. The factor that
leaves on the ``2n\times{}2n`` side is then ``\mathbb{I}_{2n} + \frac{1}{2}C(B'')^TB'``, which is
``C`` itself, because ``C`` inverts ``\mathbb{I}_{2n} - \frac{1}{2}(B'')^TB'``.

The result is the ``N\times{}N`` retraction of the lift, so the cost cannot fall below ``O(N^2n)``,
and this grouping reaches that floor: nothing larger than ``N\times{}2n`` enters a product.
Multiplying the factored left-hand side by an unfactored ``\mathbb{I} + \frac{1}{2}\bar{B}``
instead multiplies two dense ``N\times{}N`` matrices, which is ``O(N^3)`` on its own and puts the
retraction above [`geodesic`](@ref) at every size. `scripts/cayley_regrouping_cost.jl` carries the
measurement.

Both sums are written as the five-argument `mul!` into an identity rather than as `𝕀 + X`: the
identity has to be built either way, and adding into it is what makes the product's own destination
the answer instead of a temporary. `scripts/retraction_step_allocations.jl` carries that
measurement. The ``2n\times{}2n`` inverse stays an `inv` and is the allocation this cannot remove —
`lu!` and `rdiv!` would take it in place on the host, and neither is something a
`KernelAbstractions` backend is obliged to supply, where `inv` is what `cayley` already runs on
Metal through.
"""
function cayley(B::StiefelLieAlgHorMatrix)
    StiefelManifold(_cayley_matrix(B))
end

# The body both `cayley` methods share; they differ only in the manifold the result is wrapped in,
# and `manifold_type` is not used for that because each of the two carries its own docstring and the
# manual links to both by signature.
function _cayley_matrix(B::AbstractLieAlgHorMatrix)
    T = eltype(B)
    backend = KernelAbstractions.get_backend(B)
    B̂, B̄ = lift_factors(B)

    M = unit_matrix(backend, T, 2 * B.n)
    mul!(M, B̄', B̂, -T(0.5), one(T))
    retracted = unit_matrix(backend, T, B.N)
    mul!(retracted, B̂ * inv(M), B̄', one(T), one(T))

    retracted
end

@doc raw"""
    cayley(B̄::GrassmannLieAlgHorMatrix)

Compute the Cayley retraction of `B`.

This is equivalent to the method of [`cayley`](@ref) for [`StiefelLieAlgHorMatrix`](@ref).

See [`cayley(::StiefelLieAlgHorMatrix)`](@ref).
"""
function cayley(B::GrassmannLieAlgHorMatrix)
    GrassmannManifold(_cayley_matrix(B))
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
projection has the same ones, so what [`lift_from_columns`](@ref) is handed is that projection.

For a [`GrassmannLieAlgHorMatrix`](@ref) the top ``n\times{}n`` block is dropped rather than kept,
and that is not an approximation. ``M``, ``M^T`` and ``\bar{B}`` are all rational functions of
``\bar{B}`` and so commute, which gives

```math
D(\alpha) = \bar{B}\left(\mathbb{I} - \tfrac{\alpha^2}{4}\bar{B}^2\right)^{-1} ,
```

and for a Grassmann lift — where ``A \equiv \mathbb{O}``, so ``\bar{B}^2`` is block-diagonal — the
top-left block of that product is identically zero at every ``\alpha``. The drop is exact whatever
``D`` is paired against, not merely exact against a gradient that happens to be orthogonal to ``Y``.

Both inverses are taken through [`lift_factors`](@ref) and the Woodbury identity, exactly as
[`cayley`](@ref) does, so the cost is ``O(Nn^2 + n^3)`` and no ``N\times{}N`` matrix is formed.

`α = 0` returns `B` unchanged, and that is the case that matters for cost:
[`SimpleSolvers.Backtracking`](@extref) — the default of [`default_linesearch`](@ref) — evaluates
``\varphi'`` only at ``\alpha = 0``, so it never reaches the general branch.
"""
retraction_differential(R::AbstractRetraction,
    B,
    α) = error("retraction_differential is not implemented for $(typeof(R)) and $(typeof(B)); " *
               "a line search that evaluates φ' needs it, and pairing the gradient with `B` instead is " *
               "correct only where α ↦ retract(αB) is a one-parameter subgroup.")

retraction_differential(::Geodesic, B, α) = B

retraction_differential(::Cayley, B::AbstractVecOrMat, α) = B

function retraction_differential(R::Cayley, B::NetworkParameters, α)
    mapparameters(Bᵢ -> retraction_differential(R, Bᵢ, α), B)
end

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

    lift_from_columns(B, V)
end

@doc raw"""
    lift_from_columns(B, V)

Rebuild a lift of the same type as `B` from the ``N\times{}n`` block `V`, i.e. from the first ``n``
columns of a skew matrix. Used by [`retraction_differential`](@ref), which is the only place a lift
has to be assembled from its action on ``E`` rather than from its own blocks.

Not to be confused with the *canonical horizontal lift* [`GeometricOptimizers.Ω`](@ref), which maps a
tangent vector to an element of ``\mathfrak{g}^\mathrm{hor}``. This one takes the columns of a matrix
that is already in the Lie algebra.
"""
lift_from_columns(B::StiefelLieAlgHorMatrix, V::AbstractMatrix) = StiefelLieAlgHorMatrix(
    SkewSymMatrix(V[1:(B.n), :]), V[(B.n + 1):(B.N), :], B.N, B.n)

function lift_from_columns(B::GrassmannLieAlgHorMatrix, V::AbstractMatrix)
    GrassmannLieAlgHorMatrix(V[(B.n + 1):(B.N), :], B.N, B.n)
end

@doc raw"""
    retraction(R::AbstractRetraction, x)

Apply the retraction `R` to `x`, i.e. dispatch on the retraction *type* rather than calling
[`geodesic`](@ref) or [`cayley`](@ref) by name.

This is what a caller who has been handed a `retraction = …` keyword uses: the two shipped types,
[`Geodesic`](@ref) and [`Cayley`](@ref), select the two functions, and a `Geodesic` also carries the
[`AbstractExponentialAlgorithm`](@ref) its `geodesic` is evaluated with. `R(x)` is the same thing
written as a call.

`x` is an [`AbstractLieAlgHorMatrix`](@ref) on a manifold — the retractions this package is about map
``\mathfrak{g}^\mathrm{hor}\to{}G`` — a `NamedTuple` of parameters, or an ordinary array, on which
every retraction is the identity because the extended retraction on a vector space is addition and
[`update_section!`](@ref) does the adding.

An `R` and an `x` that do not go together is an error that says so, rather than a `nothing` that
fails further downstream.
"""
function retraction(R::AbstractRetraction, x::AbstractArray)
    error("retraction is not implemented for $(typeof(R)) and $(typeof(x)).")
end

retraction(::Cayley, x::AbstractArray) = cayley(x)
retraction(R::Geodesic, x::AbstractArray) = geodesic(x, R.algorithm)

(R::AbstractRetraction)(x::AbstractArray) = retraction(R, x)
