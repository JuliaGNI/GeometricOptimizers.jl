@doc raw"""
    AbstractExponentialAlgorithm

Supertype of the algorithms that [`Geodesic`](@ref) can use to evaluate the matrix exponential.

A horizontal lift factors as ``\bar{B} = B'(B'')^T`` into two ``N\times{}2n`` matrices, and the
exponential of a product in that order is

```math
\exp(B'(B'')^T) = \mathbb{I} + B'\,\mathfrak{A}((B'')^TB')\,(B'')^T,
\qquad
\mathfrak{A}(X) = \sum_{n=1}^\infty \frac{X^{n-1}}{n!},
```

so the whole computation reduces to one ``2n\times{}2n`` matrix function — the argument
``X = (B'')^TB'`` is small even when ``N`` is large. ``\mathfrak{A}`` is the function usually written
``\varphi_1(X) = (\exp(X) - \mathbb{I})X^{-1}``, though it is defined by the series and is perfectly
regular at a singular ``X``.

The subtypes differ only in how that ``2n\times{}2n`` function is evaluated, and they are
mathematically identical — every one of them returns the exponential map, so the one-parameter
subgroup property [`Geodesic`](@ref) relies on holds for all of them. They differ in accuracy at a
large lift, in cost, and in which backends they run on. On a random `StiefelLieAlgHorMatrix(20, 3)`
scaled to ``\|\bar{B}\| = 361``, against the cost of one retraction at ``N = 200``, ``n = 10``:

| | `check` | forward error | cost | backend |
|---|---|---|---|---|
| [`ScaledSquaring`](@ref) | `3.6e-14` | `1.0e-14` | `0.094 ms` | any |
| [`AugmentedPade`](@ref) | `1.5e-14` | `1.3e-14` | `0.125 ms` | CPU (dense LAPACK) |
| [`ProjectedSkew`](@ref) | `4.8e-15` | `2.0e-14` | `0.126 ms` | CPU (dense LAPACK) |
| [`TaylorSeries`](@ref) | `1.4e168` | — | `0.155 ms` | any |

"Forward error" is the relative distance to `exp(Matrix(B))`. [`ScaledSquaring`](@ref) is the default
and is the cheapest as well as the most portable, so there is rarely a reason to change it —
[`ProjectedSkew`](@ref) if `check` matters more than the last digit of the exponential, and
[`AugmentedPade`](@ref) as the independent implementation the other two are tested against.
[`TaylorSeries`](@ref) is the pre-0.2.0 behaviour and is retained only so the regression is
reproducible; it is not a usable retraction.

A new one has to supply `𝔄(X::AbstractMatrix, ::NewAlgorithm)`; `geodesic` and everything above it
then follow. An algorithm that does not go through ``\mathfrak{A}`` at all — [`ProjectedSkew`](@ref)
is the one such — supplies `geodesic(::AbstractLieAlgHorMatrix, ::NewAlgorithm)` instead.

See [`GeometricOptimizers.𝔄`](@ref) for the implementations.
"""
abstract type AbstractExponentialAlgorithm end

@doc raw"""
    ScaledSquaring(θ = 0.5) <: AbstractExponentialAlgorithm

Evaluate ``\mathfrak{A}`` by scaling and squaring, and the default.

The series for ``\mathfrak{A}`` converges for every argument but is only *accurate* for a small one:
at ``\|X\| \gg 1`` its terms cancel catastrophically, and the partial sum reaches ``2.5\cdot10^{18}``
where the result is of order one. So halve the argument until it is small, sum the series there, and
undo the halving by squaring.

The squaring is done on the ``2n\times{}2n`` factor rather than on the assembled ``N\times{}N``
matrix, which is possible because the low-rank form is closed under squaring:

```math
\left(\mathbb{I} + B'W(B'')^T\right)^2 = \mathbb{I} + B'\left(2W + WXW\right)(B'')^T,
\qquad X = (B'')^TB',
```

so one squaring of the exponential is one application of ``W \mapsto 2W + WXW``. With ``s`` chosen so
that ``\|X\|_1/2^s \leq θ``, the whole algorithm is `s` small matrix products on top of a series that
now converges in a handful of terms — cheaper than summing the unscaled series, not just more
accurate.

`θ` is the norm below which the series is summed. It barely matters: at ``\|\bar{B}\| = 155`` every
``θ \in [0.125, 4]`` gives a `check` between `9.9e-15` and `5.0e-14` and a forward error between
`6.4e-15` and `8.2e-15`. `0.5` is the measured middle of that range and needs no tuning.

This is the only algorithm that uses nothing but matrix products and norms, so it is the only one
that runs unchanged on a `KernelAbstractions` GPU backend. That is why it is the default. The norm
is taken by [`GeometricOptimizers.opnorm₁`](@ref) rather than by `LinearAlgebra.opnorm(X, 1)`, which
is a scalar-indexing loop and would give up exactly the property this paragraph claims.

!!! note "The argument is worse-conditioned than the lift"
    ``X``'s lower-left block is ``\frac{1}{4}A^2 - B^TB``, so ``\|X\| \approx \|\bar{B}\|^2/4`` while
    its spectral radius is only ``\approx\|\bar{B}\|`` — the eigenvalues of ``X`` are the nonzero
    (purely imaginary) eigenvalues of ``\bar{B}``. The factorisation is strongly non-normal, which is
    why the unscaled series does worse here than it would on ``\bar{B}`` itself, and why `s` grows
    like ``2\log_2\|\bar{B}\|`` rather than ``\log_2\|\bar{B}\|``.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct ScaledSquaring{T<:Real} <: AbstractExponentialAlgorithm
    θ::T

    function ScaledSquaring(θ::T = 0.5) where {T<:Real}
        @assert θ > zero(T) "the scaling threshold has to be positive, got $(θ)"
        new{T}(θ)
    end
end

@doc raw"""
    AugmentedPade <: AbstractExponentialAlgorithm

Evaluate ``\mathfrak{A}`` as a block of a larger *ordinary* exponential.

```math
\exp\begin{pmatrix} X & \mathbb{I} \\ \mathbb{O} & \mathbb{O} \end{pmatrix}
= \begin{pmatrix} \exp(X) & \mathfrak{A}(X) \\ \mathbb{O} & \mathbb{I} \end{pmatrix}
```

so one call to `Base.exp` on a ``4n\times{}4n`` matrix returns ``\mathfrak{A}(X)`` in its upper-right
block. That hands the numerics to Julia's own exponential — a degree-13 Padé approximant with its own
scaling and squaring, and the most heavily exercised implementation available — at the cost of
exponentiating a matrix four times the size and discarding three quarters of it.

Accurate to the same order as [`ScaledSquaring`](@ref), and about twice as expensive in the
``\mathfrak{A}`` call itself though much less than that once the ``N\times{}N`` assembly is counted.
Its value is that it introduces no new numerics: it is the reference the other algorithms are tested
against in `test/retractions/exponential_accuracy.jl`.

!!! warning "CPU only"
    `Base.exp` on a dense matrix needs LAPACK, so this does not run on a GPU backend. Use
    [`ScaledSquaring`](@ref) there.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct AugmentedPade <: AbstractExponentialAlgorithm end

@doc raw"""
    ProjectedSkew <: AbstractExponentialAlgorithm

Exponentiate the lift in a basis of its own range, where it is a small *skew-symmetric* matrix.

``\bar{B}`` is skew-symmetric of rank at most ``2n``, so its range and its row space coincide and a
thin QR of ``B'`` gives an ``N\times{}2n`` orthonormal `Q` with ``\bar{B} = QMQ^T`` for
``M = Q^T\bar{B}Q`` skew-symmetric and ``2n\times{}2n``. Then

```math
\exp(\bar{B}) = \mathbb{I} + Q\left(\exp(M) - \mathbb{I}\right)Q^T,
```

and ``\exp(M)`` is formed from an eigendecomposition: ``iM`` is Hermitian for real skew ``M``, so
``M = -iV\Lambda{}V^*`` and ``\exp(M) = \Re\left(V e^{-i\Lambda} V^*\right)``, which is orthogonal by
construction rather than by cancellation.

This is the only algorithm whose `check` does not degrade with the size of the lift: it stays between
`2.1e-15` and `5.3e-15` from ``\|\bar{B}\| = 5.8`` to ``\|\bar{B}\| = 767``, where the other two drift
from `1e-15` to `7e-14`. Orthogonality is structural here — it comes from the eigenvector matrix, not
from the accuracy of a series. The trade is the largest forward error of the three against
`exp(Matrix(B))`, and a QR plus an eigendecomposition instead of matrix products.

The gap is widest in `Float32`, where the other two are at the mercy of the format: over the same
sweep `check` reaches `2.3e-5` for [`ScaledSquaring`](@ref) and `4.2e-5` for [`AugmentedPade`](@ref)
while this stays at `1.3e-6`. Choose it when staying on the manifold matters more than agreeing with
the exponential to the last bit — a long `Float32` run, for instance, where `check` accumulates over
thousands of steps.

!!! warning "CPU only"
    `qr` and `eigen` on a dense matrix need LAPACK, so this does not run on a GPU backend. Use
    [`ScaledSquaring`](@ref) there.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct ProjectedSkew <: AbstractExponentialAlgorithm end

@doc raw"""
    TaylorSeries <: AbstractExponentialAlgorithm

Sum the series for ``\mathfrak{A}`` directly, without scaling. **This is not a usable retraction.**

It is the behaviour of every version of this package up to 0.2.0, retained only so that the
regression is reproducible from the test suite and so the working algorithms have a baseline to be
compared against. `check(geodesic(B, TaylorSeries()))`, on a random
`StiefelLieAlgHorMatrix(20, 3)` scaled up:

| ``\|\bar{B}\|`` | 0.66 | 5.8 | 17.8 | 36.5 | 78.8 | 160 | 361 | 767 |
|---|---|---|---|---|---|---|---|---|
| `check` | `4.5e-16` | `2.1e-15` | `2.6e-12` | `4.4e-7` | `8.3e10` | `4.2e55` | `1.4e168` | `NaN` |

At ``\|\bar{B}\| = 79`` the "retracted" point is not on the Stiefel manifold in any sense, and by
``767`` the series has overflowed. The series is summed on ``X = (B'')^TB'`` whose norm is
``\approx\|\bar{B}\|^2/4``, and its terms cancel: the partial sum reaches ``2.5\cdot10^{18}`` where
the result is of order one, so stopping when a *term* falls below `eps` leaves a relative error of
``\varepsilon\|\mathfrak{A}(X)\|`` rather than ``\varepsilon``. Making the termination test relative
to the partial sum instead of absolute does not help, and was measured not to change any of the
numbers above — the loss is in the cancellation, not in when the summation stops.

Use [`ScaledSquaring`](@ref), which fixes this and is also the cheaper of the two — 1.6× at
``N = 200``, ``n = 10`` and 4.6× at ``N = 500``, ``n = 50``, because the scaled series converges in a
handful of terms where the unscaled one grinds through hundreds.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct TaylorSeries <: AbstractExponentialAlgorithm end
