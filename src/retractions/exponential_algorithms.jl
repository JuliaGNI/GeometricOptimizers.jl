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
| [`ScaledSquaring`](@ref) | `3.6e-14` | `2.1e-14` | `0.087 ms` | any |
| [`NativePade`](@ref) | `3.9e-14` | `2.1e-14` | `0.091 ms` | any |
| [`AugmentedPade`](@ref) | `1.5e-14` | `2.0e-14` | `0.120 ms` | CPU (dense LAPACK) |
| [`ProjectedSkew`](@ref) | `4.8e-15` | `3.0e-14` | `0.130 ms` | CPU (dense LAPACK) |
| [`TaylorSeries`](@ref) | `1.4e168` | — | `0.149 ms` | any |

"Forward error" is the relative distance to `exp(Matrix(B))`; both it and `check` come from the same
lift, so the columns are comparable row by row. The `cost` column is one whole retraction, most of
which is the ``N\times{}N`` assembly they share; the ``\mathfrak{A}`` call on its own, which is what
actually separates the three algorithms that evaluate it, is `0.021 ms` and `201 KiB` for
[`ScaledSquaring`](@ref), `0.037 ms` and `330 KiB` for [`NativePade`](@ref), and `0.053 ms` and
`114 KiB` for [`AugmentedPade`](@ref) — note that the delegating one, which discards three quarters
of its work, is nonetheless the *lightest* allocator, because `Base.exp` reuses buffers where these
two build a fresh ``2n\times{}2n`` temporary per operation.

[`ScaledSquaring`](@ref) remains the default because it is the cheapest portable algorithm.
[`NativePade`](@ref) is the independent portable cross-check, [`ProjectedSkew`](@ref) if `check`
matters more than the last digit of the exponential, and [`AugmentedPade`](@ref) as the CPU reference
that delegates its numerics to `Base.exp`. [`TaylorSeries`](@ref) is the pre-0.2.0 behaviour and is
retained only so the regression is reproducible; it is not a usable retraction.

A new one has to supply `𝔄(X::AbstractMatrix, ::NewAlgorithm)`; `geodesic` and everything above it
then follow. An algorithm that does not go through ``\mathfrak{A}`` at all — [`ProjectedSkew`](@ref)
is the one such — supplies `geodesic(::AbstractLieAlgHorMatrix, ::NewAlgorithm)` instead.

See [`GeometricOptimizers.𝔄`](@ref) for the implementations.
"""
abstract type AbstractExponentialAlgorithm end

@doc raw"""
    ScaledSquaring(θ = 0.5) <: AbstractExponentialAlgorithm

Evaluate ``\mathfrak{A}`` by scaling and squaring, and the default.

This is the standard scaling-and-squaring framework for the matrix exponential
[higham2005scaling, higham2008functions, almohy2010new](@cite), specialised to the low-rank
factorisation used here.

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
that ``\|X\|_1/2^s \leq θ``, the whole algorithm is `s` small-matrix updates on top of a series that
now converges in a handful of terms — cheaper than summing the unscaled series, not just more
accurate.

# Algorithm

Given ``X = (B'')^TB'`` and the threshold ``θ``:

1. Set ``s = \max(0, \lceil\log_2(\|X\|_1/θ)\rceil)`` and ``α = 2^s``.
2. Sum the Taylor series at the scaled argument to obtain
   ``W = \mathfrak{A}(X/α)/α``.
3. Repeat ``W \leftarrow 2W + WXW`` exactly ``s`` times.
4. Return ``W``. It now equals ``\mathfrak{A}(X)``, and therefore
   ``\mathbb{I} + B'W(B'')^T = \exp(B'(B'')^T)``.

The factor ``1/α`` in the initial ``W`` scales ``B'`` implicitly; the recurrence then restores one
factor of two per iteration without ever assembling the ``N\times{}N`` exponential.

`θ` is the norm below which the series is summed. It barely matters: at ``\|\bar{B}\| = 155`` every
``θ \in [0.125, 4]`` — a 32-fold range — gives a `check` between `9.9e-15` and `5.0e-14` and a
forward error between `6.4e-15` and `8.2e-15`, and neither column is monotone in `θ`. Nothing in the
measurement singles out `0.5`; it needs no tuning because no value in that range does better.

Like [`TaylorSeries`](@ref) and [`NativePade`](@ref), this uses nothing but matrix products and
norms, so it runs unchanged on a `KernelAbstractions` backend — including the identities it needs,
which come from [`GeometricOptimizers.unit_matrix`](@ref) and not from `Base.one`. It remains the
default because it is the cheaper of the two usable portable algorithms: the isolated
``\mathfrak{A}`` call at ``N = 200``, ``n = 10`` is `0.021 ms` and `201 KiB` against
[`NativePade`](@ref)'s `0.037 ms` and `330 KiB`. The norm is taken by [`GeometricOptimizers.opnorm₁`](@ref) rather than by
`LinearAlgebra.opnorm(X, 1)`, which is a scalar-indexing loop and would give up exactly the property
this paragraph claims.

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
    NativePade(θ = 0.5) <: AbstractExponentialAlgorithm

Evaluate ``\mathfrak{A} = \varphi_1`` with a native degree-6 diagonal Padé approximant.

The argument is first divided by ``2^s`` until its induced one-norm is at most `θ`. On that small
argument the method evaluates

```math
\mathfrak{A}(X) \approx q_6(X)^{-1}p_6(X)
```

directly at ``2n\times{}2n``. The denominator inverse is not a dense solve: five effective
Newton--Schulz refinements start from the identity and use only matrix products. Scaling is undone
with the same low-rank squaring recursion as [`ScaledSquaring`](@ref), so no matrix larger than the
input is formed.

The degree and the threshold are paired deliberately, and unlike [`ScaledSquaring`](@ref)'s, `0.5`
here is a **ceiling** and not a preference. The Newton--Schulz count is fixed at five, so the inverse
residual is exactly ``(\mathbb{I} - q_6)^{32}``, and
``\|\mathbb{I} - q_6\|_1 \leq \sum_{k\geq1}|q_k|\theta^k = 0.256`` at ``\theta = 1/2``, putting it
at `2e-19` — below round-off, where the degree-6 Padé error already sits. Nothing bounds it above
that. Worst relative error against [`AugmentedPade`](@ref) over 400 random ``6\times6`` arguments of
one-norm exactly ``\theta``:

| ``\theta`` | 1/2 | 1 | 3/2 | 2 | 3 |
|---|---|---|---|---|---|
| relative error | `5.8e-16` | `6.4e-16` | `1.2e-10` | `1.1e-5` | `169` |

It fails *silently*: a fixed number of refinements simply stops converging, and nothing in the
result says so. [`ScaledSquaring`](@ref) has no such limit — it sums its series until the terms
vanish, which is why its own docstring can sweep ``\theta`` over `[0.125, 4]` and find nothing to
choose between — so the two thresholds are not interchangeable and this constructor rejects
``\theta > 1/2``. Lowering it is safe and merely adds squarings, which is why it is a parameter at
all.

The coefficients are not invented here. ``q_6`` is the denominator of the ``[7/6]`` Padé approximant
of ``\exp``, whose closed form is standard [higham2005scaling, higham2008functions](@cite), and
``p_6`` is ``(N - D)/x`` for that approximant's numerator ``N`` — which divides exactly, both having
constant term one, and which inherits the ``O(x^{13})`` order. The Newton--Schulz iteration for the
inverse is likewise classical [higham2008functions](@cite). What is assembled here is the *pairing*:
a portable solve-free inverse in place of the dense LU a rational approximant normally needs.

!!! note "The error criterion, and what it is not"
    ``\theta = 1/2`` is **not** taken from a backward-error table. The ``\theta_m`` of
    [higham2005scaling, almohy2010new](@cite) are derived for ``\exp`` rather than for
    ``\varphi_1``, and they bound a backward error in ``\|X\|`` — which is the least informative
    norm available here, since ``X``'s lower-left block gives ``\|X\| \approx \|\bar{B}\|^2/4``
    against a spectral radius of only ``\approx\|\bar{B}\|`` (see the note under
    [`ScaledSquaring`](@ref)). What justifies the threshold is narrower, and is stated as such: the
    Newton--Schulz residual bound above, plus the measured forward error over the norm sweep and over
    the 400 random arguments tabulated. A backward-error criterion for ``\varphi_1`` on a strongly
    non-normal argument is not settled here.

Like [`ScaledSquaring`](@ref), this uses [`GeometricOptimizers.opnorm₁`](@ref),
[`GeometricOptimizers.unit_matrix`](@ref), reductions and matrix products only, and therefore runs
without scalar indexing on a `KernelAbstractions` backend. It is an independent portable cross-check
rather than the default, and three measurements say why. Its fixed rational evaluation is `0.037 ms`
against `0.021 ms` for [`ScaledSquaring`](@ref) on the isolated ``\mathfrak{A}`` call at
``N = 200``, ``n = 10``. It allocates `330 KiB` there against `201 KiB` — `1.6×`, and exactly the
figure that does *not* stay a mere constant factor on a backend where an allocation costs a
synchronisation. And in `Float32` its `check` is the worst of the three at the top of the norm
sweep, `1.0e-4` against `4.0e-5` and `3.4e-5`; the forward error is *not* — `1.2e-5` against `1.1e-5`
and `9.6e-6` — so what degrades is the orthogonality of the retracted point, not the agreement with
the exponential. [`AugmentedPade`](@ref) remains the CPU reference that delegates all numerics to
`Base.exp`.

See [`AbstractExponentialAlgorithm`](@ref) for the comparison.
"""
struct NativePade{T<:Real} <: AbstractExponentialAlgorithm
    θ::T

    # The upper bound is not decoration. `𝔄(X, ::NativePade)` runs a *fixed* five Newton--Schulz
    # steps, and their residual `(𝕀 - q₆)³²` is below round-off only while `θ` is small: measured
    # worst relative error is `6e-16` at `θ = 1`, `1.2e-10` at `θ = 3/2`, `1.1e-5` at `θ = 2` and
    # `169` at `θ = 3`, with nothing raised. `ScaledSquaring` takes any positive `θ` because it sums
    # its series until the terms vanish; this one does a fixed amount of work, so it has to refuse.
    # The docstring above has the table and the norm bound behind `1/2`.
    function NativePade(θ::T = 0.5) where {T<:Real}
        @assert zero(T) < θ ≤ 1 // 2 "the scaling threshold has to be in (0, 1/2], got $(θ)"
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
    [`ScaledSquaring`](@ref) there, with [`NativePade`](@ref) as an independent cross-check.

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
`2.1e-15` and `5.3e-15` from ``\|\bar{B}\| = 5.8`` to ``\|\bar{B}\| = 767``, where the other three drift
from `1e-15` to `7e-14`. Orthogonality is structural here — it comes from the eigenvector matrix, not
from the accuracy of a series. The trade is the forward error against `exp(Matrix(B))`, which is the
largest of the four at all but the very largest lifts and up to 4.4× [`ScaledSquaring`](@ref)'s, and
a QR plus an eigendecomposition instead of matrix products, costing `1.1×`–`1.6×` one whole
retraction over the sizes measured.

The gap is widest in `Float32`, where the other three are at the mercy of the format: over the same
sweep `check` climbs to `4.0e-5` for [`ScaledSquaring`](@ref), `3.4e-5` for [`AugmentedPade`](@ref)
and `1.0e-4` for [`NativePade`](@ref), which of them is worst depending on the lift below the top of
that sweep, while this stays between `1.0e-6` and `3.1e-6` from one end to the other. Choose it when staying on the manifold matters more
than agreeing with the exponential to the last bit — a long `Float32` run, for instance, where
`check` accumulates over thousands of steps.

!!! warning "CPU only"
    `qr` and `eigen` on a dense matrix need LAPACK, so this does not run on a GPU backend. Use
    [`ScaledSquaring`](@ref) there, with [`NativePade`](@ref) as an independent cross-check.

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

Use [`ScaledSquaring`](@ref), which fixes this and is also the cheaper of the two — 1.7× at
``N = 200``, ``n = 10`` and 4.6× at ``N = 500``, ``n = 50``, because the scaled series converges in a
handful of terms where the unscaled one grinds through hundreds.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct TaylorSeries <: AbstractExponentialAlgorithm end
