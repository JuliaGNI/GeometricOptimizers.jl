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

Four subtypes evaluate that ``2n\times{}2n`` function, while [`ProjectedSkew`](@ref) bypasses it and
exponentiates the lift in a basis of its range. They compute the same exponential map but differ in
their approximation kernel, recovery strategy, numerical behaviour, cost, and backend requirements.
[`ScaledSquaring`](@ref) is the default; [`NativePade`](@ref) provides a direct rational alternative;
[`AugmentedPade`](@ref) delegates to Julia's dense matrix exponential; and [`TaylorSeries`](@ref) is
retained as a regression baseline rather than for normal use.

A new one has to supply `𝔄(X::AbstractMatrix, ::NewAlgorithm)`; `geodesic` and everything above it
then follow. An algorithm that does not go through ``\mathfrak{A}`` at all — [`ProjectedSkew`](@ref)
is the one such — supplies `geodesic(::AbstractLieAlgHorMatrix, ::NewAlgorithm)` instead.

See [`GeometricOptimizers.𝔄`](@ref) for the implementations.
"""
abstract type AbstractExponentialAlgorithm end

@doc raw"""
    ScaledSquaring(θ = 0.5) <: AbstractExponentialAlgorithm

Evaluate ``\mathfrak{A}`` with a Taylor kernel and low-rank modified squaring, and the default.

This is not the Padé kernel used by the conventional dense matrix-exponential algorithm. It applies
the scaling idea to the same Taylor evaluator as [`TaylorSeries`](@ref), then recovers the original
argument with a recurrence specialized to the low-rank factorization [skaflestad2009scaling](@cite).

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
that ``\|X\|_1/2^s \leq θ``, the whole algorithm is `s` small-matrix updates on top of a series at a
small argument.

# Algorithm

Given ``L = B'(B'')^T``, ``X = (B'')^TB'``, and the threshold ``θ``:

1. Set ``s = \max(0, \lceil\log_2(\|X\|_1/θ)\rceil)`` and ``α = 2^s``.
2. Sum the Taylor series at the scaled argument to obtain
   ``W_s = \mathfrak{A}(X/α)/α``, so
   ``\exp(L/α) = \mathbb{I} + B'W_s(B'')^T``.
3. For ``k=s,s-1,\ldots,1``, set ``W_{k-1} = 2W_k + W_kXW_k``. This squares the represented
   exponential and uses the original ``X``, not ``X/α``.
4. Return ``W_0 = \mathfrak{A}(X)``, so
   ``\mathbb{I} + B'W_0(B'')^T = \exp(B'(B'')^T)``.

The factor ``1/α`` follows from the scaled-exponential identity; the recurrence restores one factor
of two per iteration without assembling a dense ``N\times{}N`` exponential, square, or solve.

`θ` is the norm threshold for the Taylor kernel. Smaller values use more scaling steps; larger
values evaluate the series at a larger argument. The default is `0.5`.

The implementation uses matrix products and reductions and avoids scalar indexing in package code.
The identities come from [`GeometricOptimizers.unit_matrix`](@ref), and the norm from
[`GeometricOptimizers.opnorm₁`](@ref). Accelerator execution depends on the backend's support for
those matrix operations.

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

The degree, threshold, and fixed five Newton--Schulz refinements are paired deliberately. The
constructor requires ``0 < \theta \leq 1/2`` so the inverse iteration remains in its validated
convergence regime. Lowering the threshold is safe and adds modified-squaring steps.

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

Like [`ScaledSquaring`](@ref), this uses package-defined identities and norms, reductions, and matrix
products, and avoids scalar indexing in package code. Accelerator execution depends on the backend's
support for those operations. [`AugmentedPade`](@ref) remains the dense-CPU reference.

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
block. That hands the numerics to Julia's own dense matrix exponential, which uses a Padé-based
scaling-and-squaring algorithm [higham2005scaling, almohy2010new](@cite), at the cost of
exponentiating a matrix four times the size and discarding three quarters of it. This is the package
algorithm corresponding directly to the conventional Padé description of scaling and squaring.

Its value is that it introduces no package-specific approximation: it is the dense-CPU reference
against which the direct ``\mathfrak{A}`` implementations are tested.

!!! warning "CPU only"
    `Base.exp` on a dense matrix needs LAPACK. [`ScaledSquaring`](@ref) and [`NativePade`](@ref)
    avoid that dependency, subject to backend support for their matrix operations.

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

The construction tends to preserve orthogonality well because it exponentiates a skew-symmetric
matrix in an orthonormal basis. The trade-offs are the QR factorization and eigendecomposition, a
somewhat larger forward error in the documented benchmark sweep, and dependence on dense LAPACK.

!!! warning "CPU only"
    `qr` and `eigen` on a dense matrix need LAPACK. [`ScaledSquaring`](@ref) and
    [`NativePade`](@ref) avoid that dependency, subject to backend support for their matrix
    operations.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct ProjectedSkew <: AbstractExponentialAlgorithm end

@doc raw"""
    TaylorSeries <: AbstractExponentialAlgorithm

Sum the series for ``\mathfrak{A}`` directly, without scaling.

This is the pre-0.2.0 implementation and is retained as a regression baseline. For the non-normal
reduced matrices produced by [`lift_factors`](@ref), large intermediate terms can cancel and destroy
accuracy even though the series converges mathematically. Use [`ScaledSquaring`](@ref) for normal
operation.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct TaylorSeries <: AbstractExponentialAlgorithm end
