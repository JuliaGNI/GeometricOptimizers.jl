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

Three subtypes evaluate that ``2n\times{}2n`` function, while [`ProjectedSkew`](@ref) bypasses it and
exponentiates the lift in a basis of its range. They differ in approximation kernel, recovery
strategy, numerical behaviour, cost, and backend support:

- [`TaylorSeries`](@ref) is an unscaled Taylor baseline and is unreliable for large lifts.
- [`ScaledSquaring`](@ref) combines a Taylor kernel with low-rank modified squaring and is the
  portable default.
- [`AugmentedPade`](@ref) obtains ``\mathfrak{A}`` from a larger dense exponential and serves as an
  independent Padé-based reference.
- [`ProjectedSkew`](@ref) works directly with the skew-symmetric lift and usually gives the smallest
  orthogonality residual, but requires dense factorizations.

A new one has to supply `𝔄(X::AbstractMatrix, ::NewAlgorithm)`; `geodesic` and everything above it
then follow. An algorithm that does not go through ``\mathfrak{A}`` at all — [`ProjectedSkew`](@ref)
is the one such — supplies `geodesic(::AbstractLieAlgHorMatrix, ::NewAlgorithm)` instead.

See [`GeometricOptimizers.𝔄`](@ref) for the implementations.
"""
abstract type AbstractExponentialAlgorithm end

@doc raw"""
    ScaledSquaring(θ = 0.5) <: AbstractExponentialAlgorithm

Evaluate ``\mathfrak{A}`` with a Taylor kernel and low-rank modified squaring. This is the default
algorithm.

This is not the Padé kernel used by the conventional dense matrix-exponential algorithm. It applies
the scaling idea to the same Taylor evaluator as [`TaylorSeries`](@ref), then recovers the original
argument with a recurrence specialized to the low-rank factorization [skaflestad2009scaling](@cite).

The series for ``\mathfrak{A}`` converges for every argument, but direct floating-point summation can
lose accuracy when large intermediate terms cancel. This algorithm scales the argument before
evaluating the series and then recovers the original value by modified squaring.

The squaring is done on the ``2n\times{}2n`` factor rather than on the assembled ``N\times{}N``
matrix, which is possible because the low-rank form is closed under squaring:

```math
\left(\mathbb{I} + B'W(B'')^T\right)^2 = \mathbb{I} + B'\left(2W + WXW\right)(B'')^T,
\qquad X = (B'')^TB',
```

so one squaring of the exponential is one application of ``W \mapsto 2W + WXW``. With ``s`` chosen so
that ``\|X\|_1/2^s \leq θ``, all recovery operations remain on ``2n\times{}2n`` matrices.

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

`θ` is the norm threshold for the Taylor kernel. Smaller values use more scaling steps; larger values
evaluate the series at a larger argument. The default is `0.5`.

The implementation uses matrix products, reductions, and scalar-free array operations, so it works
on the package's GPU backends. [`GeometricOptimizers.opnorm₁`](@ref) is used instead of
`LinearAlgebra.opnorm(X, 1)` to avoid scalar indexing.

!!! note "The reduced argument can be strongly non-normal"
    ``X``'s lower-left block is ``\frac{1}{4}A^2 - B^TB``, so ``\|X\| \approx \|\bar{B}\|^2/4`` while
    its spectral radius is only ``\approx\|\bar{B}\|``. Choosing `s` from the norm can therefore
    overscale the problem, in the sense discussed by Al-Mohy and Higham [almohy2010new](@cite).

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
block. This delegates the approximation and scaling strategy to Julia's dense matrix exponential,
which uses Padé-based scaling and squaring [higham2005scaling, almohy2010new](@cite).

This algorithm is useful as an independent reference for the direct ``\mathfrak{A}``
implementations. Its main costs are the larger augmented matrix and its dependence on dense LAPACK.

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

The construction tends to preserve orthogonality well because it exponentiates a skew-symmetric
matrix in an orthonormal basis. The trade-offs are the QR factorization and eigendecomposition, a
somewhat larger forward error in the documented benchmark sweep, and dependence on dense LAPACK.

!!! warning "CPU only"
    `qr` and `eigen` on a dense matrix need LAPACK, so this does not run on a GPU backend. Use
    [`ScaledSquaring`](@ref) there.

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
