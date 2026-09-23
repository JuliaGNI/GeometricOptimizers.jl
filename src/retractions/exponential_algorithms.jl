@doc raw"""
    AbstractExponentialAlgorithm

Supertype of the algorithms that [`Geodesic`](@ref) can use to evaluate the matrix exponential.

A horizontal lift factors as ``\bar{B} = B'(B'')^T`` into two ``N\times{}2n`` matrices. Since
``(B'(B'')^T)^k = B'X^{k-1}(B'')^T`` for ``X=(B'')^TB'``, substituting into the exponential series
gives

```math
\exp(B'(B'')^T) = \mathbb{I} + B'\,\mathfrak{A}((B'')^TB')\,(B'')^T,
\qquad
\mathfrak{A}(X) = \sum_{n=1}^\infty \frac{X^{n-1}}{n!},
```

Computing ``\mathfrak{A}(X)`` is therefore the central numerical task: it preserves the low-rank
factorization and reduces the matrix function from ``N\times{}N`` to ``2n\times{}2n``.
``\mathfrak{A}`` is the notation of this implementation and of [brantner2023generalizing](@cite); the
exponential-integrator literature writes the same function ``\varphi_1``
[hochbruck2010exponential; §2.1](@cite).

Four subtypes evaluate that ``2n\times{}2n`` function, while [`ProjectedSkew`](@ref) bypasses it and
exponentiates the lift in a basis of its range. They compute the same exponential map — so the
one-parameter subgroup property [`Geodesic`](@ref) relies on holds for all of them — but differ in
their approximation kernel, recovery strategy, numerical behaviour, cost, and backend requirements.

[`ScaledSquaring`](@ref) is the default, and the cheapest algorithm with no dense-LAPACK dependency.
Reach for another only if you need what it gives you: [`NativePade`](@ref) an independent direct
calculation, [`ProjectedSkew`](@ref) orthogonality that does not degrade with the size of the lift,
[`AugmentedPade`](@ref) a dense-CPU reference that delegates its numerics to `Base.exp`.
[`TaylorSeries`](@ref) is the pre-0.2.0 behaviour and is retained only so the regression is
reproducible; it is not a usable retraction.

The [Exponential Algorithms](@ref) page derives all five, measures them against each other, and says
which to choose in [Choosing one](@ref). Its tables are recomputed on every documentation build, so
they, and not a figure quoted in a docstring, are what to trust.

A new one has to supply `𝔄(X::AbstractMatrix, ::NewAlgorithm)`; `geodesic` and everything above it
then follow. An algorithm that does not go through ``\mathfrak{A}`` at all — [`ProjectedSkew`](@ref)
is the one such — supplies `geodesic(::AbstractLieAlgHorMatrix, ::NewAlgorithm)` instead.

See [`GeometricOptimizers.𝔄`](@ref) for the implementations.
"""
abstract type AbstractExponentialAlgorithm end

@doc raw"""
    ScaledSquaring(θ = 0.5) <: AbstractExponentialAlgorithm

Evaluate ``\mathfrak{A}`` with a Taylor kernel and low-rank modified squaring, and the default.

This is not the Padé kernel used by the conventional dense matrix-exponential algorithm. Its
improvement over [`TaylorSeries`](@ref) is that it first scales the argument into a regime where the
same Taylor evaluator is accurate, then recovers the original argument with a recurrence specialized
to the low-rank factorization [skaflestad2009scaling](@cite).

The series for ``\mathfrak{A}`` converges for every argument but is only *accurate* for a small one,
so the argument is halved ``s`` times until ``\|X\|_1/2^s \leq θ``, the series is summed there, and
each halving is undone by one application of ``W \mapsto 2W + WXW``. That recurrence is the modified
squaring of [skaflestad2009scaling](@cite); it stays at ``2n\times{}2n`` because the low-rank form is
closed under squaring, so nothing is ever squared at ``N\times{}N``. The recovery steps are exact
identities, so scaling moves where Taylor is evaluated without changing the function computed.

`θ` is the norm threshold the kernel sees. Smaller values use more scaling steps; larger values
evaluate the series at a larger argument. The default `0.5` needs no tuning: across the 32-fold range
``θ \in [0.125, 4]`` neither the orthogonality of the retracted point nor the error against `exp`
moves by as much as a factor of six, and neither moves monotonically. Unlike [`NativePade`](@ref)'s,
this threshold is a preference and not a ceiling, because the inner series is summed until its terms
vanish rather than for a fixed number of steps.

This is the cheapest of the five, and it needs neither dense LAPACK nor scalar indexing in package
code — whether it runs on a given accelerator depends on that backend's support for matrix products
and reductions. What it gives up is that its orthogonality is an arithmetic outcome rather than a
structural property, so `check` drifts upwards with the size of the lift; only [`ProjectedSkew`](@ref)
avoids that.

!!! note "The argument is worse-conditioned than the lift"
    ``\|X\|_2`` grows like ``\|\bar{B}\|_2^2`` while ``\rho(X) = \|\bar{B}\|_2`` exactly, so ``X`` is
    far from normal. That is why the unscaled series does worse here than it would on ``\bar{B}``
    itself, and why `s` grows like ``2\log_2\|\bar{B}\|_2`` rather than ``\log_2\|\bar{B}\|_2``.
    [Why the reduced argument is the hard case](@ref) derives and measures both.

[2. Scaling and modified squaring](@ref) gives the algorithm step by step, with the identity behind
the recurrence and the measurements behind every claim above. See
[`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct ScaledSquaring{T <: Real} <: AbstractExponentialAlgorithm
    θ::T

    function ScaledSquaring(θ::T = 0.5) where {T <: Real}
        @assert θ > zero(T) "the scaling threshold has to be positive, got $(θ)"
        new{T}(θ)
    end
end

@doc raw"""
    NativePade(θ = 0.5) <: AbstractExponentialAlgorithm

Evaluate ``\mathfrak{A}`` with a native degree-6 diagonal Padé approximant.

**This scales and squares too.** The name distinguishes the *kernel* and not the algorithm's
relationship to scaling: `NativePade` and [`ScaledSquaring`](@ref) choose `s` by the same rule and undo
it with the same ``s`` applications of ``W \mapsto 2W + WXW``, and differ only in what they evaluate at
the scaled argument. The two are not alternatives.

Scaling controls the large-argument cancellation that makes [`TaylorSeries`](@ref) unreliable. At the
resulting small argument there is still a choice of approximation kernel, and that choice is what this
type makes. Where a degree-6 Taylor polynomial matches the series of ``\mathfrak{A}`` through order 6,
the ``[6/6]`` rational function ``p_6/q_6`` derived from the classical ``[7/6]`` approximant of
``\exp`` matches it through order 12 at the same polynomial degree. The denominator is applied by
five Newton--Schulz steps [schulz1933iterative](@cite) rather than by a dense solve, which is what
keeps the algorithm free of LAPACK — matrix products and reductions only, so whether it runs on a
given accelerator depends on that backend's support for those. That is what makes it the independent
cross-check available where [`AugmentedPade`](@ref), the dense-CPU reference, is not.

`θ` is a **ceiling** here, not a preference as it is for [`ScaledSquaring`](@ref). The Newton--Schulz
count is fixed at five, and their residual is below round-off only while the argument is small: the
bound ``\|Y\|_1 \leq 1/2`` gives ``\|I - q_6(Y)\|_1 < 0.257`` and hence a final residual below
``1.3\cdot10^{-19}``. Past ``θ \approx 1`` the inverse it computes stops being one, and it fails
*silently* — nothing in the result says so. Worst relative error against [`AugmentedPade`](@ref) over
400 random ``6\times6`` arguments of one-norm exactly ``θ``, as orders of magnitude across several
seeds:

| ``\theta`` | 1/2 | 1 | 3/2 | 2 | 3 |
|---|---|---|---|---|---|
| relative error | ``10^{-16}`` | ``10^{-16}``–``10^{-14}`` | ``10^{-10}`` | ``10^{-5}``–``10^{-3}`` | ``10^{4}``–``10^{6}`` |

Only the pattern is reproducible, not the digits: these are maxima over a random draw, and the two
rightmost columns move by orders of magnitude between seeds. Where the transition happens does not.
The constructor therefore requires ``0 < θ \leq 1/2``, where [`ScaledSquaring`](@ref) accepts any
positive value: the two thresholds are not interchangeable. Lowering this one is safe and merely adds
modified-squaring steps, which is why it is a parameter at all.

!!! note "The error criterion, and what it is not"
    ``θ = 1/2`` is **not** taken from a backward-error table. The ``\theta_m`` of
    [higham2005scaling, almohy2010new](@cite) are derived for ``\exp`` rather than for
    ``\mathfrak{A}``, and they bound a backward error in ``\|X\|`` — the least informative norm
    available here, since ``\|X\|`` is quadratic in ``\|\bar{B}\|_2`` against a spectral radius of
    exactly ``\|\bar{B}\|_2`` (see the note under [`ScaledSquaring`](@ref)). What justifies this
    threshold is narrower: the Newton--Schulz residual bound above, and the measured forward error. A
    backward-error criterion for ``\mathfrak{A}`` on an argument that far from normal is not settled
    here.

[3. Padé approximation](@ref) derives the coefficients, the Newton--Schulz iteration and the residual
bound, gives the algorithm step by step, and recomputes the table above at build time in
[What a large ``\theta`` costs `NativePade`](@ref native-pade-large-theta). See
[`AbstractExponentialAlgorithm`](@ref) for the comparison.
"""
struct NativePade{T <: Real} <: AbstractExponentialAlgorithm
    θ::T

    # The upper bound is not decoration. `𝔄(X, ::NativePade)` runs a *fixed* five Newton--Schulz
    # steps, and their residual `(𝕀 - q₆)³²` is below round-off only while `θ` is small: over 400
    # random 6×6 arguments of one-norm `θ`, the worst relative error is around `1e-16` at `θ = 1`,
    # `1e-10` at `θ = 3/2`, `1e-4` at `θ = 2` and `1e5` at `θ = 3`, with nothing raised.
    # `ScaledSquaring` takes any positive `θ` because it sums its series until the terms vanish; this
    # one does a fixed amount of work, so it has to refuse. The docstring above tabulates the same
    # measurement and derives the norm bound behind `1/2`.
    function NativePade(θ::T = 0.5) where {T <: Real}
        @assert zero(T) < θ ≤ 1 // 2 "the scaling threshold has to be in (0, 1/2], got $(θ)"
        new{T}(θ)
    end
end

@doc raw"""
    AugmentedPade <: AbstractExponentialAlgorithm

Evaluate ``\mathfrak{A}`` as a block of a larger *ordinary* exponential.

This is an independent dense-CPU reference for testing the direct implementations, not the normal
choice for a retraction.

```math
\exp\begin{pmatrix} X & \mathbb{I} \\ \mathbb{O} & \mathbb{O} \end{pmatrix}
= \begin{pmatrix} \exp(X) & \mathfrak{A}(X) \\ \mathbb{O} & \mathbb{I} \end{pmatrix}
```

so one call to `Base.exp` on a ``4n\times{}4n`` matrix returns ``\mathfrak{A}(X)`` in its upper-right
block. That hands the numerics to Julia's own dense matrix exponential, which uses a Padé-based
scaling-and-squaring algorithm [higham2005scaling, almohy2010new](@cite), at the cost of
exponentiating a matrix four times the size and discarding three quarters of it.

Its value is that it introduces no package-specific approximation, which is what makes it the
reference the other algorithms are measured against, both in the test suite and in every accuracy
table on the [Exponential Algorithms](@ref) page. Its accuracy is the same order as
[`ScaledSquaring`](@ref)'s; what it costs is the work it throws away.

!!! warning "CPU only"
    `Base.exp` on a dense matrix needs LAPACK. [`ScaledSquaring`](@ref) and [`NativePade`](@ref)
    avoid that dependency, subject to backend support for their matrix operations.

See [4. `AugmentedPade`](@ref) for the measurements and
[`AbstractExponentialAlgorithm`](@ref) for the alternatives.
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

This is the only algorithm whose `check` does not degrade with the size of the lift, because its
orthogonality is structural — it comes from the eigenvector matrix, not from the accuracy of a series.
The gap is widest in `Float32`, where the other four climb into the ``10^{-5}``s while this one stays
in the ``10^{-6}``s from one end of the norm sweep to the other.

The trade is the forward error against `exp(Matrix(B))`, the largest of the four at all but the very
largest lifts, and a QR plus an eigendecomposition instead of matrix products. Choose it when staying
on the manifold matters more than agreeing with the exponential to the last bit — a long `Float32`
run, for instance, where `check` accumulates over thousands of steps.

!!! warning "Needs the backend's `qr` and `eigen`"
    Every step runs on the lift's own backend, so the backend has to supply a `qr` and an `eigen`
    of a `Hermitian` matrix for its dense arrays. LAPACK does on the host, and CUDA.jl provides
    both through cuSOLVER, which this package does not test. Metal and JLArrays supply neither,
    and there the call raises inside `qr`.
    [`ScaledSquaring`](@ref) and [`NativePade`](@ref) need matrix products and a solve only.

See [5. `ProjectedSkew`](@ref) for the measurements and
[`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct ProjectedSkew <: AbstractExponentialAlgorithm end

@doc raw"""
    TaylorSeries <: AbstractExponentialAlgorithm

Sum the series for ``\mathfrak{A}`` directly, without scaling. **This is not a usable retraction.**

It is the behaviour of every version of this package up to 0.2.0, retained only so that the regression
is reproducible from the test suite and so the working algorithms have a baseline to be compared
against. The series is summed on ``X = (B'')^TB'``, and that reduced matrix is far from normal: its
norm is quadratic in ``\|\bar{B}\|_2`` where its spectral radius is exactly ``\|\bar{B}\|_2``. On such
an argument the terms cancel, so stopping when a *term* falls below `eps` leaves an error of size
``\varepsilon\max_m\|S_m\|`` rather than ``\varepsilon\|\mathfrak{A}(X)\|``, where ``S_m`` are the
partial sums. By ``\|\bar{B}\| \approx 79`` the partial sums exceed the result by twenty orders of
magnitude and the "retracted" point is not on the Stiefel manifold in any sense; by ``\|\bar{B}\|
\approx 770`` the summation has overflowed.

The failure is *silent*: nothing reports that the result is off the manifold. [`check`](@ref) is what
detects it, and no optimizer in this package calls it during a run — a gap rather than a decision
([#76](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/76)). Making the termination test
relative to the partial sum instead of absolute changes none of it: the loss is in the cancellation,
not in when the summation stops.

Use [`ScaledSquaring`](@ref), which fixes this and is also the cheaper of the two, because the scaled
series converges in a handful of terms where the unscaled one grinds through hundreds.

[1. Direct Taylor series](@ref) derives the failure and tabulates it over the whole norm sweep. See
[`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct TaylorSeries <: AbstractExponentialAlgorithm end
