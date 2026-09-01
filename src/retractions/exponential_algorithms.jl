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
[hochbruck2010exponential; §2.1](@cite), whose closed form ``(\exp(X) - \mathbb{I})X^{-1}`` requires an
invertible ``X`` where the series above does not. The correspondence is needed only to read
[skaflestad2009scaling](@cite), and nothing else here uses it.

Four subtypes evaluate that ``2n\times{}2n`` function, while [`ProjectedSkew`](@ref) bypasses it and
exponentiates the lift in a basis of its range. They compute the same exponential map — so the
one-parameter subgroup property [`Geodesic`](@ref) relies on holds for all of them — but differ in
their approximation kernel, recovery strategy, numerical behaviour, cost, and backend requirements.
On a random `StiefelLieAlgHorMatrix(20, 3)` scaled to ``\|\bar{B}\| = 361``, against the cost of one
retraction at ``N = 200``, ``n = 10``:

| | `check` | forward error | cost | backend |
|---|---|---|---|---|
| [`ScaledSquaring`](@ref) | `3.6e-14` | `2.0e-14` | `0.087 ms` | matrix products and reductions |
| [`NativePade`](@ref) | `3.9e-14` | `2.1e-14` | `0.091 ms` | matrix products and reductions |
| [`AugmentedPade`](@ref) | `3.0e-14` | `1.9e-14` | `0.120 ms` | CPU (dense LAPACK) |
| [`ProjectedSkew`](@ref) | `5.1e-15` | `3.1e-14` | `0.130 ms` | CPU (dense LAPACK) |
| [`TaylorSeries`](@ref) | `1.4e168` | — | `0.149 ms` | matrix products and reductions |

"Forward error" is the relative distance to `exp(Matrix(B))`; both it and `check` come from the same
lift, so the columns are comparable row by row. The `cost` column is one whole retraction,[^timings]
most of which is the ``N\times{}N`` assembly they share; the ``\mathfrak{A}`` call on its own, which is
what actually separates the three algorithms that evaluate it, is `0.021 ms` and `201 KiB` for
[`ScaledSquaring`](@ref), `0.037 ms` and `330 KiB` for [`NativePade`](@ref), and `0.053 ms` and
`114 KiB` for [`AugmentedPade`](@ref). The `backend` column says what the algorithm *needs*: whether
one of the first three runs on a given accelerator depends on that backend's support for matrix
products and reductions.

[^timings]: `minimum` of 50 repetitions on a single BLAS thread of an Apple M-series laptop. Every
    timing in this file comes from one run of `julia --project=. scripts/retraction_accuracy.jl`, which
    reproduces the whole set on your own hardware. They are illustrative: the ratios between rows are
    the informative part, and the only sound way to use the values is to remeasure on the target
    machine. The allocation figures, by contrast, are exact and machine-independent.

[`ScaledSquaring`](@ref) is the default because it is the cheapest algorithm with no dense-LAPACK
dependency. [`NativePade`](@ref) is the independent direct cross-check, [`ProjectedSkew`](@ref) if
`check` matters more than the last digit of the exponential, and [`AugmentedPade`](@ref) the
dense-CPU reference that delegates its numerics to `Base.exp`. [`TaylorSeries`](@ref) is the
pre-0.2.0 behaviour and is retained only so the regression is reproducible; it is not a usable
retraction.

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

The series for ``\mathfrak{A}`` converges for every argument but is only *accurate* for a small one:
at ``\|X\| \gg 1`` its terms can become enormous before cancelling to a result of moderate size.
Halving the argument prevents those large intermediates. The recovery steps below are exact
identities in exact arithmetic, so scaling changes where Taylor is evaluated without changing the
matrix function being computed.

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
values evaluate the series at a larger argument. The default is `0.5`, and it needs no tuning: at
``\|\bar{B}\| \approx 155`` every ``θ \in [0.125, 4]`` — a 32-fold range — gives a `check` between
`9.9e-15` and `5.0e-14` and a forward error between `6.4e-15` and `8.3e-15`, and neither column is
monotone in `θ`. Nothing in that measurement singles out `0.5`; no value in the range does
appreciably better. Unlike [`NativePade`](@ref)'s, this threshold is a preference and not a ceiling,
because the inner series is summed until its terms vanish rather than for a fixed number of steps.

It is also the cheapest algorithm here: the isolated ``\mathfrak{A}`` call at ``N = 200``, ``n = 10``
is `0.021 ms` and `201 KiB` against [`NativePade`](@ref)'s `0.037 ms` and `330 KiB` and
[`AugmentedPade`](@ref)'s `0.053 ms` and `114 KiB`, and it is `1.7×` faster than
[`TaylorSeries`](@ref) at that size and `4.6×` at ``N = 500``, ``n = 50``, because the scaled series
converges in a handful of terms where the unscaled one grinds through hundreds. Its `check` does
drift upwards with the size of the lift, from `5e-16` to about `8e-14` over the documented sweep,
because its orthogonality is an arithmetic outcome rather than a structural property; only
[`ProjectedSkew`](@ref) avoids that.

The implementation uses matrix products and reductions and avoids scalar indexing in package code.
The identities come from [`GeometricOptimizers.unit_matrix`](@ref) rather than from `Base.one`, whose
diagonal write is a scalar-indexing hazard, and the norm from
[`GeometricOptimizers.opnorm₁`](@ref) rather than from `LinearAlgebra.opnorm(X, 1)`, whose
`opnorm1` is a double loop over `X[i, j]`. Accelerator execution depends on the backend's support for
those matrix operations.

!!! note "The argument is worse-conditioned than the lift"
    ``X``'s lower-left block is ``\frac{1}{4}A^2 - B^TB``, so ``\|X\| \approx \|\bar{B}\|^2/4`` while
    its spectral radius is only ``\approx\|\bar{B}\|`` — the eigenvalues of ``X`` are the nonzero
    (purely imaginary) eigenvalues of ``\bar{B}``. The factorisation is strongly non-normal, which is
    why the unscaled series does worse here than it would on ``\bar{B}`` itself, and why `s` grows
    like ``2\log_2\|\bar{B}\|`` rather than ``\log_2\|\bar{B}\|``.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
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

Scaling controls the large-argument cancellation that makes [`TaylorSeries`](@ref) unreliable. At
the resulting small argument there is still a choice of approximation kernel, and that choice is what
this type makes. A Taylor polynomial retains one series coefficient per degree; a Padé approximant uses
a numerator and denominator whose quotient matches more coefficients at comparable polynomial degree.

For a scalar function ``f``, the ``[m/n]`` Padé approximant is the rational function ``P_m/Q_n``
chosen so that

```math
Q_n(z)f(z)-P_m(z)=O(z^{m+n+1}),
\qquad Q_n(0)=1.
```

Unlike a degree-``m`` Taylor polynomial, division by ``Q_n`` represents infinitely many powers of
``z``. For ``e^z`` the coefficients are classical, and nothing is fitted numerically. Write
``P_m=\sum_{k=0}^m a_kz^k`` and ``Q_n=\sum_{j=0}^n b_jz^j`` with ``b_0=1``; the ``z^k`` coefficient of
``Q_ne^z`` is then the convolution ``\sum_j b_j/(k-j)!``. Requiring it to vanish for
``k=m+1,\ldots,m+n`` is a system of ``n`` equations that fixes the denominator, after which the same
convolution reads the numerator off for ``k\leq m`` without constraining anything further. The
solution is

```math
a_k=\frac{(m+n-k)!}{(m+n)!}\binom{m}{k},
\qquad
b_k=(-1)^k\frac{(m+n-k)!}{(m+n)!}\binom{n}{k},
```

both halves of which follow from
``\sum_{j=0}^n(-1)^j\binom{n}{j}\binom{m+n-j}{d}=\binom{m}{d-n}`` at ``d=m+n-k``: the right-hand side
vanishes on ``k\geq m+1`` and equals ``\binom{m}{k}`` on ``k\leq m``. The two formulas are the same
expression with ``m`` and ``n`` interchanged and a sign, which is the reflection ``e^{-z}=1/e^z``.

Specializing to ``m=7`` and ``n=6`` gives the standard ``[7/6]`` exponential approximant
``\exp(z)=P^{\exp}_7(z)/Q^{\exp}_6(z)+O(z^{14})``, from which this implementation uses

```math
\mathfrak{A}(z)=\frac{\exp(z)-1}{z}
\approx\frac{P^{\exp}_7(z)-Q^{\exp}_6(z)}{zQ^{\exp}_6(z)}
=\frac{p_6(z)}{q_6(z)},
```

where ``p_6=(P^{\exp}_7-Q^{\exp}_6)/z`` and ``q_6=Q^{\exp}_6``. The constant terms of
``P^{\exp}_7`` and ``Q^{\exp}_6`` are both one, so their difference is divisible by ``z``.
Consequently ``p_6`` and ``q_6`` both have degree six, and their quotient agrees with the series of
``\mathfrak{A}`` through order 12; a degree-6 Taylor polynomial agrees only through order 6. The first
term missed is ``z^{13}/149597947699200``, about ``8\cdot10^{-19}`` at the scaled argument
``|z|\leq1/2`` evaluated here.

The scalar variable ``z`` is used only to determine the coefficients. For a matrix ``Y``, replace
each ``z^k`` by ``Y^k`` and the scalar constant by ``I``. Scalar division then becomes the matrix
equation ``q_6(Y)R=p_6(Y)``, whose solution is ``R=q_6(Y)^{-1}p_6(Y)`` when ``q_6(Y)`` is invertible;
it is not an entrywise quotient. Because both polynomials contain powers of the same ``Y``, they
commute, and the power-series match now holds through the ``Y^{12}`` term.

The algorithm chooses ``s`` such that ``Y=X/2^s`` satisfies ``\|Y\|_1\leq θ`` and evaluates

```math
p_6(Y)={}
I+\frac{Y}{26}+\frac{5Y^2}{156}+\frac{Y^3}{858}
+\frac{Y^4}{5720}+\frac{Y^5}{205920}+\frac{Y^6}{8648640},
```

```math
q_6(Y)={}
I-\frac{6Y}{13}+\frac{5Y^2}{52}-\frac{5Y^3}{429}
+\frac{Y^4}{1144}-\frac{Y^5}{25740}+\frac{Y^6}{1235520}.
```

The denominator is applied without a dense solve. Newton--Schulz is Newton's method applied to the
matrix equation ``Z^{-1}-q_6(Y)=0`` [schulz1933iterative](@cite). Since
``D(Z^{-1})[H]=-Z^{-1}HZ^{-1}``, its correction is
``H=Z-Zq_6(Y)Z``, and therefore, starting from ``Z_0=I``,

```math
Z_{j+1}=Z_j(2I-q_6(Y)Z_j),
\qquad
I-q_6(Y)Z_{j+1}=(I-q_6(Y)Z_j)^2.
```

The update uses only matrix multiplication; it requires an initial guess whose inverse residual has
norm below one. Here scaling makes ``Z_0=I`` sufficient. The implementation computes
``Z_1=2I-q_6(Y)`` and four further updates. Because
``\|Y\|_1\leq1/2`` implies ``\|I-q_6(Y)\|_1<0.257``, the final residual is bounded by
``0.257^{32}<1.3\cdot10^{-19}``. It then sets
``W=Z_5p_6(Y)/2^s`` and applies ``W\mapsto2W+WXW`` exactly ``s`` times. This is the same low-rank
modified-squaring recovery as [`ScaledSquaring`](@ref); no matrix larger than the input is formed.

The degree, threshold, and fixed five Newton--Schulz refinements are paired deliberately, and unlike
[`ScaledSquaring`](@ref)'s, `0.5` here is a **ceiling** and not a preference. The refinement count is
fixed, so past ``\theta \approx 1`` the inverse it computes stops being one, and it fails *silently*
— nothing in the result says so. Worst relative error against [`AugmentedPade`](@ref) over 400 random
``6\times6`` arguments of one-norm exactly ``\theta``, as orders of magnitude across several seeds:

| ``\theta`` | 1/2 | 1 | 3/2 | 2 | 3 |
|---|---|---|---|---|---|
| relative error | ``10^{-16}`` | ``10^{-16}``–``10^{-14}`` | ``10^{-10}`` | ``10^{-5}``–``10^{-3}`` | ``10^{4}``–``10^{6}`` |

Only the pattern is reproducible, not the digits: these are maxima over a random draw, and the two
rightmost columns move by orders of magnitude between seeds. Where the transition happens does not.
One such draw is recomputed at build time in
[What a large ``\theta`` costs `NativePade`](@ref native-pade-large-theta).

The constructor therefore requires ``0 < \theta \leq 1/2``, where [`ScaledSquaring`](@ref) accepts any
positive value: the two thresholds are not interchangeable. Lowering this one is safe and merely adds
modified-squaring steps, which is why it is a parameter at all.

The Padé coefficients and Newton--Schulz iteration are standard
[higham2005scaling, higham2008functions, schulz1933iterative](@cite). What is assembled here is the pairing: a
``\mathfrak{A}`` approximant derived from the exponential approximant, a portable solve-free inverse,
and low-rank modified squaring.

!!! note "The error criterion, and what it is not"
    ``\theta = 1/2`` is **not** taken from a backward-error table. The ``\theta_m`` of
    [higham2005scaling, almohy2010new](@cite) are derived for ``\exp`` rather than for
    ``\mathfrak{A}``, and they bound a backward error in ``\|X\|`` — which is the least informative
    norm available here, since ``X``'s lower-left block gives ``\|X\| \approx \|\bar{B}\|^2/4``
    against a spectral radius of only ``\approx\|\bar{B}\|`` (see the note under
    [`ScaledSquaring`](@ref)). What justifies the threshold is narrower, and is stated as such: the
    Newton--Schulz residual bound above, plus the measured forward error over the norm sweep and over
    the 400 random arguments tabulated. A backward-error criterion for ``\mathfrak{A}`` on a strongly
    non-normal argument is not settled here.

Like [`ScaledSquaring`](@ref), this uses package-defined identities and norms, reductions, and matrix
products, and avoids scalar indexing in package code. Accelerator execution depends on the backend's
support for those operations, which is what makes it the independent cross-check available where
[`AugmentedPade`](@ref) — still the dense-CPU reference — is not.

It is that cross-check rather than the default, and three measurements say why. Its fixed rational
evaluation is `0.037 ms` against `0.021 ms` for [`ScaledSquaring`](@ref) on the isolated
``\mathfrak{A}`` call at ``N = 200``, ``n = 10``. It allocates `330 KiB` there against `201 KiB` —
`1.6×`, and an allocation count is the figure least likely to stay a constant factor on a backend
where allocating can cost a synchronisation. And in `Float32` its `check` is the worst of the three at
the top of the norm sweep, `1.0e-4` against `4.0e-5` and `4.1e-5`; the forward error is *not* —
`1.2e-5` against `1.1e-5` and `9.3e-6` — so what degrades is the orthogonality of the retracted point,
not the agreement with the exponential.

See [`AbstractExponentialAlgorithm`](@ref) for the comparison.
"""
struct NativePade{T <: Real} <: AbstractExponentialAlgorithm
    θ::T

    # The upper bound is not decoration. `𝔄(X, ::NativePade)` runs a *fixed* five Newton--Schulz
    # steps, and their residual `(𝕀 - q₆)³²` is below round-off only while `θ` is small: measured
    # worst relative error is `6e-16` at `θ = 1`, `1.2e-10` at `θ = 3/2`, `1.1e-5` at `θ = 2` and
    # `169` at `θ = 3`, with nothing raised. `ScaledSquaring` takes any positive `θ` because it sums
    # its series until the terms vanish; this one does a fixed amount of work, so it has to refuse.
    # The docstring above has the table and the norm bound behind `1/2`.
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
exponentiating a matrix four times the size and discarding three quarters of it. This is the package
algorithm corresponding directly to the conventional Padé description of scaling and squaring.

Its value is that it introduces no package-specific approximation: it is the reference the other
algorithms are measured against, both in the test suite and in every accuracy table on the
[Exponential Algorithms](@ref) page, and its accuracy is the same order as [`ScaledSquaring`](@ref)'s
— `3.0e-14` `check` and `1.9e-14` forward error at ``\|\bar{B}\| = 361``. It is also, unexpectedly, the
lightest allocator of the three algorithms that evaluate ``\mathfrak{A}``, at `114 KiB` against
`201 KiB` and `330 KiB`, because `Base.exp` reuses buffers where the two native algorithms build a
fresh ``2n\times{}2n`` temporary per operation.

Its disadvantages are substantial for normal use: it exponentiates a matrix four times the size needed
by the direct methods and discards three quarters of the result, which costs about `2.5×`
[`ScaledSquaring`](@ref)'s ``\mathfrak{A}`` call — `0.053 ms` against `0.021 ms`, though much less
than that once the ``N\times{}N`` assembly is counted — and it requires dense LAPACK.

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

This is the only algorithm whose `check` does not degrade with the size of the lift: it stays between
`2.2e-15` and `5.1e-15` from ``\|\bar{B}\| = 5.8`` to ``\|\bar{B}\| = 767``, where the other four
drift from `5e-16` to `8e-14`. Orthogonality is structural here — it comes from the eigenvector
matrix, not from the accuracy of a series — and the gap is widest in `Float32`, where the others climb
into the ``10^{-5}``s while this stays between `1.2e-6` and `3.1e-6` from one end of the sweep to the
other.

The trade is the forward error against `exp(Matrix(B))`, which is the largest of the four at all but
the very largest lifts and up to about `4.3×` [`ScaledSquaring`](@ref)'s, and a QR plus an
eigendecomposition instead of matrix products, costing `1.1×`–`1.6×` one whole retraction over the
sizes measured. Choose it when staying on the manifold matters more than agreeing with the exponential
to the last bit — a long `Float32` run, for instance, where `check` accumulates over thousands of
steps.

!!! warning "CPU only"
    `qr` and `eigen` on a dense matrix need LAPACK. [`ScaledSquaring`](@ref) and
    [`NativePade`](@ref) avoid that dependency, subject to backend support for their matrix
    operations.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct ProjectedSkew <: AbstractExponentialAlgorithm end

@doc raw"""
    TaylorSeries <: AbstractExponentialAlgorithm

Sum the series for ``\mathfrak{A}`` directly, without scaling. **This is not a usable retraction.**

It is the behaviour of every version of this package up to 0.2.0, retained only so that the regression
is reproducible from the test suite and so the working algorithms have a baseline to be compared
against. The series is summed on ``X = (B'')^TB'``, formed from the factors
[`lift_factors`](@ref) returns, and that reduced matrix is strongly non-normal: its norm is
``\approx\|\bar{B}\|^2/4`` where its spectral radius is only ``\approx\|\bar{B}\|``. On such an
argument the terms cancel — at ``\|\bar{B}\| \approx 79`` the intermediate partial sums exceed the
result by some twenty orders of magnitude, reaching ``5\cdot10^{20}`` where the answer is of order one,
and the summation takes 168 terms to reach `eps` where the scaled series takes a handful — so stopping
when a *term* falls below `eps` leaves a relative error of
``\varepsilon\|\mathfrak{A}(X)\|`` rather than ``\varepsilon``.
`check(geodesic(B, TaylorSeries()))`, on a random `StiefelLieAlgHorMatrix(20, 3)` scaled up:

| ``\|\bar{B}\|`` | 0.66 | 5.8 | 17.8 | 36.5 | 78.8 | 160 | 361 | 767 |
|---|---|---|---|---|---|---|---|---|
| `check` | `4.5e-16` | `2.1e-15` | `2.6e-12` | `4.4e-7` | `8.3e10` | `4.2e55` | `1.4e168` | `NaN` |

At ``\|\bar{B}\| = 79`` the "retracted" point is not on the Stiefel manifold in any sense, and by
``767`` the series has overflowed. The failure is *silent*: nothing reports that the result is off the
manifold. [`check`](@ref) is what detects it, and no optimizer in this package calls it during a run —
a gap rather than a decision, and one this table is the strongest argument for closing
([#76](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/76)). Making the termination test
relative to the partial sum instead of absolute was measured to change none of the numbers above — the
loss is in the cancellation, not in when the summation stops.

Use [`ScaledSquaring`](@ref), which fixes this and is also the cheaper of the two — `1.7×` at
``N = 200``, ``n = 10`` and `4.6×` at ``N = 500``, ``n = 50``, because the scaled series converges in
a handful of terms where the unscaled one grinds through hundreds.

See [`AbstractExponentialAlgorithm`](@ref) for the alternatives.
"""
struct TaylorSeries <: AbstractExponentialAlgorithm end
