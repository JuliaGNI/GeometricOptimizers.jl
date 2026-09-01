```@meta
CurrentModule = GeometricOptimizers
```

# Exponential Algorithms

[`Geodesic`](@ref) is the exponential map, and evaluating a matrix exponential is a numerical
problem rather than a formula one writes down. Since 0.2.0 the retraction therefore carries an
*algorithm* saying how that exponential is evaluated, and there are five of them. They compute the
same map. They differ in accuracy at a large step, in cost, and in which backends they run on.

This page is that numerical problem: why the exponential needs an algorithm at all, the two
independent choices any such algorithm makes, what each of the five does, what they measure, and
which one to reach for. The retractions themselves — what they are, and how [`Cayley`](@ref) and
[`Geodesic`](@ref) differ as maps — are on the [Retractions](@ref) page, and the optimizer that uses
them on [Optimization on Homogeneous Spaces](@ref).

## The exponential needs an algorithm

### The notation this page uses

Everything below is stated in the notation of
[Both retractions factor the lift](@ref) on the [Retractions](@ref) page, which is where it is
derived. Briefly:

- ``\bar{B} \in \mathbb{R}^{N\times{}N}`` is the **horizontal lift**, a skew-symmetric matrix of rank
  at most ``2n``. It is the direction an [`OptimizerMethod`](@ref) produces and the argument the
  retraction is applied to; concretely it is a [`StiefelLieAlgHorMatrix`](@ref) or a
  [`GrassmannLieAlgHorMatrix`](@ref).
- ``B'`` and ``B''`` are the two ``N\times{}2n`` **thin factors** of that lift, so that
  ``\bar{B} = B'(B'')^T``. [`lift_factors`](@ref) returns them; for the Stiefel manifold they are
  the two matrices in
  ```math
  \begin{bmatrix} A & -B^T \\ B & \mathbb{O} \end{bmatrix}
  = \underbrace{\begin{bmatrix} \tfrac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix}}_{B'}
    \underbrace{\begin{bmatrix} \mathbb{I} & \mathbb{O} \\ \tfrac{1}{2}A & -B^T \end{bmatrix}}_{(B'')^T},
  ```
  and for a [`GrassmannLieAlgHorMatrix`](@ref) the same expression with ``A \equiv \mathbb{O}``.
- ``X := (B'')^TB' \in \mathbb{R}^{2n\times{}2n}`` is the **reduced matrix**. It is small even when
  ``N`` is large, and it is the argument every matrix function on this page is evaluated on. It is
  also considerably worse-behaved than ``\bar{B}``, which is the subject of
  [1. Direct Taylor series](@ref) below.

### The reduction

The geodesic requires ``\exp(\bar{B})``, but forming and exponentiating the full ``N\times{}N`` lift
would throw away its low-rank structure. Writing ``\bar{B}=B'(B'')^T`` and
``X=(B'')^TB'`` gives, for every ``k\geq 1``,

```math
\left(B'(B'')^T\right)^k = B'X^{k-1}(B'')^T.
```

Substitution into the exponential series therefore gives the exact identity

```math
\exp\left(B'(B'')^T\right) = \mathbb{I} + B'\,\mathfrak{A}(X)\,(B'')^T,
\qquad
\mathfrak{A}(X) = \sum_{k=1}^\infty \frac{X^{k-1}}{k!},
```

so **computing ``\mathfrak{A}(X)`` is the central numerical task**: once it is available, the
``N\times{}N`` exponential is assembled using only the two thin factors, and the only matrix function
evaluated anywhere is ``\mathfrak{A}`` on a ``2n\times{}2n`` argument.[^1] That is what preserves the
cost advantage when ``n\ll N``, and it is why this page is about ``\mathfrak{A}`` throughout rather
than about ``\exp``.

There are two separate algorithmic choices to make in evaluating it:

1. how to approximate ``\mathfrak{A}`` at a **small** argument — the *kernel*; and
2. whether to **scale** a large argument down and recover the original value afterwards — the
   *recovery*.

Taylor and Padé are kernels. Scaling and modified squaring is a framework around a kernel, not a
competing approximation, so the two choices are independent. Classical dense matrix-exponential
routines normally combine a Padé kernel with scaling and squaring
[higham2005scaling, almohy2010new](@cite); Taylor kernels can also be effective when scaling keeps
the argument small [skaflestad2009scaling](@cite).

All five algorithms in this package are:

| Package algorithm | Object evaluated | Kernel | Recovery | Backend |
|---|---|---|---|---|
| [`TaylorSeries`](@ref) | ``\mathfrak{A}(X)`` | Taylor series | none | matrix products and reductions |
| [`ScaledSquaring`](@ref) | ``\mathfrak{A}(X)`` | Taylor series | low-rank modified squaring | matrix products and reductions |
| [`NativePade`](@ref) | ``\mathfrak{A}(X)`` | degree-6 Padé | low-rank modified squaring | matrix products and reductions |
| [`AugmentedPade`](@ref) | ``\mathfrak{A}(X)`` as a block of a ``4n\times{}4n`` exponential | delegated to `Base.exp` | delegated to `Base.exp` | CPU (dense LAPACK) |
| [`ProjectedSkew`](@ref) | ``\exp(M)`` for the ``2n\times{}2n`` skew ``M = Q^T\bar{B}Q`` | eigendecomposition | none | CPU (dense LAPACK) |

Note that the two algorithms a reader is most likely to reach for, [`ScaledSquaring`](@ref) and
[`NativePade`](@ref), differ **only in the kernel column**. Both scale the argument down and both
recover it with the same modified-squaring recurrence; the names are historical and unfortunate in
this respect, since `NativePade` scales and squares exactly as much as `ScaledSquaring` does.

The Backend column says what the algorithm needs, not what it is known to run on: the first three
require only matrix products and reductions, so whether they run on a particular accelerator depends
on that backend's support for those operations, while the last two call into dense LAPACK and are
therefore CPU-only.

[`Geodesic`](@ref) makes the choice visible:

```julia
Geodesic(ScaledSquaring())   # the default, and `Geodesic()`
Geodesic(NativePade())
Geodesic(AugmentedPade())    # the dense-CPU reference
Geodesic(ProjectedSkew())
Geodesic(TaylorSeries())     # the pre-0.2.0 behaviour; not a usable retraction
```

All five return the same exponential map, so the one-parameter subgroup property holds for every one
of them. Two of the five are not ordinary choices, and both are documented here for a reason beyond
being constructible:

- [`TaylorSeries`](@ref) is the **small-argument kernel that [`ScaledSquaring`](@ref) uses**. The
  section on it is not a deprecation notice but the case for scaling: it is what the kernel does when
  the argument is *not* small, and therefore what the framework of the next section exists to
  prevent.
- [`AugmentedPade`](@ref) is the route that hands ``\mathfrak{A}`` to a **native LAPACK exponential**
  via one embedding. That route is available if wanted, and is strongly discouraged: it exponentiates
  a matrix four times the size needed, discards three quarters of the result, and gives up the
  backend portability of the direct algorithms for no gain in accuracy.

[^1]: In the exponential-integrator literature ``\mathfrak{A}`` is written ``\varphi_1``, the first of
      the ``\varphi``-functions ``\varphi_{k+1}(z) = (\varphi_k(z) - 1/k!)/z`` with
      ``\varphi_0 = \exp`` [hochbruck2010exponential; §2.1](@cite), and its closed form is
      ``\varphi_1(X) = (\exp(X) - \mathbb{I})X^{-1}``. That form is only valid at an invertible ``X``,
      while the series above is not, so this page uses ``\mathfrak{A}`` and the series — which is also
      the notation of the implementation and of
      [brantner2023generalizing](@cite). The names of the recovery formulas in
      [skaflestad2009scaling](@cite) refer to the ``\varphi``-functions, and that is the only place the
      correspondence is needed here.

```@setup retractions
using GeometricOptimizers
using GeometricOptimizers: geodesic, cayley, check, ScaledSquaring, NativePade, AugmentedPade, ProjectedSkew, TaylorSeries
using GeometricOptimizers: lift_factors, 𝔄, _native_pade_polynomials, unit_matrix, opnorm₁
using LinearAlgebra: norm
using Markdown
using Printf
import Random

# A `Markdown.MD` rather than a string: Documenter renders it as a table instead of as the code
# block a printed string would give.
function table(header, rows)
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", repeat("---|", length(header)))
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
    Markdown.parse(String(take!(io)))
end

# `TaylorSeries` overflows at the top of the sweep, so both formatters have to survive a `NaN` and
# an `Inf` — a bare `@sprintf` of one is fine, but the guard makes the intent explicit.
sci(x) = isfinite(x) ? (@sprintf "%.2e" x) : string(x)
fixed(x) = isfinite(x) ? (@sprintf "%.2f" x) : string(x)

"""
A sweep of horizontal lifts of increasing norm, all drawn from the same seed — the same eight
`scripts/retraction_accuracy.jl` sweeps, so that every table on this page agrees row for row.
"""
function sweep(T)
    Random.seed!(1234)
    [T(s) * rand(StiefelLieAlgHorMatrix{T}, 20, 3) for s in (0.1, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0, 120.0)]
end

lifts = sweep(Float64)
```

## 1. Direct Taylor series

[`TaylorSeries`](@ref) evaluates ``\mathfrak{A}`` directly from its defining series, without scaling,
and stops when a term falls below `eps`. The series converges for every matrix, but convergence alone
does not guarantee an accurate floating-point sum.

### Why the reduced argument is the hard case

The reduction of the previous section bought a small matrix, and it did so at a price: ``X`` is a
considerably worse argument for a power series than ``\bar{B}`` is. ``X``'s lower-left block is
``\tfrac{1}{4}A^2 - B^TB``, quadratic in the entries of the lift, so its norm is quadratic in
``\|\bar{B}\|``. Its *spectrum*, on the other hand, is not: the nonzero eigenvalues of ``X`` are the
nonzero, purely imaginary eigenvalues of the skew matrix ``\bar{B}``, whose modulus is bounded by
``\|\bar{B}\|``. Writing ``\rho`` for the **spectral radius**, ``\rho(X) := \max_i|\lambda_i(X)|``,

```math
\|X\| \approx \tfrac{1}{4}\|\bar{B}\|^2
\qquad\text{while}\qquad
\rho(X) \approx \|\bar{B}\|.
```

A matrix whose norm is quadratically larger than its spectral radius is **strongly non-normal**, and
that gap is exactly what a truncated power series is bad at. The size of the terms is governed by the
norm, while the size of the answer is governed by the spectrum, so the intermediate terms become
enormous before cancelling down to a result of moderate size. Every digit of that cancellation is a
digit lost:

```math
\text{digits lost} \approx \log_{10}
\frac{\max_m\left\|\sum_{k=1}^{m}X^{k-1}/k!\right\|}{\|\mathfrak{A}(X)\|}.
```

At ``\|\bar{B}\| \approx 79`` the reduced matrix has ``\|X\|_1 \approx 2.7\cdot10^3`` against
``\rho(X) \approx 51``. The partial sums peak at ``5\cdot10^{20}`` where ``\|\mathfrak{A}(X)\|`` is
``0.99``, so that ratio is ``5\cdot10^{20}``: **the summation loses about 21 decimal digits where
`Float64` has 16.** There is nothing left, and the computed ``\mathfrak{A}(X)`` comes out with a
relative error of ``1.5\cdot10^{5}`` — not a lost digit or two, but no correct digits at all. The 168
terms it takes before one falls below `eps` are 168 chances to accumulate that loss, and the
termination test cannot detect it: a *term* below `eps` bounds the truncation error, and what has gone
wrong is the rounding error, which is of size ``\varepsilon\cdot5\cdot10^{20}`` and not
``\varepsilon``.

That the direct series is not a method for the matrix exponential has been known since
[moler1978nineteen](@cite) — restated with twenty-five more years of evidence in
[moler2003nineteen](@cite). What is specific here is the direction of the effect: the factorisation
that made the problem cheap also made its argument *worse* than the matrix one started with, so this
package meets the failure earlier than a dense implementation would.

How badly it goes is worth measuring rather than asserting. `check(geodesic(B, TaylorSeries()))`,
i.e. ``\|Y^TY - \mathbb{I}\|`` of the retracted point, over a random `StiefelLieAlgHorMatrix(20, 3)`
scaled up:

```@example retractions
table(["‖B̄‖", "`check`"],
      [[fixed(norm(Matrix(B))), sci(check(geodesic(B, TaylorSeries())))] for B in lifts])
```

By ``\|\bar{B}\| \approx 79`` the "retracted" point is not on the Stiefel manifold in any sense, and
at the top of the sweep the series has overflowed outright. The [`Cayley`](@ref) and
[`ScaledSquaring`](@ref) columns of the same sweep, for comparison, are in
[Staying on the manifold](@ref) below and stay at round-off throughout.

!!! danger "This is not a usable retraction"
    `TaylorSeries` is retained as a regression baseline for the pre-0.2.0 implementation — the
    behaviour of every release up to 0.2.0, and of the algorithm as published in earlier versions of
    [brantner2023generalizing](@cite). It can silently lose orthogonality for large lifts. Do not
    select it for optimization. Its role in the current package is as the small-argument kernel of
    [`ScaledSquaring`](@ref), where the argument it is handed is guaranteed small.

Two things about the failure are worth recording rather than merely deprecating. First, it is
*silent*: an optimizer using it takes a step, gets a matrix back, and nothing anywhere reports that
the matrix is no longer on the manifold. No optimizer in this package currently verifies that a
retracted point is still on its manifold, which is a gap rather than a decision — see
[#76](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/76). Second, the obvious first fix does
not work: making the termination test relative to the partial sum rather than absolute was measured to
change *none* of the numbers above, at any lift norm, for the reason the displayed ratio gives. The
loss is in the cancellation inside the sum, not in the point at which the summation stops. Making the
argument small is what helps, and that is the next section.

## 2. Scaling and modified squaring

The previous section blames the size of the argument, so the remedy is to evaluate the kernel
somewhere else. Scaling and squaring computes a matrix function at ``X/2^s`` and then reconstructs its
value at ``X`` — for ``\exp`` by repeated squaring, ``\exp(X) = (\exp(X/2^s))^{2^s}``. For
``\mathfrak{A}`` the recovery formula is not squaring but a *modified* squaring
[skaflestad2009scaling](@cite), and it is worth deriving because it is the one step of these
algorithms that has no counterpart in the classical dense routine.

### The doubling identity for ``\mathfrak{A}``

Everything follows from one identity. From ``\exp(2Y) - \mathbb{I} = (\exp(Y) -
\mathbb{I})(\exp(Y) + \mathbb{I})`` and ``\exp(Y) = \mathbb{I} + Y\mathfrak{A}(Y)``,

```math
\mathfrak{A}(2Y)
= \frac{\exp(2Y) - \mathbb{I}}{2Y}
= \mathfrak{A}(Y)\,\frac{\exp(Y) + \mathbb{I}}{2}
= \mathfrak{A}(Y)\,\frac{2\mathbb{I} + Y\mathfrak{A}(Y)}{2},
```

that is

```math
\mathfrak{A}(2Y) = \mathfrak{A}(Y) + \tfrac{1}{2}Y\mathfrak{A}(Y)^2.
```

The middle step divides by ``Y``, but the two ends do not: both sides are power series in ``Y``, so
the identity holds as an identity of power series and therefore at every ``Y``, singular or not. It is
an **exact algebraic identity, not an approximation** — the ``\mathfrak{A}`` case of the modified
squaring of [skaflestad2009scaling](@cite).

Now put

```math
W_k := \frac{\mathfrak{A}(X/2^k)}{2^k},
```

which is the object both algorithms below actually carry: not ``\mathfrak{A}`` of anything, but
``\mathfrak{A}`` at the ``k``-times-halved argument, divided by ``2^k``. That scaling is what makes
the recovery step clean. Applying the doubling identity at ``Y = X/2^k`` gives

```math
W_{k-1}
= \frac{\mathfrak{A}(X/2^{k-1})}{2^{k-1}}
= \frac{1}{2^{k-1}}\left(\mathfrak{A}(Y) + \tfrac{1}{2}Y\mathfrak{A}(Y)^2\right)
= 2W_k + W_kXW_k,
```

where the last equality uses ``Y = X/2^k`` to turn ``Y\mathfrak{A}(Y)^2/2^k`` into ``W_kXW_k``. So one
recovery step is one application of

```math
W \longleftarrow 2W + WXW,
```

at ``2n\times{}2n``, with the **original** ``X`` and not the scaled one. Starting from ``W_s`` and
applying it ``s`` times lands on ``W_0 = \mathfrak{A}(X)``, which is what was wanted.[^2]

### The framework, and the two algorithms in it

Scaling and modified squaring is a *framework*: it says nothing about how ``W_s`` is obtained, only
how to get from ``W_s`` to ``W_0``. Filling in a kernel gives an algorithm, and this package has two:

| | kernel for ``W_s`` | recovery | ``\theta`` |
|---|---|---|---|
| [`ScaledSquaring`](@ref) | the Taylor series of [1. Direct Taylor series](@ref) | ``W\leftarrow2W+WXW``, ``s`` times | a preference, any positive value |
| [`NativePade`](@ref) | the degree-6 Padé kernel of [3. Padé approximation](@ref) | ``W\leftarrow2W+WXW``, ``s`` times | a ceiling, ``\theta\leq1/2`` |

They differ in one column. In particular **[`NativePade`](@ref) scales and squares too**, with the
same rule for ``s`` and the same recurrence; the algorithm names suggest a contrast that does not
exist.

The names are a historical record rather than a taxonomy, and they are worth reading carefully because
of what they do *not* offer. `ScaledSquaring` means scaling with a Taylor kernel and `NativePade` means
scaling with a Padé kernel: the two axes are welded together, so there is no way to ask for the
framework with a kernel of your choosing, and in particular `ScaledSquaring` has no Padé support of any
kind. Separating the two — a kernel argument to one scaling-and-squaring algorithm, with Taylor the
default — is the natural shape for this code and is
[#63](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/63), along with the question of which
kernel is the better choice, which this page does not settle.

What scaling buys the Taylor kernel is exactly what [1. Direct Taylor series](@ref) measured. The
series is summed only where ``\|X/2^s\|_1\leq\theta``. There the terms decrease monotonically from the
first one, so the cancellation ratio of that section is of order one rather than ``5\cdot10^{20}``,
and the sum converges in a handful of terms rather than 168. In this way the framework makes an
otherwise unusable kernel usable.

### The complete `ScaledSquaring` algorithm

Given the reduced matrix ``X = (B'')^TB'`` and the threshold ``\theta``:

1. **Choose the scaling.** Set ``s = \max(0, \lceil\log_2(\|X\|_1/\theta)\rceil)``, so that
   ``\|X/2^s\|_1 \leq \theta``.
2. **Evaluate the kernel.** Sum the Taylor series of [1. Direct Taylor series](@ref) at ``X/2^s`` and
   divide by ``2^s``, giving ``W_s = \mathfrak{A}(X/2^s)/2^s``.
3. **Recover.** Apply ``W \leftarrow 2W + WXW`` exactly ``s`` times, with the original ``X``.
4. **Return** ``W_0 = \mathfrak{A}(X)``.

The division by ``2^s`` in step 2 is part of the definition of ``W_s`` and not an approximation, and
neither is step 3. Every operation is ``2n\times{}2n``; nothing ``N\times{}N`` is squared, inverted or
exponentiated anywhere.[^2]

!!! note "Which errors the recovery contributes, and which it does not"
    Steps 2 and 3 make errors of two different kinds, and only one of them belongs to the recovery.
    The *approximation* (truncation) error is committed entirely in step 2, by the kernel; step 3
    introduces none, because the doubling identity is exact. What step 3 does contribute is
    **rounding** error: each application of ``W \leftarrow 2W + WXW`` roughly doubles whatever error
    it is handed, so an error ``\epsilon`` in ``W_s`` reaches ``W_0`` at about ``2^s\epsilon``. That is
    why the halving count matters beyond its cost, and it is the subject of the note at the end of this
    section. The trade is nonetheless overwhelmingly favourable: the error being amplified is
    round-off at a small argument, not the ``5\cdot10^{20}\varepsilon`` the unscaled sum commits.

### The threshold ``\theta``

`θ` defaults to `0.5`. A smaller value performs more scaling steps; a larger value asks the kernel to
handle a larger argument. That default is **not** taken from the literature: the ``\theta_m`` tables of
[higham2005scaling, almohy2010new](@cite) are derived for ``\exp`` rather than for ``\mathfrak{A}``,
and they bound a backward error in ``\|X\|`` — the least informative norm available here, for the
reason [Why the reduced argument is the hard case](@ref) gives. How `0.5` was in fact arrived at
differs between the two algorithms:

- For [`NativePade`](@ref) it is **forced**. Its five Newton–Schulz steps are only accurate while
  ``\|I - q_6(Y)\|_1 < 1``, and ``\theta \leq 1/2`` is what guarantees that with room to spare. See
  [Applying the denominator without a matrix solve](@ref) for the bound and
  [What a large ``\theta`` costs `NativePade`](@ref) for the measured failures above it.
- For `ScaledSquaring` it is an **empirical choice, and a nearly free one**, because the algorithm
  turns out to be insensitive to it. The measurement is in
  [Sensitivity to the threshold `θ`](@ref) below, recomputed at build time rather than quoted here:
  over the 32-fold range ``\theta \in [0.125, 4]`` both `check` and the forward error move by less
  than a factor of six, and neither is monotone in ``\theta``. Nothing in that table singles out
  `0.5`; it is a reasonable value rather than a tuned one, and it is one lift and one seed rather
  than a general error bound.

**Advantages.** It is the cheapest of the algorithms on the ``\mathfrak{A}`` call itself, and it comes
as close to `exp(Matrix(B))` as [`AugmentedPade`](@ref) does, which is as close as anything here gets;
the two measurements are [What they cost](@ref) and [Agreeing with the exponential](@ref) below. It
needs only matrix products, norms and a kernel-written identity, so it has no dense-LAPACK dependency.
It is the default because it is the cheapest of the algorithms that have none.[^3]

**Disadvantages.** Its orthogonality is the outcome of an arithmetic cancellation rather than a
structural property, so `check` drifts upwards with the size of the lift — by two to three orders of
magnitude over [Staying on the manifold](@ref), and considerably further in [`Float32`](@ref). Only
[`ProjectedSkew`](@ref) avoids that drift. And it takes about twice the squarings it needs, for the
reason in the note below.

!!! note "The halving count is loose"
    `s` is chosen from a **norm**, ``s = \lceil\log_2(\|X\|_1/\theta)\rceil``, because that is the
    quantity available without an eigendecomposition. What actually limits the kernel is the
    *spectrum*: as [Why the reduced argument is the hard case](@ref) sets out,
    ``\|X\|_1 \approx \|\bar{B}\|^2/4`` where ``\rho(X) \approx \|\bar{B}\|``, so this rule takes
    ``s \approx 2\log_2\|\bar{B}\|`` halvings where ``\log_2\|\bar{B}\|`` would suffice — about twice
    as many as necessary. By the note above, those extra steps cost both time and a factor of
    ``2^{s}`` in amplified round-off.

    Choosing a scaling parameter from a norm that overestimates what the spectrum requires is the same
    underlying cause as the *overscaling* that Al-Mohy and Higham identify for the dense exponential
    [almohy2010new; §3](@cite), and their remedy — replacing ``\|X\|`` by estimates of
    ``\|X^k\|^{1/k}``, which approach ``\rho(X)`` — would apply here too. It is not the same
    phenomenon, though: their analysis is a backward-error statement about ``\exp`` on a general
    matrix, whereas the gap here is a structural property of the ``2n\times{}2n`` reduction, present
    at every lift and quantified exactly by ``\|X\| \approx \|\bar{B}\|^2/4`` against
    ``\rho(X) \approx \|\bar{B}\|``.

    The rule is left alone because a tighter one needs the spectral radius, and an eigenvalue
    computation would forfeit exactly the freedom from dense LAPACK that makes this the default
    algorithm; ``\|X^k\|^{1/k}`` estimates would not, and are the obvious thing to try.
    [`NativePade`](@ref) takes ``s`` the same way and inherits all of this.

[^2]: The recurrence has an equivalent reading on the assembled exponential, which is where the name
      "squaring" comes from and which is how the implementation comments put it: the low-rank form is
      closed under squaring,
      ``(\mathbb{I} + B'W(B'')^T)^2 = \mathbb{I} + B'(2W + WXW)(B'')^T``, so squaring the
      ``N\times{}N`` exponential *is* the ``2n\times{}2n`` update ``W \mapsto 2W + WXW``. The
      derivation above avoids ``\exp`` because nothing on this page needs it: ``\mathfrak{A}`` is what
      is computed, and the doubling identity is a statement about ``\mathfrak{A}`` alone.

[^3]: "Only matrix products, norms and a kernel-written identity" is a portability requirement rather
      than a stylistic preference, and it is why two apparently redundant helpers exist. A
      `KernelAbstractions` array cannot serve a scalar index, so the norm is taken by
      [`GeometricOptimizers.opnorm₁`](@ref) as a reduction rather than by
      `LinearAlgebra.opnorm(X, 1)`, whose `LinearAlgebra.opnorm1` is a double loop over `X[i, j]`, and
      identities come from [`GeometricOptimizers.unit_matrix`](@ref) rather than from `Base.one`, whose
      diagonal write is the same hazard one level down. One such call anywhere in the algorithm would
      make it CPU-only, which is the whole distinction between the first three rows of the table above
      and the last two. Whether a given accelerator then runs it still depends on that backend's
      support for the matrix operations that remain.

## 3. Padé approximation

Scaling is settled. The previous section made the argument small and gave an exact recovery for the
original one, and it did so without committing to any particular kernel. What it did not settle is
*which* approximation to evaluate at the small argument. [`ScaledSquaring`](@ref) evaluates a Taylor
series there because that is what it already had; [`NativePade`](@ref) evaluates a rational function
instead, and this section is why that is a different and defensible choice.

!!! note "`NativePade` is the previous section's framework with a different kernel"
    Everything in [2. Scaling and modified squaring](@ref) applies to `NativePade` unchanged: the same
    ``s = \max(0, \lceil\log_2(\|X\|_1/\theta)\rceil)``, the same ``W_s = \mathfrak{A}(X/2^s)/2^s``
    read as a definition, the same ``s`` applications of ``W \leftarrow 2W + WXW``, and the same
    doubling identity behind them. Only steps 2's *contents* differ. The name `NativePade` names the
    kernel, not the algorithm's relationship to scaling; the two are not alternatives, and
    [What scaling buys the Padé kernel](@ref) below measures what happens without it.

A degree-``m`` Taylor polynomial simply keeps the first ``m+1`` terms,

```math
T_m(z)=\sum_{k=0}^m c_kz^k,
\qquad
f(z)-T_m(z)=O(z^{m+1}).
```

Its coefficients are fixed one by one by the series, so matching more terms requires increasing the
polynomial degree. Padé takes a different route: it chooses a numerator *and* a denominator together,
so that their quotient matches more series coefficients without raising the numerator to the same
degree. It is therefore an alternative small-argument kernel, not an alternative to scaling.

To see how this works, first consider a scalar analytic function ``f``. Its ``[m/n]`` Padé
approximant is the rational function

```math
R_{m,n}(z) = \frac{P_m(z)}{Q_n(z)},
\qquad
\deg P_m\leq m,\quad \deg Q_n\leq n,\quad Q_n(0)=1,
```

and is defined by the matching condition

```math
Q_n(z)f(z)-P_m(z)=O\left(z^{m+n+1}\right).
```

Equating powers of ``z`` determines the coefficients of ``P_m`` and ``Q_n``. The denominator is what
makes this different from truncating a Taylor series: when the quotient is expanded again as a power
series, even low-degree ``P_m`` and ``Q_n`` generate infinitely many powers. For example,

```math
\frac{1+z/2}{1-z/2}=1+z+\frac{z^2}{2}+\frac{z^3}{4}+\cdots
```

matches ``e^z`` through degree two, whereas a polynomial with numerator degree one can match only
``1+z``. More generally, a ``[m/n]`` Padé approximant matches through degree ``m+n`` whenever its
matching condition is solvable — for ``e^z`` it always is, as the next section shows. This
is why a rational kernel can capture much more local information than a Taylor polynomial with a
similar polynomial degree [higham2005scaling, higham2008functions](@cite). The scalar variable ``z``
serves only to fix the coefficients; a matrix takes its place once they are known.

### Deriving the coefficients for ``\mathfrak{A}``

Nothing is fitted to ``\mathfrak{A}``, and nothing is fitted numerically. The coefficients come from
the matching condition above, applied to ``f=\exp`` at ``m=7`` and ``n=6``; one subtraction then
converts that approximant into one for ``\mathfrak{A}``. Write its numerator and denominator as
``P^{\exp}_7`` and ``Q^{\exp}_6``:

```math
e^z = \frac{P^{\exp}_7(z)}{Q^{\exp}_6(z)} + O(z^{14}),
\qquad P^{\exp}_7(0)=Q^{\exp}_6(0)=1.
```

Keep ``m`` and ``n`` general for the derivation, since one formula covers every degree:

```math
P_m(z)=\sum_{k=0}^m a_kz^k,
\qquad
Q_n(z)=\sum_{j=0}^n b_jz^j,
\qquad b_0=1.
```

Because ``e^z=\sum_{r\geq0}z^r/r!``, the ``z^k`` coefficient of ``Q_n(z)e^z`` is the **convolution at
degree ``k``**,

```math
c_k := \sum_{j=0}^{n}\frac{b_j}{(k-j)!},
```

reading ``1/(k-j)!`` as zero when ``j>k``. That single quantity is what the rest of the derivation is
about. The matching condition ``Q_ne^z-P_m=O(z^{m+n+1})`` says that the ``z^k`` coefficient of
``Q_ne^z-P_m`` vanishes for every ``k\leq m+n``, and since ``P_m`` has no term above degree ``m``, it
says two different things on the two ranges of ``k``:

```math
a_k=c_k
\quad (k=0,\ldots,m),
\qquad
c_k=0
\quad (k=m+1,\ldots,m+n).
```

Only the right-hand block is a system to be solved: ``n`` equations in the ``n`` unknowns
``b_1,\ldots,b_n``. The left-hand block constrains nothing, since each ``a_k`` occurs in one equation
and nowhere else — with the denominator known it reads the numerator off. The construction is a
statement about the denominator alone. For ``m=7`` and ``n=6`` the ranges are the degrees
``0,\ldots,7`` and ``8,\ldots,13``.

The solution is

```math
a_k=\frac{(m+n-k)!}{(m+n)!}\binom{m}{k},
\qquad
b_k=(-1)^k\frac{(m+n-k)!}{(m+n)!}\binom{n}{k},
```

and one binomial identity settles both blocks at once. Substituting this ``b_j`` into the convolution
at degree ``k`` and multiplying through by ``(m+n)!`` to clear the constant gives, in full,

```math
\begin{aligned}
(m+n)!\,c_k
&= (m+n)!\sum_{j=0}^n\frac{1}{(k-j)!}\cdot(-1)^j\frac{(m+n-j)!}{(m+n)!}\binom{n}{j}\\
&= \sum_{j=0}^n(-1)^j\binom{n}{j}\frac{(m+n-j)!}{(k-j)!}\\
&= d!\sum_{j=0}^n(-1)^j\binom{n}{j}\binom{m+n-j}{d},
\qquad d := m+n-k.
\end{aligned}
```

The last step is the only one that needs a word. With ``d = m+n-k`` fixed, the exponent difference
``(m+n-j) - (k-j) = d`` does not depend on ``j``, so ``(m+n-j)!/(k-j)!`` is always a product of ``d``
consecutive integers,

```math
\frac{(m+n-j)!}{(k-j)!}
= (m+n-j)(m+n-j-1)\cdots(k-j+1)
= d!\binom{m+n-j}{d},
```

which also reproduces the convention above: for ``j>k`` the binomial has upper index ``m+n-j<d`` and
vanishes, matching ``1/(k-j)! = 0``. What remains is the claim

```math
\sum_{j=0}^n(-1)^j\binom{n}{j}\binom{m+n-j}{d}=\binom{m}{d-n}.
```

This is the binomial theorem, applied twice and then read off one degree at a time. Introduce a
bookkeeping variable ``x``, unrelated to the ``z`` of the approximant and used only to carry
coefficients. The alternating sum on the left is a binomial expansion in the *quantity* ``(1+x)``:

```math
\sum_{j=0}^n(-1)^j\binom{n}{j}(1+x)^{m+n-j}
=(1+x)^m\sum_{j=0}^n\binom{n}{j}(1+x)^{n-j}(-1)^j
=(1+x)^m\bigl((1+x)-1\bigr)^n
=x^n(1+x)^m.
```

Now compare the coefficient of ``x^d`` on the two ends. On the left, expanding each
``(1+x)^{m+n-j}`` by the binomial theorem gives ``\sum_j(-1)^j\binom{n}{j}\binom{m+n-j}{d}``; on the
right, ``x^n(1+x)^m`` contributes ``\binom{m}{d-n}``. That proves the claim, and it splits exactly
where the matching condition splits:

- if ``k\geq m+1``, then ``d-n=m-k<0``, and ``x^n(1+x)^m`` has no ``x^d`` term at all, so ``c_k=0`` —
  all ``n`` denominator equations hold;
- if ``k\leq m``, the coefficient is ``\binom{m}{m-k}=\binom{m}{k}``, and restoring the ``d!/(m+n)!``
  gives ``c_k = \frac{(m+n-k)!}{(m+n)!}\binom{m}{k}``, which is the claimed ``a_k``.

The two formulas are one expression with ``m`` and ``n`` interchanged and a sign,
``a_k(m,n)=(-1)^kb_k(n,m)``: the reflection that ``e^{-z}=1/e^z`` induces on the table. They are
classical [higham2005scaling, higham2008functions](@cite); the derivation appears here so the numbers
below are traceable rather than quoted. At ``m=7`` and ``n=6``:

```math
\begin{array}{r|ccc}
k & a_k & b_k & a_k-b_k\\\hline
0 & 1 & 1 & 0\\
1 & \tfrac{7}{13} & -\tfrac{6}{13} & 1\\
2 & \tfrac{7}{52} & \tfrac{5}{52} & \tfrac{1}{26}\\
3 & \tfrac{35}{1716} & -\tfrac{5}{429} & \tfrac{5}{156}\\
4 & \tfrac{7}{3432} & \tfrac{1}{1144} & \tfrac{1}{858}\\
5 & \tfrac{7}{51480} & -\tfrac{1}{25740} & \tfrac{1}{5720}\\
6 & \tfrac{7}{1235520} & \tfrac{1}{1235520} & \tfrac{1}{205920}\\
7 & \tfrac{1}{8648640} & 0 & \tfrac{1}{8648640}
\end{array}
```

The last column is the promised subtraction. Since ``\mathfrak{A}(z)=(e^z-1)/z``,

```math
\begin{aligned}
\mathfrak{A}(z)
&= \frac{e^z-1}{z}\\
&= \frac{P^{\exp}_7(z)-Q^{\exp}_6(z)}{z\,Q^{\exp}_6(z)} + O(z^{13})\\
&= \frac{p_6(z)}{q_6(z)} + O(z^{13}),
\end{aligned}
```

where

```math
p_6(z)=\frac{P^{\exp}_7(z)-Q^{\exp}_6(z)}{z},
\qquad
q_6(z)=Q^{\exp}_6(z).
```

The ``k=0`` entry of that column vanishes, both constant terms being one, so the difference is
divisible by ``z`` and ``p_6`` is a polynomial. Its coefficients are the remaining entries shifted
down one degree, ``(p_6)_k=a_{k+1}-b_{k+1}`` for ``k=0,\ldots,6`` with ``b_7=0``. Numerator and
denominator both have degree six, so this is a ``[6/6]`` approximant of ``\mathfrak{A}``, and it
agrees with

```math
\mathfrak{A}(z)=1+\frac{z}{2!}+\frac{z^2}{3!}+\cdots
```

through the ``z^{12}`` term, where truncating that series after ``z^6`` agrees only through ``z^6``.
The first term missed is small: in exact arithmetic

```math
q_6(z)\mathfrak{A}(z)-p_6(z)=\frac{z^{13}}{149597947699200}+O(z^{14}),
```

so at the scaled argument [`NativePade`](@ref) evaluates, ``|z|\leq1/2``, that term is about
``8\cdot10^{-19}`` — the same order as the inverse-iteration residual derived below, and far under the
`Float64` rounding of a result of size one. The polynomials the implementation uses are therefore

```math
\begin{aligned}
p_6(z)={}&1+\frac{z}{26}+\frac{5z^2}{156}+\frac{z^3}{858}
          +\frac{z^4}{5720}+\frac{z^5}{205920}+\frac{z^6}{8648640},\\
q_6(z)={}&1-\frac{6z}{13}+\frac{5z^2}{52}-\frac{5z^3}{429}
          +\frac{z^4}{1144}-\frac{z^5}{25740}+\frac{z^6}{1235520}.
\end{aligned}
```

### From scalar Padé to matrix Padé

The scalar calculation above has done its only job: it has determined the coefficients. Matrix
functions defined by power series use those same scalar coefficients, with each scalar power ``z^k``
replaced by the matrix power ``Y^k``. Thus ``p_6(z)`` and ``q_6(z)`` become the matrix polynomials
``p_6(Y)`` and ``q_6(Y)`` displayed above with ``z`` replaced by ``Y`` and ``1`` replaced by ``I``.
The matching statement also transfers directly: the expansion of the rational matrix function agrees
with

```math
\mathfrak{A}(Y)=I+\frac{Y}{2!}+\frac{Y^2}{3!}+\cdots
```

through the ``Y^{12}`` term.

The quotient is not taken entry by entry. For scalars, ``r=p/q`` means ``qr=p``. For matrices, the
corresponding operation is therefore to solve

```math
q_6(Y)R=p_6(Y),
```

or, when ``q_6(Y)`` is invertible, to write

```math
R=q_6(Y)^{-1}p_6(Y)\approx\mathfrak{A}(Y).
```

Because ``p_6(Y)`` and ``q_6(Y)`` are polynomials in the same matrix, they commute; the order shown is
the one used by the implementation. Notice that no inverse of ``Y`` occurs. Unlike the expression
``(e^Y-I)Y^{-1}``, the Padé formula remains directly usable when ``Y`` is singular. In
[`_native_pade_polynomials`](@ref), the code first forms ``Y^2`` and ``Y^4`` and groups the displayed
polynomials around those shared powers. This evaluates both degree-6 polynomials with matrix products
instead of constructing the powers one by one.

### Applying the denominator without a matrix solve

A conventional evaluation of this rational matrix function would solve

```math
q_6(Y)W=p_6(Y).
```

!!! info "Newton--Schulz here is *not* the standard practice — LU is"
    Pairing a Padé denominator with a Newton–Schulz inverse is a choice specific to this package, and
    it should not be read as the textbook recipe. The Padé coefficients and the Newton–Schulz
    iteration are each entirely standard on their own — the coefficients from
    [higham2005scaling](@cite) and [higham2008functions; §10.3](@cite), and the iteration from
    [schulz1933iterative](@cite), analysed as a matrix-inverse iteration in
    [higham2008functions; §7.2](@cite) — but the *conventional* way to apply a Padé denominator, and
    what `Base.exp` and every dense library routine do, is a pivoted LU followed by triangular
    solves. One would not normally form an explicit inverse at all.

    `NativePade` departs from that for one reason: Newton–Schulz needs only matrix multiplication and
    addition, so it keeps the algorithm independent of backend-specific factorization support, which
    is the entire point of having a direct implementation next to [`AugmentedPade`](@ref). It is a
    portability choice and not a faster solver — five matrix products may well cost more than an LU on
    a CPU, and the measurement that would settle that has not been made. It is
    [#67](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/67).

    What makes it *viable*, as opposed to merely portable, is that scaling supplies the
    initial-residual bound derived below, which fixes both the iteration count and the resulting
    error in advance. Without scaling there would be no such bound and no fixed count.

To derive the iteration [schulz1933iterative](@cite), analysed in [higham2008functions; §7.2](@cite),
temporarily write
``A=q_6(Y)`` and seek a matrix ``Z`` satisfying ``Z^{-1}-A=0``. The Fréchet derivative of matrix
inversion is

```math
D(Z^{-1})[H]=-Z^{-1}HZ^{-1}.
```

One Newton correction therefore solves

```math
-Z_j^{-1}H_jZ_j^{-1}=A-Z_j^{-1},
```

which gives ``H_j=Z_j-Z_jAZ_j`` and hence

```math
Z_0=I,
\qquad
Z_{j+1}=Z_j+H_j=Z_j\left(2I-AZ_j\right).
```

This rearrangement is Newton's method using no factorization or linear solve. It is not safe from an
arbitrary starting point. Writing ``E_j := I - AZ_j`` for the **inverse residual** at step ``j``, the
condition is that the *initial* one be a contraction in some submultiplicative norm:

```math
\|E_0\| = \|I - AZ_0\| < 1.
```

For the present choice ``A=q_6(Y)`` and ``Z_0=I`` that reads ``\|I-q_6(Y)\|<1``, which is exactly what
scaling will deliver below. Since ``AZ_j=I-E_j`` and ``2I-AZ_j=I+E_j``, the update gives

```math
AZ_{j+1}=(I-E_j)(I+E_j)=I-E_j^2,
\qquad
E_{j+1}=E_j^2.
```

Consequently ``E_j=E_0^{2^j}`` in exact arithmetic and
``\|E_j\|\leq\|E_0\|^{2^j}``: once ``\|E_0\|<1``, the iteration converges quadratically, squaring
the residual at every step.[^4] The code writes the first step explicitly as ``Z_1=2I-q_6(Y)`` and
performs four more, giving
``E_5=(I-q_6(Y))^{32}``. Scaling ensures ``\|Y\|_1\leq 1/2``; from the displayed coefficients,

```math
\|I-q_6(Y)\|_1
\leq \sum_{k=1}^6 |(q_6)_k|\,\|Y\|_1^k
\leq \sum_{k=1}^6 \frac{|(q_6)_k|}{2^k}
= 0.2563204283419127\ldots
<0.257,
```

so ``\|E_5\|_1 \leq 0.257^{32}<1.3\cdot10^{-19}``. This explains both the fixed five Newton--Schulz
steps and the constructor restriction ``0<\theta\leq 1/2``: together they make the solve-free inverse
accurate to approximately `Float64` precision using matrix multiplication alone. It is also the same
order as the ``8\cdot10^{-19}`` kernel truncation error derived above, so neither half of the accuracy
argument is the weaker one.

[^4]: The condition ``\|E_0\| < 1`` and the language of contraction invite a comparison with the
      Banach fixed-point theorem, but the two are not the same statement and the identification is not
      one the literature makes. Banach would give ``\|E_{j+1}\| \leq L\|E_j\|`` for a fixed
      ``L<1``, i.e. *linear* convergence at a rate the map supplies. What holds here is the exact
      identity ``E_{j+1}=E_j^2``, hence ``\|E_{j+1}\| \leq \|E_j\|^2`` — quadratic convergence, with
      the rate improving at every step, which is the ordinary local behaviour of Newton's method and
      strictly stronger than a contraction estimate. The role of ``\|E_0\|<1`` is to identify the
      *basin* of that local convergence, not to supply a contraction constant. See
      [higham2008functions; §7.2](@cite) for the standard analysis.

### The complete `NativePade` algorithm

Given the reduced matrix ``X=(B'')^TB'``:

1. Choose ``s=\max(0,\lceil\log_2(\|X\|_1/\theta)\rceil)`` and set ``\alpha=2^s``.
2. Set ``Y=X/\alpha``, so ``\|Y\|_1\leq\theta\leq1/2``, and evaluate ``p_6(Y)`` and ``q_6(Y)``.
3. Compute ``Z_5\approx q_6(Y)^{-1}`` with the five Newton--Schulz steps above.
4. Form ``W_s=Z_5p_6(Y)/\alpha``. The division by ``\alpha`` is required because

   ```math
   \exp\left(B'(B'')^T/\alpha\right)
   =I+B'\left[\mathfrak{A}(X/\alpha)/\alpha\right](B'')^T.
   ```

5. Apply ``W\leftarrow2W+WXW`` exactly ``s`` times — the low-rank modified-squaring steps of the
   previous section, unchanged and for the same reason. After the last one,
   ``W\approx\mathfrak{A}(X)``.

Padé is therefore the *small-argument kernel* in `NativePade`, not a replacement for scaling and
squaring. Steps 1 and 5 are shared with [`ScaledSquaring`](@ref) verbatim; steps 2 to 4 are the only
difference between the two algorithms. Scaling makes the rational approximation and its solve-free
denominator application reliable; modified squaring transports that small-argument result back to the
original ``X``.

### What scaling buys the Padé kernel

[1. Direct Taylor series](@ref) is what the Taylor kernel does at an argument that is not small. The
same measurement for the Padé kernel, over the same eight lifts, makes the point that the two kernels
are in the same position: neither is usable on its own.

Two unscaled variants are worth separating, because an unscaled Padé kernel fails in two independent
ways. The first evaluates ``q_6(X)^{-1}p_6(X)`` with an exact dense solve, so its error is the
*approximation* error of the ``[6/6]`` rational function alone — the best the kernel could possibly do
without scaling, and not something `NativePade` could run on a GPU backend anyway. The second is what
`NativePade` would actually compute with ``s=0``: the same rational function with the five
Newton–Schulz steps, which lose their residual bound the moment ``\|Y\|_1`` exceeds ``1/2``. Relative
error against [`AugmentedPade`](@ref):

```@example retractions
# The kernel of the previous subsections on its own, with an exact denominator solve — the best an
# unscaled `[6/6]` can do. `\` is a dense LAPACK solve and is used only for this comparison.
function pade_exact_solve(X)
    p, q = _native_pade_polynomials(X, unit_matrix(X))
    q \ p
end

# The same kernel with the denominator applied the way `NativePade` applies it.
function pade_newton_schulz(X)
    𝕀 = unit_matrix(X)
    p, q = _native_pade_polynomials(X, 𝕀)
    q⁻¹ = 2 * 𝕀 - q
    for _ in 1:4
        q⁻¹ = q⁻¹ * (2 * 𝕀 - q * q⁻¹)
    end
    q⁻¹ * p
end

rows = map(lifts) do B
    B′, B′′ = lift_factors(B)
    X = B′′' * B′
    reference = 𝔄(X, AugmentedPade())
    relative(W) = norm(W - reference) / norm(reference)
    [fixed(norm(Matrix(B))), fixed(opnorm₁(X)),
     sci(relative(pade_exact_solve(X))), sci(relative(pade_newton_schulz(X))),
     sci(relative(𝔄(X, NativePade())))]
end

table(["‖B̄‖", "‖X‖₁", "kernel, exact solve", "kernel, Newton–Schulz", "`NativePade`"], rows)
```

The kernel column with the exact solve is accurate at the first lift, has lost half its digits by
``\|X\|_1 \approx 16``, and is wrong by ``O(1)`` — a relative error above one, meaning no correct
digits at all — from ``\|X\|_1 \approx 600`` upwards. It does not overflow the way the Taylor kernel
does, because a rational function stays bounded where a truncated series does not; it simply
approximates a different function. The Newton–Schulz column is worse and worse for the separate
reason of the previous subsection: past ``\|Y\|_1 = 1/2`` the residual no longer contracts, so five
squarings of a quantity larger than one amplify rather than converge.

The last column is the same kernel with steps 1 and 5 of the algorithm around it, and it holds
``10^{-14}`` across the sweep. What the framework supplies is the guarantee that the kernel is only
ever asked for ``\|Y\|_1 \leq \theta``, where both failure modes are absent.

### What a large ``\theta`` costs `NativePade`

``\theta`` means something different in the two algorithms, and the difference is the reason
`NativePade(θ)` **rejects** ``\theta > 1/2`` where `ScaledSquaring(θ)` accepts any positive value.
`ScaledSquaring` sums its series until the terms vanish, so a larger ``\theta`` only asks it to sum
more terms; [Sensitivity to the threshold `θ`](@ref) shows what that costs, which is almost nothing
over a 32-fold range. `NativePade` does a *fixed* five Newton–Schulz steps against a fixed rational
approximant, and both were sized for ``\|Y\|_1 \leq 1/2``. Above that, the iteration count is simply
too small, and no quantity the algorithm computes says so.

Worst relative error of the kernel against [`AugmentedPade`](@ref) over 400 random ``6\times6``
arguments of one-norm exactly ``\theta`` — that is, the kernel asked for exactly what a threshold of
``\theta`` would hand it:

```@example retractions
Random.seed!(2024)  # nothing else on this page draws after `lifts`, so this is self-contained

worst(θ) = maximum(1:400) do _
    Y = randn(6, 6)
    Y = θ * Y / opnorm₁(Y)
    reference = 𝔄(Y, AugmentedPade())
    norm(pade_newton_schulz(Y) - reference) / norm(reference)
end

θs = [0.5, 1.0, 1.5, 2.0, 3.0]
table(vcat("``\\theta``", string.(θs)), [vcat("worst relative error", sci.(worst.(θs)))])
```

Only the pattern is reproducible, not the digits: these are maxima over a random draw, and the two
rightmost entries move by orders of magnitude between seeds. What does not move is where the
transition happens — full accuracy up to ``\theta = 1``, several digits gone by ``3/2``, none left by
``2``. The default sits a factor of two below the first of those, which is the room the residual bound
of [Applying the denominator without a matrix solve](@ref) quantifies.

Lowering ``\theta`` below ``1/2`` is safe and only adds modified-squaring steps. Raising it is not
available, and that asymmetry is deliberate: the two thresholds are not interchangeable, even though
both constructors spell the argument `θ`.

**Advantages.** It never forms a matrix larger than ``2n\times{}2n`` and uses only reductions, matrix
products and a kernel-written identity, so it is the algorithm [`ScaledSquaring`](@ref) can be checked
against on a backend that forbids scalar indexing. The two share their scaling rule and their recovery
step, but nothing in the kernel: a truncated series against a rational function, and a division by
``2^s`` against a Newton–Schulz inverse. Agreement between them is therefore evidence about the two
kernels, though not about the framework they have in common. Its accuracy is indistinguishable from
[`ScaledSquaring`](@ref)'s and [`AugmentedPade`](@ref)'s in `Float64` throughout
[Agreeing with the exponential](@ref).

**Disadvantages.** Its fixed rational evaluation does more small matrix products than
[`ScaledSquaring`](@ref) — around `1.8×` the ``\mathfrak{A}`` call in [What they cost](@ref), still
below [`AugmentedPade`](@ref) — and it allocates the most of the three algorithms that evaluate
``\mathfrak{A}`` directly. An allocation count is the figure least likely to stay a constant factor on
a backend where allocating can cost a synchronisation rather than a `malloc`, and the ordering of the
three is itself unexplained; both are
[#77](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/77).

The `Float64` indifference does not carry to `Float32`, where it has the worst `check` of the three
while its *forward* error there is no outlier at all. What degrades is the orthogonality of the
retracted point rather than the agreement with the exponential; both are in [`Float32`](@ref) below.

!!! note "What justifies `θ = 1/2`, and what does not"
    Not a backward-error table. The ``\theta_m`` of [higham2005scaling, almohy2010new](@cite) are
    derived for ``\exp`` rather than for ``\mathfrak{A}``, and they bound a backward error in
    ``\|X\|`` — the least informative norm available here, since
    ``\|X\| \approx \|\bar{B}\|^2/4`` against a spectral radius of only
    ``\approx\|\bar{B}\|``. What justifies the threshold is narrower, and is stated as such: the
    Newton--Schulz residual bound above, the ``8\cdot10^{-19}`` kernel truncation error, and the
    measured forward error over the norm sweep and over the 400 random arguments tabulated. A
    backward-error criterion for ``\mathfrak{A}`` on a strongly non-normal argument is a genuine gap,
    and nothing on this page closes it.

## 4. `AugmentedPade`

[`AugmentedPade`](@ref) is the one algorithm here that chooses neither a kernel nor a recovery: it
delegates both. For the ``4n\times{}4n`` augmented matrix,

```math
\exp\begin{pmatrix} X & \mathbb{I} \\ \mathbb{O} & \mathbb{O} \end{pmatrix}
= \begin{pmatrix} \exp(X) & \mathfrak{A}(X) \\ \mathbb{O} & \mathbb{I} \end{pmatrix},
```

which is the standard device for recovering ``\mathfrak{A}`` from a routine that computes only ``\exp``
[sidje1998expokit, higham2008functions](@cite). One call to `Base.exp` therefore returns
``\mathfrak{A}(X)`` in its upper-right block, and the numerics are Julia's own dense matrix
exponential — a Padé approximant with its own scaling and squaring
[higham2005scaling, almohy2010new](@cite). This is the package algorithm that corresponds directly to
the conventional Padé description of scaling and squaring, and the one place a reader can compare the
two sections above against a textbook implementation.

**Advantages.** It introduces no package-specific approximation at all. Everything delicate is done by
the most heavily exercised matrix-exponential implementation available, which is why it is the
reference every accuracy table on this page is measured against. Its accuracy matches
[`ScaledSquaring`](@ref)'s throughout [Agreeing with the exponential](@ref), and it is, unexpectedly,
the *lightest* allocator of the three algorithms that evaluate ``\mathfrak{A}`` — see
[What they cost](@ref) — because `Base.exp` works in a few reused buffers where both native algorithms
produce a fresh ``2n\times{}2n`` temporary per operation.

**Disadvantages.** It exponentiates a matrix four times the size needed and discards three quarters of
the result, which makes the ``\mathfrak{A}`` call the most expensive of the three, though much less so
once the ``N\times{}N`` assembly around it is counted. And `Base.exp` on a dense matrix needs LAPACK,
so it does not run on a GPU backend.

!!! note "A reference, not a normal choice"
    `AugmentedPade` is constructible and supported, and it is documented here because it is the
    reference the direct implementations are measured against, both in the test suite and in the tables
    below. It is not what to select for an optimization run: it costs more than
    [`ScaledSquaring`](@ref), is no more accurate, and needs dense LAPACK.

## 5. `ProjectedSkew`

[`ProjectedSkew`](@ref) does not go through ``\mathfrak{A}`` at all. It nonetheless exploits the same
rank-``2n`` structure as the other four, and nothing ``N\times{}N`` is exponentiated here either; it
simply reaches that structure by a change of basis rather than by [The reduction](@ref). It
exponentiates the lift in a basis of the lift's own range, where the lift is a small skew-symmetric
matrix.

``\bar{B}`` is skew-symmetric of rank at most ``2n``, so its range and its row space coincide and
both sit inside the range of ``B'``. A thin QR of ``B'`` gives an ``N\times{}2n`` orthonormal ``Q``
with ``\bar{B} = QMQ^T``, where ``M = Q^T\bar{B}Q`` is skew-symmetric and ``2n\times{}2n``, and

```math
\exp(\bar{B}) = \mathbb{I} + Q\left(\exp(M) - \mathbb{I}\right)Q^T.
```

``\exp(M)`` is then formed from an eigendecomposition rather than from a series: ``iM`` is Hermitian
for real skew ``M``, so ``M = -iV\Lambda{}V^*`` with ``V`` unitary and ``\Lambda`` real, and

```math
\exp(M) = \Re\left(V e^{-i\Lambda} V^*\right),
```

which is orthogonal **by construction** — a product of a unitary matrix, a diagonal of unit-modulus
numbers, and a unitary matrix — rather than by the cancellation of a series. That is the property the
whole algorithm exists for, and it is what the measurements below show: the orthogonality of the
result does not depend on how accurately anything was summed.

Unlike the other four, `ProjectedSkew` bypasses ``\mathfrak{A}`` and specialises [`geodesic`](@ref)
directly. It therefore has no `𝔄(X, ProjectedSkew())` method — worth knowing if you call
``\mathfrak{A}`` yourself.

**Advantages.** It is the only algorithm whose `check` does not degrade with the size of the lift: its
column in [Staying on the manifold](@ref) is level where the other four climb by two orders of
magnitude. The gap is widest in [`Float32`](@ref), where the others are at the mercy of the format and
this one is not. That is the case for choosing it: a long `Float32` run, where the departure from the
manifold accumulates over thousands of steps and staying on the manifold matters more than agreeing
with the exponential to the last bit.

**Disadvantages.** It usually has the largest forward error against `exp(Matrix(B))`, in both formats
and at all but the top of the sweep — see [Agreeing with the exponential](@ref). It needs a `qr` and
an `eigen` instead of matrix products, which costs more than [`ScaledSquaring`](@ref) per retraction
in [What they cost](@ref) and rules out a GPU backend.

The trade is a real exchange rather than a ranking: orthogonality is structural here and accuracy
against the exponential is not, and the two tables below disagree with each other for exactly that
reason.

## Using them

None of these types is exported, so import the ones you use:

```jldoctest retraction-usage
using GeometricOptimizers
using GeometricOptimizers: Geodesic, Cayley, ScaledSquaring, NativePade, AugmentedPade, ProjectedSkew, check
import Random
Random.seed!(123)

Y = rand(StiefelManifold, 5, 3)
B = GeometricOptimizers.global_rep(GlobalSection(Y), rand(5, 3))

check(Geodesic()(B)) < 1e-14, check(Cayley()(B)) < 1e-14

# output

(true, true)
```

A retraction is passed to [`Optimizer`](@ref) as a keyword argument, and the algorithm travels inside
it:

```julia
optimizer = Optimizer(ps, L; algorithm  = Adam(Float32),
                             retraction = Geodesic(ProjectedSkew()))
```

`Geodesic()` is `Geodesic(ScaledSquaring())`, and `ScaledSquaring(θ)` accepts a scaling threshold;
the default is `0.5`:

```jldoctest retraction-usage
Geodesic().algorithm == ScaledSquaring(0.5)

# output

true
```

The algorithm reaches the lower-level entry points too. [`geodesic`](@ref) takes it as an optional
last argument, both for a tangent vector at a point and for a horizontal lift:

```jldoctest retraction-usage
using GeometricOptimizers: geodesic, rgrad

Δ = rgrad(Y, rand(5, 3))

check(geodesic(Y, 300 * Δ, ProjectedSkew())) < 1e-13

# output

true
```

and [`GeometricOptimizers.𝔄`](@ref) can be called on a bare matrix, which is the level at which four
of the five algorithms are implemented — [`ProjectedSkew`](@ref) being the exception, as above:

```jldoctest retraction-usage
using GeometricOptimizers: 𝔄
import Random
Random.seed!(123)

X = randn(6, 6)

isapprox(𝔄(X, ScaledSquaring()), 𝔄(X, NativePade()); rtol = 1e-12) &&
    isapprox(𝔄(X, NativePade()), 𝔄(X, AugmentedPade()); rtol = 1e-12)

# output

true
```

If what you want is the exponential itself rather than ``\mathfrak{A}``,
[`GeometricOptimizers.𝔄exp`](@ref) assembles it — ``\exp(B'(B'')^T) = \mathbb{I} +
B'\mathfrak{A}(B', B'')(B'')^T``, at a cost still set by ``n`` rather than by ``N``, since the only
matrix function evaluated is ``\mathfrak{A}`` on the ``2n\times{}2n`` product. It is what
[`geodesic`](@ref) computes before wrapping the result in a [`Manifold`](@ref), and it defaults to
[`ScaledSquaring`](@ref) for the same reason `geodesic` does:

```jldoctest retraction-usage
using GeometricOptimizers: 𝔄exp, lift_factors
import Random
Random.seed!(1234)

B = 60 * rand(StiefelLieAlgHorMatrix, 20, 3)      # ‖B̄‖ ≈ 393
B̂, B̄ = lift_factors(B)

isapprox(𝔄exp(B̂, B̄), exp(Matrix(B)); rtol = 1e-10)

# output

true
```

The difference between scaled and unscaled Taylor evaluation becomes visible for a large lift:

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: Geodesic, TaylorSeries, check
import Random
Random.seed!(1234)

B = 60 * rand(StiefelLieAlgHorMatrix, 20, 3)      # ‖B̄‖ ≈ 393

check(Geodesic()(B)) < 1e-12, check(Geodesic(TaylorSeries())(B)) < 1e-12

# output

(true, false)
```

## 6. Numerical comparison

Everything in this section other than the timings is recomputed when this page is built, so the
figures are those of the version of the package the documentation was built from rather than a quote
that can go stale. `scripts/retraction_accuracy.jl` produces the same tables — including the
timings — from the command line.

The figures quoted in the algorithm sections above are read off one build of these tables. They are
there to make each section's argument concrete, and the tables here are what to trust: round-off-level
numbers depend on the BLAS and the machine, so the last digit of a quotation will not always match the
table below it. What does not move is the shape of each column, and that is what the sections claim.

```@setup retractions
using GeometricOptimizers
using GeometricOptimizers: geodesic, cayley, check, ScaledSquaring, NativePade, AugmentedPade, ProjectedSkew, TaylorSeries
using LinearAlgebra: norm
using Markdown
using Printf
import Random

# A `Markdown.MD` rather than a string: Documenter renders it as a table instead of as the code
# block a printed string would give.
function table(header, rows)
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", repeat("---|", length(header)))
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
    Markdown.parse(String(take!(io)))
end

# `TaylorSeries` overflows at the top of the sweep, so both formatters have to survive a `NaN` and
# an `Inf` — a bare `@sprintf` of one is fine, but the guard makes the intent explicit.
sci(x) = isfinite(x) ? (@sprintf "%.2e" x) : string(x)
fixed(x) = isfinite(x) ? (@sprintf "%.2f" x) : string(x)

"""
A sweep of horizontal lifts of increasing norm, all drawn from the same seed — the same eight
`scripts/retraction_accuracy.jl` sweeps, so that the tables below and the ones it prints agree row
for row.
"""
function sweep(T)
    Random.seed!(1234)
    [T(s) * rand(StiefelLieAlgHorMatrix{T}, 20, 3) for s in (0.1, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0, 120.0)]
end
```

### Staying on the manifold

[`check`](@ref) of the retracted point, ``\|Y^TY - \mathbb{I}\|``, on a random
`StiefelLieAlgHorMatrix(20, 3)` scaled up. This is the quantity a retraction is supposed to keep at
round-off, and it is the one the test suite asserts on. [`Cayley`](@ref) is in the last column as the
reference, since it evaluates no matrix function at all.

```@example retractions
lifts = sweep(Float64)

table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`AugmentedPade`", "`ProjectedSkew`", "`TaylorSeries`", "`Cayley`"],
      [[fixed(norm(Matrix(B))),
        sci(check(geodesic(B, ScaledSquaring()))),
        sci(check(geodesic(B, NativePade()))),
        sci(check(geodesic(B, AugmentedPade()))),
        sci(check(geodesic(B, ProjectedSkew()))),
        sci(check(geodesic(B, TaylorSeries()))),
        sci(check(cayley(B)))] for B in lifts])
```

`TaylorSeries` fails by many orders of magnitude and is the column the `!!! danger` above quotes.
Every other column stays at round-off across the whole `Float64` sweep, and only
[`ProjectedSkew`](@ref)'s is *level*. The other four — [`Cayley`](@ref) included, and it is the one
that drifts furthest — grow by two to three orders of magnitude from one end to the other, because
their orthogonality is an arithmetic outcome where [`ProjectedSkew`](@ref)'s is structural. Round-off
at ``\|\bar{B}\| \approx 770`` is still round-off, so this separates the algorithms without
condemning any of the four usable ones.

### Agreeing with the exponential

Relative distance to `exp(Matrix(B))`, i.e. to the exponential of the full ``N\times{}N`` lift. This
is a different question from the one above — a retraction that re-orthonormalised its result would
have a perfect `check` and be wrong here — and the test suite asserts both.

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`AugmentedPade`", "`ProjectedSkew`"],
      [begin
           reference = exp(Matrix(B))
           err(algorithm) = norm(Matrix(geodesic(B, algorithm)) - reference) / norm(reference)
           [fixed(norm(Matrix(B))), sci(err(ScaledSquaring())), sci(err(NativePade())),
            sci(err(AugmentedPade())), sci(err(ProjectedSkew()))]
       end for B in lifts])
```

All four grow slowly with the norm of the lift, and the ordering is roughly the reverse of the
previous table: [`ProjectedSkew`](@ref) is the furthest from the exponential at all but the largest of
these norms. [`ScaledSquaring`](@ref), [`NativePade`](@ref) and [`AugmentedPade`](@ref) sit within a
factor of `1.6` of each other on every row, so the trade is between that group and
[`ProjectedSkew`](@ref): one is orthogonal by construction, the others agree with `exp` more closely.
The four converge again at the top of the sweep, where the reference `exp(Matrix(B))` is itself no more
accurate than what is being measured against it. The table reports an experiment, not an error bound;
the ordering can depend on the matrix and the floating-point format, which is what the next section
is for.

### `Float32`

The same orthogonality residual in `Float32`, the format the MNIST example in
[Optimization on Homogeneous Spaces](@ref) actually runs in. Nothing can do better than about
``10^{-6}`` here, but the four do not degrade alike.

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`AugmentedPade`", "`ProjectedSkew`"],
      [[fixed(norm(Matrix(B))),
        sci(check(geodesic(B, ScaledSquaring()))),
        sci(check(geodesic(B, NativePade()))),
        sci(check(geodesic(B, AugmentedPade()))),
        sci(check(geodesic(B, ProjectedSkew())))] for B in sweep(Float32)])
```

[`ProjectedSkew`](@ref) is flat here too, and by the top of the sweep it is one to two orders of
magnitude below the others. Which of the other three is worst depends on the lift over most of the
range, but not at the top: there [`NativePade`](@ref) is the worst of them by a factor of about
``2.5``, which is the one place the `Float64` indifference of the previous table does not carry over.

And the forward error in the same format, which gives a different ranking and is worth having next to
it. The reference is `exp` of the lift promoted to `Float64`, with the difference taken there as well:
`exp` of a `Float32` matrix is itself only `Float32`-accurate, so comparing against it would measure
the reference as much as the algorithm.

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`AugmentedPade`", "`ProjectedSkew`"],
      [begin
           reference = exp(Matrix{Float64}(Matrix(B)))
           err(algorithm) = norm(Matrix{Float64}(Matrix(geodesic(B, algorithm))) - reference) /
                            norm(reference)
           [fixed(norm(Matrix(B))), sci(err(ScaledSquaring())), sci(err(NativePade())),
            sci(err(AugmentedPade())), sci(err(ProjectedSkew()))]
       end for B in sweep(Float32)])
```

Here [`ProjectedSkew`](@ref) is the *worst* of the four at every norm — by `1.4×` to `3.9×` over most
of the sweep and by `23×` at the smallest lift, where it is the only one not at `Float32` round-off —
and the three ``\mathfrak{A}`` algorithms are within `1.7×` of each other throughout,
[`NativePade`](@ref) included, its `check` outlier above notwithstanding. Taken together the two
tables say what the trade actually is in `Float32`: [`ProjectedSkew`](@ref) buys orthogonality at the
price of agreement, and it is the only one of the four for which that is a structural exchange rather
than an accident of the arithmetic. Whether it matters in an optimization run depends on how the two
errors accumulate in that application.

### Sensitivity to the threshold `θ`

[`ScaledSquaring`](@ref)'s only parameter, swept over a 32-fold range on one lift:

```@example retractions
Random.seed!(99)
B = 30 * rand(StiefelLieAlgHorMatrix{Float64}, 20, 3)
reference = exp(Matrix(B))

table(["θ", "`check`", "error vs `exp`"],
      [begin
           Y = geodesic(B, ScaledSquaring(θ))
           [string(θ), sci(check(Y)), sci(norm(Matrix(Y) - reference) / norm(reference))]
       end for θ in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)])
```

Both columns move by less than a factor of six across the whole 32-fold range, and not monotonically.
The default `0.5` sits inside that band and nothing in the measurement singles it out, which is the
point: it needs no tuning because no value in the range does appreciably better. This is one lift and
one seed, so it supports `0.5` as a reasonable default rather than establishing an optimal threshold,
and it says nothing about [`NativePade`](@ref), whose `θ` is a ceiling and not a preference.

### What they cost

Unlike everything above, these are timings and therefore machine-dependent, so they are quoted rather
than measured at build time. `minimum` of 50 repetitions, a single BLAS thread, on an Apple M-series
laptop; `julia --project=. scripts/retraction_accuracy.jl` reproduces them on yours, and every figure
in this section comes from one run of it. They are illustrative, and the only sound way to use them is
to remeasure on the target hardware.

The ``\mathfrak{A}`` call on its own, at ``N = 200``, ``n = 10``, is what separates the three
algorithms that evaluate it:

| | `ScaledSquaring` | `NativePade` | `AugmentedPade` |
|---|---|---|---|
| runtime | `0.021 ms` | `0.037 ms` | `0.053 ms` |
| allocated | `201 KiB` | `330 KiB` | `114 KiB` |

The allocation row is the one figure here that is *not* machine-dependent — `@allocated` is exact —
and it does not rank the three the way runtime does. [`AugmentedPade`](@ref), which builds a
``4n\times{}4n`` matrix and discards three quarters of the result, allocates the least, because
`Base.exp` works in a few reused buffers where both native algorithms produce a fresh
``2n\times{}2n`` temporary per operation. On a CPU that is a detail. On a backend where an allocation
costs a synchronisation it may not be, which is worth knowing about the algorithm whose whole purpose
is to be portable. Neither native algorithm reuses buffers, and both could; that is
[#77](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/77).

One whole retraction, which adds the ``N\times{}N`` assembly they all share, in milliseconds:

| ``N``, ``n`` | 10, 2 | 20, 3 | 50, 5 | 100, 5 | 200, 10 | 500, 10 | 500, 50 | 1000, 20 |
|---|---|---|---|---|---|---|---|---|
| `Geodesic(ScaledSquaring())` | 0.003 | 0.005 | 0.014 | 0.023 | 0.087 | 0.396 | 3.03 | 2.51 |
| `Geodesic(NativePade())` | 0.004 | 0.005 | 0.013 | 0.023 | 0.091 | 0.410 | 3.33 | 2.57 |
| `Geodesic(AugmentedPade())` | 0.003 | 0.006 | 0.016 | 0.027 | 0.120 | 0.464 | 5.98 | 2.72 |
| `Geodesic(ProjectedSkew())` | 0.004 | 0.008 | 0.023 | 0.028 | 0.130 | 0.464 | 4.04 | 2.89 |
| `Geodesic(TaylorSeries())` | 0.003 | 0.006 | 0.019 | 0.033 | 0.149 | 0.505 | 14.1 | 3.49 |
| `Cayley()` | 0.002 | 0.004 | 0.016 | 0.056 | 0.361 | 4.87 | 6.24 | 38.8 |

and the same in KiB allocated:

| ``N``, ``n`` | 10, 2 | 20, 3 | 50, 5 | 100, 5 | 200, 10 | 500, 10 | 500, 50 | 1000, 20 |
|---|---|---|---|---|---|---|---|---|
| `Geodesic(ScaledSquaring())` | 13.1 | 33.1 | 130 | 338 | 1354 | 6653 | 13813 | 26419 |
| `Geodesic(NativePade())` | 19.8 | 45.9 | 162 | 370 | 1480 | 6782 | 16776 | 26940 |
| `Geodesic(AugmentedPade())` | 12.0 | 29.6 | 117 | 322 | 1278 | 6552 | 11011 | 25982 |
| `Geodesic(ProjectedSkew())` | 15.1 | 33.3 | 122 | 335 | 1315 | 6685 | 10769 | 26417 |
| `Geodesic(TaylorSeries())` | 12.2 | 31.1 | 131 | 345 | 1483 | 6893 | 27586 | 28615 |
| `Cayley()` | 12.9 | 32.9 | 144 | 475 | 1881 | 10546 | 13383 | 41694 |

[`ScaledSquaring`](@ref) and [`NativePade`](@ref) are level on runtime to within the run-to-run noise
at every size but one; the isolated call above is where the extra rational work shows. The exception is
``n = 50``, the one column where the ``2n\times{}2n`` argument is large enough for the difference to
survive the assembly: `3.33` against `3.03`, with [`AugmentedPade`](@ref)'s `5.98` as it pays for the
``4n\times{}4n`` embedding. Allocations separate them everywhere and by more, up to `1.5×` at
``n = 50``, since that is the metric the shared assembly dilutes least at small ``N``.
[`ProjectedSkew`](@ref) stays close on both — a QR and an eigendecomposition of a ``2n\times{}2n``
matrix are not expensive things, and it is the *lightest* of the five at ``n = 50``. [`Cayley`](@ref)
is level with the exponential up to ``N \approx 50`` and loses by a factor of 15 by ``N = 1000``,
which is the ``O(N^3)`` against ``O(N^2n)`` of the [Retractions](@ref) page.

## Choosing one

[`Cayley`](@ref) remains the package default. It requires only a small linear solve and is robust for
large steps. [`Geodesic`](@ref) computes the exponential map and has the one-parameter subgroup
property. In the measurements above it also becomes cheaper once the ambient dimension is large
relative to the manifold dimension. Choose between them according to which map the algorithm needs,
then benchmark representative problem sizes if cost matters.

For the algorithm: **[`ScaledSquaring`](@ref), i.e. the default, unless one of the alternatives has
the property you specifically need.**

| | choose it when | at the price of |
|---|---|---|
| [`ScaledSquaring`](@ref) | almost always; it is the default | `check` drifting up with the size of the lift |
| [`NativePade`](@ref) | you want an independent direct calculation on a backend that forbids scalar indexing | `1.8×` the isolated ``\mathfrak{A}`` runtime and `1.6×` its allocations, the same accuracy in `Float64`, the worst `check` of the three in `Float32`, and `θ` bounded by `1/2` |
| [`ProjectedSkew`](@ref) | staying on the manifold matters more than the last bit of the exponential — a long `Float32` run, where `check` accumulates over thousands of steps | `1.1×`–`1.6×` the cost, the largest forward error in either format, CPU only |
| [`AugmentedPade`](@ref) | you want a second opinion from an implementation that introduces no numerics of its own | roughly `2.5×` the cost of the ``\mathfrak{A}`` call, no better than [`ScaledSquaring`](@ref) on accuracy, CPU only |
| [`TaylorSeries`](@ref) | never; it exists so the pre-0.2.0 regression stays reproducible | leaving the manifold silently above ``\Vert\bar{B}\Vert \approx 50`` |

[`ScaledSquaring`](@ref), [`NativePade`](@ref) and [`TaylorSeries`](@ref) avoid dense LAPACK and scalar
indexing in package code; [`AugmentedPade`](@ref) and [`ProjectedSkew`](@ref) do not. Whether any of
the first three runs on a particular accelerator depends on that backend's matrix-multiplication and
reduction support. [`ScaledSquaring`](@ref) is the default among them because it does less work, and
[`NativePade`](@ref) is the independent cross-check that was previously missing there.

## Adding one

A new algorithm is a subtype of [`AbstractExponentialAlgorithm`](@ref) that supplies

```julia
GeometricOptimizers.𝔄(X::AbstractMatrix, ::NewAlgorithm)
```

Everything above it — [`geodesic`](@ref) for lifts and for tangent vectors, `retraction`,
`update_section!`, the optimizer — then follows, for both manifolds, with no further methods. An
algorithm that does not evaluate ``\mathfrak{A}`` at all supplies

```julia
GeometricOptimizers.geodesic(B::AbstractLieAlgHorMatrix, ::NewAlgorithm)
```

instead, which is what [`ProjectedSkew`](@ref) does. A new *retraction* is a subtype of
[`AbstractRetraction`](@ref) supplying `retraction(::NewRetraction, x)`; the callable form `R(x)`
comes for free.

Whichever it is, `test/retractions/exponential_accuracy.jl` is where it earns its place: the sweep
there asserts that a retracted point stays on the manifold at every lift norm *and* that it still
agrees with `exp(Matrix(B))`, which together rule out both of the ways an exponential can be wrong
here.



## Reference

```@bibliography
Pages = ["exponential_algorithms.md"]
Canonical = false
```
