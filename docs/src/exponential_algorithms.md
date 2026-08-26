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

so computing ``\mathfrak{A}(X)`` is the central numerical task: once it is available, the
``N\times{}N`` exponential is assembled using only the two thin factors. The argument ``X`` is only
``2n\times{}2n``, which preserves the cost advantage when ``n\ll N``. ``\mathfrak{A}`` is the
function usually written ``\varphi_1(X) = \left(\exp(X) - \mathbb{I}\right)X^{-1}``, though its
series definition remains valid when ``X`` is singular.

There are two separate algorithmic choices:

1. how to approximate the matrix function at a small argument; and
2. whether to scale a large argument down and recover the original value afterwards.

Taylor and Padé are approximation kernels. Scaling and squaring is a framework around such a kernel,
not a competing approximation. Classical dense matrix-exponential routines normally combine a Padé
approximant with scaling and squaring [higham2005scaling, almohy2010new](@cite). Taylor kernels can
also be effective when scaling keeps the argument small [skaflestad2009scaling](@cite).

All five algorithms in this package are:

| Package algorithm | Object evaluated | Kernel | Recovery | Backend |
|---|---|---|---|---|
| [`TaylorSeries`](@ref) | ``\mathfrak{A}(X)`` | Taylor series | none | matrix products and reductions |
| [`ScaledSquaring`](@ref) | ``\mathfrak{A}(X)`` | Taylor series | low-rank modified squaring | matrix products and reductions |
| [`NativePade`](@ref) | ``\mathfrak{A}(X)`` | degree-6 Padé | low-rank modified squaring | matrix products and reductions |
| [`AugmentedPade`](@ref) | ``\mathfrak{A}(X)`` as a block of a ``4n\times{}4n`` exponential | delegated to `Base.exp` | delegated to `Base.exp` | CPU (dense LAPACK) |
| [`ProjectedSkew`](@ref) | projected lift exponential | eigendecomposition | none | CPU (dense LAPACK) |

The Backend column says what the algorithm needs, not what it is known to run on: the first three
require only matrix products and reductions, so whether they run on a particular accelerator depends
on that backend's support for those operations, while the last two call into dense LAPACK and are
therefore CPU-only.

These algorithms return the same exponential map, so the one-parameter subgroup property above holds
for every one of them. Two of the five are not ordinary choices — [`TaylorSeries`](@ref) is a
regression baseline and [`AugmentedPade`](@ref) a reference implementation — and both are documented
here anyway, because both are constructible and both appear in the test suite.

## 1. Direct Taylor series

[`TaylorSeries`](@ref) evaluates ``\mathfrak{A}`` directly from its defining series, without scaling,
and stops when a term falls below `eps`. The series converges for every matrix, but convergence alone
does not guarantee an accurate floating-point sum.

The reduced argument here is particularly difficult. ``X``'s lower-left block is
``\tfrac{1}{4}A^2 - B^TB``, so

```math
\|X\| \approx \tfrac{1}{4}\|\bar{B}\|^2
\qquad\text{while}\qquad
\rho(X) \approx \|\bar{B}\|,
```

because the eigenvalues of ``X`` are the nonzero, purely imaginary eigenvalues of the skew matrix
``\bar{B}``. A norm quadratically larger than the spectral radius is a strongly non-normal matrix, and
on such an argument the intermediate terms become enormous before cancelling to a result of moderate
size: at ``\|\bar{B}\| \approx 79`` the partial sums exceed the answer by some twenty orders of
magnitude, reaching ``4\cdot10^{21}`` where the result is of order one, and 174 terms are needed before
one falls below `eps`. Stopping there therefore leaves a relative error of
``\varepsilon\|\mathfrak{A}(X)\|`` rather than ``\varepsilon``. That the direct series is not a
method for the matrix exponential is a very old observation [moler2003nineteen](@cite); what is
specific here is that the factorisation makes the argument *worse* than the matrix one started with.

How badly this goes is worth measuring rather than asserting. `check(geodesic(B, TaylorSeries()))`
on a random `StiefelLieAlgHorMatrix(20, 3)` scaled up:

| ``\|\bar{B}\|`` | 0.66 | 5.8 | 17.8 | 36.5 | 78.8 | 160 | 361 | 767 |
|---|---|---|---|---|---|---|---|---|
| `check` | `4.5e-16` | `2.1e-15` | `2.6e-12` | `4.4e-7` | `8.3e10` | `4.2e55` | `1.4e168` | `NaN` |

At ``\|\bar{B}\| \approx 79`` the "retracted" point is not on the Stiefel manifold in any sense, and
by ``767`` the series has overflowed. The same figures are recomputed in
[Staying on the manifold](@ref) below, so this table is a quotation of that measurement rather than a
separate claim.

!!! danger "This is not a usable retraction"
    `TaylorSeries` is retained as a regression baseline for the pre-0.2.0 implementation. It can
    silently lose orthogonality for large lifts. Do not select it for optimization.

Two things about it are worth recording rather than merely deprecating. The failure is *silent*: an
optimizer using it takes a step, gets a matrix back, and nothing anywhere reports that the matrix is
not on the manifold — which is why the defect survived until [`check`](@ref) was made generic over
[`Manifold`](@ref) instead of being defined for [`StiefelManifold`](@ref) alone. And the obvious first
fix does not work: making the termination test relative to the partial sum rather than absolute was
measured to change *none* of the numbers above, at any lift norm. The loss is in the cancellation
inside the sum, not in the point at which the summation stops. Scaling the argument down is what
helps, which is the next section.

[`Geodesic`](@ref) makes the algorithm choice visible:

```julia
Geodesic(ScaledSquaring())   # the default, and `Geodesic()`
Geodesic(NativePade())
Geodesic(AugmentedPade())    # the dense-CPU reference
Geodesic(ProjectedSkew())
Geodesic(TaylorSeries())     # the pre-0.2.0 behaviour; not a usable retraction
```

## 2. Scaling and modified squaring

Scaling and squaring first evaluates a matrix function at a smaller argument and then reconstructs
the value at the original argument. The classical exponential algorithm repeatedly squares
``\exp(X/2^s)``. For ``\varphi``-functions such as ``\mathfrak{A}=\varphi_1``, the corresponding
recovery formulas are usually called *modified squaring* [skaflestad2009scaling](@cite).

[`ScaledSquaring`](@ref) uses a Taylor kernel, while [`NativePade`](@ref) uses a Padé kernel. Both
scale the argument first and use the same recovery recurrence. `ScaledSquaring`'s advantage over
[`TaylorSeries`](@ref) comes entirely from that scaling: the Taylor series is evaluated only where
``\|X/2^s\|_1\leq\theta``, so its terms remain modest and the catastrophic cancellation of the
unscaled series is avoided. In exact arithmetic the subsequent modified-squaring steps are algebraic
identities and not additional approximations, so scaling changes *where* the kernel is evaluated
without changing the function being computed. In floating point they are not free — each squaring
amplifies whatever error it is handed, which is what the note at the end of this section is about —
but the error they amplify is far smaller than the one the unscaled series commits. In this way
`ScaledSquaring` makes the otherwise unreliable Taylor kernel usable for the reduced matrices
encountered here.

The low-rank form is closed under squaring:

```math
\left(\mathbb{I} + B'W(B'')^T\right)^2 = \mathbb{I} + B'\left(2W + WXW\right)(B'')^T,
```

so recovery needs only ``2n\times{}2n`` matrix products. It never squares an ``N\times{}N`` matrix.

Let ``L = B'(B'')^T``, ``X = (B'')^TB'``, and ``\alpha = 2^s``. The scaled exponential is

```math
\exp(L/\alpha)
= \mathbb{I} + B'\left[\frac{\mathfrak{A}(X/\alpha)}{\alpha}\right](B'')^T.
```

Thus the complete computation is:

1. **Choose the scaling.** Set
   ``s = \max(0, \lceil\log_2(\|X\|_1/\theta)\rceil)`` and ``\alpha = 2^s``.
2. **Evaluate the scaled problem.** Compute
   ``W_s = \mathfrak{A}(X/\alpha)/\alpha`` with the Taylor kernel. The argument now satisfies
   ``\|X/\alpha\|_1 \leq \theta``.
3. **Undo the scaling.** If
   ``\exp(L/2^k) = \mathbb{I} + B'W_k(B'')^T``, then squaring gives
   ``W_{k-1} = 2W_k + W_kXW_k``. Apply this update for ``k=s,s-1,\ldots,1``. The recurrence uses the
   original ``X``, not ``X/\alpha``.
4. **Return the result.** After the loop, ``W_0 = \mathfrak{A}(X)``, so
   ``\mathbb{I} + B'W_0(B'')^T = \exp(B'(B'')^T)``.

The initial division by ``\alpha`` follows directly from the scaled-exponential identity and is not
an approximation. The recurrence uses the original ``X`` because it represents repeated squaring of
``\exp(L/2^s)``. Every recovery step remains a ``2n\times{}2n`` update, so no dense ``N\times{}N``
exponential, square, or solve is formed.

The threshold `θ` defaults to `0.5`. A smaller value performs more scaling steps; a larger value asks
the Taylor kernel to handle a larger argument. Measured at ``\|\bar{B}\| \approx 155``, every
``\theta \in [0.125, 4]`` — a 32-fold range — gives a `check` between ``9.9\cdot10^{-15}`` and
``5.0\cdot10^{-14}`` and a forward error between ``6.4\cdot10^{-15}`` and ``8.3\cdot10^{-15}``, and
neither column is monotone in ``\theta``. That sweep is recomputed at build time in
[Sensitivity to the threshold `θ`](@ref) below. Nothing in it singles out `0.5`; it is a reasonable
value rather than a tuned one, and the sweep is an empirical check over one lift rather than a
general error bound.

**Advantages.** It is the cheapest of the algorithms on the ``\mathfrak{A}`` call itself — `0.021 ms`
at ``N = 200``, ``n = 10``, against [`NativePade`](@ref)'s `0.037 ms` and [`AugmentedPade`](@ref)'s
`0.053 ms` — and as close to `exp(Matrix(B))` as [`AugmentedPade`](@ref), which is as close as
anything here gets: `2.0e-14` against `1.9e-14` at ``\|\bar{B}\| = 361``. It needs only matrix
products, norms and a kernel-written identity, so it has no dense-LAPACK dependency. It is the
default because it is the cheapest of the algorithms that have none.

**Disadvantages.** Its orthogonality is the outcome of an arithmetic cancellation rather than a
structural property, so `check` drifts upwards with the size of the lift — from ``10^{-15}`` to around
``7\cdot10^{-14}`` over the sweep below, and considerably further in `Float32`. Only
[`ProjectedSkew`](@ref) avoids that drift. And it takes about twice the squarings it needs, for the
reason in the note below.

The implementation uses matrix products and reductions and avoids scalar indexing in package code.
The norm is taken by [`GeometricOptimizers.opnorm₁`](@ref) rather than by
`LinearAlgebra.opnorm(X, 1)`, whose `LinearAlgebra.opnorm1` is a double loop over `X[i, j]`, and the
identities come from [`GeometricOptimizers.unit_matrix`](@ref) rather than from `Base.one`, whose
diagonal write is the same hazard one level down. Execution on an accelerator still depends on the
backend's support for those matrix operations.

!!! note "The halving count is loose"
    The implementation chooses `s` from ``\|X\|_1``, i.e.
    ``s = \lceil\log_2(\|X\|_1/\theta)\rceil``. For these reduced matrices that norm is roughly
    ``\|\bar{B}\|^2/4`` where the spectral radius is only ``\approx\|\bar{B}\|``, so
    ``s \approx 2\log_2\|\bar{B}\|`` where ``\log_2\|\bar{B}\|`` would do, and the rule performs
    about twice the squarings it needs. Each squaring amplifies the error it is handed, so this costs
    both time and accuracy. It is the overscaling phenomenon discussed for the matrix exponential by
    Al-Mohy and Higham [almohy2010new](@cite). It is left alone because the tighter bound needs the
    spectral radius, and an eigenvalue computation would forfeit exactly the freedom from dense
    LAPACK that makes this the default algorithm. [`NativePade`](@ref) takes ``s`` the same way and
    inherits all of this.

## 3. Padé approximation

Scaling is settled: the previous section made the argument small and gave an exact recovery for the
original one. What it did not settle is *which* approximation to evaluate at the small argument.
[`ScaledSquaring`](@ref) evaluates a Taylor series there because that is what it already had;
[`NativePade`](@ref) evaluates a rational function instead, and this section is why that is a
different and defensible choice. A degree-``m`` Taylor polynomial simply keeps the first ``m+1``
terms,

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

Because ``e^z=\sum_{r\geq0}z^r/r!``, the ``z^k`` coefficient of ``Q_n(z)e^z`` is the convolution
``\sum_j b_j/(k-j)!``, reading ``1/(k-j)!`` as zero when ``j>k``. The matching condition
``Q_ne^z-P_m=O(z^{m+n+1})`` then says two different things, one on each of the two ranges of ``k``
that it covers:

```math
a_k=\sum_{j=0}^{n}\frac{b_j}{(k-j)!}
\quad (k=0,\ldots,m),
\qquad
\sum_{j=0}^{n}\frac{b_j}{(k-j)!}=0
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

and one binomial identity settles both blocks at once. Substitute this ``b_j`` into the convolution at
degree ``k`` and clear the constant:

```math
(m+n)!\sum_{j=0}^n\frac{b_j}{(k-j)!}
=\sum_{j=0}^n(-1)^j\binom{n}{j}\frac{(m+n-j)!}{(k-j)!}.
```

Put ``d=m+n-k``. The ratio ``(m+n-j)!/(k-j)!`` is a product of ``d`` consecutive integers, that is
``d!\binom{m+n-j}{d}``, which also reproduces the convention above: for ``j>k`` the binomial has
``m+n-j<d`` and vanishes. So the right-hand side is ``d!`` times

```math
\sum_{j=0}^n(-1)^j\binom{n}{j}\binom{m+n-j}{d}=\binom{m}{d-n}.
```

This is the binomial theorem read at a single degree. Introduce a bookkeeping variable ``x``, unrelated
to the ``z`` of the approximant and used only to extract a coefficient:

```math
\sum_{j=0}^n(-1)^j\binom{n}{j}(1+x)^{m+n-j}
=(1+x)^m\bigl((1+x)-1\bigr)^n
=x^n(1+x)^m,
```

and the coefficient of ``x^d`` on the left is ``\sum_j(-1)^j\binom{n}{j}\binom{m+n-j}{d}``, on the
right ``\binom{m}{d-n}``. That single evaluation splits exactly where the matching condition splits.
If ``k\geq m+1``, then ``d-n=m-k<0`` and ``x^n(1+x)^m`` has no ``x^d`` term, so the sum vanishes and
all ``n`` denominator equations hold. If ``k\leq m``, the coefficient is
``\binom{m}{m-k}=\binom{m}{k}``; restoring ``d!/(m+n)!`` gives the claimed ``a_k``.

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
\mathfrak{A}(z)
= \frac{e^z-1}{z}
= \frac{P^{\exp}_7(z)-Q^{\exp}_6(z)}{zQ^{\exp}_6(z)} + O(z^{13})
= \frac{p_6(z)}{q_6(z)} + O(z^{13}),
\qquad
p_6(z)=\frac{P^{\exp}_7(z)-Q^{\exp}_6(z)}{z},
\quad
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

!!! info "Why Newton--Schulz rather than LU?"
    For dense CPU matrices, the standard choice is pivoted LU followed by triangular solves; one
    would not normally form an explicit inverse. `NativePade` instead uses Newton--Schulz because
    its updates need only matrix multiplication and addition, keeping the direct algorithm
    independent of backend-specific factorization support. This is a portability choice, not a
    generally faster solver: five matrix products may cost more than LU on a CPU. It is viable here
    because scaling gives the initial-residual bound below, fixing both the iteration count and the
    resulting error.

To derive the iteration [schulz1933iterative](@cite), temporarily write
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
arbitrary starting point: convergence requires the initial inverse residual to be smaller than one
in a submultiplicative norm.

For the present choice ``A=q_6(Y)`` and ``Z_0=I``, define ``E_j=I-AZ_j``. Since
``AZ_j=I-E_j`` and ``2I-AZ_j=I+E_j``, the update gives

```math
AZ_{j+1}=(I-E_j)(I+E_j)=I-E_j^2,
\qquad
E_{j+1}=E_j^2.
```

Consequently ``E_j=E_0^{2^j}`` in exact arithmetic and
``\|E_j\|\leq\|E_0\|^{2^j}``: once ``\|E_0\|<1``, the iteration converges quadratically, squaring
the residual at every step. The code writes the first step explicitly as ``Z_1=2I-q_6(Y)`` and
performs four more, giving
``E_5=(I-q_6(Y))^{32}``. Scaling ensures ``\|Y\|_1\leq 1/2``; from the displayed coefficients,

```math
\|I-q_6(Y)\|_1
\leq \sum_{k=1}^6 |(q_6)_k|\,\|Y\|_1^k
<0.257,
```

so ``\|E_5\|_1<0.257^{32}<1.3\cdot10^{-19}``. This explains both the fixed five Newton--Schulz
steps and the constructor restriction ``0<\theta\leq 1/2``: together they make the solve-free inverse
accurate to approximately `Float64` precision using matrix multiplication alone.

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

**Advantages.** It never forms a matrix larger than ``2n\times{}2n`` and uses only reductions, matrix
products and a kernel-written identity, so it is the independent implementation
[`ScaledSquaring`](@ref) can be checked against on a backend that forbids scalar indexing — which is
what `test/retractions/exponential_accuracy.jl` does, on a `JLArray`. Its numerics share nothing with
`ScaledSquaring`'s beyond the scaling and the recovery, so agreement between the two is evidence
rather than tautology. At ``\|\bar{B}\| = 361`` its `check` is `3.9e-14` and its forward error
`2.1e-14`, indistinguishable in `Float64` from both [`ScaledSquaring`](@ref) and
[`AugmentedPade`](@ref).

**Disadvantages.** Its fixed rational evaluation does more small matrix products than
[`ScaledSquaring`](@ref): at ``N = 200``, ``n = 10`` the ``\mathfrak{A}`` call costs `0.037 ms`
against `0.021 ms`, though it stays under [`AugmentedPade`](@ref)'s `0.053 ms`. It also allocates the
most of the three that evaluate ``\mathfrak{A}`` directly — `330 KiB` against `201 KiB` and
`114 KiB` — and an allocation count is the figure least likely to stay a constant factor on a backend
where allocating can cost a synchronisation rather than a `malloc`.

The `Float64` indifference above does not carry to `Float32`. At the top of the norm sweep its `check`
is ``1.0\cdot10^{-4}`` against ``4.0\cdot10^{-5}`` for [`ScaledSquaring`](@ref) and
``4.1\cdot10^{-5}`` for [`AugmentedPade`](@ref) — the worst of the three — while its *forward* error
there is ``1.2\cdot10^{-5}`` against ``1.1\cdot10^{-5}`` and ``9.3\cdot10^{-6}``, which is no
outlier at all. What degrades in `Float32` is the orthogonality of the retracted point rather than the
agreement with the exponential. Both figures are recomputed in [`Float32`](@ref) below.

!!! warning "`θ` is a ceiling here, not a preference"
    `ScaledSquaring(θ)` accepts any positive threshold and stays accurate over a 32-fold range,
    because it sums its series until the terms vanish. `NativePade` does a *fixed* five
    Newton--Schulz steps, so past ``\theta \approx 1`` the inverse it computes stops being one.
    Worst relative error against [`AugmentedPade`](@ref) over 400 random ``6\times6`` arguments of
    one-norm exactly ``\theta``:

    | ``\theta`` | 1/2 | 1 | 3/2 | 2 | 3 |
    |---|---|---|---|---|---|
    | relative error | `5.8e-16` | `6.4e-16` | `1.5e-10` | `1.1e-5` | `242` |

    The two rightmost entries are the worst of a random draw and move by a factor of a few between
    seeds; their magnitudes do not, and the magnitudes are the point.

    Nothing in the result says so — a fixed number of refinements simply stops converging, silently.
    `NativePade(θ)` therefore refuses ``\theta > 1/2``, where `ScaledSquaring` accepts any positive
    value. The two thresholds are not interchangeable. Lowering this one is safe and only adds
    modified-squaring steps.

!!! note "What justifies `θ = 1/2`, and what does not"
    Not a backward-error table. The ``\theta_m`` of [higham2005scaling, almohy2010new](@cite) are
    derived for ``\exp`` rather than for ``\varphi_1``, and they bound a backward error in
    ``\|X\|`` — the least informative norm available here, since
    ``\|X\| \approx \|\bar{B}\|^2/4`` against a spectral radius of only
    ``\approx\|\bar{B}\|``. What justifies the threshold is narrower, and is stated as such: the
    Newton--Schulz residual bound above, the ``8\cdot10^{-19}`` kernel truncation error, and the
    measured forward error over the norm sweep and over the 400 random arguments tabulated. A
    backward-error criterion for ``\varphi_1`` on a strongly non-normal argument is one of the things
    [#52](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/52) asked for and this does not
    settle.

## 4. `AugmentedPade`

[`AugmentedPade`](@ref) is the one algorithm here that chooses neither a kernel nor a recovery: it
delegates both. For the ``4n\times{}4n`` augmented matrix,

```math
\exp\begin{pmatrix} X & \mathbb{I} \\ \mathbb{O} & \mathbb{O} \end{pmatrix}
= \begin{pmatrix} \exp(X) & \mathfrak{A}(X) \\ \mathbb{O} & \mathbb{I} \end{pmatrix},
```

which is the standard device for getting a ``\varphi`` function out of an exponential routine
[sidje1998expokit, higham2008functions](@cite). One call to `Base.exp` therefore returns
``\mathfrak{A}(X)`` in its upper-right block, and the numerics are Julia's own dense matrix
exponential — a Padé approximant with its own scaling and squaring
[higham2005scaling, almohy2010new](@cite). This is the package algorithm that corresponds directly to
the conventional Padé description of scaling and squaring, and the one place a reader can compare the
two sections above against a textbook implementation.

**Advantages.** It introduces no package-specific approximation at all. Everything delicate is done
by the most heavily exercised matrix-exponential implementation available, which is why it is the CPU
reference in `test/retractions/exponential_accuracy.jl`. Its accuracy is the same order as
[`ScaledSquaring`](@ref)'s — `3.0e-14` `check` and `1.9e-14` forward error at ``\|\bar{B}\| = 361``
— and it is, unexpectedly, the *lightest* allocator of the three algorithms that evaluate
``\mathfrak{A}``, at `114 KiB` against `201 KiB` and `330 KiB`, because `Base.exp` works in a few
reused buffers where both native algorithms produce a fresh ``2n\times{}2n`` temporary per operation.

**Disadvantages.** It exponentiates a matrix four times the size needed and discards three quarters
of the result, so the ``\mathfrak{A}`` call is about `2.5×` [`ScaledSquaring`](@ref)'s — `0.053 ms`
against `0.021 ms` — though much less than that once the ``N\times{}N`` assembly around it is
counted. And `Base.exp` on a dense matrix needs LAPACK, so it does not run on a GPU backend.

!!! note "A reference, not a normal choice"
    `AugmentedPade` is constructible and supported, and it is documented here because the test suite
    and `scripts/retraction_accuracy.jl` both use it as the reference the direct implementations are
    measured against. It is not what to select for an optimization run: it costs more than
    [`ScaledSquaring`](@ref), is no more accurate, and needs dense LAPACK.

## 5. `ProjectedSkew`

[`ProjectedSkew`](@ref) does not go through ``\mathfrak{A}`` at all. It exponentiates the lift in a
basis of the lift's own range, where it is a small skew-symmetric matrix.

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

**Advantages.** It is the only algorithm whose `check` does not degrade with the size of the lift.
Over the sweep below it stays between ``2\cdot10^{-15}`` and ``5\cdot10^{-15}`` from
``\|\bar{B}\| \approx 6`` to ``\|\bar{B}\| \approx 770``, where the other four drift from
``5\cdot10^{-16}`` to around ``8\cdot10^{-14}``. The gap is widest in `Float32`, where the others are at the
mercy of the format: over the same sweep their `check` climbs into the ``10^{-5}``s — into the
``10^{-4}``s for [`NativePade`](@ref) — while this stays at a few ``10^{-6}`` from one end to the
other. That is the case for choosing it: a long `Float32` run, where the departure from the manifold
accumulates over thousands of steps and staying on the manifold matters more than agreeing with the
exponential to the last bit.

**Disadvantages.** It usually has the largest forward error against `exp(Matrix(B))` — up to about
`4.3×` [`ScaledSquaring`](@ref)'s, and largest at all but the top of the sweep. It needs a `qr` and an
`eigen` instead of matrix products, which costs `1.1×`–`1.6×` one whole retraction over the sizes
measured and rules out a GPU backend.

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
is to be portable.

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
