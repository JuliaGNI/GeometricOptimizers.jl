```@meta
CurrentModule = GeometricOptimizers
```

# Retractions

A retraction is how every step in this package is taken. An [`OptimizerMethod`](@ref) produces a
direction in the horizontal component ``\mathfrak{g}^\mathrm{hor}`` of the Lie algebra, and the
retraction turns that direction back into a point of the manifold. Two of them ship with the
package — [`Cayley`](@ref), the Cayley transform, and [`Geodesic`](@ref), the exponential map — and
since 0.2.0 [`Geodesic`](@ref) additionally carries an *algorithm* that says how the exponential is
evaluated. There are four of those, and the choice between them is a numerical one: they compute the
same map and differ in accuracy at a large step, in cost, and in which backends they run on.

This page collects the theory of all of it: what the retractions are, why the exponential needs an
algorithm at all, what the four algorithms do, and which one to reach for. The optimizer that uses
them is described on the [Optimization on Homogeneous Spaces](@ref) page.

## Where a retraction sits in the algorithm

The optimizer never moves a point of the manifold directly. It keeps a section
``\Lambda^{(t)} \in G`` with ``Y^{(t)} = \Lambda^{(t)}E``, and one step is

```math
\Lambda^{(t+1)} \gets \Lambda^{(t)}\,\mathrm{retraction}(W^{(t)}),
\qquad
Y^{(t+1)} \gets \Lambda^{(t+1)}E,
```

with ``W^{(t)} \in \mathfrak{g}^\mathrm{hor}`` the direction the method produced. So what a
retraction has to return is an element of the *group*, not of the manifold, and the point follows
from it. This is the *extended* retraction of [brantner2023generalizing](@cite); `update_section!`
performs the first line and `apply_section` the second. Because the retracted element is in ``G``, a
retracted point is on the manifold by construction, and [`check`](@ref) — which measures
``\|Y^TY - \mathbb{I}\|`` — returns nothing but accumulated round-off.

An instance is passed to [`Optimizer`](@ref) as `retraction = Cayley()` or `retraction = Geodesic()`.
The default is [`Cayley`](@ref).

## Both retractions factor the lift

A horizontal lift is sparse, and both retractions exploit that in the same way. For the Stiefel
manifold, [`lift_factors`](@ref) writes

```math
\bar{B} = \begin{bmatrix} A & -B^T \\ B & \mathbb{O} \end{bmatrix}
        = \begin{bmatrix} \tfrac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix}
          \begin{bmatrix} \mathbb{I} & \mathbb{O} \\ \tfrac{1}{2}A & -B^T \end{bmatrix}
        =: B'(B'')^T,
```

with two ``N\times{}2n`` factors — for a [`GrassmannLieAlgHorMatrix`](@ref) the same expression with
``A \equiv \mathbb{O}``. Whatever matrix function a retraction needs is then evaluated on the
``2n\times{}2n`` product

```math
X := (B'')^TB',
```

which is small even when ``N`` is large. The matrix function is therefore priced by the number of
columns ``n`` and not by the dimension ``N`` of the ambient space; what is left of the retraction is
the assembly around it, which is ``O(N^2n)`` and not the ``O(N^3)`` an ``N\times{}N`` matrix function
would cost. And, as the next sections show, the factorisation is also where the *accuracy* of the
exponential is decided, because ``X`` is a considerably worse-behaved matrix than ``\bar{B}`` is.

## `Cayley` and `Geodesic`

[`Cayley`](@ref) is the Cayley transform,

```math
\mathrm{Cayley}(\bar{B}) = \left(\mathbb{I} - \tfrac{1}{2}\bar{B}\right)^{-1}
                           \left(\mathbb{I} + \tfrac{1}{2}\bar{B}\right),
```

which maps a skew-symmetric matrix into ``SO(N)`` exactly. [`cayley`](@ref) never forms the
``N\times{}N`` inverse: with the factorisation above it inverts a ``2n\times{}2n`` matrix instead. No
matrix *function* is involved anywhere, only a solve, so there is no series to cancel and no step
size at which the transform breaks down the way an unscaled series does. That is not the same as
being insensitive to the size of the lift: `check` still climbs from ``10^{-15}`` to
``3\cdot10^{-13}`` over the sweep [below](@ref "Staying on the manifold"), which is the largest drift
of anything on this page other than [`TaylorSeries`](@ref).

[`Geodesic`](@ref) is the exponential map,

```math
\mathrm{Geodesic}(\bar{B}) = \exp(\bar{B}),
```

i.e. the true geodesic of the manifold [edelman1998geometry](@cite). The difference that matters to
the rest of the package is that ``\alpha \mapsto \exp(\alpha\bar{B})`` is a **one-parameter
subgroup**,

```math
\exp\left((\alpha + \beta)\bar{B}\right) = \exp(\alpha\bar{B})\exp(\beta\bar{B}),
```

and ``\alpha \mapsto \mathrm{Cayley}(\alpha\bar{B})`` is not. Everything in sight is a rational
function of ``\bar{B}``, so it all commutes and the two products can be compared directly: with
``t = \alpha/2`` and ``s = \beta/2``,

```math
\mathrm{Cayley}(\alpha\bar{B})\,\mathrm{Cayley}(\beta\bar{B})
  = \big[\mathbb{I} + (t + s)\bar{B} + ts\bar{B}^2\big]
    \big[\mathbb{I} - (t + s)\bar{B} + ts\bar{B}^2\big]^{-1} ,
```

against ``[\mathbb{I} + (t+s)\bar{B}][\mathbb{I} - (t+s)\bar{B}]^{-1}`` for
``\mathrm{Cayley}((\alpha+\beta)\bar{B})``. The two agree only where ``ts\bar{B}^2 = \mathbb{O}``, and
the gap is not small: on a random ``\mathfrak{g}^\mathrm{hor}`` element of ``\operatorname{St}(6,3)``
with ``\|\bar{B}\| = 2.99`` it is ``1.28`` at ``\alpha = \beta = 1``, where the same difference for
``\exp`` is ``8\times10^{-16}``.

So the generator of the curve's velocity turns with ``\alpha`` instead of staying ``\bar{B}``. That
is what [`retraction_differential`](@ref) supplies, and with it [`trial_slope`](@ref) is the exact
derivative of a line search's merit function under *either* retraction. Before 0.2.0 the slope was
paired against ``\bar{B}`` regardless, which made it first-order under [`Cayley`](@ref) — 8.9% off at
``\alpha = 0.5`` and 36% at ``\alpha = 1`` on the ``\operatorname{St}(6,3)`` problem of
[Linesearches on Manifolds](@ref), which is where that measurement lives.

!!! info "It is the parameterisation and not the approximation"
    The natural reading of "`Cayley` is a retraction and `Geodesic` is the exponential map" is that
    the slope was wrong because the *curve* was approximate. That is not the mechanism, and the
    smallest counterexample separates them. Take ``N = 2`` and ``\bar{B} = J = \left(\begin{smallmatrix}0 & -1\\ 1 & 0\end{smallmatrix}\right)``.
    Then

    ```math
    \mathrm{Cayley}(\alpha{}J) = \exp\big(2\arctan(\tfrac{\alpha}{2})\,J\big)
    ```

    to the last bit — the Cayley curve is the geodesic, *exactly*, with nothing approximate about it.
    It is traversed at a different speed:

    | ``\alpha`` | 0 | 0.5 | 1 | 2 | 4 |
    |---|---|---|---|---|---|
    | angle turned, `Cayley` | 0 | 0.4900 | 0.9273 | 1.5708 | 2.2143 |
    | angle turned, `Geodesic` | 0 | 0.5 | 1 | 2 | 4 |
    | ``d\theta/d\alpha`` for `Cayley` | 1 | 0.9412 | 0.8 | 0.5 | 0.2 |

    and that last row is exactly ``D(\alpha) = \bar{B}(\mathbb{I} - \frac{\alpha^2}{4}\bar{B}^2)^{-1}``
    at ``J^2 = -\mathbb{I}``, i.e. ``J/(1 + \alpha^2/4)``. Pairing the gradient against ``J`` rather
    than against ``D(\alpha)`` therefore overstates ``\varphi'`` by ``1 + \alpha^2/4`` — 6.2% at
    ``\alpha = 0.5``, 25% at ``\alpha = 1``, 100% at ``\alpha = 2``. A curve can be exactly right and
    still give the wrong ``\varphi'``, because ``\varphi'`` is a derivative with respect to the
    *parameter*.

    That also says how to read the percentages above: to leading order the error is
    ``\alpha^2\lambda^2/4`` for an eigenvalue ``\pm{}i\lambda`` of ``\bar{B}``, so it grows with the
    step *and* with the size of the lift, and a figure quoted without its problem means little. The
    same measurement on the ``\operatorname{St}(3,1)`` sphere of `manifold_linesearch_tests.jl` gives
    4.5%, 18%, 72% and 288% at ``\alpha = 0.25, 0.5, 1, 2``.

The retraction used to separate the two polynomial line searches on the SVD problem, where they left
the manifold under [`Cayley`](@ref) for every optimizer method and stayed on it under
[`Geodesic`](@ref). That was issue A1b, and the exact differential closed only one of its four cases:
the cause is the size of the step those searches extrapolate to, not the slope they extrapolate from.
Bounding the step closes it — see [`DEFAULT_STEP_CEILING`](@ref) — and the retraction no longer
separates them. It was the *amplifier* rather than the cause, which is what the `check` table further
down measures.

Cost no longer separates them the way it once did. [`cayley`](@ref) finishes with a product of two
``N\times{}N`` matrices, which is ``O(N^3)``, where [`geodesic`](@ref) only assembles
``\mathbb{I} + B'\mathfrak{A}(X)(B'')^T`` at ``O(N^2n)``, so since 0.2.0 [`Geodesic`](@ref) is the
cheaper of the two for ``N \gtrsim 50``; the table under [What they cost](@ref) has the figures.

## The exponential needs an algorithm

Exponentiating a full ``N\times{}N`` matrix would throw away the sparsity of the lift. The
factorisation avoids it: the exponential of a product taken in this order is

```math
\exp\left(B'(B'')^T\right) = \mathbb{I} + B'\,\mathfrak{A}(X)\,(B'')^T,
\qquad
\mathfrak{A}(X) = \sum_{k=1}^\infty \frac{X^{k-1}}{k!},
```

so the whole computation reduces to one ``2n\times{}2n`` matrix function. ``\mathfrak{A}`` is the
function usually written ``\varphi_1(X) = \left(\exp(X) - \mathbb{I}\right)X^{-1}``, though it is
defined by the series and is perfectly regular at a singular ``X``.

Evaluating ``\mathfrak{A}`` by summing that series is the obvious thing to do and it is what every
version of this package up to 0.2.0 did. It is also wrong for any but a small argument, and the
argument here is not small. ``X``'s lower-left block is ``\tfrac{1}{4}A^2 - B^TB``, so

```math
\|X\| \approx \tfrac{1}{4}\|\bar{B}\|^2
\qquad\text{while}\qquad
\rho(X) \approx \|\bar{B}\|,
```

because the eigenvalues of ``X`` are the nonzero — purely imaginary — eigenvalues of the skew matrix
``\bar{B}``. A norm quadratically larger than the spectral radius is a strongly non-normal matrix,
and on such an argument the terms of the series cancel catastrophically: at
``\|\bar{B}\| \approx 79`` the partial sum reaches ``2.5\cdot10^{18}`` where the result is of order
one. Stopping the summation when a *term* falls below `eps` then leaves a relative error of
``\varepsilon\|\mathfrak{A}(X)\|`` rather than ``\varepsilon``, and the retracted point is not on the
manifold in any sense. That the direct series is not a method for the matrix exponential is a very
old observation [moler2003nineteen](@cite); what is specific here is that the factorisation makes the
argument *worse* than the matrix one started with.

The remedy is a choice, and [`Geodesic`](@ref) makes it one the caller can see:

```julia
Geodesic(ScaledSquaring())   # the default, and `Geodesic()`
Geodesic(AugmentedPade())
Geodesic(ProjectedSkew())
Geodesic(TaylorSeries())     # the pre-0.2.0 behaviour; not a usable retraction
```

All four are subtypes of [`AbstractExponentialAlgorithm`](@ref) and all four return the exponential
map, so the one-parameter subgroup property above holds for every one of them. What follows is what
each does and what it trades.

## `ScaledSquaring`

[`ScaledSquaring`](@ref) is the default. The series is only inaccurate for a large argument, so halve
the argument until it is small, sum the series there, and undo the halving by squaring — the standard
remedy for a matrix exponential [higham2005scaling, higham2008functions](@cite), and what
`Base.exp` itself does.

The one thing that needs care is that squaring must not cost ``O(N^3)``. It does not, because the
low-rank form is closed under squaring:

```math
\left(\mathbb{I} + B'W(B'')^T\right)^2 = \mathbb{I} + B'\left(2W + WXW\right)(B'')^T,
```

so one squaring of the assembled exponential is one application of ``W \mapsto 2W + WXW`` at
``2n\times{}2n``, and no ``N\times{}N`` matrix is ever formed, let alone squared. With ``s`` chosen so
that ``\|X\|_1/2^s \leq \theta``, the algorithm is `s` small matrix products on top of a series that
now converges in a handful of terms. That makes it *cheaper* than summing the unscaled series, not
merely more accurate — by 1.7× at ``N = 200``, ``n = 10`` and 4.6× at ``N = 500``, ``n = 50``.

The threshold `θ` is the algorithm's one parameter — positional, `ScaledSquaring(0.5)`, and defaulted
to `0.5` — and it barely matters: at ``\|\bar{B}\| \approx 155`` every ``\theta \in [0.125, 4]`` — a
32-fold range — gives a `check` between ``9.9\cdot10^{-15}`` and ``5.0\cdot10^{-14}`` and a forward
error between ``6.4\cdot10^{-15}`` and ``8.2\cdot10^{-15}``. That sweep is [measured at build
time](@ref "The threshold `θ` needs no tuning") below; there is no reason to tune it.

**Advantages.** The fastest, or tied fastest, of the four at every size measured below, and as close
to `exp(Matrix(B))` as [`AugmentedPade`](@ref), which is as close as anything here gets. And —
because it uses nothing but matrix products and norms — the only usable algorithm that runs unchanged
on a `KernelAbstractions` GPU backend, which is why it is the default. Keeping that property is also
why the norm is taken by [`GeometricOptimizers.opnorm₁`](@ref) rather than by
`LinearAlgebra.opnorm(X, 1)`: the latter is a scalar-indexing double loop, and scalar indexing is
exactly what a GPU array cannot serve.

**Disadvantages.** Its orthogonality is the outcome of an arithmetic cancellation rather than a
structural property, so `check` does drift upwards with the size of the lift — from ``10^{-15}`` to
around ``7\cdot10^{-14}`` over the sweep below, and considerably further in `Float32`. Only
[`ProjectedSkew`](@ref) avoids that drift; [`Cayley`](@ref) has more of it. And it takes about twice
the squarings it needs:

!!! note "The halving count is loose"
    ``s`` is taken from the norm, ``s = \lceil\log_2(\|X\|_1/\theta)\rceil``, and
    ``\|X\| \approx \|\bar{B}\|^2/4`` — so ``s \approx 2\log_2\|\bar{B}\|`` where
    ``\log_2\|\bar{B}\|`` would do, since the spectral radius is only ``\approx\|\bar{B}\|``. Each
    squaring amplifies the error, so this costs both time and accuracy. It is left alone because the
    tighter bound needs the spectral radius, and an eigenvalue computation would forfeit precisely
    the freedom from dense LAPACK that makes this the default algorithm.

## `AugmentedPade`

[`AugmentedPade`](@ref) evaluates ``\mathfrak{A}`` as a block of a larger *ordinary* exponential. For
the ``4n\times{}4n`` augmented matrix,

```math
\exp\begin{pmatrix} X & \mathbb{I} \\ \mathbb{O} & \mathbb{O} \end{pmatrix}
= \begin{pmatrix} \exp(X) & \mathfrak{A}(X) \\ \mathbb{O} & \mathbb{I} \end{pmatrix},
```

which is the standard device for getting a ``\varphi`` function out of an exponential routine
[sidje1998expokit, higham2008functions](@cite). One call to `Base.exp` therefore returns
``\mathfrak{A}(X)`` in its upper-right block. That hands the numerics to Julia's own exponential — a
degree-13 Padé approximant with its own scaling and squaring [higham2005scaling,
almohy2010new](@cite) — at the cost of exponentiating a matrix four times the size and discarding
three quarters of it.

**Advantages.** It introduces no new numerics at all. Everything delicate is done by the most
heavily exercised matrix-exponential implementation available, which is why it is the reference the
other two are tested against in `test/retractions/exponential_accuracy.jl`. Accuracy is the same
order as [`ScaledSquaring`](@ref)'s.

**Disadvantages.** Three quarters of the work is thrown away, so the ``\mathfrak{A}`` call itself is
about twice as expensive as [`ScaledSquaring`](@ref)'s — though much less than twice once the
``N\times{}N`` assembly around it is counted. In `Float32` it and [`ScaledSquaring`](@ref) trade last
place across the sweep below — it is the worse of the two on most rows, [`ScaledSquaring`](@ref) at
the very top — and at the large lifts both are an order of magnitude behind
[`ProjectedSkew`](@ref). And `Base.exp` on a dense matrix needs LAPACK:

!!! warning "CPU only"
    Neither this nor [`ProjectedSkew`](@ref) runs on a GPU backend. Use [`ScaledSquaring`](@ref)
    there.

## `ProjectedSkew`

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
numbers, and a unitary matrix — rather than by cancellation.

**Advantages.** It is the only algorithm whose `check` does not degrade with the size of the lift.
Over the sweep below it stays between ``2\cdot10^{-15}`` and ``5\cdot10^{-15}`` from
``\|\bar{B}\| \approx 6`` to ``\|\bar{B}\| \approx 770``, where the other two drift from ``10^{-15}``
to around ``7\cdot10^{-14}``. The gap is widest in `Float32`, where the other two are at the mercy of
the format: over the same sweep their `check` climbs into the ``10^{-5}``s while this stays at a few
``10^{-6}`` from one end to the other. That is the case for choosing it — a long `Float32` run, where
the departure from the manifold accumulates over thousands of steps and staying on the manifold
matters more than agreeing with the exponential to the last bit.

**Disadvantages.** It usually has the largest forward error of the three against `exp(Matrix(B))` —
up to about 4.5× [`ScaledSquaring`](@ref)'s, and largest at all but the top of the sweep measured
below. It needs a `qr` and an `eigen` instead of matrix
products, which costs 1.2×–1.9× over the sizes measured below and rules out a GPU backend. And
because it bypasses ``\mathfrak{A}``, it is the one algorithm that specialises
[`geodesic`](@ref) directly rather than supplying a method of [`GeometricOptimizers.𝔄`](@ref) — worth
knowing if you call ``\mathfrak{A}`` yourself, since `𝔄(X, ProjectedSkew())` does not exist.

## `TaylorSeries`

[`TaylorSeries`](@ref) sums the series for ``\mathfrak{A}`` directly, without scaling, terminating
when a term falls below `eps`. It is the behaviour of every version of this package up to 0.2.0.

!!! danger "This is not a usable retraction"
    It is retained only so that the regression is reproducible from the test suite and so the
    working algorithms have a baseline to be compared against. Its column in the first table
    below is what it does: already at ``10^{-12}`` by ``\|\bar{B}\| \approx 18``, off the manifold
    by any standard at ``37``, meaningless at ``79``, and overflowed to `NaN` by ``767``. Do not
    select it.

Two things about it are worth recording rather than merely deprecating. The failure is *silent*: an
optimizer using it takes a step, gets a matrix back, and nothing anywhere reports that the matrix is
not on the manifold — which is why the defect survived until [`check`](@ref) was made generic over
[`Manifold`](@ref) instead of being defined for [`StiefelManifold`](@ref) alone. And the obvious
first fix does not work: making the termination test relative to the partial sum rather than absolute
was measured to change *none* of the numbers below, at any lift norm. The loss is the cancellation
inside the sum, not the point at which the summation stops. Scaling the argument down is the only
thing that helps, which is [`ScaledSquaring`](@ref) — and that is also 1.7× to 4.6× *faster* here,
because the scaled series converges in a handful of terms where the unscaled one grinds through
hundreds.

## Using them

None of these types is exported, so import the ones you use:

```jldoctest retraction-usage
using GeometricOptimizers
using GeometricOptimizers: Geodesic, Cayley, ScaledSquaring, AugmentedPade, ProjectedSkew, check
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

`Geodesic()` is `Geodesic(ScaledSquaring())`, and `ScaledSquaring(θ)` takes the scaling threshold if
you want to override the default `0.5` — which, per the sweep below, you do not need to:

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

and [`GeometricOptimizers.𝔄`](@ref) can be called on a bare matrix, which is the level at which three
of the four algorithms are implemented:

```jldoctest retraction-usage
using GeometricOptimizers: 𝔄
import Random
Random.seed!(123)

X = randn(6, 6)

isapprox(𝔄(X, ScaledSquaring()), 𝔄(X, AugmentedPade()); rtol = 1e-12)

# output

true
```

Where the algorithms part company is a large step, and that is the whole reason the default changed:

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

## What they cost and how accurate they are

Everything in this section other than the timings is recomputed when this page is built, so the
figures are those of the version of the package the documentation was built from rather than a quote
that can go stale. `scripts/retraction_accuracy.jl` produces the same tables — including the
timings — from the command line.

```@setup retractions
using GeometricOptimizers
using GeometricOptimizers: geodesic, cayley, check, ScaledSquaring, AugmentedPade, ProjectedSkew, TaylorSeries
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

table(["‖B̄‖", "`ScaledSquaring`", "`AugmentedPade`", "`ProjectedSkew`", "`TaylorSeries`", "`Cayley`"],
      [[fixed(norm(Matrix(B))),
        sci(check(geodesic(B, ScaledSquaring()))),
        sci(check(geodesic(B, AugmentedPade()))),
        sci(check(geodesic(B, ProjectedSkew()))),
        sci(check(geodesic(B, TaylorSeries()))),
        sci(check(cayley(B)))] for B in lifts])
```

Every column but `TaylorSeries`'s stays at round-off, and only [`ProjectedSkew`](@ref)'s is *level*.
The other three — [`Cayley`](@ref) included, and it is the one that drifts furthest — grow by two to
three orders of magnitude across the sweep, because their orthogonality is an arithmetic outcome
while [`ProjectedSkew`](@ref)'s is structural. Round-off at ``\|\bar{B}\| \approx 770`` is still
round-off, so this separates the algorithms without condemning any of the three usable ones.

### Agreeing with the exponential

Relative distance to `exp(Matrix(B))`, i.e. to the exponential of the full ``N\times{}N`` lift. This
is a different question from the one above — a retraction that re-orthonormalised its result would
have a perfect `check` and be wrong here — and the test suite asserts both.

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`AugmentedPade`", "`ProjectedSkew`"],
      [begin
           reference = exp(Matrix(B))
           err(algorithm) = norm(Matrix(geodesic(B, algorithm)) - reference) / norm(reference)
           [fixed(norm(Matrix(B))), sci(err(ScaledSquaring())),
            sci(err(AugmentedPade())), sci(err(ProjectedSkew()))]
       end for B in lifts])
```

All three grow slowly with the norm of the lift, and the ordering is roughly the reverse of the
previous table: [`ProjectedSkew`](@ref) is the furthest from the exponential at all but the largest
of these norms. [`ScaledSquaring`](@ref) and [`AugmentedPade`](@ref) are indistinguishable — each is
the closer of the two on half the rows — so the trade is between the pair of them and
[`ProjectedSkew`](@ref): one is orthogonal by construction, the others agree with `exp` more closely.
The three converge again at the top of the sweep, where the reference `exp(Matrix(B))` is itself no
more accurate than what is being measured against it.

### `Float32`

The same `check`, in the format the MNIST experiment of
[Optimization on Homogeneous Spaces](@ref) actually runs in. Nothing can do better than about
``10^{-6}`` here, but the three do not degrade alike.

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`AugmentedPade`", "`ProjectedSkew`"],
      [[fixed(norm(Matrix(B))),
        sci(check(geodesic(B, ScaledSquaring()))),
        sci(check(geodesic(B, AugmentedPade()))),
        sci(check(geodesic(B, ProjectedSkew())))] for B in sweep(Float32)])
```

[`ProjectedSkew`](@ref) is flat here too, an order of magnitude below the others at the top of the
sweep — where which of the other two is worse depends on the lift, and neither is close. In a
`Float64` run that difference is academic; in a `Float32` one over thousands of steps it is the
reason to choose it.

### The threshold `θ` needs no tuning

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

Both columns move by less than a factor of six across the whole range, and not monotonically. The
default of `0.5` sits in that band; nothing in the measurement singles it out, which is the point.

### What they cost

Unlike everything above, these are timings and therefore machine-dependent, so they are quoted rather
than measured at build time. One retraction, `minimum` of 50 repetitions, a single BLAS thread, on an
Apple M-series laptop; `julia --project=. scripts/retraction_accuracy.jl` reproduces them on yours.
Milliseconds:

| ``N``, ``n`` | 10, 2 | 20, 3 | 50, 5 | 100, 5 | 200, 10 | 500, 10 | 500, 50 | 1000, 20 |
|---|---|---|---|---|---|---|---|---|
| `Geodesic(ScaledSquaring())` | 0.003 | 0.005 | 0.012 | 0.022 | 0.097 | 0.449 | 3.16 | 2.61 |
| `Geodesic(AugmentedPade())` | 0.003 | 0.006 | 0.021 | 0.027 | 0.130 | 0.474 | 6.18 | 2.83 |
| `Geodesic(ProjectedSkew())` | 0.005 | 0.009 | 0.023 | 0.033 | 0.140 | 0.525 | 4.16 | 3.07 |
| `Geodesic(TaylorSeries())` | 0.003 | 0.006 | 0.019 | 0.040 | 0.164 | 0.550 | 14.6 | 3.60 |
| `Cayley()` | 0.002 | 0.004 | 0.015 | 0.064 | 0.388 | 5.16 | 6.54 | 39.2 |

[`ScaledSquaring`](@ref) is the fastest, or tied fastest, at every size. [`ProjectedSkew`](@ref)
costs 1.2×–1.9× of it, never more — a QR and an eigendecomposition of a ``2n\times{}2n`` matrix are
not expensive things — and [`AugmentedPade`](@ref) is between the two except at ``n = 50``, where
exponentiating a ``200\times{}200`` augmented matrix begins to tell. [`Cayley`](@ref) is level with
the exponential up to ``N \approx 50`` and loses by a factor of 15 by ``N = 1000``, which is the
``O(N^3)`` against ``O(N^2n)`` of the previous section.

## Choosing one

For the retraction: **[`Geodesic`](@ref) unless you have a reason for [`Cayley`](@ref)**. It is the
exponential map and the cheaper of the two at any size worth worrying about, and it survives an
implausibly large step with a `check` an order of magnitude smaller — which is what issue A1b turned
on. That argument is weaker now than it was: bounding the step ([`DEFAULT_STEP_CEILING`](@ref)) means
an implausibly large step is no longer taken under either retraction, so the tolerance `Geodesic` has
for one is insurance rather than a live difference. A derivative-based line search is exact under
either since 0.2.0, so that is no longer part of the argument. [`Cayley`](@ref) remains the package default, needs no matrix function at all, and
is unconditionally stable.

For the algorithm: **[`ScaledSquaring`](@ref), i.e. the default, unless one of the two special cases
applies.**

| | choose it when | at the price of |
|---|---|---|
| [`ScaledSquaring`](@ref) | almost always; it is the default | `check` drifting up with the size of the lift |
| [`ProjectedSkew`](@ref) | staying on the manifold matters more than the last bit of the exponential — a long `Float32` run, where `check` accumulates over thousands of steps | 1.2×–1.9× the cost, usually the largest forward error, CPU only |
| [`AugmentedPade`](@ref) | you want a second opinion from an implementation that introduces no numerics of its own | roughly 2× the cost of the ``\mathfrak{A}`` call, no better than [`ScaledSquaring`](@ref) on accuracy, CPU only |
| [`TaylorSeries`](@ref) | never; it exists so the pre-0.2.0 regression stays reproducible | leaving the manifold silently above ``\Vert\bar{B}\Vert \approx 50`` |

And on a GPU backend the question does not arise: [`ScaledSquaring`](@ref) is the only *usable*
algorithm free of dense LAPACK — [`TaylorSeries`](@ref) is too, and is no more a retraction there
than anywhere else — and that is the reason it is the default.

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
Pages = ["retractions.md"]
Canonical = false
```
