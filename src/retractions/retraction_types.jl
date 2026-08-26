@doc raw"""
    AbstractRetraction

Supertype of the retractions, i.e. of the maps that turn an element of a horizontal Lie algebra back
into a point of the manifold.

A retraction is how every step in this package is taken: an [`OptimizerMethod`](@ref) produces a
direction ``B`` in the horizontal component of ``\mathfrak{g}``, and
``\Lambda \mapsto \Lambda\cdot\mathrm{retract}(B)`` moves the global section — and with it the point —
without ever leaving the manifold. An instance is passed to [`Optimizer`](@ref) as
`retraction = Cayley()`; the default is [`Cayley`](@ref).

Concrete subtypes are [`Geodesic`](@ref) and [`Cayley`](@ref). A new one has to supply
`retraction(::NewRetraction, x)`; the callable form `R(x)` and everything in `update_section!` then
follow.

For every manifold layer one has to specify a retraction that takes the layer and elements of the
(global) tangent space.
"""
abstract type AbstractRetraction end

@doc raw"""
    Cayley <: AbstractRetraction

The Cayley transform as a retraction, and the default one.

```math
\mathrm{Cayley}(B) = \left(\mathbb{I} - \frac{1}{2}B\right)^{-1}\left(\mathbb{I} + \frac{1}{2}B\right)
```

For ``B`` in the horizontal component this maps into the Lie group, so the retracted point stays on
the manifold to round-off. [`cayley`](@ref) never forms the ``N\times{}N`` inverse: it factors
``B = B'(B'')^T`` into two ``N\times{}2n`` matrices with [`lift_factors`](@ref) and inverts a
``2n\times{}2n`` matrix instead.

!!! note "Cost depends on the matrix dimensions"
    `cayley` finishes with a product of two ``N\times{}N`` matrices, which is ``O(N^3)``, whereas
    `geodesic` assembles ``\mathbb{I} + B'\mathfrak{A}(X)(B'')^T`` at ``O(N^2n)``. One benchmark of
    `ScaledSquaring` against `Cayley` gives:

    | ``N``, ``n`` | 20, 3 | 50, 5 | 100, 5 | 200, 10 | 500, 10 | 1000, 20 |
    |---|---|---|---|---|---|---|
    | `Geodesic` | `0.005 ms` | `0.014 ms` | `0.023 ms` | `0.087 ms` | `0.40 ms` | `2.5 ms` |
    | `Cayley` | `0.004 ms` | `0.016 ms` | `0.056 ms` | `0.36 ms` | `4.9 ms` | `39 ms` |

    The two are comparable at the smaller sizes in this table, while the dense product in `Cayley`
    dominates at larger `N` — a factor of 15 by ``N = 1000``. Timings are machine-dependent; see
    [What they cost](@ref) for the setup and the full table. `Cayley` remains useful — it is
    unconditionally stable and needs no matrix function at all — but cost is no longer a reason to
    prefer it.

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: Cayley, check

Y = rand(StiefelManifold, 5, 3)
B = GeometricOptimizers.global_rep(GlobalSection(Y), rand(5, 3))
check(Cayley()(B)) < 1e-14        # the retracted element is still on the manifold

# output

true
```

!!! warning "It is a retraction, not the exponential map"
    ``\alpha \mapsto \mathrm{Cayley}(\alpha{}B)`` is *not* a one-parameter subgroup — only
    [`Geodesic`](@ref) is. It agrees with the geodesic to first order at ``\alpha = 0`` and departs
    from it as the step grows, so the generator of the curve's velocity turns with ``\alpha`` instead
    of staying ``B``. [`retraction_differential`](@ref) is what supplies it, and with it
    [`trial_slope`](@ref) is the exact derivative of a line search's merit under either retraction.
    Before 0.2.0 the slope was paired against ``B`` regardless, which made it only a first-order
    approximation under `Cayley`.

See [`cayley`](@ref) for the implementation and [`Geodesic`](@ref) for the alternative.
"""
struct Cayley <: AbstractRetraction end

@doc raw"""
    Geodesic(algorithm = ScaledSquaring()) <: AbstractRetraction

The exponential map as a retraction, i.e. the *true* geodesic of the manifold.

```math
\mathrm{Geodesic}(B) = \exp(B)
```

Because this is the matrix exponential, ``\alpha \mapsto \mathrm{Geodesic}(\alpha{}B)`` is a
one-parameter subgroup: it follows the geodesic through the point in the direction ``B`` exactly, and
``\mathrm{Geodesic}((\alpha+\beta)B) = \mathrm{Geodesic}(\alpha{}B)\mathrm{Geodesic}(\beta{}B)``. That
is the property [`Cayley`](@ref) lacks, and the reason a derivative-based line search is exact here.

[`geodesic`](@ref) exploits the sparsity of a horizontal lift rather than exponentiating the full
``N\times{}N`` matrix: the only matrix function it evaluates is on a ``2n\times{}2n`` argument. Since
0.2.0 that also makes it the cheaper of the two for ``N \gtrsim 50`` — see the note on
[`Cayley`](@ref).

`algorithm` selects how the exponential is evaluated. All of them compute the same map — the choice
is one of accuracy at a large lift, cost, and backend support — and the default
[`ScaledSquaring`](@ref) is the one to use unless there is a reason not to. See
[`AbstractExponentialAlgorithm`](@ref) for the comparison.

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: Geodesic, check

Y = rand(StiefelManifold, 5, 3)
B = GeometricOptimizers.global_rep(GlobalSection(Y), rand(5, 3))
check(Geodesic()(B)) < 1e-14      # the retracted element is still on the manifold

# output

true
```

A large lift is where the algorithms part company, and the reason the default is what it is:

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: Geodesic, ScaledSquaring, TaylorSeries, check
import Random
Random.seed!(1234)

B = 60 * rand(StiefelLieAlgHorMatrix, 20, 3)      # ‖B̄‖ ≈ 393

check(Geodesic()(B)) < 1e-12, check(Geodesic(TaylorSeries())(B)) < 1e-12

# output

(true, false)
```

See [`geodesic`](@ref) for the implementation and [`Cayley`](@ref) for the rational alternative, and
[Exponential Algorithms](@ref) for the choice of `algorithm`.
"""
struct Geodesic{AT<:AbstractExponentialAlgorithm} <: AbstractRetraction
    algorithm::AT
end

Geodesic() = Geodesic(ScaledSquaring())
