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
``B = B'(B'')^T`` into two ``N\times{}2n`` matrices and inverts a ``2n\times{}2n`` matrix instead,
which is what makes it cheaper than [`Geodesic`](@ref) when ``n \ll N``.

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
    from it as the step grows, which is why [`trial_slope`](@ref) is the exact derivative of a line
    search's merit under `Geodesic` but only a first-order one under `Cayley`.

See [`cayley`](@ref) for the implementation and [`Geodesic`](@ref) for the alternative.
"""
struct Cayley <: AbstractRetraction end

@doc raw"""
    Geodesic <: AbstractRetraction

The exponential map as a retraction, i.e. the *true* geodesic of the manifold.

```math
\mathrm{Geodesic}(B) = \exp(B)
```

Because this is the matrix exponential, ``\alpha \mapsto \mathrm{Geodesic}(\alpha{}B)`` is a
one-parameter subgroup: it follows the geodesic through the point in the direction ``B`` exactly, and
``\mathrm{Geodesic}((\alpha+\beta)B) = \mathrm{Geodesic}(\alpha{}B)\mathrm{Geodesic}(\beta{}B)``. That
is the property [`Cayley`](@ref) lacks, and the reason a derivative-based line search is exact here.

[`geodesic`](@ref) exploits the sparsity of a horizontal lift rather than exponentiating the full
``N\times{}N`` matrix, but it is still the more expensive of the two.

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

See [`geodesic`](@ref) for the implementation and [`Cayley`](@ref) for the cheaper alternative.
"""
struct Geodesic <: AbstractRetraction end
