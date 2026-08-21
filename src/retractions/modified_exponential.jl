update_algorithm = "while norm(Aⁿ) > ε
mul!(A_temp, Aⁿ, A)
Aⁿ .= A_temp
rmul!(Aⁿ, T(inv(n)))

𝔄A += Aⁿ
n += 1
end"

@doc (raw"""
    𝔄(A)

Compute ``\mathfrak{A}(A) := \sum_{n=1}^\infty \frac{1}{n!} (A)^{n-1}.``

# Implementation

This uses a Taylor expansion that iteratively adds terms with

```julia
""" * update_algorithm * raw"""

```

until the norm of `Aⁿ` becomes smaller than machine precision.
The counter `n` in the above algorithm is initialized as `2`
The matrices `Aⁿ` and `𝔄` are initialized as the identity matrix.

!!! warning "Only accurate for a small argument"
    The series converges for every `A` but cancels catastrophically for ``\|A\| \gg 1``, so this
    method alone is not a usable exponential — see [`TaylorSeries`](@ref) for what it does at a
    large argument. It is used here as the inner summation of [`ScaledSquaring`](@ref), which calls
    it only on an argument that has been halved until its norm is below `θ`. Reach for it directly
    only if you know the argument is small.
""")
function 𝔄(A::AbstractMatrix)
    T = eltype(A)
    Aⁿ = one(A)
    𝔄A = one(A)
    A_temp = zero(A)
    n = 2
    ε = eps(T)
    while norm(Aⁿ) > ε
        LinearAlgebra.mul!(A_temp, Aⁿ, A)
        Aⁿ .= A_temp
        LinearAlgebra.rmul!(Aⁿ, T(inv(n)))

        𝔄A += Aⁿ
        n += 1
    end
    𝔄A
end

@doc raw"""
    opnorm₁(X)

The induced 1-norm of `X`, i.e. its largest absolute column sum, as a reduction.

`LinearAlgebra.opnorm(X, 1)` is the natural spelling and is *not* used, because
`LinearAlgebra.opnorm1` is a double loop over `X[i, j]`. Scalar indexing is precisely what an array
on a GPU backend cannot serve, and being free of it is the reason [`ScaledSquaring`](@ref) is the
default algorithm — so the one norm that algorithm takes has to be expressible as `sum` and
`maximum`, which every `KernelAbstractions` backend specializes.

The two agree to a few `eps`, not bitwise: `opnorm1` accumulates each column sequentially in at
least `Float64`, whereas `sum` is pairwise and accumulates in `eltype(X)`. The value is only ever
used to pick the number of halvings `s = ⌈log₂(‖X‖₁/θ)⌉`, so a difference of an ulp can at most
shift `s` by one, and only for an argument that lands exactly on a power of two.
"""
opnorm₁(X::AbstractMatrix) = isempty(X) ? zero(real(eltype(X))) : maximum(sum(abs, X; dims=1))

@doc raw"""
    𝔄(X, algorithm)

Compute ``\mathfrak{A}(X)`` with the requested [`AbstractExponentialAlgorithm`](@ref).

All algorithms compute the same function and differ only in accuracy at a large `X`, in cost, and in
which backends they run on. See [`AbstractExponentialAlgorithm`](@ref) for the comparison and
[`ScaledSquaring`](@ref), which is the default.

# Examples

The four agree wherever the unscaled series is still accurate, and only three of them agree beyond
that:

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: 𝔄, ScaledSquaring, AugmentedPade, TaylorSeries
import Random
Random.seed!(123)

X = randn(6, 6)

isapprox(𝔄(X, ScaledSquaring()), 𝔄(X, AugmentedPade()); rtol = 1e-12) &&
    isapprox(𝔄(X, ScaledSquaring()), 𝔄(X, TaylorSeries()); rtol = 1e-12)

# output

true
```
"""
𝔄(X::AbstractMatrix, ::TaylorSeries) = 𝔄(X)

function 𝔄(X::AbstractMatrix, algorithm::ScaledSquaring)
    # `X` is halved `s` times so that the Taylor series is summed on an argument of norm ≤ θ, where
    # it converges in a handful of terms and does not cancel. Initially
    # `exp(B̂B̄ᵗ/2^s) = I + B̂(𝔄(X/2^s)/2^s)B̄ᵗ`. Squaring this represented exponential stays
    # low-rank:
    #
    #     (I + B̂WB̄ᵗ)² = I + B̂(2W + WXW)B̄ᵗ,
    #
    # so each recovery step is `W ↦ 2W + WXW` at 2n × 2n, with the original `X`. After `s` steps
    # `W = 𝔄(X)`. Nothing is ever squared at N × N.
    nrm = opnorm₁(X)
    s = nrm > algorithm.θ ? ceil(Int, log2(nrm / algorithm.θ)) : 0
    scale = eltype(X)(2)^s

    W = 𝔄(X / scale) / scale
    for _ in 1:s
        W = 2 * W + W * X * W
    end

    W
end

function 𝔄(X::AbstractMatrix, ::AugmentedPade)
    # exp([X I; 0 0]) == [exp(X) 𝔄(X); 0 I], so Julia's Padé-based, scaling-and-squaring matrix
    # exponential returns `𝔄(X)` in the upper-right block.
    m = size(X, 1)
    T = eltype(X)
    augmented = [X one(X); zeros(T, m, m) zeros(T, m, m)]

    exp(augmented)[1:m, (m+1):(2m)]
end

@doc raw"""
    𝔄(B̂, B̄)
    𝔄(B̂, B̄, algorithm)

Compute ``\mathfrak{A}(B', B'') := \sum_{n=1}^\infty \frac{1}{n!} ((B'')^TB')^{n-1}.``

This expression has the property ``\mathbb{I} +  B'\mathfrak{A}(B', B'')(B'')^T = \exp(B'(B'')^T).``

Note that the argument ``(B'')^TB'`` is only ``2n\times{}2n``, so this is where the cost of a
retraction is set by ``n`` rather than by ``N``.

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: 𝔄
import Random
Random.seed!(123)

B = rand(StiefelLieAlgHorMatrix, 10, 2)
B̂ = hcat(vcat(.5 * B.A, B.B), vcat(one(B.A), zero(B.B)))
B̄ = hcat(vcat(one(B.A), zero(B.B)), vcat(-.5 * B.A, -B.B))

one(B̂ * B̄') + B̂ * 𝔄(B̂, B̄) * B̄' ≈ exp(Matrix(B))

# output

true
```
"""
function 𝔄(B̂::AbstractMatrix, B̄::AbstractMatrix)
    𝔄(B̄' * B̂)
end

function 𝔄(B̂::AbstractMatrix, B̄::AbstractMatrix, algorithm::AbstractExponentialAlgorithm)
    𝔄(B̄' * B̂, algorithm)
end

@doc raw"""
    𝔄exp(B̂, B̄, algorithm = ScaledSquaring())

Compute ``\exp(B'(B'')^T)`` as ``\mathbb{I} + B'\mathfrak{A}(B', B'')(B'')^T``, i.e. the identity
[`𝔄`](@ref) exists for, packaged as the exponential it computes.

This is what makes a geodesic retraction cheap: the argument handed to [`𝔄`](@ref) is
``(B'')^TB'``, which is ``2n\times{}2n``, so the cost is set by ``n`` and not by ``N`` even though
the result is ``N\times{}N``. [`geodesic`](@ref) computes this same product inline — it needs to wrap
the result in `manifold_type(B)` and to take the lift factors apart itself — so this is for callers
that want the matrix exponential of a low-rank product on its own.

`algorithm` is forwarded to [`𝔄`](@ref), which supplies [`TaylorSeries`](@ref),
[`ScaledSquaring`](@ref) and [`AugmentedPade`](@ref). [`ProjectedSkew`](@ref) is *not* among them: it
is a [`geodesic`](@ref)-level algorithm with its own branch there and no `𝔄` method, so
`𝔄exp(B̂, B̄, ProjectedSkew())` fails inside `𝔄` exactly as `𝔄(B̂, B̄, ProjectedSkew())` does.

# Implementation

The default is [`ScaledSquaring`](@ref) and not the unscaled series, for the reason given under
[`geodesic`](@ref): the series cancels catastrophically once ``\|\bar{B}\| \gtrsim 50``, which is not
a regime a function that presents itself as an exponential may quietly get wrong. Relative error
against `exp(Matrix(B))` for `B = scale * rand(StiefelLieAlgHorMatrix, 10, 2)`, as
`test/retractions/exponential_accuracy.jl` draws it:

| `scale` | 1 | 10 | 50 | 100 |
| --- | --- | --- | --- | --- |
| ``\|\bar{B}\|`` | 3.8 | 36.3 | 145.8 | 324.9 |
| [`TaylorSeries`](@ref) | 5.3e-16 | 1.1e-7 | 1.8e24 | 1.7e79 |
| [`ScaledSquaring`](@ref) | 4.0e-16 | 1.9e-15 | 7.8e-15 | 1.9e-14 |

Note that this differs from `𝔄(B̂, B̄)`, which has no `algorithm` argument at all and is the
unscaled series. That one is a kernel, and [`𝔄`](@ref) carries the warning; this one is a result.

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: 𝔄exp
import Random
Random.seed!(123)

B = rand(StiefelLieAlgHorMatrix, 10, 2)
B̂ = hcat(vcat(.5 * B.A, B.B), vcat(one(B.A), zero(B.B)))
B̄ = hcat(vcat(one(B.A), zero(B.B)), vcat(-.5 * B.A, -B.B))

𝔄exp(B̂, B̄) ≈ exp(Matrix(B))

# output

true
```
"""
function 𝔄exp(B̂::AbstractMatrix, B̄::AbstractMatrix, algorithm::AbstractExponentialAlgorithm=ScaledSquaring())
    I + B̂ * 𝔄(B̂, B̄, algorithm) * B̄'
end
