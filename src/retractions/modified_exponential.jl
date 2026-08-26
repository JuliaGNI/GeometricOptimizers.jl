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
    The series converges for every `A`, but cancellation can make direct summation inaccurate for
    ``\|A\| \gg 1`` — see [`TaylorSeries`](@ref) for what it does at a large argument. This method
    is therefore intended as a small-argument kernel. It is used by [`ScaledSquaring`](@ref) only
    after the argument has been divided until its norm is below `θ`. Reach for it directly only if
    you know the argument is small.
""")
function 𝔄(A::AbstractMatrix)
    T = eltype(A)
    Aⁿ = unit_matrix(A)
    𝔄A = copy(Aⁿ)
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
`LinearAlgebra.opnorm1` is a double loop over `X[i, j]`. Scalar indexing is precisely what an array on
a GPU backend cannot serve, and being free of it is why [`ScaledSquaring`](@ref) and
[`NativePade`](@ref) have no dense-LAPACK dependency at all — so the one norm they take has to be
expressible as `sum` and `maximum`. Accelerator execution still depends on the array backend's support
for those reductions.

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

The five agree wherever the unscaled series is still accurate, and only four of them agree beyond
that:

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: 𝔄, ScaledSquaring, NativePade, AugmentedPade, TaylorSeries
import Random
Random.seed!(123)

X = randn(6, 6)

isapprox(𝔄(X, ScaledSquaring()), 𝔄(X, NativePade()); rtol = 1e-12) &&
    isapprox(𝔄(X, NativePade()), 𝔄(X, AugmentedPade()); rtol = 1e-12) &&
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

@doc raw"""
    _native_pade_polynomials(X, 𝕀)

Evaluate the degree-6 numerator ``p_6(X)`` and denominator ``q_6(X)`` used by [`NativePade`](@ref).

If ``P^{\exp}_7/Q^{\exp}_6`` is the ``[7/6]`` Padé approximant of the exponential, then

```math
p_6(z)=\frac{P^{\exp}_7(z)-Q^{\exp}_6(z)}{z},
\qquad
q_6(z)=Q^{\exp}_6(z),
```

so ``q_6(X)^{-1}p_6(X)`` agrees with ``\mathfrak{A}(X)=\varphi_1(X)`` through the ``X^{12}`` term.
The implementation shares ``X^2`` and ``X^4`` between the two polynomials and groups the remaining
terms to avoid forming every matrix power separately. `𝕀` must be the multiplicative identity with
the same size, element type, and backend as `X`.

This is an internal kernel; [`NativePade`](@ref) supplies scaling, applies the denominator, and undoes
the scaling with modified squaring.
"""
function _native_pade_polynomials(X::AbstractMatrix, 𝕀::AbstractMatrix)
    T = eltype(X)
    X² = X * X
    X⁴ = X² * X²

    p = 𝕀 + T(1 // 26) * X +
        X² * (T(5 // 156) * 𝕀 + T(1 // 858) * X) +
        X⁴ * (T(1 // 5720) * 𝕀 + T(1 // 205920) * X + T(1 // 8648640) * X²)
    q = 𝕀 - T(6 // 13) * X +
        X² * (T(5 // 52) * 𝕀 - T(5 // 429) * X) +
        X⁴ * (T(1 // 1144) * 𝕀 - T(1 // 25740) * X + T(1 // 1235520) * X²)

    p, q
end

function 𝔄(X::AbstractMatrix, algorithm::NativePade)
    nrm = opnorm₁(X)
    s = nrm > algorithm.θ ? ceil(Int, log2(nrm / algorithm.θ)) : 0
    scale = eltype(X)(2)^s
    𝕀 = unit_matrix(X)
    p, q = _native_pade_polynomials(X / scale, 𝕀)

    # `q₆` differs from the identity by at most `Σ|qₖ|θᵏ = 0.256` in one-norm, which is what the
    # constructor's bound `θ ≤ 1/2` buys, so the dense solve `q⁻¹p` can be a Newton--Schulz iteration
    # instead: `q⁻¹ ↦ q⁻¹(2𝕀 - q·q⁻¹)` squares the residual `𝕀 - q·q⁻¹` at every step. From `q⁻¹ = 𝕀`
    # the first step is just `2𝕀 - q`, and four more take the residual to `(𝕀 - q)³²` —
    # `0.256³² ≈ 2e-19`, below `Float64` round-off. Matrix products only, so this is the part that
    # stays portable where a dense solve would not.
    q⁻¹ = 2 * 𝕀 - q
    for _ in 1:4
        q⁻¹ = q⁻¹ * (2 * 𝕀 - q * q⁻¹)
    end

    # The squaring recursion of `ScaledSquaring` above, unchanged and for the same reason: `W`
    # absorbs the `2^-s`, so `s` applications of `W ↦ 2W + WXW` undo the scaling at 2n × 2n.
    W = q⁻¹ * p / scale
    for _ in 1:s
        W = 2 * W + W * X * W
    end

    W
end

function 𝔄(X::AbstractMatrix, ::AugmentedPade)
    # exp([X I; 0 0]) == [exp(X) 𝔄(X); 0 I], so Julia's Padé-based, scaling-and-squaring matrix
    # exponential — the most heavily exercised implementation available — returns `𝔄(X)` in the
    # upper-right block. Nothing delicate happens here, which is what makes this the reference.
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
[`ScaledSquaring`](@ref), [`NativePade`](@ref) and [`AugmentedPade`](@ref). [`ProjectedSkew`](@ref)
is *not* among them: it is a [`geodesic`](@ref)-level algorithm with its own branch there and no `𝔄`
method, so `𝔄exp(B̂, B̄, ProjectedSkew())` fails inside `𝔄` exactly as `𝔄(B̂, B̄, ProjectedSkew())`
does.

# Implementation

The default is [`ScaledSquaring`](@ref) rather than the unscaled series because cancellation makes the
latter unreliable once ``\|\bar{B}\| \gtrsim 50``, which is not a regime a function that presents
itself as an exponential may quietly get wrong. Relative error
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
