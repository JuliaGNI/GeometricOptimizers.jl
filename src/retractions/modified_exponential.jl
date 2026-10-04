@doc raw"""
    𝔄(A)

Compute ``\mathfrak{A}(A) := \sum_{n=1}^\infty \frac{1}{n!} (A)^{n-1}.``

# Implementation

The partial sum is accumulated term by term, each term obtained from its predecessor by
``A^{n-1}/n! = (A^{n-2}/(n-1)!)\cdot{}A/n``, until the term's norm falls below machine precision.
Both start at the identity, which is the ``n = 1`` term. The recurrence and the accumulation are
in place, so the loop allocates nothing.

!!! warning "Only accurate for a small argument"
    The series converges for every `A`, but cancellation can make direct summation inaccurate for
    ``\|A\| \gg 1`` — see [`TaylorSeries`](@ref) for what it does at a large argument. This method
    is therefore intended as a small-argument kernel. It is used by [`ScaledSquaring`](@ref) only
    after the argument has been divided until its norm is below `θ`. Reach for it directly only if
    you know the argument is small.
"""
function 𝔄(A::AbstractMatrix)
    # the identity is the first term, so it is handed in as both `𝕀` and `term`
    𝕀 = unit_matrix(A)
    _taylor_𝔄!(similar(A), A, 𝕀, 𝕀, similar(A))
end

# The partial sum of `𝔄(A)`, written into `𝔄A` with `term` and `next` as scratch; `𝕀` is the
# identity of `A`'s size, and is read only before `term` is first written, so the two may be one
# array. `𝔄(A)` above hands in fresh arrays, `𝔄!` the workspace's.
function _taylor_𝔄!(𝔄A, A, 𝕀, term, next)
    T = eltype(A)
    copyto!(term, 𝕀)
    fill!(next, zero(T))
    copyto!(𝔄A, term)
    n = 2
    while norm(term) > eps(real(T))
        LinearAlgebra.mul!(next, term, A, T(inv(n)), zero(T))
        term, next = next, term
        𝔄A .+= term
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
opnorm₁(X::AbstractMatrix) = isempty(X) ? zero(real(eltype(X))) :
                             maximum(sum(abs, X; dims = 1))

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

@doc raw"""
    _scaled_kernel!(scratch, X, algorithm)

Evaluate ``\mathfrak{A}(X)`` for ``\|X\|_1 \leq θ`` into `scratch.𝔄X`, i.e. the small-argument
kernel that [`ScaledSquaring`](@ref) and [`NativePade`](@ref) differ in.

[`ScaledSquaring`](@ref) sums the Taylor series, [`NativePade`](@ref) evaluates the ``[6/6]`` Padé
approximant. Everything else the two algorithms do — choosing the number of halvings, and undoing
them — is the shared framework in `𝔄!(::Any, ::AbstractMatrix, ::ScaledAlgorithm)` below. `scratch`
is as there.
"""
function _scaled_kernel!(scratch, X::AbstractMatrix, ::ScaledSquaring)
    _taylor_𝔄!(scratch.𝔄X, X, scratch.𝕀_small2, scratch.s₂, scratch.s₃)
end

@doc raw"""
    ScaledAlgorithm

The [`AbstractExponentialAlgorithm`](@ref)s built as a small-argument kernel inside scaling and
modified squaring, i.e. [`ScaledSquaring`](@ref) and [`NativePade`](@ref). They share the `θ` field
and the `𝔄!` method below, and differ only in their [`_scaled_kernel!`](@ref).
"""
const ScaledAlgorithm = Union{ScaledSquaring, NativePade}

# The arrays `𝔄!` writes into, for a call that has no `RetractionWorkspace`: the same names as that
# type's fields, fresh. `similar` with an element type and a size, so a structured `X` gets a dense
# array of its backend.
function _𝔄_scratch(X::AbstractMatrix{T}) where {T}
    m = size(X, 1)
    fresh() = similar(X, T, (m, m))
    (𝔄X = fresh(), 𝕀_small2 = unit_matrix(X), colsum = similar(X, T, (1, m)),
        s₁ = fresh(), s₂ = fresh(), s₃ = fresh(), s₄ = fresh(), s₅ = fresh(), s₆ = fresh(),
        s₇ = fresh(), s₈ = fresh())
end

𝔄(X::AbstractMatrix, algorithm::ScaledAlgorithm) = 𝔄!(_𝔄_scratch(X), X, algorithm)

# `opnorm₁` with the column sums written into `colsum`, a `1 × m` array, rather than a fresh one.
# `sum!` and the `dims = 1` reduction of `opnorm₁` are the same reduction, so the two agree bit for
# bit, and with them the halving count. `colsum` has the element type of `X`, so on a complex `X` the
# sums are complex with a zero imaginary part, and the norm is the largest real part.
function _opnorm₁!(colsum::AbstractMatrix, X::AbstractMatrix)
    isempty(X) ? zero(real(eltype(X))) : maximum(real, sum!(abs, colsum, X))
end

function 𝔄!(scratch, X::AbstractMatrix, algorithm::ScaledAlgorithm)
    # `X` is halved `s` times so that the kernel sees an argument of norm ≤ θ, where it is accurate.
    # Initially `exp(B̂B̄ᵗ/2^s) = I + B̂(𝔄(X/2^s)/2^s)B̄ᵗ`. Squaring this represented exponential stays
    # low-rank:
    #
    #     (I + B̂WB̄ᵗ)² = I + B̂(2W + WXW)B̄ᵗ,
    #
    # so each recovery step is `W ↦ 2W + WXW` at 2n × 2n, with the original `X`. After `s` steps
    # `W = 𝔄(X)`. Nothing is ever squared at N × N.
    #
    # `s₁` holds the scaled argument and `𝔄X` is `W`; the kernel uses `s₂` to `s₈`, and the recovery
    # step `s₂` and `s₃` again. `WXW` is `(WX)W`, the order `W * X * W` takes for three square
    # matrices of one size.
    nrm = _opnorm₁!(scratch.colsum, X)
    s = nrm > algorithm.θ ? ceil(Int, log2(nrm / algorithm.θ)) : 0
    scale = eltype(X)(2)^s

    scratch.s₁ .= X ./ scale
    W = _scaled_kernel!(scratch, scratch.s₁, algorithm)
    W ./= scale
    for _ in 1:s
        LinearAlgebra.mul!(scratch.s₂, W, X)
        LinearAlgebra.mul!(scratch.s₃, scratch.s₂, W)
        W .= 2 .* W .+ scratch.s₃
    end

    W
end

@doc raw"""
    _native_pade_polynomials(X, 𝕀)

Evaluate the degree-6 numerator ``p_6(X)`` and denominator ``q_6(X)`` used by [`NativePade`](@ref).
It allocates its result and scratch and calls `_native_pade_polynomials!`, which writes into
given arrays.

If ``P^{\exp}_7/Q^{\exp}_6`` is the ``[7/6]`` Padé approximant of the exponential, then

```math
p_6(z)=\frac{P^{\exp}_7(z)-Q^{\exp}_6(z)}{z},
\qquad
q_6(z)=Q^{\exp}_6(z),
```

so ``q_6(X)^{-1}p_6(X)`` agrees with ``\mathfrak{A}(X)`` through the ``X^{12}`` term.
The implementation shares ``X^2`` and ``X^4`` between the two polynomials and groups the remaining
terms to avoid forming every matrix power separately. `𝕀` must be the multiplicative identity with
the same size, element type, and backend as `X`.

This is an internal kernel; [`NativePade`](@ref) supplies scaling, applies the denominator, and undoes
the scaling with modified squaring.
"""
function _native_pade_polynomials(X::AbstractMatrix, 𝕀::AbstractMatrix)
    _native_pade_polynomials!(
        similar(X), similar(X), X, 𝕀, similar(X), similar(X), similar(X),
        similar(X))
end

# Each sum is one broadcast, and a broadcast of `a .+ b .+ c .+ d` adds left to right, as the sum of
# four arrays does; the products are the same products. So this is `p` and `q` as the closed
# expressions
#
#     p = 𝕀 + c₁X + X²(c₂𝕀 + c₃X) + X⁴(c₄𝕀 + c₅X + c₆X²)
#
# and its `q` counterpart write them, to the bit. `p` holds the `X⁴` product until the last sum reads
# it, and `q` likewise.
function _native_pade_polynomials!(
        p, q, X::AbstractMatrix, 𝕀::AbstractMatrix, X², X⁴, inner,
        product)
    T = eltype(X)
    LinearAlgebra.mul!(X², X, X)
    LinearAlgebra.mul!(X⁴, X², X²)

    inner .= T(5 // 156) .* 𝕀 .+ T(1 // 858) .* X
    LinearAlgebra.mul!(product, X², inner)
    inner .= T(1 // 5720) .* 𝕀 .+ T(1 // 205920) .* X .+ T(1 // 8648640) .* X²
    LinearAlgebra.mul!(p, X⁴, inner)
    p .= 𝕀 .+ T(1 // 26) .* X .+ product .+ p

    inner .= T(5 // 52) .* 𝕀 .- T(5 // 429) .* X
    LinearAlgebra.mul!(product, X², inner)
    inner .= T(1 // 1144) .* 𝕀 .- T(1 // 25740) .* X .+ T(1 // 1235520) .* X²
    LinearAlgebra.mul!(q, X⁴, inner)
    q .= 𝕀 .- T(6 // 13) .* X .+ product .+ q

    p, q
end

function _scaled_kernel!(scratch, X::AbstractMatrix, ::NativePade)
    𝕀 = scratch.𝕀_small2
    p, q = _native_pade_polynomials!(
        scratch.s₆, scratch.s₇, X, 𝕀, scratch.s₂, scratch.s₃, scratch.s₄, scratch.s₅)

    # `q₆` differs from the identity by at most `Σ|qₖ|θᵏ = 0.2563… < 0.257` in one-norm, which is
    # what the constructor's bound `θ ≤ 1/2` buys, so the dense solve `q⁻¹p` can be a Newton--Schulz
    # iteration instead: `q⁻¹ ↦ q⁻¹(2𝕀 - q·q⁻¹)` squares the residual `𝕀 - q·q⁻¹` at every step. From
    # `q⁻¹ = 𝕀` the first step is just `2𝕀 - q`, and four more take the residual to `(𝕀 - q)³²` —
    # `1.3e-19`, below `Float64` round-off. Matrix products only, so this is the part that stays
    # portable where a dense solve would not.
    #
    # `s₄` and `s₅` were the polynomials' scratch and are free again: `s₄` holds `2𝕀 - q·q⁻¹`, and
    # `q⁻¹` alternates between `s₈` and `s₅`.
    q⁻¹, next, residual = scratch.s₈, scratch.s₅, scratch.s₄
    q⁻¹ .= 2 .* 𝕀 .- q
    for _ in 1:4
        LinearAlgebra.mul!(residual, q, q⁻¹)
        residual .= 2 .* 𝕀 .- residual
        LinearAlgebra.mul!(next, q⁻¹, residual)
        q⁻¹, next = next, q⁻¹
    end

    LinearAlgebra.mul!(scratch.𝔄X, q⁻¹, p)
end

function 𝔄(X::AbstractMatrix, ::AugmentedPade)
    # exp([X I; 0 0]) == [exp(X) 𝔄(X); 0 I], so Julia's Padé-based, scaling-and-squaring matrix
    # exponential — the most heavily exercised implementation available — returns `𝔄(X)` in the
    # upper-right block. Nothing delicate happens here, which is what makes this the reference.
    m = size(X, 1)
    T = eltype(X)
    augmented = [X one(X); zeros(T, m, m) zeros(T, m, m)]

    exp(augmented)[1:m, (m + 1):(2m)]
end

@doc raw"""
    𝔄!(ws, X, algorithm)

[`GeometricOptimizers.𝔄`](@ref)`(X, algorithm)` written into `ws.𝔄X`, and returned, with the
scratch of the [`RetractionWorkspace`](@ref) `ws` in place of fresh temporaries.

`X` is ``2n\times{}2n`` for the ``n`` that `ws` was built for, and is not one of the buffers the
algorithm writes; `ws.X` is the buffer meant for it. Every step and every product is the one the
allocating method takes, in the same order, so the two agree bit for bit.

[`ScaledSquaring`](@ref), [`NativePade`](@ref) and [`TaylorSeries`](@ref) then allocate nothing,
with `mul!` and broadcasts only, so no scalar indexing either. [`AugmentedPade`](@ref) writes `X` into
the ``4n\times{}4n`` `ws.augmented`, whose identity and zero blocks were written when `ws` was built,
and calls `Base.exp` on it: it allocates what that `exp` allocates, and nothing besides. Any other
algorithm falls back to the allocating `𝔄` and copies the answer in.
"""
function 𝔄!(ws, X::AbstractMatrix, algorithm::AbstractExponentialAlgorithm)
    copyto!(ws.𝔄X, 𝔄(X, algorithm))
end

𝔄!(ws, X::AbstractMatrix, ::TaylorSeries) = _taylor_𝔄!(ws.𝔄X, X, ws.𝕀_small2, ws.s₂, ws.s₃)

function 𝔄!(ws, X::AbstractMatrix, ::AugmentedPade)
    m = size(X, 1)
    @views begin
        ws.augmented[1:m, 1:m] .= X
        copyto!(ws.𝔄X, exp(ws.augmented)[1:m, (m + 1):(2m)])
    end
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
