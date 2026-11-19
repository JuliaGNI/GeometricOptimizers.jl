# `𝔄!`, the in-place `𝔄` the retraction workspace evaluates a geodesic with (issue #77).
#
# Three properties, each in both precisions:
#
#   * the answer is the allocating algorithm's to the bit, against the copy of the allocating bodies
#     in `test/helpers/reference_retractions.jl`, at an argument that needs no halving (`s = 0`) and
#     at one that needs several (`s ≥ 1`), where the recovery step runs;
#   * `ScaledSquaring`, `NativePade` and `TaylorSeries` allocate nothing, on the `20 × 20` argument
#     of issue #77;
#   * `AugmentedPade` allocates what `exp` of its `4n × 4n` matrix allocates, and nothing besides, to
#     a tolerance far below one extra buffer.
#
# The first holds on a real argument. On a complex argument `ScaledSquaring` and `NativePade` agree
# with the reference copy to `rtol = 10eps`, not to the bit: the norm that picks the halving count
# can differ in the last bit there, so at a norm on a threshold the count can shift by one. They
# still agree to the bit with the allocating `𝔄` of `src/`, which runs `𝔄!` in fresh scratch.
# `geodesic` of a complex lift with either is `exp` of the lift.
#
# `@allocated` is inside a function whose arguments are its parameters, and the call is made once
# before it is measured, for the reason the head of `test/integration/flat_buffer_allocations.jl` gives.

using GeometricOptimizers
using GeometricOptimizers: 𝔄, 𝔄!, RetractionWorkspace, ScaledSquaring, NativePade,
                           AugmentedPade, TaylorSeries, opnorm₁, geodesic, lift_factors,
                           _balance!, StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix,
                           SkewSymMatrix
using LinearAlgebra: Diagonal
using KernelAbstractions: CPU
using JLArrays: JLArray, JLBackend
import JLArrays
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/reference_retractions.jl")
include("../helpers/allocations.jl")

# a `2n × 2n` argument with one-norm `nrm`
function argument(::Type{T}, n, nrm, seed) where {T}
    X = randn(Random.Xoshiro(seed), T, 2n, 2n)
    X .*= T(nrm) / opnorm₁(X)
    X
end

workspace(::Type{T}, n) where {T} = RetractionWorkspace(CPU(), T, 2n, n)

_measured_𝔄!(ws, X, algorithm) = (𝔄!(ws, X, algorithm); @allocated 𝔄!(ws, X, algorithm))
_measured_𝔄(X, algorithm) = (𝔄(X, algorithm); @allocated 𝔄(X, algorithm))
_measured_exp(A) = (exp(A); @allocated exp(A))

reference(X, algorithm) = reference_𝔄(X, algorithm)
reference(X, ::TaylorSeries) = reference_𝔄(X)

const ALGORITHMS = (ScaledSquaring(), NativePade(), AugmentedPade(), TaylorSeries())

# `0.1` is below every `θ`, so `s = 0`; `40` needs `s = 6` or `7` at the default `θ = 1/2`, after the
# balancing.
const NORMS = (0.1, 40.0)

@testset "𝔄! is the allocating 𝔄 to the bit, $T" for T in REAL_ELTYPES
    for algorithm in ALGORITHMS, n in (1, 3, 10), (k, nrm) in enumerate(NORMS)
        ws = workspace(T, n)
        X = argument(T, n, nrm, 100n + k)
        result = 𝔄!(ws, X, algorithm)

        @test result === ws.𝔄X
        @test eltype(result) == T
        @test result == reference(X, algorithm)
        # the allocating method of `src/`, which delegates to `𝔄!` for three of the four
        @test result == 𝔄(X, algorithm)
    end
end

# The fixtures of the test above do reach the recovery step: a dropped step would otherwise pass. The
# halving count reads the norm of the balanced argument.
@testset "the large argument needs halving, $T" for T in REAL_ELTYPES
    for n in (1, 3, 10)
        X = argument(T, n, last(NORMS), 100n + 2)
        X′ = similar(X)
        _balance!(X′, similar(X, 1, 2n), X)
        @test eltype(opnorm₁(X′)) == T
        @test opnorm₁(X′) > ScaledSquaring().θ
        @test opnorm₁(X′) > NativePade().θ
    end
end

# The balancing `X′ = D⁻¹XD`, `D = diag(𝕀, t𝕀)`, on the reduced argument of a lift, where it matters:
# the identity block gives `‖X‖₁ ≥ 1` however small the lift, and the lower-left block is quadratic in
# the lift. `t` is a power of two, so `X′` is `D⁻¹XD` to the bit; it is within `√2` of balancing the
# two off-diagonal blocks, so their norms are within a factor of two; and it takes `‖X′‖₁` below `θ`
# for a small lift, and an order of magnitude below `‖X‖₁` for a large one.
@testset "the balancing of a reduced argument, $T" for T in REAL_ELTYPES
    for LT in (StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix),
        (k, scale) in enumerate((1e-3, 1.0, 100.0))

        B = T(scale) * rand(Random.Xoshiro(600 + k), LT{T}, 20, 3)
        B̂, B̄ = lift_factors(B)
        X = B̄' * B̂
        X′ = similar(X)
        t = _balance!(X′, similar(X, 1, 6), X)

        @test t isa T
        @test significand(t) == 1
        @test X′ == Diagonal([1, 1, 1, t, t, t]) \ X * Diagonal([1, 1, 1, t, t, t])
        a, b = opnorm₁(X′[1:3, 4:6]), opnorm₁(X′[4:6, 1:3])
        @test 1 / 2 ≤ a / b ≤ 2
        @test opnorm₁(X) ≥ 1
        scale < 1 && @test opnorm₁(X′) < ScaledSquaring().θ
        scale > 1 && @test opnorm₁(X′) < opnorm₁(X) / 10
    end
end

@testset "an argument balancing leaves as it is, $T" for T in REAL_ELTYPES
    odd = argument(T, 3, 40, 700)[1:5, 1:5]
    zero_block = argument(T, 3, 40, 701)
    zero_block[4:6, 1:3] .= 0
    for X in (odd, zero_block)
        X′ = similar(X)
        @test _balance!(X′, similar(X, 1, size(X, 2)), X) === one(T)
        @test X′ == X
    end
end

# `𝔄` of a `2 × 2` matrix is `c₀𝕀 + c₁X`, the interpolant of `φ(x) = (eˣ - 1)/x` at its two
# eigenvalues, here in 256-bit arithmetic.
function two_by_two_𝔄(X)
    setprecision(BigFloat, 256) do
        A = big.(X)
        h = (A[1, 1] + A[2, 2]) / 2
        r = sqrt(h^2 - (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]))
        μ₁, μ₂ = h + r, h - r
        φ(x) = expm1(x) / x
        c₁ = (φ(μ₁) - φ(μ₂)) / (μ₁ - μ₂)
        c₀ = (μ₁ * φ(μ₂) - μ₂ * φ(μ₁)) / (μ₁ - μ₂)
        c₀ * [1 0; 0 1] + c₁ * A
    end
end

# Off-diagonal blocks so far apart that `b / a` overflows to `Inf` (the first two) or underflows to
# `0` (the last two) in the element type, which `t` avoids by taking the difference of the
# logarithms. Every entry is then within `1.9eps(T)` of the closed form above, relative; the test
# allows `8eps(T)`. `t` overflows only for a block norm below `floatmin`, as in the last argument,
# and that is left unbalanced. There its subnormal entry underflows to zero when `𝔄!` scales `X′` for
# the kernel and back, which the reference does not do, so the two are not compared to the bit.
@testset "the balancing of blocks of very different norms, $(nameof(typeof(algorithm)))" for algorithm in (
    ScaledSquaring(), NativePade())
    for X in (Float32[0.1 1.0f-30; 1.0f10 0.2], [0.1 1e-300; 1e10 0.2],
        Float32[0.1 1.0f30; 1.0f-30 0.2], [0.1 1e300; 1e-300 0.2])
        T = eltype(X)
        t = _balance!(similar(X), similar(X, 1, 2), X)
        @test isfinite(t) && !iszero(t)
        result = 𝔄(X, algorithm)
        @test result == reference(X, algorithm)
        @test all(isapprox.(result, two_by_two_𝔄(X); rtol = 8eps(T)))
    end

    X = Float32[0.1 nextfloat(0.0f0); 1.0f32 0.2]
    @test _balance!(similar(X), similar(X, 1, 2), X) === 1.0f0
    result = 𝔄(X, algorithm)
    @test all(isfinite, result)
    @test isapprox(result, two_by_two_𝔄(X); rtol = 8eps(Float32))
end

# A second call on the same workspace starts from what the first left in the scratch, so a buffer
# that is read before it is written shows here.
@testset "𝔄! does not depend on what the workspace held, $T" for T in REAL_ELTYPES
    for algorithm in ALGORITHMS
        ws = workspace(T, 3)
        large, small = argument(T, 3, 40, 1), argument(T, 3, 0.1, 2)
        𝔄!(ws, large, algorithm)
        result = 𝔄!(ws, small, algorithm)
        @test eltype(result) == T
        @test result == reference(small, algorithm)
    end
end

# A complex argument: its column sums of `abs` are real, but `colsum` has the element type of `X`, so
# the norm is the largest real part. `isless` has no method on two complex numbers. Base sums into
# that complex `colsum` in another order than `opnorm₁` sums into a real array, so the two norms can
# differ in the last bit, and at a norm on a threshold `θ ⋅ 2ˢ` the halving count can shift by one.
# The answer then differs at round-off: at most `3eps` relative in 20 such `ComplexF32` draws for
# either algorithm. The tests allow `10eps`, and check the norm against `opnorm₁` to `2eps`.
#
# The purely imaginary argument has real parts zero, so a norm of the real parts alone is 0 there:
# it halves nothing, and the kernel then sees an argument of norm 40.
const COMPLEX = map(complex, REAL_ELTYPES)

# a purely imaginary `2n × 2n` argument with one-norm `nrm`
function imaginary_argument(::Type{T}, n, nrm, seed) where {T}
    X = T.(im .* randn(Random.Xoshiro(seed), real(T), 2n, 2n))
    X .*= real(T)(nrm) / opnorm₁(X)
    X
end

complex_norm(X) = GeometricOptimizers._opnorm₁!(similar(X, (1, size(X, 2))), X)

@testset "𝔄! is the allocating 𝔄 on a complex argument, $T" for T in COMPLEX
    for algorithm in (ScaledSquaring(), NativePade()),
        X in (argument(T, 3, 0.1, 401), argument(T, 3, 40, 402), imaginary_argument(T, 3, 40, 403))

        ws = workspace(T, 3)
        result = 𝔄!(ws, X, algorithm)

        @test isapprox(complex_norm(X), opnorm₁(X); rtol = 2eps(real(T)))
        @test eltype(result) == T
        @test isapprox(result, reference(X, algorithm); rtol = 10eps(real(T)))
        # the allocating method of `src/` runs `𝔄!` in fresh scratch, with the same norm
        @test result == 𝔄(X, algorithm)
    end
end

# The norm on the threshold `θ ⋅ 2³`, where the halving count of `𝔄!` and of the reference can differ.
# On Julia 1.13 the count differs in 10 of the 200 `ComplexF32` draws, the first at seed 11, and in
# none of the `ComplexF64` draws.
@testset "𝔄! is the allocating 𝔄 on a complex argument at a threshold norm, $T" for T in COMPLEX
    for algorithm in (ScaledSquaring(), NativePade()), seed in 1:200

        ws = workspace(T, 3)
        X = argument(T, 3, algorithm.θ * 8, seed)
        result = 𝔄!(ws, X, algorithm)

        @test isapprox(complex_norm(X), opnorm₁(X); rtol = 2eps(real(T)))
        @test eltype(result) == T
        @test isapprox(result, reference(X, algorithm); rtol = 10eps(real(T)))
        @test result == 𝔄(X, algorithm)
    end
end

# `geodesic` of a complex lift runs `𝔄` through the norm above. The answer is `exp` of the lift.
@testset "geodesic of a complex lift with $(nameof(typeof(algorithm))), $T" for T in COMPLEX,
    algorithm in (ScaledSquaring(), NativePade())

    rng = Random.Xoshiro(500)
    for C in (StiefelLieAlgHorMatrix(SkewSymMatrix(randn(rng, T, 3, 3)), randn(rng, T, 5, 3), 8, 3),
        GrassmannLieAlgHorMatrix(randn(rng, T, 5, 3), 8, 3))
        Y = geodesic(C, algorithm)
        @test eltype(Y) == T
        @test Matrix(Y) ≈ exp(Matrix(C))
    end
end

@testset "𝔄! allocates nothing for $(nameof(typeof(algorithm))), $T" for T in REAL_ELTYPES,
    algorithm in (ScaledSquaring(), NativePade(), TaylorSeries())

    ws = workspace(T, 10)
    for nrm in NORMS
        X = argument(T, 10, nrm, 7)
        @test _measured_𝔄!(ws, X, algorithm) == 0
        result = @inferred 𝔄!(ws, X, algorithm)
        @test result === ws.𝔄X
        @test eltype(result) == T
    end
    # the control: the barrier does see an allocation
    @test _measured_𝔄(argument(T, 10, 0.1, 7), algorithm) > 0
end

# Under a tolerance, for the reason `test/helpers/allocations.jl` gives: on Windows the two readings
# are up to 96 bytes apart. 256 bytes is well below one extra `20 × 20` buffer, 1 600 bytes in
# `Float32`.
@testset "AugmentedPade allocates what exp allocates, $T" for T in REAL_ELTYPES
    ws = workspace(T, 10)
    for nrm in NORMS
        X = argument(T, 10, nrm, 8)
        measured = _measured_𝔄!(ws, X, AugmentedPade())
        @test n_independent(measured, _measured_exp(ws.augmented); tolerance = 256)
        @test measured > 0
        @test eltype(ws.𝔄X) == T
    end
end

# The portable algorithms on a device array, with scalar indexing an error. JLArrays multiplies with
# a generic loop and not with BLAS, so a BLAS may take the sums in another order. On OpenBLAS 0.3.30
# arm64 the two agree to the bit in every row; the test allows `100eps(T)` relative, the rounding of
# a product chain taken in another order. `TaylorSeries` only at the small argument, where its sum
# does not cancel.
@testset "𝔄! runs on a JLArray without scalar indexing, $T" for T in REAL_ELTYPES
    JLArrays.allowscalar(false)
    for (algorithm, nrm) in ((ScaledSquaring(), 0.1), (ScaledSquaring(), 40.0),
        (NativePade(), 0.1), (NativePade(), 40.0), (TaylorSeries(), 0.1))
        X = argument(T, 3, nrm, 9)
        ws = RetractionWorkspace(JLBackend(), T, 6, 3)
        result = 𝔄!(ws, JLArray(X), algorithm)
        @test result isa JLArray{T}
        @test eltype(result) == T
        @test isapprox(Array(result), reference(X, algorithm); rtol = 100eps(T))
    end

    # The random arguments above have off-diagonal blocks of one size, where the balancing factor is
    # one: `_balance!` multiplies the blocks by one and `_unbalance!` skips them. A reduced argument
    # has a factor far from one.
    B̂, B̄ = lift_factors(T(30) * rand(Random.Xoshiro(10), StiefelLieAlgHorMatrix{T}, 20, 3))
    X = B̄' * B̂
    @test _balance!(similar(X), similar(X, 1, 6), X) != 1
    for algorithm in (ScaledSquaring(), NativePade())
        ws = RetractionWorkspace(JLBackend(), T, 6, 3)
        result = 𝔄!(ws, JLArray(X), algorithm)
        @test result isa JLArray{T}
        @test isapprox(Array(result), reference(X, algorithm); rtol = 100eps(T))
    end
end
