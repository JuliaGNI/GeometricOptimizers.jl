# `𝔄!`, the in-place `𝔄` the retraction workspace evaluates a geodesic with (issue #77).
#
# Three properties, each in both precisions:
#
#   * the answer is the allocating algorithm's to the bit, against the copy of the allocating bodies
#     in `test/helpers/reference_retractions.jl`, at an argument that needs no halving (`s = 0`) and
#     at one that needs several (`s ≥ 1`), where the recovery step runs;
#   * `ScaledSquaring`, `NativePade` and `TaylorSeries` allocate nothing, on the `20 × 20` argument
#     of issue #77;
#   * `AugmentedPade` allocates what `exp` of its `4n × 4n` matrix allocates, and nothing besides.
#
# The first holds on a real argument. On a complex argument `ScaledSquaring` and `NativePade` agree
# with the reference copy to `rtol = 10eps`, not to the bit: the norm that picks the halving count
# can differ in the last bit there, so at a norm on a threshold the count can shift by one. They
# still agree to the bit with the allocating `𝔄` of `src/`, which runs `𝔄!` in fresh scratch.
# `geodesic` of a complex lift with either is `exp` of the lift.
#
# `@allocated` is inside a function whose arguments are its parameters, and the call is made once
# before it is measured, for the reason the head of `test/flat_buffer_allocations.jl` gives.

using GeometricOptimizers
using GeometricOptimizers: 𝔄, 𝔄!, RetractionWorkspace, ScaledSquaring, NativePade,
                           AugmentedPade, TaylorSeries, opnorm₁, geodesic,
                           StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix, SkewSymMatrix
using KernelAbstractions: CPU
using JLArrays: JLArray, JLBackend
import JLArrays
using Test
import Random

include("../helpers/reference_retractions.jl")

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

# `0.1` is below every `θ`, so `s = 0`; `40` needs `s = 7` at the default `θ = 1/2`.
const NORMS = (0.1, 40.0)

@testset "𝔄! is the allocating 𝔄 to the bit, $T" for T in (Float32, Float64)
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

# The fixture of the test above does reach the recovery step: a dropped step would otherwise pass.
@testset "the large argument needs halving, $T" for T in (Float32, Float64)
    X = argument(T, 3, last(NORMS), 302)
    @test opnorm₁(X) > ScaledSquaring().θ
    @test opnorm₁(X) > NativePade().θ
end

# A second call on the same workspace starts from what the first left in the scratch, so a buffer
# that is read before it is written shows here.
@testset "𝔄! does not depend on what the workspace held, $T" for T in (Float32, Float64)
    for algorithm in ALGORITHMS
        ws = workspace(T, 3)
        large, small = argument(T, 3, 40, 1), argument(T, 3, 0.1, 2)
        𝔄!(ws, large, algorithm)
        @test 𝔄!(ws, small, algorithm) == reference(small, algorithm)
    end
end

# A complex argument: its column sums of `abs` are real, but `colsum` has the element type of `X`, so
# the norm is the largest real part. `isless` has no method on two complex numbers. Base sums into
# that complex `colsum` in another order than `opnorm₁` sums into a real array, so the two norms can
# differ in the last bit, and at a norm on a threshold `θ ⋅ 2ˢ` the halving count by one. The answer
# then differs at round-off: at most `3eps` relative in 20 such `ComplexF32` draws for either
# algorithm. The tests allow `10eps`, and check the norm against `opnorm₁` to `2eps`.
#
# The purely imaginary argument has real parts zero, so a norm of the real parts alone is 0 there:
# it halves nothing, and the kernel then sees an argument of norm 40.
const COMPLEX = (ComplexF32, ComplexF64)

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

@testset "𝔄! allocates nothing for $(nameof(typeof(algorithm))), $T" for T in (Float32, Float64),
    algorithm in (ScaledSquaring(), NativePade(), TaylorSeries())

    ws = workspace(T, 10)
    for nrm in NORMS
        X = argument(T, 10, nrm, 7)
        @test _measured_𝔄!(ws, X, algorithm) == 0
        @test (@inferred 𝔄!(ws, X, algorithm)) === ws.𝔄X
    end
    # the control: the barrier does see an allocation
    @test _measured_𝔄(argument(T, 10, 0.1, 7), algorithm) > 0
end

@testset "AugmentedPade allocates what exp allocates, $T" for T in (Float32, Float64)
    ws = workspace(T, 10)
    for nrm in NORMS
        X = argument(T, 10, nrm, 8)
        measured = _measured_𝔄!(ws, X, AugmentedPade())
        @test measured == _measured_exp(ws.augmented)
        @test measured > 0
    end
end

# The portable algorithms on a device array, with scalar indexing an error. JLArrays multiplies with
# a generic loop and not with BLAS, so a BLAS may take the sums in another order. On OpenBLAS 0.3.30
# arm64 the two agree to the bit in every row; the test allows `100eps(T)` relative, the rounding of
# a product chain taken in another order. `TaylorSeries` only at the small argument, where its sum
# does not cancel.
@testset "𝔄! runs on a JLArray without scalar indexing, $T" for T in (Float32, Float64)
    JLArrays.allowscalar(false)
    for (algorithm, nrm) in ((ScaledSquaring(), 0.1), (ScaledSquaring(), 40.0),
        (NativePade(), 0.1), (NativePade(), 40.0), (TaylorSeries(), 0.1))
        X = argument(T, 3, nrm, 9)
        ws = RetractionWorkspace(JLBackend(), T, 6, 3)
        result = 𝔄!(ws, JLArray(X), algorithm)
        @test result isa JLArray{T}
        @test isapprox(Array(result), reference(X, algorithm); rtol = 100eps(T))
    end
end
