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
# `@allocated` is inside a function whose arguments are its parameters, and the call is made once
# before it is measured, for the reason the head of `test/flat_buffer_allocations.jl` gives.

using GeometricOptimizers
using GeometricOptimizers: 𝔄, 𝔄!, RetractionWorkspace, ScaledSquaring, NativePade,
                           AugmentedPade, TaylorSeries, opnorm₁
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
# a generic loop and not with BLAS, so the sums are taken in another order and the two agree to
# round-off, not to the bit: `√eps(T)` relative, which the recovery chain of `s = 7` steps stays well
# inside. `TaylorSeries` only at the small argument, where its sum does not cancel.
@testset "𝔄! runs on a JLArray without scalar indexing, $T" for T in (Float32, Float64)
    JLArrays.allowscalar(false)
    for (algorithm, nrm) in ((ScaledSquaring(), 0.1), (ScaledSquaring(), 40.0),
        (NativePade(), 0.1), (NativePade(), 40.0), (TaylorSeries(), 0.1))
        X = argument(T, 3, nrm, 9)
        ws = RetractionWorkspace(JLBackend(), T, 6, 3)
        result = 𝔄!(ws, JLArray(X), algorithm)
        @test result isa JLArray{T}
        @test isapprox(Array(result), reference(X, algorithm); rtol = sqrt(eps(T)))
    end
end
