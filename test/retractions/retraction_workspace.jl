# What a retraction taken in a `RetractionWorkspace` allocates, and that it answers what the
# allocating methods answer: the geodesic through `𝔄!` (issue #77), and the `Cayley` differential
# through `retraction_differential!` (issue #73).
#
# Every allocation assertion is an exact zero or an equality between two measurements in one
# process. `@allocated` is inside a function whose arguments are its parameters, and the call is made
# once before it is measured, for the reason the head of `test/flat_buffer_allocations.jl` gives.

using GeometricOptimizers
using GeometricOptimizers: 𝔄, RetractionWorkspace, ScaledSquaring, NativePade,
                           AugmentedPade,
                           TaylorSeries, lift_factors, lift_factors!, retraction_matrix!,
                           retraction_differential, retraction_differential!,
                           retraction_workspace,
                           _similar
using LinearAlgebra: I, mul!
using Test
import Random

include("../helpers/reference_retractions.jl")

const LIFT_TYPES = (StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix)
manifold(::Type{StiefelLieAlgHorMatrix}) = StiefelManifold
manifold(::Type{GrassmannLieAlgHorMatrix}) = GrassmannManifold

function fixture(LT, ::Type{T}, N, n; seed = N + n, scale = 1) where {T}
    B = T(scale) * rand(Random.Xoshiro(seed), LT{T}, N, n)
    Y = rand(Random.Xoshiro(seed + 1), manifold(LT){T}, N, n)
    (B = B, ws = retraction_workspace(Y))
end

_measured_lift_factors(ws, B) = (lift_factors!(ws, B); @allocated lift_factors!(ws, B))
function _measured_retraction(ws, R, B)
    retraction_matrix!(ws, R, B)
    @allocated retraction_matrix!(ws, R, B)
end
function _measured_differential!(D, ws, R, B, α)
    retraction_differential!(D, ws, R, B, α)
    @allocated retraction_differential!(D, ws, R, B, α)
end
function _measured_differential(R, B, α)
    retraction_differential(R, B, α)
    @allocated retraction_differential(R, B, α)
end

# `geodesic` as `src/` wrote it before `𝔄!`, on the copied `𝔄`
function reference_geodesic(B, algorithm)
    T = eltype(B)
    B̂, B̄ = lift_factors(B)
    retracted = Matrix{T}(I, B.N, B.N)
    W = algorithm isa TaylorSeries ? reference_𝔄(B̄' * B̂) : reference_𝔄(B̄' * B̂, algorithm)
    mul!(retracted, B̂ * W, B̄', one(T), one(T))
    retracted
end

@testset "lift_factors! allocates nothing, for a $(nameof(LT)){$T}" for LT in LIFT_TYPES,
    T in (Float32, Float64)

    f = fixture(LT, T, 6, 3)
    @test _measured_lift_factors(f.ws, f.B) == 0
end

@testset "the geodesic in a workspace is the allocating one, for a $(nameof(LT)){$T}" for LT in LIFT_TYPES,
    T in (Float32, Float64)

    for algorithm in (ScaledSquaring(), NativePade(), AugmentedPade(), TaylorSeries()),
        (N, n) in ((6, 3), (6, 1), (20, 10))

        f = fixture(LT, T, N, n; scale = 3)
        result = retraction_matrix!(f.ws, Geodesic(algorithm), f.B)
        @test eltype(result) == T
        @test result == reference_geodesic(f.B, algorithm)
    end
end

# A geodesic adds nothing to what writing the lift's factors costs, which is zero: the `𝔄` it
# evaluates is `𝔄!`, in the workspace.
@testset "a geodesic in a workspace allocates what lift_factors! does, for a $(nameof(LT)){$T}" for LT in LIFT_TYPES,
    T in (Float32, Float64)

    for algorithm in (ScaledSquaring(), NativePade(), TaylorSeries()), scale in (0.01, 3)

        f = fixture(LT, T, 20, 10; scale = scale)
        @test _measured_retraction(f.ws, Geodesic(algorithm), f.B) ==
              _measured_lift_factors(f.ws, f.B)
        @test (@inferred retraction_matrix!(f.ws, Geodesic(algorithm), f.B)) ===
              f.ws.retracted
    end
end

const ALPHAS = (0.5, -0.3, 2.0)

@testset "retraction_differential! is the allocating differential to the bit, for a $(nameof(LT)){$T}" for LT in LIFT_TYPES,
    T in (Float32, Float64)

    for (N, n) in ((6, 3), (6, 1), (20, 4), (6, 6)), α in ALPHAS

        f = fixture(LT, T, N, n)
        D = _similar(f.B)
        result = retraction_differential!(D, f.ws, Cayley(), f.B, α)

        @test result === D
        @test eltype(result) == T
        @test result == reference_cayley_differential(f.B, α)
        @test result == retraction_differential(Cayley(), f.B, α)
    end
end

@testset "at α = 0 the differential is B, copied, for a $(nameof(LT)){$T}" for LT in LIFT_TYPES,
    T in (Float32, Float64)

    f = fixture(LT, T, 6, 3)
    D = _similar(f.B)
    @test retraction_differential!(D, f.ws, Cayley(), f.B, 0) === D
    @test D == f.B
    @test D !== f.B
    @test retraction_differential(Cayley(), f.B, 0) === f.B
end

@testset "a zero lift, a reused workspace, for a $(nameof(LT)){$T}" for LT in LIFT_TYPES,
    T in (Float32, Float64)

    # a zero lift at α ≠ 0: every quantity is exact, and the in-place answer is the allocating one
    f = fixture(LT, T, 6, 3)
    zero_lift = zero(f.B)
    D = _similar(f.B)
    @test retraction_differential!(D, f.ws, Cayley(), zero_lift, 0.5) ==
          reference_cayley_differential(zero_lift, 0.5)

    # the differential reads nothing a geodesic left in the shared scratch
    other = fixture(LT, T, 6, 3; seed = 99, scale = 5).B
    retraction_matrix!(f.ws, Geodesic(), other)
    @test retraction_differential!(D, f.ws, Cayley(), f.B, 0.5) ==
          reference_cayley_differential(f.B, 0.5)
end

# `\` raises on a matrix with an `Inf` or a `NaN`, through the `chkfinite` of its `lu`; the in-place
# solve raises the same error through the `chkfinite` of `getrf!`.
@testset "a NaN or an Inf raises as `\\` does, for a $(nameof(LT)){$T}" for LT in LIFT_TYPES,
    T in (Float32, Float64)

    f = fixture(LT, T, 6, 3)
    D = _similar(f.B)
    for α in (T(NaN), T(Inf))
        @test_throws ArgumentError reference_cayley_differential(f.B, α)
        @test_throws ArgumentError retraction_differential!(D, f.ws, Cayley(), f.B, α)
    end
    nonfinite = copy(f.B)
    nonfinite.B[1, 1] = T(NaN)
    @test_throws ArgumentError reference_cayley_differential(nonfinite, 0.5)
    @test_throws ArgumentError retraction_differential!(D, f.ws, Cayley(), nonfinite, 0.5)
end

@testset "retraction_differential! allocates nothing, for a $(nameof(LT)){$T}" for LT in LIFT_TYPES,
    T in (Float32, Float64)

    f = fixture(LT, T, 6, 3)
    D = _similar(f.B)
    for α in (0.5, 0)
        @test _measured_differential!(D, f.ws, Cayley(), f.B, α) == 0
        @test (@inferred retraction_differential!(D, f.ws, Cayley(), f.B, α)) === D
    end
    # the control: the barrier does see an allocation
    @test _measured_differential(Cayley(), f.B, 0.5) > 0
end
