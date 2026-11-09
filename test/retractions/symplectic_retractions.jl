# The retractions of a `SymplecticLieAlgHorMatrix`: what `test/retractions/retractions.jl`,
# `test/retractions/retraction_workspace.jl` and the differential testset of
# `test/integration/manifold_linesearch_tests.jl` check for the Stiefel and Grassmann lifts. A
# retraction of a symplectic lift preserves the Poisson tensor `𝕁̄` of the adapted basis rather than
# being orthogonal.
using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, ScaledSquaring, NativePade, AugmentedPade,
                           TaylorSeries, ProjectedSkew, StiefelProjection, check, lift_factors,
                           lift_factors!, retraction_matrix!, retraction_differential,
                           retraction_differential!, retraction_workspace
using LinearAlgebra: I, norm
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/symplectic_points.jl")

const SIZES = ((4, 2), (6, 2), (8, 4), (10, 4))

lift(rng, T, N, n; scale = 1) = T(scale) * rand(rng, SymplecticLieAlgHorMatrix{T}, N, n)

@testset "the lift factorisation reproduces the lift, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1)
    for (N, n) in SIZES
        B = lift(rng, T, N, n)
        B̂, B̄ = lift_factors(B)
        @test size(B̂) == (N, 2n)
        @test B̂ * B̄' == Matrix(B)
    end
end

@testset "the Cayley retraction is the Cayley transform and preserves 𝕁̄, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(2)
    for (N, n) in SIZES
        B = lift(rng, T, N, n)
        𝕁 = adapted_poisson(T, N, n)
        R = Cayley()(B)
        @test R isa SymplecticStiefelManifold{T}
        @test R.A ≈ (I - Matrix(B) / 2) \ (I + Matrix(B) / 2)
        @test norm(R.A' * 𝕁 * R.A - 𝕁) < 64eps(T) * norm(R.A)^2
    end
end

@testset "the geodesic is the exponential and preserves 𝕁̄, $(nameof(typeof(algorithm))), $T" for T in REAL_ELTYPES,
    algorithm in (ScaledSquaring(), NativePade(), AugmentedPade(), TaylorSeries())

    rng = Random.Xoshiro(3)
    for (N, n) in SIZES
        B = lift(rng, T, N, n)
        𝕁 = adapted_poisson(T, N, n)
        R = Geodesic(algorithm)(B)
        @test R isa SymplecticStiefelManifold{T}
        @test R.A ≈ exp(Matrix(B))
        @test norm(R.A' * 𝕁 * R.A - 𝕁) < 256eps(T) * norm(R.A)^2
    end
end

@testset "ProjectedSkew refuses a symplectic lift, and says why" begin
    B = rand(Random.Xoshiro(4), SymplecticLieAlgHorMatrix{Float64}, 6, 2)
    @test_throws "Hamiltonian" Geodesic(ProjectedSkew())(B)
end

# The point API: `geodesic(U, Δ)` and `cayley(U, Δ)` go through the section and the lift, land on the
# manifold, and a step of `√eps(T)` along either is the tangent vector to first order.
@testset "a step of √eps(T) along each retraction is the tangent vector, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(5)
    for (N2, n2) in SIZES
        U = conditioned_point(rng, T, N2, n2)
        Δ = rgrad(U, randn(rng, T, N2, n2))
        Δ ./= norm(Δ)
        h = sqrt(eps(T))
        for retract in (GeometricOptimizers.geodesic, GeometricOptimizers.cayley)
            U₂ = retract(U, Δ)
            @test U₂ isa SymplecticStiefelManifold{T}
            @test check(U₂) < 1024eps(T)
            @test norm((retract(U, h * Δ).A - U.A) / h - Δ) < 16sqrt(eps(T))
        end
    end
end

@testset "a retraction in a workspace is the allocating one, $(nameof(typeof(R))), $T" for T in REAL_ELTYPES,
    R in (Cayley(), Geodesic())

    rng = Random.Xoshiro(6)
    for (N2, n2) in SIZES
        B = lift(rng, T, N2, n2)
        ws = retraction_workspace(conditioned_point(rng, T, N2, n2), R)
        @test retraction_matrix!(ws, R, B) ≈ R(B).A
        lift_factors!(ws, B)
        @test (lift_factors!(ws, B); @allocated lift_factors!(ws, B)) == 0
    end
end

# `D(α)` against a central difference of the retraction it is the derivative of, as the Stiefel and
# Grassmann lifts are checked in `test/integration/manifold_linesearch_tests.jl`.
@testset "retraction_differential is d/dα retract(αB), $T" for T in REAL_ELTYPES
    B = lift(Random.Xoshiro(7), T, 8, 4; scale = 1 // 2)
    E = StiefelProjection(B)
    frame(R, α) = Matrix(R(α * B))
    h = T == Float32 ? T(1e-2) : T(1e-5)
    tolerance = T == Float32 ? T(1e-2) : T(1e-7)
    for R in (Geodesic(), Cayley()), α in T.((0.25, 0.5, 1.0, 2.0))
        difference = (frame(R, α + h) * E - frame(R, α - h) * E) / (2h)
        velocity = frame(R, α) * Matrix(retraction_differential(R, B, α)) * E
        @test eltype(velocity) == T
        @test norm(velocity - difference) / norm(difference) < tolerance
    end
    @test retraction_differential(Cayley(), B, zero(T)) === B

    # and in a workspace it is the allocating one
    ws = retraction_workspace(conditioned_point(Random.Xoshiro(8), T, 8, 4), Cayley())
    D = similar(B)
    @test retraction_differential!(D, ws, Cayley(), B, T(0.5)) ≈
          retraction_differential(Cayley(), B, T(0.5))
end
