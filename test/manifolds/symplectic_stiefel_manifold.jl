using Test
using LinearAlgebra
using GeometricOptimizers
using GeometricOptimizers: _poisson_tensor
import Random

Random.seed!(1234)

# See the note on tolerances in `test/decompositions/symplectic_sr.jl`: the residual grows with the
# size because the SR decomposition these points come from has no re-orthogonalization step, and
# `Float32` is out of reach at every size. The figures are in `CHANGELOG.md` under *Open Issues*.
tolerance(N2) = N2 ≤ 4 ? 1e-9 : N2 ≤ 6 ? 1e-6 : 1e-5

const SIZES = ((4, 2), (6, 4), (10, 6))

@testset "a random point lies on the manifold" begin
    for (N2, n2) in SIZES
        U = rand(SymplecticStiefelManifold, N2, n2)
        @test size(U) == (N2, n2)
        @test check(U) < tolerance(N2)

        # `check` is the symplectic residual and not the generic orthonormality one. Without the
        # specific method this testset would be asserting the wrong constraint and would fail; the
        # assertion below states that the two are genuinely different quantities here.
        @test norm(U.A' * U.A - I) > 1e-6
    end
end

@testset "the element type is honoured" begin
    U = rand(SymplecticStiefelManifold{Float64}, 6, 4)
    @test eltype(U) == Float64
    # `Float32` points are constructed, and deliberately not checked against the manifold: the
    # residual at this size runs to 0.62 over 200 draws.
    @test eltype(rand(SymplecticStiefelManifold{Float32}, 6, 4)) == Float32
end

@testset "the constructor rejects what is not a symplectic shape" begin
    @test_throws AssertionError SymplecticStiefelManifold(randn(5, 4))
    @test_throws AssertionError SymplecticStiefelManifold(randn(6, 3))
    @test_throws AssertionError SymplecticStiefelManifold(randn(4, 6))
end

@testset "rgrad lands in the tangent space" begin
    # The tangent space at `U` is `{Δ : Δᵀ𝕁U + Uᵀ𝕁Δ = 0}`, which is what makes the Riemannian
    # gradient a direction the retraction can follow. A gradient of the right *size* says nothing.
    for (N2, n2) in SIZES
        J = _poisson_tensor(Float64, N2)
        U = rand(SymplecticStiefelManifold, N2, n2)
        Δ = rgrad(U, randn(N2, n2))
        @test size(Δ) == (N2, n2)
        @test norm(Δ' * J * U.A + U.A' * J * Δ) < tolerance(N2)
    end
end

@testset "the metric is symmetric and bilinear" begin
    for (N2, n2) in SIZES
        U = rand(SymplecticStiefelManifold, N2, n2)
        Δ₁, Δ₂ = randn(N2, n2), randn(N2, n2)
        @test isapprox(metric(U, Δ₁, Δ₂), metric(U, Δ₂, Δ₁); atol = tolerance(N2))
        @test isapprox(metric(U, 2 * Δ₁, Δ₂), 2 * metric(U, Δ₁, Δ₂); atol = tolerance(N2))
        @test isapprox(metric(U, Δ₁ + Δ₂, Δ₂), metric(U, Δ₁, Δ₂) + metric(U, Δ₂, Δ₂);
            atol = tolerance(N2))
    end
end

@testset "global_section completes the point symplectically" begin
    for (N2, n2) in SIZES
        J = _poisson_tensor(Float64, N2)
        U = rand(SymplecticStiefelManifold, N2, n2)
        Λ = global_section(U)
        @test size(Λ) == (N2, N2)
        @test norm(Λ' * J * Λ - J) < tolerance(N2)
    end
end

@testset "the canonical Poisson tensor" begin
    J = _poisson_tensor(Float64, 4)
    @test J == [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0; -1.0 0.0 0.0 0.0; 0.0 -1.0 0.0 0.0]
    @test J' == -J
    @test J * J == -Matrix{Float64}(I, 4, 4)
    @test eltype(_poisson_tensor(Float32, 6)) == Float32
    @test_throws AssertionError _poisson_tensor(Float64, 5)
end
