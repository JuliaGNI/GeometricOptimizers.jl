using Test
using LinearAlgebra: norm
using GeometricOptimizers
import Random

include("../helpers/eltypes.jl")
include("../helpers/grassmann_test_help.jl")
include("../helpers/symplectic_points.jl")

# `Ω(Y, Δ) * Y` and the tangent vector `Δ` it has to give back.
function stiefel_Ω(rng, N::Integer, n::Integer, T::Type)
    Y = rand(rng, StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(rng, T, N, n))
    GeometricOptimizers.Ω(Y, Δ) * Y.A, Δ
end

function grassmann_Ω(rng, N::Integer, n::Integer, T::Type)
    Y = rand(rng, GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(rng, T, N, n))
    GeometricOptimizers.Ω(Y, Δ) * Y.A, Δ
end

@testset "Ω, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for N in 3:5
        for n in 1:N
            ΩY, Δ = stiefel_Ω(rng, N, n, T)
            @test eltype(ΩY) == T
            @test ΩY ≈ Δ

            ΩY, Δ = grassmann_Ω(rng, N, n, T)
            @test eltype(ΩY) == T
            grassmann_test_help(ΩY ≈ Δ, N, n)
        end
    end
end

# The symplectic lift is Hamiltonian, maps `U` to `Δ`, and is the lift `global_rep` represents:
# `λ(U)⁻¹Ω(U, Δ)λ(U)` is the dense `global_rep(λ(U), Δ)`. The section is symplectic in the adapted
# basis, `λᵀ𝕁λ = 𝕁̄`, so its inverse is `𝕁̄ᵀλᵀ𝕁`.
@testset "Ω on the symplectic Stiefel manifold, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(124)
    for (N2, n2) in ((4, 2), (6, 4), (8, 4), (10, 4))
        U = conditioned_point(rng, T, N2, n2)
        Δ = rgrad(U, randn(rng, T, N2, n2))
        J = poisson(T, N2)
        W = GeometricOptimizers.Ω(U, Δ)

        @test eltype(W) == T
        @test W * U.A ≈ Δ
        @test norm(W' * J + J * W) < 64eps(T) * norm(W)

        λU = GlobalSection(U)
        M = Matrix(λU)
        M⁻¹ = adapted_poisson(T, N2, n2)' * M' * J
        @test M⁻¹ * W * M ≈ Matrix(global_rep(λU, Δ))
    end
end
