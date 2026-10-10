using Test
using LinearAlgebra: norm
using GeometricOptimizers
import Random

include("../helpers/eltypes.jl")
include("../helpers/grassmann_test_help.jl")

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
