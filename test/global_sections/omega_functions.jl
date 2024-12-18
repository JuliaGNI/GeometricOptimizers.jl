using Test 
using LinearAlgebra: norm
using GeometricOptimizers
using GeometricOptimizers: geodesic, cayley
import Random

Random.seed!(123)

function stiefel_Ω(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    @test GeometricOptimizers.Ω(Y, Δ) * Y.A ≈ Δ
end

function grassmann_Ω(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(GrassmannManifold{T})
    Δ = rgrad(Y, rand(T, N, n))
    @test GeometricOptimizers.Ω(Y, Δ) * Y.A ≈ Δ
end

T = Float32

for N ∈ 3:5
    for n ∈ 1:N
        @test (N, n, T)
        grassmann_test_help(geodesic_retraction_for_grassmann_manifold(N, n, T), N, n)
        grassmann_test_help(cayley_retraction_for_grassmann_manifold(N, n, T), N, n)
    end
end