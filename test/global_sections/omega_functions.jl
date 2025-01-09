using Test 
using LinearAlgebra: norm
using GeometricOptimizers
import Random

Random.seed!(123)

include("../grassmann_test_help.jl")

function stiefel_Ω(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    GeometricOptimizers.Ω(Y, Δ) * Y.A ≈ Δ
end

function grassmann_Ω(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    GeometricOptimizers.Ω(Y, Δ) * Y.A ≈ Δ
end

T = Float32

for N ∈ 3:5
    for n ∈ 1:N
        @test stiefel_Ω(N, n, T)
        grassmann_test_help(grassmann_Ω(N, n, T), N, n)
    end
end