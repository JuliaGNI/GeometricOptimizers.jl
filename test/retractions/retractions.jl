using Test 
using LinearAlgebra: norm
using GeometricOptimizers
using GeometricOptimizers: geodesic, cayley
import Random

Random.seed!(123)

include("../grassmann_test_help.jl")

function geodesic_retraction_for_stiefel_manifold(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = geodesic(Y, Δ / 1000)
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function cayley_retraction_for_stiefel_manifold(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = cayley(Y, Δ / 1000)
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function geodesic_retraction_for_grassmann_manifold(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = geodesic(Y, Δ / 1000)
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function cayley_retraction_for_grassmann_manifold(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = cayley(Y, Δ / 1000)
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

T = Float32

for N ∈ 3:5
    for n ∈ 1:N
        @test geodesic_retraction_for_stiefel_manifold(N, n, T)
        @test cayley_retraction_for_stiefel_manifold(N, n, T)
        grassmann_test_help(geodesic_retraction_for_grassmann_manifold(N, n, T), N, n)
        grassmann_test_help(cayley_retraction_for_grassmann_manifold(N, n, T), N, n)
    end
end