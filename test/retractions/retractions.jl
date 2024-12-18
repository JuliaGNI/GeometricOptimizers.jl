using Test 
using LinearAlgebra: norm
using GeometricOptimizers
using GeometricOptimizers: geodesic, cayley
import Random

Random.seed!(123)

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

function grassmann_test_help(result::Bool, N::Integer, n::Integer)
    if N > n
        @test result
    elseif N == n 
        @test !result
    else
        error("N has to be greater than n")
    end
end

for N ∈ 3:5
    for n ∈ 1:N
        @test geodesic_retraction_for_stiefel_manifold(N, n, T)
        @test cayley_retraction_for_stiefel_manifold(N, n, T)
        grassmann_test_help(geodesic_retraction_for_grassmann_manifold(N, n, T), N, n)
        grassmann_test_help(cayley_retraction_for_grassmann_manifold(N, n, T), N, n)
    end
end