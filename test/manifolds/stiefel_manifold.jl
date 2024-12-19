using Test 
using LinearAlgebra
using GeometricOptimizers
using GeometricOptimizers: Ω, metric, geodesic
import Random

Random.seed!(123)

n = 1
A_hor = StiefelLieAlgHorMatrix(A_skew, n)

for i in 1:n
    for j in 1:N 
        @test abs(A_hor[i,j] - A_skew[i,j]) < 1e-10
    end 
end

for i in (n+1):N 
    for j in 1:n 
        @test abs(A_hor[i,j] - A_skew[i,j]) < 1e-10
    end
    for j in (n+1):N 
        @test abs(A_hor[i,j]) < 1e-10
    end
end

function metric_test(N, n, T)
    Y = rand(StiefelManifold{T}, N, n)
    Δ₁ = rgrad(Y, rand(T, N, n))
    Δ₂ = rgrad(Y, rand(T, N, n))
    @test T(.5) * tr(Ω(Y, Δ₁)' * Ω(Y, Δ₂)) ≈ metric(Y, Δ₁, Δ₂)
end

for N in (20, 10)
    for n in (5, 3)
        for T in (Float64, Float32)
            metric_test(N, n, T)
        end
    end
end