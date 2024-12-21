using Test 
using LinearAlgebra
using GeometricOptimizers
using GeometricOptimizers: Ω, metric, geodesic
import Random

Random.seed!(123)

function correct_format(n::Integer, N::Integer, T::DataType)
    A_skew = rand(SkewSymMatrix{T}, N)
    A_hor = StiefelLieAlgHorMatrix(A_skew, n)

    for i in 1:n
        for j in 1:N 
            @test abs(A_hor[i, j] - A_skew[i, j]) < eps(T)
        end 
    end

    for i in (n + 1):N 
        for j in 1:n 
            @test abs(A_hor[i, j] - A_skew[i, j]) < eps(T)
        end
        for j in (n + 1):N 
            @test abs(A_hor[i, j]) < eps(T)
        end
    end
end

function metric_test(n::Integer, N::Integer, T::DataType)
    Y = rand(StiefelManifold{T}, N, n)
    Δ₁ = rgrad(Y, rand(T, N, n))
    Δ₂ = rgrad(Y, rand(T, N, n))
    @test T(.5) * tr(Ω(Y, Δ₁)' * Ω(Y, Δ₂)) ≈ metric(Y, Δ₁, Δ₂)
end

for N in (20, 10)
    for n in (5, 3)
        for T in (Float64, Float32)
            correct_format(n, N, T)
            metric_test(n, N, T)
        end
    end
end