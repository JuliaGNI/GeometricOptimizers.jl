using GeometricOptimizers
using Test
import Random

Random.seed!(123)

function skew_symmetrization_operation(N::Integer, T::DataType=Float32)
    A = rand(T, N,N)
    A_skew = SkewSymMatrix(A)

    for i in 1:N 
        for j in 1:N 
            @test abs(.5*(A - A')[i,j] - A_skew[i,j]) < 1e-10
        end
    end
end

#check if symmetric matrix works for 1×1 matrices 
W = rand(1,1)
S = SymmetricMatrix(W)
@test abs(W[1,1] - S[1,1]) < 1e-10

#check if built-in projection, matrix addition & subtraction works   
function skew_mat_add_sub(N::Integer, T::DataType=Float32)
    anti_symmetrize(W) = .5 * (W - W')
    W₁ = rand(T, n, n)
    S₁ = SkewSymMatrix(W₁)
    W₂ = rand(T, n, n)
    S₂ = SkewSymMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SkewSymMatrix
    @test typeof(S₄) <: SkewSymMatrix
    @test all(abs.(anti_symmetrize(W₁ + W₂) .- S₃) .< 1e-10)
    @test all(abs.(anti_symmetrize(W₁ - W₂) .- S₄) .< 1e-10)
end

# this function tests if the matrix multiplication for the SkewSym Matrix is the same as the implied one.
function skew_mat_mul(n::Integer, T::DataType=Float64)
    S = rand(SkewSymMatrix{T}, n)
    A = rand(T, n, n)
    SA1 = S * A 
    SA2 = Matrix{T}(S) * A 
    @test isapprox(SA1, SA2)
end

# tests if multiplication from the right also works correctly
function skew_mat_mul_from_the_right(n::Integer, T::DataType=Float64)
    S = rand(SkewSymMatrix{T}, n)
    A = rand(T, n, n)
    AS1 = A * S 
    AS2 = A * Matrix{T}(S)
    @test isapprox(AS1, AS2)
end

for T ∈ (Float32, Float64)
    for N ∈ 2:5
        skew_symmetrization_operation(N, T)
        skew_mat_add_sub(N, T)
        skew_mat_mul(N, T)
        skew_mat_mul_from_the_right(N, T)
    end
end