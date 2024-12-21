using GeometricOptimizers
using GeometricOptimizers: map_to_Skew
using Test
import Random

Random.seed!(123)

function skew_symmetrization_operation(N::Integer, T::DataType=Float32)
    A = rand(T, N,N)
    A_skew = SkewSymMatrix(A)

    for i in 1:N 
        for j in 1:N 
            @test abs(.5*(A - A')[i,j] - A_skew[i,j]) < eps(T)
        end
    end
end

#check if symmetric matrix works for 1×1 matrices
function check_if_symmetric_matrix_works_for_1x1_matrices(T::DataType)
    W = rand(T, 1, 1)
    S = SkewSymMatrix(W)
    # a 1×1-skew-symmetric matrix is 0
    @test abs(S[1,1]) < eps(T)
end

#check if built-in projection, matrix addition & subtraction works   
function skew_mat_add_sub(N::Integer, T::DataType=Float32)
    anti_symmetrize(W) = .5 * (W - W')
    W₁ = rand(T, N, N)
    S₁ = SkewSymMatrix(W₁)
    W₂ = rand(T, N, N)
    S₂ = SkewSymMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SkewSymMatrix
    @test typeof(S₄) <: SkewSymMatrix
    @test all(abs.(anti_symmetrize(W₁ + W₂) .- S₃) .< eps(T))
    @test all(abs.(anti_symmetrize(W₁ - W₂) .- S₄) .< eps(T))
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
function skew_mat_mul_from_the_right(N::Integer, T::DataType=Float64)
    S = rand(SkewSymMatrix{T}, N)
    A = rand(T, N, N)
    AS1 = A * S 
    AS2 = A * Matrix{T}(S)
    @test isapprox(AS1, AS2)
end

function check_map_to_Skew(N::Integer, T::DataType=Float64)
    A = rand(SkewSymMatrix{T}, N)
    @test A.S ≈ map_to_Skew(A)
end

function scalar_multiplication(n::Integer, T::DataType)
    A = rand(T, n, n)
    α = rand(T)

    # SkewSymMatrix
    Aα_sym = SkewSymMatrix(α * A)
    Aα_sym2 = α * SkeySymMatrix(A)
    @test Aα_sym ≈ Aα_sym2
    @test typeof(Aα_sym) <: SkewSymMatrix{T}
    @test typeof(Aα_sym2) <: SkewSymMatrix{T}
end

function test_random_array_generation(n::Int, N::Int, T::DataType)
    A_sym = rand(SkewSymMatrix{T}, n)
    @test typeof(A_sym) <: SkewSymMatrix{T}
    @test eltype(A_sym) == T
end

for T ∈ (Float32, Float64)
    check_if_symmetric_matrix_works_for_1x1_matrices(T)
    for N ∈ 2:5
        skew_symmetrization_operation(N, T)
        skew_mat_add_sub(N, T)
        skew_mat_mul(N, T)
        skew_mat_mul_from_the_right(N, T)
    end
end