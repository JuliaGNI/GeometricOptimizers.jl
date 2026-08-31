using GeometricOptimizers
using GeometricOptimizers: map_to_Skew
using Test
import Random

Random.seed!(123)

function skew_symmetrization_operation(N::Integer, T::DataType = Float32)
    A = rand(T, N, N)
    A_skew = SkewSymMatrix(A)

    for i in 1:N
        for j in 1:N
            @test abs(0.5*(A - A')[i, j] - A_skew[i, j]) < eps(T)
        end
    end
end

#check if symmetric matrix works for 1×1 matrices
function check_if_symmetric_matrix_works_for_1x1_matrices(T::DataType)
    W = rand(T, 1, 1)
    S = SkewSymMatrix(W)
    # a 1×1-skew-symmetric matrix is 0
    @test abs(S[1, 1]) < eps(T)
end

#check if built-in projection, matrix addition & subtraction works   
function skew_mat_add_sub(N::Integer, T::DataType = Float32)
    anti_symmetrize(W) = 0.5 * (W - W')
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
function skew_mat_mul(n::Integer, T::DataType = Float64)
    S = rand(SkewSymMatrix{T}, n)
    A = rand(T, n, n)
    SA1 = S * A
    SA2 = Matrix{T}(S) * A
    @test isapprox(SA1, SA2)
end

# tests if multiplication from the right also works correctly
function skew_mat_mul_from_the_right(N::Integer, T::DataType = Float64)
    S = rand(SkewSymMatrix{T}, N)
    A = rand(T, N, N)
    AS1 = A * S
    AS2 = A * Matrix{T}(S)
    @test isapprox(AS1, AS2)
end

function check_map_to_Skew(N::Integer, T::DataType = Float64)
    A = rand(SkewSymMatrix{T}, N)
    @test A.S ≈ map_to_Skew(A)
end

# `SkewSymMatrix(α * A) == α * SkewSymMatrix(A)`, i.e. skew-symmetrization is linear.
#
# This was `SkeySymMatrix(A)` until the loop at the bottom of the file started calling it: a typo in
# a function nothing invoked, so the suite passed and the property went untested. It is tested in
# `GeometricMachineLearning` (`test/arrays/scalar_multiplication_for_custom_arrays.jl`), which is
# where it was found when those tests were folded into this suite.
function scalar_multiplication(n::Integer, T::DataType)
    A = rand(T, n, n)
    α = rand(T)

    # SkewSymMatrix
    Aα_sym = SkewSymMatrix(α * A)
    Aα_sym2 = α * SkewSymMatrix(A)
    @test Aα_sym ≈ Aα_sym2
    @test typeof(Aα_sym) <: SkewSymMatrix{T}
    @test typeof(Aα_sym2) <: SkewSymMatrix{T}
end

# `SkewSymMatrix(A + B) == SkewSymMatrix(A) + SkewSymMatrix(B)`, the other half of linearity.
# `skew_mat_add_sub` above adds two matrices that are already `SkewSymMatrix`es; this one adds the
# dense matrices first, so it tests the constructor and not only `+`.
function addition_is_linear(n::Integer, T::DataType)
    A = rand(T, n, n)
    B = rand(T, n, n)

    AB = SkewSymMatrix(A + B)
    AB₂ = SkewSymMatrix(A) + SkewSymMatrix(B)
    @test AB ≈ AB₂
    @test typeof(AB) <: SkewSymMatrix{T}
    @test typeof(AB₂) <: SkewSymMatrix{T}
end

function test_random_array_generation(n::Int, N::Int, T::DataType)
    A_sym = rand(SkewSymMatrix{T}, n)
    @test typeof(A_sym) <: SkewSymMatrix{T}
    @test eltype(A_sym) == T
end

# `SkewSymMatrix` is exported, so both of these are public API. The parametric method was once
# introduced *in place of* the non-parametric one, which made `zeros(SkewSymMatrix, n)` fall
# through to `Base.zeros(::Type, ::Int)` and throw `MethodError: no method matching
# zero(::Type{SkewSymMatrix})` — and took `zeros(::Type{StiefelLieAlgHorMatrix}, N, n)`, its
# only in-repo caller, down with it.
@testset "zeros for SkewSymMatrix" begin
    for n in 2:5
        A = zeros(SkewSymMatrix, n)
        @test A isa SkewSymMatrix{Float64}
        @test size(A) == (n, n)
        @test all(iszero, A)

        for T in (Float32, Float64)
            A_T = zeros(SkewSymMatrix{T}, n)
            @test A_T isa SkewSymMatrix{T}
            @test size(A_T) == (n, n)
            @test all(iszero, A_T)
        end
    end
end

for T in (Float32, Float64)
    check_if_symmetric_matrix_works_for_1x1_matrices(T)
    for N in 2:5
        skew_symmetrization_operation(N, T)
        skew_mat_add_sub(N, T)
        skew_mat_mul(N, T)
        skew_mat_mul_from_the_right(N, T)
        # `check_map_to_Skew`, `scalar_multiplication` and `test_random_array_generation` were
        # defined above but never called, which is how the typo in the second one survived. Their
        # properties were covered only by `GeometricMachineLearning`'s copies of these tests.
        check_map_to_Skew(N, T)
        scalar_multiplication(N, T)
        addition_is_linear(N, T)
        test_random_array_generation(N, N + 5, T)
    end
end

# The storage layout is public: `vec` returns it and the two-argument constructor takes it. Spelling
# it out for one matrix pins the index arithmetic, which a change that kept `vec` and the constructor
# consistent with *each other* would otherwise slip past. From
# `GeometricMachineLearning`'s `test/arrays/{triangular,constructor_tests_for_custom_arrays}.jl`.
@testset "storage layout" begin
    M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
    @test vec(SkewSymMatrix(M)) ≈ [1.5, 3.0, 1.5, 4.5, 3.0, 1.5]

    @test SkewSymMatrix([1, 2, 3, 4, 5, 6], 4) == [0 -1 -2 -4; 1 0 -3 -5; 2 3 0 -6; 4 5 6 0]
    @test SkewSymMatrix(vec(SkewSymMatrix(M)), 4) ≈ SkewSymMatrix(M)
end
