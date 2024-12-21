using GeometricOptimizers
using GeometricOptimizers: StiefelProjection
using LinearAlgebra: I
import Random

Random.seed!(123)

@doc raw"""
This function tests addition for various custom arrays, i.e. if \(A + B\) is performed in the correct way. 
"""
function add_and_sub(n::Int, N::Int, T::Type)
    C = rand(T, N, N)
    D = rand(T, N, N)

    # StiefelLieAlgHorMatrix
    CD_slahm = StiefelLieAlgHorMatrix(C + D, n)
    CD_slahm2 = StiefelLieAlgHorMatrix(C, n) + StiefelLieAlgHorMatrix(D, n)
    @test CD_slahm ≈ CD_slahm2
    @test typeof(CD_slahm) <: StiefelLieAlgHorMatrix{T}
    @test typeof(CD_slahm2) <: StiefelLieAlgHorMatrix{T}

    CD_slahm_sub = StiefelLieAlgHorMatrix(C - D, n)
    CD_slahm2_sub = StiefelLieAlgHorMatrix(C, n) - StiefelLieAlgHorMatrix(D, n)
    @test CD_slahm_sub ≈ CD_slahm2_sub
    @test typeof(CD_slahm_sub) <: StiefelLieAlgHorMatrix{T}
    @test typeof(CD_slahm2_sub) <: StiefelLieAlgHorMatrix{T}
end

function stiefel_lie_alg_projection(n::Integer, N::Integer, T::DataType=Float32)
    E = StiefelProjection(T, N, n)
    projection(W::SkewSymMatrix) = W - (I - E * E') * W * (I - E * E')
    W₁ = SkewSymMatrix(rand(T, N, N))
    S₁ = StiefelLieAlgHorMatrix(W₁, n)
    W₂ = SkewSymMatrix(rand(T, N, N))
    S₂ = StiefelLieAlgHorMatrix(W₂, n)
    A = rand(T, N, N)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: StiefelLieAlgHorMatrix
    @test typeof(S₄) <: StiefelLieAlgHorMatrix
    @test all(abs.(projection(W₁ + W₂) .- S₃) .< eps(T))
    @test all(abs.(projection(W₁ - W₂) .- S₄) .< eps(T))
    # check custom addition
    @test S₁ + A ≈ Matrix(S₁) + A
    @test A + S₁ ≈ Matrix(S₁) + A
end

function stiefel_lie_alg_vectorization_test(n::Integer, N::Integer, T::DataType=Float32)
    A = rand(StiefelLieAlgHorMatrix{T}, N, n)
    @test isapprox(StiefelLieAlgHorMatrix(vec(A), N, n), A)
end

function scalar_multiplication(n::Integer, N::Integer, T::DataType=Float32)
    C = rand(T, N, N)
    α = rand(T)

    # StiefelLieAlgHorMatrix
    Cα_slahm = StiefelLieAlgHorMatrix(α * C, n)
    Cα_slahm2 = α * StiefelLieAlgHorMatrix(C, n)
    @test Cα_slahm ≈ Cα_slahm2
    @test typeof(Cα_slahm) <: StiefelLieAlgHorMatrix{T}
    @test typeof(Cα_slahm2) <: StiefelLieAlgHorMatrix{T}
end

function random_array_generation(n::Integer, N::Integer, T::DataType)
    A_stiefel_hor = rand(StiefelLieAlgHorMatrix{T}, N, n)
    @test typeof(A_stiefel_hor) <: StiefelLieAlgHorMatrix{T}
    @test eltype(A_stiefel_hor) == T
end

for T ∈ (Float32, Float64)
    for N ∈ 3:5
        for n ∈ 1:N
            add_and_sub(n, N, T)
            stiefel_lie_alg_projection(n, N, T)
            stiefel_lie_alg_vectorization_test(n, N, T)
            scalar_multiplication(n, N, T)
            random_array_generation(n, N, T)
        end
    end
end
