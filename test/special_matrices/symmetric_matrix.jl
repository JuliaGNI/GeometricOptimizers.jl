using GeometricOptimizers
using Test
import Random

Random.seed!(123)

# import ChainRulesTestUtils

symmetrize(W::AbstractMatrix{T}) where {T} = T(0.5) * (W + W')

function sym_mat_add_sub(n::Integer, T::DataType)
    W₁ = rand(T, n, n)
    S₁ = SymmetricMatrix(W₁)
    W₂ = rand(T, n, n)
    S₂ = SymmetricMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SymmetricMatrix
    @test typeof(S₄) <: SymmetricMatrix
    @test all(abs.(symmetrize(W₁ + W₂) - S₃) .< 2 * eps(T))
    @test all(abs.(symmetrize(W₁ - W₂) - S₄) .< 2 * eps(T))
end

function random_generation(N::Integer, T::DataType = Float64)
    A_sym = rand(SymmetricMatrix{T}, N)
    @test typeof(A_sym) <: SymmetricMatrix{T}
    @test eltype(A_sym) == T
end

function multiplication(n::Integer = 5, T::DataType = Float32)
    A = rand(SymmetricMatrix{T}, n)
    b = rand(T, n)
    B = rand(T, n, n)
    # test if the custom multiplication is performed the right way
    @test A * b ≈ Matrix{T}(A) * b
    @test A * B ≈ Matrix{T}(A) * B
end

function calling_symmetric_matrix(n::Integer = 5, T::DataType = Float32)
    B = rand(T, n, n)
    @test isapprox(SymmetricMatrix(B), 0.5*(B + B'))
end

function test_pullback_routine(n::Integer = 5, T::DataType = Float32)
    A = rand(SymmetricMatrix{T}, n)
    B = rand(T, n, n)

    @test ChainRulesTestUtils.rrule(*, A, B)
end

function scalar_multiplication(n::Integer, T::DataType)
    A = rand(T, n, n)
    α = rand(T)

    # SymmetricMatrix
    Aα_sym = SymmetricMatrix(α * A)
    Aα_sym2 = α * SymmetricMatrix(A)
    @test Aα_sym ≈ Aα_sym2
    @test typeof(Aα_sym) <: SymmetricMatrix{T}
    @test typeof(Aα_sym2) <: SymmetricMatrix{T}
end

for T in (Float32, Float64)
    for n in 1:5
        sym_mat_add_sub(n, T)
        random_generation(n, T)
        multiplication(n, T)
        calling_symmetric_matrix(n, T)
        scalar_multiplication(n, T)
    end
end
# see the note on `storage layout` in `skew_symmetric.jl`
@testset "storage layout" begin
    M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
    @test SymmetricMatrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4) ==
          [1 2 4 7; 2 3 5 8; 4 5 6 9; 7 8 9 10]
    @test SymmetricMatrix(vec(SymmetricMatrix(M)), 4) ≈ SymmetricMatrix(M)
end
