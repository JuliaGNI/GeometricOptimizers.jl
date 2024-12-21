using GeometricOptimizers
using Test
import Random

Random.seed!(123)

# import ChainRulesTestUtils

symmetrize(W::AbstractMatrix{T}) where T = T(.5) * (W + W')

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

function random_generation(N::Integer, T::DataType=Float64)
    A_sym = rand(SymmetricMatrix{T}, N)
    @test typeof(A_sym) <: SymmetricMatrix{T}
    @test eltype(A_sym) == T
end

function multiplication(n::Integer=5, T::DataType=Float32)
    A = rand(SymmetricMatrix{T}, n)
    b = rand(T, n)
    B = rand(T, n, n)
    # test if the custom multiplication is performed the right way
    @test A * b ≈ Matrix{T}(A) * b
    @test A * B ≈ Matrix{T}(A) * B
end

function calling_symmetric_matrix(n::Integer=5, T::DataType=Float32)
    B = rand(T, n, n)
    @test isapprox(SymmetricMatrix(B), .5*(B + B'))
end

function test_pullback_routine(n::Integer=5, T::DataType=Float32)
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

for T ∈ (Float32, Float64)
    for n ∈ 1:5
        sym_mat_add_sub(n, T)
        random_generation(n, T)
        multiplication(n, T)
        calling_symmetric_matrix(n, T)
        scalar_multiplication(n, T)
    end
end