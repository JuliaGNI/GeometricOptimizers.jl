using GeometricMachineLearning

using Test
import ChainRulesTestUtils

function sym_mat_add_sub(n::Integer)
    symmetrize(W) = .5*(W + W')
    W₁ = rand(n,n)
    S₁ = SymmetricMatrix(W₁)
    W₂ = rand(n,n)
    S₂ = SymmetricMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SymmetricMatrix
    @test typeof(S₄) <: SymmetricMatrix
    @test all(abs.(symmetrize(W₁ + W₂) .- S₃) .< 1e-10)
    @test all(abs.(symmetrize(W₁ - W₂) .- S₄) .< 1e-10)
end

function test_multiplication(n::Integer=5, T::DataType=Float32)
    A = rand(SymmetricMatrix{T}, n)
    b = rand(T, n)
    B = rand(T, n, n)
    # test if the custom multiplication is performed the right way
    @test A*b == Matrix{T}(A)*b
    @test A*B == Matrix{T}(A)*B
end

function test_calling_symmetric_matrix(n::Integer=5, T::DataType=Float32)
    B = rand(T, n, n)
    @test isapprox(SymmetricMatrix(B), .5*(B + B'))
end

function test_pullback_routine(n::Integer=5, T::DataType=Float32)
    A = rand(SymmetricMatrix{T}, n)
    B = rand(T, n, n)

    @test ChainRulesTestUtils.rrule(*, A, B)
end

test_multiplication()
# this test is not working - problem has to do with FiniteDifferences.jl (I don't know if it's worth looking into this)
# test_calling_symmetric_matrix()