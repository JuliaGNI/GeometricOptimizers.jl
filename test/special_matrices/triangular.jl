using GeometricOptimizers
using GeometricOptimizers: AbstractTriangular
using LinearAlgebra: tr
using Test
import Random

Random.seed!(1234)

# There was no test file for the triangular types at all until the `GeometricMachineLearning` tests
# of them were folded into this suite; `test/arrays/triangular.jl` there is where most of this comes
# from. The half of that file that tested `mat_tensor_mul` and its pullback stayed behind — those are
# GML's kernels, not this package's.

@testset "the two triangles and the diagonal partition the matrix" begin
    # `A - L - U` leaves exactly the diagonal, so summing it is the trace. This is the property that
    # says the two constructors take the *strict* triangles and agree on where the split is.
    for T ∈ (Float32, Float64), n ∈ 2:5
        A = rand(T, n, n)
        @test tr(A) ≈ sum(A - LowerTriangular(A) - UpperTriangular(A))
    end
end

@testset "multiplication agrees with the dense matrix" begin
    for T ∈ (Float32, Float64), n ∈ 2:5
        Aₗ = rand(LowerTriangular{T}, n)
        Aᵤ = rand(UpperTriangular{T}, n)
        B = rand(T, n, n)
        b = rand(T, n)

        @test Aₗ * B ≈ Matrix{T}(Aₗ) * B
        @test Aᵤ * B ≈ Matrix{T}(Aᵤ) * B
        @test B * Aₗ ≈ B * Matrix{T}(Aₗ)
        @test B * Aᵤ ≈ B * Matrix{T}(Aᵤ)
        @test Aₗ * b ≈ Matrix{T}(Aₗ) * b
        @test Aᵤ * b ≈ Matrix{T}(Aᵤ) * b
    end
end

@testset "addition and scalar multiplication are linear" begin
    for T ∈ (Float32, Float64), n ∈ 2:5
        A = rand(T, n, n)
        B = rand(T, n, n)
        α = rand(T)

        for MT ∈ (LowerTriangular, UpperTriangular)
            @test MT(A + B) ≈ MT(A) + MT(B)
            @test MT(α * A) ≈ α * MT(A)
            @test typeof(MT(A) + MT(B)) <: MT{T}
            @test typeof(α * MT(A)) <: MT{T}
        end
    end
end

@testset "random generation" begin
    for T ∈ (Float32, Float64), n ∈ 2:5, MT ∈ (LowerTriangular, UpperTriangular)
        A = rand(MT{T}, n)
        @test typeof(A) <: MT{T}
        @test eltype(A) == T
        @test size(A) == (n, n)
        @test A isa AbstractTriangular
    end
end

# The storage layout is public: `vec` returns it and the two-argument constructor takes it, so a
# change to the index arithmetic that kept them consistent with each other would still be breaking.
# Spelling the layout out for one matrix is what pins it.
@testset "storage layout" begin
    M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]

    @test LowerTriangular(M) == [0 0 0 0; 5 0 0 0; 9 10 0 0; 13 14 15 0]
    @test UpperTriangular(M) == [0 2 3 4; 0 0 7 8; 0 0 0 12; 0 0 0 0]

    @test vec(LowerTriangular(M)) == [5, 9, 10, 13, 14, 15]
    @test vec(UpperTriangular(M)) == [2, 3, 7, 4, 8, 12]

    # and the round trip: the vector the second constructor takes is the one `vec` returns
    @test LowerTriangular(vec(LowerTriangular(M)), 4) == LowerTriangular(M)
    @test UpperTriangular(vec(UpperTriangular(M)), 4) == UpperTriangular(M)

    @test LowerTriangular([1, 2, 3, 4, 5, 6], 4) == [0 0 0 0; 1 0 0 0; 2 3 0 0; 4 5 6 0]
    @test UpperTriangular([1, 2, 3, 4, 5, 6], 4) == [0 1 2 4; 0 0 3 5; 0 0 0 6; 0 0 0 0]
end
