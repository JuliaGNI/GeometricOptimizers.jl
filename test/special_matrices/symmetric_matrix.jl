using GeometricOptimizers
using GeometricOptimizers: map_to_S
using LinearAlgebra: transpose
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

# see the note on the same testset in `skew_symmetric.jl`
@testset "an integer matrix is projected into float(T)" begin
    for T in (Int8, Int16, Int32, Int64, Int128, BigInt,
        UInt8, UInt16, UInt32, UInt64, UInt128)
        A = T[0 1 2; 3 0 4; 5 6 0]
        @test eltype(map_to_S(A)) === float(T)
        @test eltype(SymmetricMatrix(A)) === float(T)
        @test SymmetricMatrix(A) ≈ (float(T).(A) .+ float(T).(A)') ./ 2
    end
    A = Bool[0 1 1; 0 0 1; 1 0 0]
    @test eltype(map_to_S(A)) === Float64
    @test eltype(SymmetricMatrix(A)) === Float64
    @test SymmetricMatrix(A) ≈ (Float64.(A) .+ Float64.(A)') ./ 2
end

# see the note on `storage layout` in `skew_symmetric.jl`
@testset "storage layout" begin
    M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
    @test SymmetricMatrix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4) ==
          [1 2 4 7; 2 3 5 8; 4 5 6 9; 7 8 9 10]
    @test SymmetricMatrix(vec(SymmetricMatrix(M)), 4) ≈ SymmetricMatrix(M)
end

# see `the projection and the product are transposes on a complex element type` in
# `skew_symmetric.jl`: `SymmetricMatrix` is the set `{M : Mᵀ = M}`, so the projection and
# `*(::AbstractMatrix, ::SymmetricMatrix)` both spell it `transpose`. With `adjoint` the product
# returns `B·conj(A)`, which the real path cannot tell from `B·A`.
@testset "the projection and the product are transposes on a complex element type" begin
    A = randn(ComplexF64, 4, 4)
    S = SymmetricMatrix(A)
    B = randn(ComplexF64, 3, 4)
    v = randn(ComplexF64, 4)

    @test Matrix(S) ≈ (A + transpose(A)) / 2
    @test transpose(Matrix(S)) == Matrix(S)
    @test B * S ≈ B * Matrix(S)
    @test v' * S ≈ v' * Matrix(S)
    @test transpose(v) * S ≈ transpose(v) * Matrix(S)

    # A symmetric matrix is its own transpose; only a real one is its own adjoint. `S'` returned `S`
    # for every element type, which answers `S' == S` for a complex `S` — false.
    @test Matrix(S') ≈ Matrix(S)'
    @test Matrix(S') ≉ transpose(Matrix(S))

    Ar = randn(4, 4)
    Sr = SymmetricMatrix(Ar)
    Br = randn(3, 4)
    @test Matrix(Sr) ≈ (Ar + transpose(Ar)) / 2
    @test Matrix(Sr) ≈ (Ar + Ar') / 2
    @test Br * Sr ≈ Br * Matrix(Sr)
    # and the real path keeps the method that returns the matrix itself, rather than a wrapper
    @test Sr' === Sr
end

# A row vector on the left is the one shape `*(::AbstractMatrix, ::SymmetricMatrix)` does not settle
# on its own: `LinearAlgebra` has its own method for that left operand, narrower there and wider on
# the right, so neither wins. The two tie-breakers beside that product in
# `src/special_matrices/symmetric.jl` settle it. `test/ambiguities.jl` cannot cover this pair,
# because one of its two methods is not this package's.
@testset "a row vector times a SymmetricMatrix" begin
    for T in (Float32, Float64), N in 2:5

        A = rand(SymmetricMatrix{T}, N)
        v = rand(T, N)

        @test v' * A ≈ v' * Matrix(A)
        @test transpose(v) * A ≈ transpose(v) * Matrix(A)
        @test size(v' * A) == (1, N)
    end
end
