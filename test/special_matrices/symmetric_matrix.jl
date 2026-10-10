using GeometricOptimizers
using GeometricOptimizers: map_to_S, freeparameters
using LinearAlgebra: transpose
using Test
import Random

include("../helpers/eltypes.jl")

# import ChainRulesTestUtils

symmetrize(W::AbstractMatrix{T}) where {T} = T(0.5) * (W + W')

function sym_mat_add_sub(rng, n::Integer, T::DataType)
    W₁ = rand(rng, T, n, n)
    S₁ = SymmetricMatrix(W₁)
    W₂ = rand(rng, T, n, n)
    S₂ = SymmetricMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SymmetricMatrix
    @test typeof(S₄) <: SymmetricMatrix
    @test eltype(S₃) == T
    @test eltype(S₄) == T
    # entries of `rand` are in `[0, 1)`: the reference rounds two sums below 2 (`eps(T)/2` each) and
    # their sum below 4 (`eps(T)`) and halves exactly, so it is off by at most `eps(T)`; `S₁ ± S₂`
    # rounds two sums below 2 (`eps(T)/2` each, then halved) and one sum below 2 (`eps(T)/2`), so
    # by at most `eps(T)` as well. Together `2 eps(T)`.
    @test all(abs.(symmetrize(W₁ + W₂) - S₃) .< 2 * eps(T))
    @test all(abs.(symmetrize(W₁ - W₂) - S₄) .< 2 * eps(T))
end

function random_generation(rng, N::Integer, T::DataType)
    A_sym = rand(rng, SymmetricMatrix{T}, N)
    @test typeof(A_sym) <: SymmetricMatrix{T}
    @test eltype(A_sym) == T
end

function multiplication(rng, n::Integer, T::DataType)
    A = rand(rng, SymmetricMatrix{T}, n)
    b = rand(rng, T, n)
    B = rand(rng, T, n, n)
    # test if the custom multiplication is performed the right way
    @test eltype(A * b) == T
    @test eltype(A * B) == T
    @test A * b ≈ Matrix{T}(A) * b
    @test A * B ≈ Matrix{T}(A) * B
end

function calling_symmetric_matrix(rng, n::Integer, T::DataType)
    B = rand(rng, T, n, n)
    @test eltype(SymmetricMatrix(B)) == T
    @test isapprox(SymmetricMatrix(B), (B + B') / 2)
end

function test_pullback_routine(n::Integer = 5, T::DataType = Float32)
    A = rand(SymmetricMatrix{T}, n)
    B = rand(T, n, n)

    @test ChainRulesTestUtils.rrule(*, A, B)
end

function scalar_multiplication(rng, n::Integer, T::DataType)
    A = rand(rng, T, n, n)
    α = rand(rng, T)

    # SymmetricMatrix
    Aα_sym = SymmetricMatrix(α * A)
    Aα_sym2 = α * SymmetricMatrix(A)
    @test eltype(Aα_sym2) == T
    @test Aα_sym ≈ Aα_sym2
    @test typeof(Aα_sym) <: SymmetricMatrix{T}
    @test typeof(Aα_sym2) <: SymmetricMatrix{T}
end

@testset "SymmetricMatrix projection and arithmetic, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for n in 1:5
        sym_mat_add_sub(rng, n, T)
        random_generation(rng, n, T)
        multiplication(rng, n, T)
        calling_symmetric_matrix(rng, n, T)
        scalar_multiplication(rng, n, T)
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
    @test SymmetricMatrix(freeparameters(SymmetricMatrix(M)), 4) ≈ SymmetricMatrix(M)
    @test vec(SymmetricMatrix(M)) == vec(Matrix(SymmetricMatrix(M)))
end

# see `the projection and the product are transposes on a complex element type` in
# `skew_symmetric.jl`: `SymmetricMatrix` is the set `{M : Mᵀ = M}`, so the projection and
# `*(::AbstractMatrix, ::SymmetricMatrix)` both spell it `transpose`. With `adjoint` the product
# returns `B·conj(A)`, which the real path cannot tell from `B·A`.
@testset "the projection and the product are transposes on a complex element type, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    A = randn(rng, Complex{T}, 4, 4)
    S = SymmetricMatrix(A)
    B = randn(rng, Complex{T}, 3, 4)
    v = randn(rng, Complex{T}, 4)

    @test eltype(S) == Complex{T}
    @test eltype(B * S) == Complex{T}
    @test Matrix(S) ≈ (A + transpose(A)) / 2
    @test transpose(Matrix(S)) == Matrix(S)
    @test B * S ≈ B * Matrix(S)
    @test v' * S ≈ v' * Matrix(S)
    @test transpose(v) * S ≈ transpose(v) * Matrix(S)

    # A symmetric matrix is its own transpose; only a real one is its own adjoint. `S'` returned `S`
    # for every element type, which answers `S' == S` for a complex `S` — false.
    @test Matrix(S') ≈ Matrix(S)'
    @test Matrix(S') ≉ transpose(Matrix(S))

    Ar = randn(rng, T, 4, 4)
    Sr = SymmetricMatrix(Ar)
    Br = randn(rng, T, 3, 4)
    @test eltype(Br * Sr) == T
    @test Matrix(Sr) ≈ (Ar + transpose(Ar)) / 2
    @test Matrix(Sr) ≈ (Ar + Ar') / 2
    @test Br * Sr ≈ Br * Matrix(Sr)
    # and the real path keeps the method that returns the matrix itself, rather than a wrapper
    @test Sr' === Sr
end

# A row vector on the left is the one shape `*(::AbstractMatrix, ::SymmetricMatrix)` does not settle
# on its own: `LinearAlgebra` has its own method for that left operand, narrower there and wider on
# the right, so neither wins. The two row-vector methods in `src/ambiguities.jl` settle it.
@testset "a row vector times a SymmetricMatrix, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for N in 2:5
        A = rand(rng, SymmetricMatrix{T}, N)
        v = rand(rng, T, N)

        @test eltype(v' * A) == T
        @test v' * A ≈ v' * Matrix(A)
        @test transpose(v) * A ≈ transpose(v) * Matrix(A)
        @test size(v' * A) == (1, N)
    end
end
