using GeometricOptimizers
using GeometricOptimizers: StiefelProjection
using LinearAlgebra: I
using Test
import Random

include("../helpers/eltypes.jl")

@doc raw"""
This function tests addition for various custom arrays, i.e. if \(A + B\) is performed in the correct way.
"""
function add_and_sub(rng, n::Int, N::Int, T::Type)
    C = rand(rng, T, N, N)
    D = rand(rng, T, N, N)

    # StiefelLieAlgHorMatrix
    CD_slahm = StiefelLieAlgHorMatrix(C + D, n)
    CD_slahm2 = StiefelLieAlgHorMatrix(C, n) + StiefelLieAlgHorMatrix(D, n)
    @test eltype(CD_slahm2) == T
    @test CD_slahm ≈ CD_slahm2
    @test typeof(CD_slahm) <: StiefelLieAlgHorMatrix{T}
    @test typeof(CD_slahm2) <: StiefelLieAlgHorMatrix{T}

    CD_slahm_sub = StiefelLieAlgHorMatrix(C - D, n)
    CD_slahm2_sub = StiefelLieAlgHorMatrix(C, n) - StiefelLieAlgHorMatrix(D, n)
    @test eltype(CD_slahm2_sub) == T
    @test CD_slahm_sub ≈ CD_slahm2_sub
    @test typeof(CD_slahm_sub) <: StiefelLieAlgHorMatrix{T}
    @test typeof(CD_slahm2_sub) <: StiefelLieAlgHorMatrix{T}
end

function stiefel_lie_alg_projection(rng, n::Integer, N::Integer, T::DataType)
    E = StiefelProjection(T, N, n)
    projection(W::SkewSymMatrix) = W - (I - E * E') * W * (I - E * E')
    W₁ = SkewSymMatrix(rand(rng, T, N, N))
    S₁ = StiefelLieAlgHorMatrix(W₁, n)
    W₂ = SkewSymMatrix(rand(rng, T, N, N))
    S₂ = StiefelLieAlgHorMatrix(W₂, n)
    A = rand(rng, T, N, N)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: StiefelLieAlgHorMatrix
    @test typeof(S₄) <: StiefelLieAlgHorMatrix
    @test eltype(S₃) == T
    @test eltype(S₄) == T
    # `I - E * E'` holds only zeros and ones, so the projection selects entries of `W₁ ± W₂` and adds
    # exact zeros, and both sides round the same sums of the same stored entries; `eps(T)` admits
    # no more than one rounding of an entry below 1
    @test all(abs.(projection(W₁ + W₂) .- S₃) .< eps(T))
    @test all(abs.(projection(W₁ - W₂) .- S₄) .< eps(T))
    # check custom addition
    @test S₁ + A ≈ Matrix(S₁) + A
    @test A + S₁ ≈ Matrix(S₁) + A
end

function stiefel_lie_alg_vectorization_test(rng, n::Integer, N::Integer, T::DataType)
    A = rand(rng, StiefelLieAlgHorMatrix{T}, N, n)
    A′ = StiefelLieAlgHorMatrix(vcat(parent(A.A), vec(A.B)), N, n)
    @test eltype(A′) == T
    @test isapprox(A′, A)
end

function scalar_multiplication(rng, n::Integer, N::Integer, T::DataType)
    C = rand(rng, T, N, N)
    α = rand(rng, T)

    # StiefelLieAlgHorMatrix
    Cα_slahm = StiefelLieAlgHorMatrix(α * C, n)
    Cα_slahm2 = α * StiefelLieAlgHorMatrix(C, n)
    @test eltype(Cα_slahm2) == T
    @test Cα_slahm ≈ Cα_slahm2
    @test typeof(Cα_slahm) <: StiefelLieAlgHorMatrix{T}
    @test typeof(Cα_slahm2) <: StiefelLieAlgHorMatrix{T}
end

function random_array_generation(rng, n::Integer, N::Integer, T::DataType)
    A_stiefel_hor = rand(rng, StiefelLieAlgHorMatrix{T}, N, n)
    @test typeof(A_stiefel_hor) <: StiefelLieAlgHorMatrix{T}
    @test eltype(A_stiefel_hor) == T
end

# The non-parametric method delegates to `zeros(SkewSymMatrix, n)` and to `zeros(N - n, n)`,
# so it is `Float64` throughout. It broke once when `zeros(SkewSymMatrix, n)` was removed in
# favour of the parametric method only, and nothing in the suite noticed.
function zeros_array_generation(n::Integer, N::Integer, T::DataType)
    A = zeros(StiefelLieAlgHorMatrix{T}, N, n)
    @test A isa StiefelLieAlgHorMatrix{T}
    @test size(A) == (N, N)
    @test all(iszero, A)

    A₆₄ = zeros(StiefelLieAlgHorMatrix, N, n)
    @test A₆₄ isa StiefelLieAlgHorMatrix{Float64}
    @test size(A₆₄) == (N, N)
    @test all(iszero, A₆₄)
end

# `getindex` builds the upper-right block as `-B.B[j, i]`, entrywise and without conjugating, so the
# lift is skew-*symmetric* rather than skew-Hermitian. `+(::StiefelLieAlgHorMatrix,
# ::AbstractMatrix)` rebuilds that block and spells it `-transpose(B.B)` for that reason. With
# `-B.B'` the sum disagrees with the dense sum on a complex element type, and agrees on a real one —
# which is the difference the real path cannot see, and why this testset is here. See the matching
# one in `test/special_matrices/skew_symmetric.jl`.
@testset "the sum rebuilds the block as a transpose on a complex element type, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    C = StiefelLieAlgHorMatrix(
        SkewSymMatrix(randn(rng, Complex{T}, 2, 2)), randn(rng, Complex{T}, 2, 2), 4, 2)
    D = randn(rng, Complex{T}, 4, 4)

    @test eltype(C + D) == Complex{T}
    @test transpose(Matrix(C)) == -Matrix(C)
    @test C + D ≈ Matrix(C) + D
    @test D + C ≈ D + Matrix(C)
end

@testset "StiefelLieAlgHorMatrix projection and arithmetic, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for N in 3:5
        for n in 1:N
            add_and_sub(rng, n, N, T)
            stiefel_lie_alg_projection(rng, n, N, T)
            stiefel_lie_alg_vectorization_test(rng, n, N, T)
            scalar_multiplication(rng, n, N, T)
            random_array_generation(rng, n, N, T)
            zeros_array_generation(n, N, T)
        end
    end
end
