# The tests of `test/lie_algebras/stiefel_lie_algebra_horizontal.jl`, for the symplectic lift, and
# the one property that has no Stiefel counterpart: the lift is Hamiltonian with respect to the
# Poisson tensor `𝕁̄ = diag(𝕁ₙ, 𝕁_{N-n})` of the basis the section is stored in.
using GeometricOptimizers
using GeometricOptimizers: StiefelProjection
using LinearAlgebra: I, norm
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/symplectic_points.jl")

# `X⁺ = 𝕁ᵀXᵀ𝕁` for an `m × p` matrix `X`, with the Poisson tensors of its two sizes
symplectic_adjoint(X) = poisson(eltype(X), size(X, 2))' * transpose(X) * poisson(eltype(X), size(X, 1))

# A Hamiltonian `N × N` matrix with respect to `𝕁̄`, i.e. an element of the Lie algebra the lift is
# the horizontal part of: `𝕁̄S` for a symmetric `S`.
function hamiltonian(rng, ::Type{T}, N, n) where {T}
    S = randn(rng, T, N, N)
    adapted_poisson(T, N, n) * (S + S')
end

const SIZES = ((4, 2), (6, 2), (6, 4), (8, 4), (6, 6))

function add_and_sub(rng, N, n, T)
    C = rand(rng, T, N, N)
    D = rand(rng, T, N, N)

    CD = SymplecticLieAlgHorMatrix(C + D, n)
    CD₂ = SymplecticLieAlgHorMatrix(C, n) + SymplecticLieAlgHorMatrix(D, n)
    @test eltype(CD₂) == T
    @test CD ≈ CD₂
    @test CD isa SymplecticLieAlgHorMatrix{T}
    @test CD₂ isa SymplecticLieAlgHorMatrix{T}

    CD_sub = SymplecticLieAlgHorMatrix(C - D, n)
    CD₂_sub = SymplecticLieAlgHorMatrix(C, n) - SymplecticLieAlgHorMatrix(D, n)
    @test eltype(CD₂_sub) == T
    @test CD_sub ≈ CD₂_sub
    @test CD₂_sub isa SymplecticLieAlgHorMatrix{T}
end

# The projection onto the horizontal space keeps the first `n` rows and columns of a Hamiltonian
# matrix and drops the rest, as for the Stiefel lift.
function symplectic_lie_alg_projection(rng, N, n, T)
    E = StiefelProjection(T, N, n)
    projection(W) = W - (I - E * E') * W * (I - E * E')
    W₁ = hamiltonian(rng, T, N, n)
    W₂ = hamiltonian(rng, T, N, n)
    S₁ = SymplecticLieAlgHorMatrix(W₁, n)
    S₂ = SymplecticLieAlgHorMatrix(W₂, n)
    A = rand(rng, T, N, N)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test S₃ isa SymplecticLieAlgHorMatrix{T}
    @test S₄ isa SymplecticLieAlgHorMatrix{T}
    @test S₃ ≈ projection(W₁ + W₂)
    @test S₄ ≈ projection(W₁ - W₂)
    @test S₁ + A ≈ Matrix(S₁) + A
    @test A + S₁ ≈ Matrix(S₁) + A
end

function symplectic_lie_alg_vectorization_test(rng, N, n, T)
    A = rand(rng, SymplecticLieAlgHorMatrix{T}, N, n)
    A′ = SymplecticLieAlgHorMatrix(
        vcat(vec(A.A₁), parent(A.B₁), parent(A.C₁), vec(A.B)), N, n)
    @test eltype(A′) == T
    @test A′ == A
end

function scalar_multiplication(rng, N, n, T)
    C = rand(rng, T, N, N)
    α = rand(rng, T)
    Cα = SymplecticLieAlgHorMatrix(α * C, n)
    Cα₂ = α * SymplecticLieAlgHorMatrix(C, n)
    @test eltype(Cα₂) == T
    @test Cα ≈ Cα₂
    @test Cα₂ isa SymplecticLieAlgHorMatrix{T}
end

function random_array_generation(rng, N, n, T)
    A = rand(rng, SymplecticLieAlgHorMatrix{T}, N, n)
    @test A isa SymplecticLieAlgHorMatrix{T}
    @test eltype(A) == T
    @test size(A) == (N, N)
end

function zeros_array_generation(N, n, T)
    A = zeros(SymplecticLieAlgHorMatrix{T}, N, n)
    @test A isa SymplecticLieAlgHorMatrix{T}
    @test size(A) == (N, N)
    @test all(iszero, A)

    A₆₄ = zeros(SymplecticLieAlgHorMatrix, N, n)
    @test A₆₄ isa SymplecticLieAlgHorMatrix{Float64}
    @test all(iszero, A₆₄)
end

# Every lift is in the Lie algebra of the group the section's retractions land in, and its upper
# right block is minus the symplectic adjoint of its lower left one.
function hamiltonian_structure(rng, N, n, T)
    B = rand(rng, SymplecticLieAlgHorMatrix{T}, N, n)
    𝕁 = adapted_poisson(T, N, n)
    M = Matrix(B)
    @test M' * 𝕁 + 𝕁 * M == zeros(T, N, N)
    @test M[1:n, (n + 1):N] == -symplectic_adjoint(B.B)
    @test M[(n + 1):N, (n + 1):N] == zeros(T, N - n, N - n)
end

# The products go through the blocks rather than through `getindex`.
function products(rng, N, n, T)
    B = rand(rng, SymplecticLieAlgHorMatrix{T}, N, n)
    C = randn(rng, T, N, 3)
    @test B * C ≈ Matrix(B) * C
    @test C' * B ≈ C' * Matrix(B)
    c = randn(rng, T, N)
    @test B * c ≈ Matrix(B) * c
end

@testset "SymplecticLieAlgHorMatrix projection and arithmetic, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for (N, n) in SIZES
        add_and_sub(rng, N, n, T)
        symplectic_lie_alg_projection(rng, N, n, T)
        symplectic_lie_alg_vectorization_test(rng, N, n, T)
        scalar_multiplication(rng, N, n, T)
        random_array_generation(rng, N, n, T)
        zeros_array_generation(N, n, T)
        hamiltonian_structure(rng, N, n, T)
        products(rng, N, n, T)
    end
end

# The symplectic adjoint is spelled with `transpose`, so the lift is Hamiltonian over a complex
# element type in the sense of `transpose` and not of `adjoint`, and the sum with a dense matrix
# agrees with the dense sum. The counterpart of the complex testset of the Stiefel lift.
@testset "the symplectic adjoint is a transpose on a complex element type, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    C = rand(rng, SymplecticLieAlgHorMatrix{Complex{T}}, 6, 2)
    D = randn(rng, Complex{T}, 6, 6)
    𝕁 = adapted_poisson(T, 6, 2)

    @test eltype(C + D) == Complex{T}
    @test transpose(Matrix(C)) * 𝕁 + 𝕁 * Matrix(C) == zeros(Complex{T}, 6, 6)
    @test C + D ≈ Matrix(C) + D
    @test D + C ≈ D + Matrix(C)
end

@testset "an odd size is refused" begin
    @test_throws AssertionError SymplecticLieAlgHorMatrix(randn(2, 2), SymmetricMatrix(randn(2, 2)),
        SymmetricMatrix(randn(2, 2)), randn(3, 4), 7, 4)
    @test_throws AssertionError SymplecticLieAlgHorMatrix(randn(6, 6), 3)
end
