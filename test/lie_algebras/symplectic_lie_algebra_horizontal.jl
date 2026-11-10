# The tests of `test/lie_algebras/stiefel_lie_algebra_horizontal.jl`, for the symplectic lift, and
# the one property that has no Stiefel counterpart: the lift is Hamiltonian with respect to the
# Poisson tensor `𝕁̄ = diag(𝕁ₙ, 𝕁_{N-n})` of the basis the section is stored in.
using GeometricOptimizers
using GeometricOptimizers: StiefelProjection
using LinearAlgebra: I, Symmetric, cond, norm, rank
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/symplectic_points.jl")

# `X⁺ = 𝕁ᵀXᵀ𝕁` for an `m × p` matrix `X`, with the Poisson tensors of its two sizes
function symplectic_adjoint(X)
    poisson(eltype(X), size(X, 2))' * transpose(X) * poisson(eltype(X), size(X, 1))
end

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

# `getindex` takes integers, so a slice goes through the `AbstractArray` fallback.
function slices(rng, N, n, T)
    B = rand(rng, SymplecticLieAlgHorMatrix{T}, N, n)
    @test B[1:n, 1:n] == Matrix(B)[1:n, 1:n]
    @test B[:, 1] == Matrix(B)[:, 1]
    @test B[(n + 1):N, :] == Matrix(B)[(n + 1):N, :]
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
        slices(rng, N, n, T)
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

# The dimension of the space, `4Nn − n(2n − 1)` for a `2N × 2n` point, is that of the manifold, and it
# comes from Bendokat and Zimmermann and the book, not from the code. `V ↦ VE` maps the lifts
# injectively onto the tangent space at `E`, `{Δ : Δᵀ𝕁̄E + Eᵀ𝕁̄Δ = 0}` in the adapted basis: the
# images of a basis of the stored entries are tangent, of full rank, and as many as the dimension of
# that space, which the null space of its linear condition gives.
@testset "V ↦ VE maps the lifts onto the tangent space at E, $T" for T in REAL_ELTYPES
    for (N2, n2) in (SIZES..., (10, 4))
        N, n = N2 ÷ 2, n2 ÷ 2
        dimension = 4N * n - n * (2n - 1)
        k = n2 ÷ 2
        stored = k^2 + k * (k + 1) + (N2 - n2) * n2
        @test stored == dimension

        E = Matrix{T}(StiefelProjection(T, N2, n2))
        𝕁 = adapted_poisson(T, N2, n2)
        basis = [SymplecticLieAlgHorMatrix(T.(1:stored .== i), N2, n2) for i in 1:stored]
        @test rank(reduce(hcat, vec.(Matrix.(basis)))) == dimension
        images = [B * E for B in basis]
        @test all(Δ -> norm(Δ' * 𝕁 * E + E' * 𝕁 * Δ) == 0, images)
        @test rank(reduce(hcat, vec.(images))) == dimension

        # the tangent condition as a linear map on `vec(Δ)`, and the dimension of its null space
        unit(i) = reshape(T.(1:(N2 * n2) .== i), N2, n2)
        condition = reduce(hcat, [vec(Δ' * 𝕁 * E + E' * 𝕁 * Δ) for Δ in unit.(1:(N2 * n2))])
        @test N2 * n2 - rank(condition) == dimension
    end
end

# `Ad_h` maps the space to itself for `h = diag(I, S)` in the stabiliser of `E`, `S` symplectic: the
# conjugate `hXh⁻¹` of a lift is its own projection onto the space. Two controls must fail: an `h`
# outside the stabiliser, `exp` of a lift, and a vertical element `diag(𝕆, 𝕁H)`, whose projection is
# zero.
@testset "Ad_h preserves the lifts for h in the stabiliser of E, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(321)
    for (N2, n2) in ((6, 2), (8, 4), (10, 4))
        m = N2 - n2
        X = Matrix(rand(rng, SymplecticLieAlgHorMatrix{T}, N2, n2))
        S = exp(poisson(T, m) * T.(Symmetric(randn(rng, m, m))) / 4)
        h = cat(Matrix{T}(I, n2, n2), S; dims = (1, 2))
        @test h' * adapted_poisson(T, N2, n2) * h ≈ adapted_poisson(T, N2, n2)
        Y = h * X / h
        @test isapprox(Matrix(SymplecticLieAlgHorMatrix(Y, n2)), Y;
            rtol = 64eps(T) * cond(S)^2)

        g = exp(Matrix(rand(rng, SymplecticLieAlgHorMatrix{T}, N2, n2)) / 4)
        Z = g * X / g
        @test !isapprox(Matrix(SymplecticLieAlgHorMatrix(Z, n2)), Z; rtol = T(0.01))

        H = T.(Symmetric(randn(rng, m, m)))
        V = cat(zeros(T, n2, n2), poisson(T, m) * H; dims = (1, 2))
        @test iszero(Matrix(SymplecticLieAlgHorMatrix(V, n2)))
        @test !iszero(V)
    end
end

@testset "an odd size is refused" begin
    @test_throws AssertionError SymplecticLieAlgHorMatrix(
        randn(2, 2), SymmetricMatrix(randn(2, 2)),
        SymmetricMatrix(randn(2, 2)), randn(3, 4), 7, 4)
    @test_throws AssertionError SymplecticLieAlgHorMatrix(randn(6, 6), 3)
end
