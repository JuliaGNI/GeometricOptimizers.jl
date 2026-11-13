# The manifold of biorthogonal pairs `𝓑(n, N) ≅ GL(N)/GL(N - n)`, its global tangent space
# `BiorthogonalLieAlgHorMatrix`, its section and its retractions.
#
# The group acts by `g⋅(Φ, Ψ) = (gΦ, g⁻ᵀΨ)`, so every statement about the second half of a pair is
# checked against the dense inverse transpose of the frame, which the package itself never forms.
using GeometricOptimizers
using GeometricOptimizers: check, global_section, Ω, apply_section, gradient_rep!,
                           retraction_differential, retraction_differential!,
                           update_section!, RetractionWorkspace, StiefelProjection, _dot,
                           zero_tangent, solution_scale
using LinearAlgebra: I, cond, dot, norm
using Test
import Random

include("../helpers/eltypes.jl")

const SIZES = ((4, 1), (6, 2), (7, 3), (5, 5))

halves(A) = (n = size(A, 2) ÷ 2; (A[:, 1:n], A[:, (n + 1):(2n)]))
frame(λY) = Matrix(λY)
dual_frame(λY) = (n = size(λY.Y, 2) ÷ 2; hcat(λY.Y[:, (n + 1):(2n)], halves(λY.λ)[2]))
# the tangent vector `λX⋅(E, E) = (λXE, -λ⁻ᵀXᵀE)` of a lift at a section
function tangent(λY, X)
    n = X.n
    hcat(frame(λY) * Matrix(X)[:, 1:n], -dual_frame(λY) * Matrix(X)'[:, 1:n])
end

@testset "points, $(T), $(N)×$(n)" for T in REAL_ELTYPES, (N, n) in SIZES
    rng = Random.Xoshiro(1)
    Y = rand(rng, BiorthogonalManifold{T}, N, 2n)
    @test Y isa BiorthogonalManifold{T, Matrix{T}}
    @test size(Y) == (N, 2n)
    Φ, Ψ = halves(Y)
    # `Ψ` is `Φ(ΦᵀΦ)⁻¹` plus a part orthogonal to `Φ`, so the residual grows with the condition
    # number of `ΦᵀΦ`; the square case `n = N` draws a square Gaussian, whose tail is heavy
    @test check(Y) < 100 * eps(T) * cond(Φ)^2
    @test Ψ' * Φ ≈ I rtol = 100 * eps(T) * cond(Φ)^2
    @test copy(Y) == Y
    @test copy(Y) isa BiorthogonalManifold{T}
    @test_throws AssertionError BiorthogonalManifold(rand(T, N, 3))
    # a Stiefel point is a biorthogonal pair
    U = rand(rng, StiefelManifold{T}, N, n)
    @test check(BiorthogonalManifold(hcat(U.A, U.A))) < 100eps(T)
    # the manifold is not compact, so the scale of a relative change is measured; it is `√(2n)` at
    # a Stiefel point and above it elsewhere
    @test solution_scale(Y) ≈ norm(Y.A)
    @test solution_scale(Y) > √T(2n)
    @test solution_scale(BiorthogonalManifold(hcat(U.A, U.A))) ≈ √T(2n)
end

@testset "a complex element type is refused" begin
    @test_throws ArgumentError BiorthogonalManifold(zeros(ComplexF64, 6, 4))
    @test_throws ArgumentError rand(BiorthogonalManifold{ComplexF64}, 6, 4)
end

@testset "rgrad is the orthogonal projection onto the tangent space, $(N)×$(n)" for (N, n) in SIZES
    rng = Random.Xoshiro(2)
    Y = rand(rng, BiorthogonalManifold{Float64}, N, 2n)
    Φ, Ψ = halves(Y)
    ∇L = randn(rng, N, 2n)
    Δ = rgrad(Y, ∇L)
    ΔΦ, ΔΨ = halves(Δ)
    # tangent
    @test norm(Ψ' * ΔΦ + ΔΨ' * Φ) < 1e-12
    # idempotent
    @test rgrad(Y, Δ) ≈ Δ
    # the residual is normal: orthogonal to every tangent vector
    λY = GlobalSection(Y)
    for _ in 1:3
        V = tangent(λY, rand(rng, BiorthogonalLieAlgHorMatrix{Float64}, N, n))
        @test abs(dot(∇L - Δ, V)) < 1e-10 * norm(∇L) * norm(V)
        @test metric(Y, Δ, V) ≈ dot(∇L, V)
    end
    @test_throws DimensionMismatch rgrad(Y, randn(N, n))
end

@testset "the section and its dual, $(T), $(N)×$(n)" for T in REAL_ELTYPES, (N, n) in SIZES
    rng = Random.Xoshiro(3)
    Y = rand(rng, BiorthogonalManifold{T}, N, 2n)
    λ = global_section(Y)
    @test size(λ) == (N, 2(N - n))
    λY = GlobalSection(Y)
    K, K̃ = halves(λY.λ)
    Φ, Ψ = halves(Y)
    @test K' * K ≈ I
    @test norm(Ψ' * K) < 100 * eps(T) * N
    @test frame(λY) * dual_frame(λY)' ≈ I rtol = 100 * eps(T) * cond(Φ)^2
    @test size(λY) == (N, N)
    @test frame(λY)[:, 1:n] == Φ

    # at a Stiefel point the dual frame is the frame, and the section is the Stiefel one
    U = rand(rng, StiefelManifold{T}, N, n)
    λU = GlobalSection(BiorthogonalManifold(hcat(U.A, U.A)))
    K, K̃ = halves(λU.λ)
    @test K̃ ≈ K atol = 100eps(T)
    @test frame(λU)' * frame(λU) ≈ I atol = 100eps(T)
end

# A valid pair whose `ΨᵀΨ` rounds to a singular matrix, so that CholeskyQR2 cannot orthonormalize `Ψ`
# and the section takes the Householder QR.
@testset "the section of a pair CholeskyQR2 cannot orthonormalize" begin
    Ψ = [1.0 1.0; 0.0 2.0^-30; 0.0 0.0]
    Φ = [1.0 0.0; -2.0^30 2.0^30; 0.0 0.0]
    @test Ψ' * Φ == I
    @test GeometricOptimizers._cholesky_qr2(Ψ) === nothing
    λY = GlobalSection(BiorthogonalManifold(hcat(Φ, Ψ)))
    K, K̃ = halves(λY.λ)
    @test K' * K ≈ I atol = 8eps()
    @test norm(Ψ' * K) < 8eps() * norm(Ψ)
    @test norm(dual_frame(λY)' * frame(λY) - I) < 8eps() * norm(Φ) * norm(Ψ)
end

@testset "global_rep, gradient_rep! and Ω, $(N)×$(n)" for (N, n) in SIZES
    rng = Random.Xoshiro(4)
    Y = rand(rng, BiorthogonalManifold{Float64}, N, 2n)
    λY = GlobalSection(Y)
    ∇L = randn(rng, N, 2n)
    Δ = rgrad(Y, ∇L)
    X = global_rep(λY, Δ)
    @test X isa BiorthogonalLieAlgHorMatrix{Float64}
    @test tangent(λY, X) ≈ Δ
    @test Ω(Y, Δ) ≈ frame(λY) * Matrix(X) / frame(λY)
    ΩΔ = Ω(Y, Δ)
    Φ, Ψ = halves(Y)
    @test hcat(ΩΔ * Φ, -ΩΔ' * Ψ) ≈ Δ

    # the derivative of the loss along `λX⋅(E, E)`, for every lift `X`
    D = zero_tangent(Y)
    gradient_rep!(D, λY, Δ)
    for _ in 1:5
        B = rand(rng, BiorthogonalLieAlgHorMatrix{Float64}, N, n)
        @test _dot(D, B) ≈ dot(∇L, tangent(λY, B))
    end

    # `global_rep` overwrites what its destination held
    D = zero_tangent(Y)
    foreach(b -> fill!(b, NaN), parent(D))
    GeometricOptimizers.global_rep!(D, λY, Δ)
    @test D ≈ X

    # `_dot` on the stored blocks is not the metric, and `_metric_apply!` is its Gram matrix
    B, C = (rand(rng, BiorthogonalLieAlgHorMatrix{Float64}, N, n) for _ in 1:2)
    MB = GeometricOptimizers._metric_apply!(zero_tangent(Y), λY, B)
    @test _dot(MB, C) ≈ metric(Y, tangent(λY, B), tangent(λY, C))
    @test !(_dot(B, C) ≈ metric(Y, tangent(λY, B), tangent(λY, C)))

    @test_throws DimensionMismatch Ω(Y, randn(rng, N, 2n + 1))
end

@testset "Ω reduces to the canonical lift of the Stiefel manifold" begin
    rng = Random.Xoshiro(5)
    U = rand(rng, StiefelManifold{Float64}, 7, 3)
    Δ = rgrad(U, randn(rng, 7, 3))
    @test Ω(BiorthogonalManifold(hcat(U.A, U.A)), hcat(Δ, Δ)) ≈ Matrix(Ω(U, Δ))
end

@testset "the lift, $(N)×$(n)" for (N, n) in SIZES
    rng = Random.Xoshiro(6)
    W = randn(rng, N, N)
    X = BiorthogonalLieAlgHorMatrix(W, n)
    projected = copy(W)
    projected[(n + 1):N, (n + 1):N] .= 0
    @test X == projected
    @test Matrix(X) == projected
    @test BiorthogonalLieAlgHorMatrix(vcat(vec(X.A), vec(X.B), vec(X.C)), N, n) == X
    @test GeometricOptimizers.manifold_type(X) == BiorthogonalManifold

    C = randn(rng, N, 4)
    @test X * C ≈ projected * C
    @test C' * X ≈ C' * projected
    c = randn(rng, N)
    @test X * c ≈ projected * c
    @test c' * X ≈ c' * projected

    Y = rand(rng, BiorthogonalLieAlgHorMatrix{Float64}, N, n)
    @test X + Y ≈ projected + Matrix(Y)
    @test X - Y ≈ projected - Matrix(Y)
    @test (X + Y) isa BiorthogonalLieAlgHorMatrix
    @test 2X ≈ 2projected
    @test -X ≈ -projected
    @test zero(X) == zeros(N, N)
    @test similar(X) isa BiorthogonalLieAlgHorMatrix{Float64}
    @test zeros(BiorthogonalLieAlgHorMatrix{Float32}, N, n) == zeros(Float32, N, N)
    @test _dot(X, Y) ≈ dot(projected, Matrix(Y))
end

@testset "retractions, $(nameof(typeof(R))), $(N)×$(n)" for R in (Cayley(), Geodesic()),
    (N, n) in SIZES

    rng = Random.Xoshiro(7)
    X = rand(rng, BiorthogonalLieAlgHorMatrix{Float64}, N, n) * 0.3
    pair = R(X)
    @test pair isa BiorthogonalManifold{Float64}
    @test size(pair) == (N, 2N)
    g, g⁻ᵀ = halves(pair)
    dense = R isa Cayley ? (I - Matrix(X) / 2) \ (I + Matrix(X) / 2) : exp(Matrix(X))
    @test g ≈ dense
    @test g⁻ᵀ ≈ inv(dense)'
    @test check(pair) < 1e-13

    # the point `λ(Y)R(X)⋅(E, E)`
    Y = rand(rng, BiorthogonalManifold{Float64}, N, 2n)
    Δ = rgrad(Y, randn(rng, N, 2n)) * 0.1
    Y₂ = R isa Cayley ? cayley(Y, Δ) : geodesic(Y, Δ)
    @test Y₂ isa BiorthogonalManifold{Float64}
    @test check(Y₂) < 1e-12
    @test Y₂ != Y

    # the transport of the section, with and without a workspace
    λ₁ = GlobalSection(Y)
    λ₂ = GlobalSection(Y)
    copyto!(λ₂.λ, λ₁.λ)
    G = frame(λ₁)
    update_section!(λ₁, X, R)
    update_section!(λ₂, X, R, RetractionWorkspace(Y, R))
    @test frame(λ₁) ≈ G * dense
    @test dual_frame(λ₁) ≈ inv(G * dense)'
    @test frame(λ₂) ≈ frame(λ₁)
    @test dual_frame(λ₂) ≈ dual_frame(λ₁)
    @test check(λ₁.Y) < 1e-12
end

@testset "the Cayley differential, $(N)×$(n)" for (N, n) in SIZES
    rng = Random.Xoshiro(8)
    X = rand(rng, BiorthogonalLieAlgHorMatrix{Float64}, N, n) * 0.3
    Y = rand(rng, BiorthogonalManifold{Float64}, N, 2n)
    λY = GlobalSection(Y)
    function curve(α)
        λ = GlobalSection(Y)
        copyto!(λ.λ, λY.λ)
        update_section!(λ, α * X, Cayley())
        λ
    end
    α, h = 0.7, cbrt(eps())
    Λ = curve(α)
    D = retraction_differential(Cayley(), X, α)
    @test D isa BiorthogonalLieAlgHorMatrix{Float64}
    @test (curve(α + h).Y.A - curve(α - h).Y.A) / 2h ≈ tangent(Λ, D) rtol = 1e-8
    D₂ = zero(X)
    retraction_differential!(D₂, RetractionWorkspace(Y, Cayley()), Cayley(), X, α)
    @test D₂ == D
    @test retraction_differential(Cayley(), X, 0) === X
    @test retraction_differential(Geodesic(), X, α) === X
end

@testset "ProjectedSkew refuses a biorthogonal lift" begin
    X = rand(BiorthogonalLieAlgHorMatrix{Float64}, 5, 2)
    @test_throws ArgumentError geodesic(X, GeometricOptimizers.ProjectedSkew())
end

# Under `X ↦ diag(X, -Xᵀ)` the biorthogonal lift is a symplectic one, in the basis `[Φ Ψ K K̃]` that a
# symplectic section of `diag(Φ, Ψ)` is stored in, and the retractions agree.
@testset "the biorthogonal lift is a block-diagonal symplectic lift" begin
    N, n = 6, 2
    m = N - n
    X = rand(Random.Xoshiro(9), BiorthogonalLieAlgHorMatrix{Float64}, N, n)
    ℬ = zeros(2m, 2n)
    ℬ[1:m, 1:n] = X.B
    ℬ[(m + 1):(2m), (n + 1):(2n)] = -X.C'
    S = SymplecticLieAlgHorMatrix(X.A, zeros(SymmetricMatrix{Float64}, n),
        zeros(SymmetricMatrix{Float64}, n), ℬ, 2N, 2n)
    perm = vcat(1:n, (2n + 1):(2n + m), (n + 1):(2n), (2n + m + 1):(2N))
    O = zeros(N, N)
    @test Matrix(S)[perm, perm] == [Matrix(X) O; O -Matrix(X)']
    g, g⁻ᵀ = halves(cayley(X))
    @test cayley(S).A[perm, perm] ≈ [g O; O g⁻ᵀ]
end
