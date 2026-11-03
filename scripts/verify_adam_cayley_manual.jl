# Checks the formulas of the manual pages "Standard Neural Network Optimizers" and "Retractions"
# against the code, each with the wrong form as a control that has to fail.
#
# Run with the repository as the active project:
#
#     julia --startup-file=no --project=. scripts/verify_adam_cayley_manual.jl
#
# What is checked:
#
#   * the two weights of the Adam second moment sum to one, and a run of `Adam` follows them;
#   * the direction of `Adam` divides by `√m₂ + δ`, with `δ` outside the root;
#   * the Woodbury line for `(𝕀 - ½B̄)⁻¹` has the `2n × 2n` matrix `𝕀 - ½(B'')ᵀB'` in the middle;
#   * `Cayley(B̄) = 𝕀 + B'(𝕀 - ½(B'')ᵀB')⁻¹(B'')ᵀ`, which is what `cayley` computes;
#   * the schedule of `AdamOptimizerWithDecay(n)` counts `optimization_step!` calls and goes on
#     falling after `n`.
#
# The Cayley reference is the definition, `(𝕀 - ½B̄)⁻¹(𝕀 + ½B̄)`, solved densely. The Adam reference
# is the recursion written out here, which shares no code with `update!`.

using GeometricOptimizers
using GeometricOptimizers: Adam, AdamOptimizerWithDecay, Static, TrainingOptimizer, cayley,
                           first_moment, lift_factors, optimization_step!, second_moment,
                           step_size
using LinearAlgebra: I, cond, norm
using Random: Xoshiro
using Test

relative(a, b) = norm(a - b) / norm(b)

@testset "the weights of the Adam second moment" begin
    ρ₁, ρ₂ = 0.9, 0.99
    weights(numerator_base, t) = (
        (ρ₂ - numerator_base^t) / (1 - ρ₂^t), (1 - ρ₂) / (1 - ρ₂^t))

    for t in 1:50
        @test sum(weights(ρ₂, t)) ≈ 1 rtol = 8eps()
    end
    # the control: with `ρ₁^t` in the numerator the weights do not sum to one
    @test abs(sum(weights(ρ₁, 2)) - 1) > 0.01
    @test abs(sum(weights(ρ₁, 50)) - 1) > 1e-4
end

@testset "a run of Adam follows the printed recursion and direction" begin
    β₁, β₂, δ, η = 0.9, 0.99, 1e-2, 1e-3     # a δ large enough to tell the two placements apart
    rng = Xoshiro(11)
    x = randn(rng, 8)
    opt = TrainingOptimizer(x; algorithm = Adam(; β₁, β₂, δ), linesearch = Static(η))

    m₁, m₂ = zero(x), zero(x)
    wrong₂ = zero(x)
    for t in 1:20
        g = 1.0e-2 .* randn(rng, 8)
        x_before = copy(x)
        optimization_step!(x, opt, g)

        m₁ .= (β₁ - β₁^t) / (1 - β₁^t) .* m₁ .+ (1 - β₁) / (1 - β₁^t) .* g
        m₂ .= (β₂ - β₂^t) / (1 - β₂^t) .* m₂ .+ (1 - β₂) / (1 - β₂^t) .* g .^ 2
        wrong₂ .= (β₂ - β₁^t) / (1 - β₂^t) .* wrong₂ .+ (1 - β₂) / (1 - β₂^t) .* g .^ 2

        @test relative(first_moment(opt.state), m₁) < 64eps()
        @test relative(second_moment(opt.state), m₂) < 64eps()
        @test relative(x - x_before, -η .* m₁ ./ (sqrt.(m₂) .+ δ)) < 1e-11

        # the controls: `ρ₁^t` in the numerator, and `δ` inside the root
        t ≥ 2 && @test relative(wrong₂, m₂) > 1e-3
        @test relative(x - x_before, -η .* m₁ ./ sqrt.(m₂ .+ δ)) > 1e-3
    end
end

@testset "the Woodbury line and the final Cayley formula on St($n, $N)" for (N, n) in ((
    6, 3), (8, 2),
    (20, 4))
    rng = Xoshiro(N)
    B = rand(rng, StiefelLieAlgHorMatrix{Float64}, N, n)
    B̂, B̄ᵀ = lift_factors(B)
    dense = Matrix(B)
    @test B̂ * B̄ᵀ' ≈ dense rtol = 8eps()

    𝕀 = Matrix{Float64}(I, N, N)
    𝕀₂ₙ = Matrix{Float64}(I, 2n, 2n)
    X = B̄ᵀ' * B̂
    C = inv(𝕀₂ₙ - X / 2)
    reference_inverse = inv(𝕀 - dense / 2)
    reference = (𝕀 - dense / 2) \ (𝕀 + dense / 2)
    tol = 64 * eps() * max(cond(𝕀 - dense / 2), cond(𝕀₂ₙ - X / 2))

    @test relative(𝕀 + B̂ * C * B̄ᵀ' / 2, reference_inverse) < tol
    @test relative(𝕀 + B̂ * C * B̄ᵀ', reference) < tol
    @test relative(Matrix(cayley(B)), reference) < tol

    # the controls. The middle factor `(𝕀 - ½B'(B'')ᵀ)⁻¹` is `N × N`, so it fits between `B'` and
    # `(B'')ᵀ` at `N = 2n` alone, and it is wrong there as well.
    wrong_middle = inv(𝕀 - B̂ * B̄ᵀ' / 2)
    if N == 2n
        @test relative(𝕀 + B̂ * wrong_middle * B̄ᵀ' / 2, reference_inverse) > 0.01
    else
        @test_throws DimensionMismatch B̂ * wrong_middle
    end
    @test relative(𝕀 + B̂ * C * B̄ᵀ' * (𝕀 + dense / 2) / 2, reference) > 0.01
end

@testset "the schedule counts steps and has no floor" begin
    η₁, η₂, n = 1e-2, 1e-4, 10
    pairing = AdamOptimizerWithDecay(n; η₁, η₂)
    x = ones(4)
    opt = TrainingOptimizer(x; pairing...)

    # a constant gradient gives a direction of magnitude one per component, so the length of a
    # step is the step size
    g = fill(1.0, 4)
    steps = map(1:25) do _
        x_before = copy(x)
        optimization_step!(x, opt, g)
        norm(x - x_before) / 2
    end

    γ = exp(log(η₂ / η₁) / n)
    @test all(relative(steps[t], η₁ * γ^t) < 1e-7 for t in 1:25)
    @test relative(steps[n], η₂) < 1e-7
    @test steps[25] < η₂ / 10
    @test step_size(pairing.linesearch, 25) ≈ η₁ * γ^25 rtol = 8eps()
    # the control: a floor at `η₂` would hold the last step there
    @test !(relative(steps[25], η₂) < 1e-7)
end
