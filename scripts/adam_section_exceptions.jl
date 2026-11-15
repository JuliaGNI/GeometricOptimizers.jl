# Where the path of `Adam` does not depend on the section, and where it still does. The `Adam`
# docstring, the Weights on Manifolds page and `CHANGELOG.md` state these exceptions; this script is
# their evidence, and it exits with status 1 when one of them fails.
#
#   julia --startup-file=no --project=. scripts/adam_section_exceptions.jl
#
# Part 1 runs `Adam` on `St(n, N)` after two seeds of the global RNG, so that `global_section` draws
# two completions, and checks that the two completions differ before it compares the paths. On
# `St(3, 4)` the one complementary column is `±Y⊥`, and seeds 1 and 4 draw opposite signs: the two
# paths must agree. On `St(2, 4)` seeds 1 and 2 draw completions that differ by more than a sign, and
# the paths must not agree. On `St(3, 4)` the column is also flipped by hand after seed 1. `St(4, 4)`
# is a structural case, not a comparison: its complement is empty, so `global_section` draws nothing.
# `f(Y) = -½ tr(YᵀSYD)`, and a fresh `Xoshiro(7)` draws `S` and `Y₀` for each size.
#
# Part 2 runs `Adam` and `GradientMethod` on `Gr(n, N)` for `N - n = 1`, with the completion `Y⊥`
# written into the section, and compares the projection `YYᵀ` after a change of the representative
# `Y ↦ YQ`, `Q ∈ O(n)` random and `Q = -1` for `n = 1`, and after the change `Y⊥ ↦ -Y⊥` of the
# completion. The sign of the completion must move neither method. The representative must not move
# `GradientMethod`, and must move `Adam` for `n ≥ 2` only: for `Gr(1, 2)` the change is a sign.
# `f(Y) = -½ tr(YᵀSY)` with `S` symmetric, and one `Xoshiro(7)` draws `S`, `Y₀` and `Q` for each size
# in turn.
#
# Every run is 60 training steps at step size 0.01. A path difference counts as zero below 1e-12.

using GeometricOptimizers
using GeometricOptimizers: section
using LinearAlgebra
using Printf
using Random

const η, STEPS = 0.01, 60
const TOL = 1.0e-12

const failures = String[]
check(property, name) = property || push!(failures, name)

function stiefel_path(N, n, seed; flip = false)
    rng = Xoshiro(7)
    Q = Matrix(qr(randn(rng, N, N)).Q)
    S = Symmetric(Q * Diagonal(collect(N:-1.0:1)) * Q')
    D = Diagonal(collect(n:-1.0:1))
    Y₀ = Matrix(qr(randn(rng, N, n)).Q)[:, 1:n]
    Random.seed!(seed)
    Y = StiefelManifold(copy(Y₀))
    opt = TrainingOptimizer(Y; algorithm = Adam(), linesearch = η)
    flip && (section(opt.state).λ .*= -1)
    λ = copy(section(opt.state).λ)
    for _ in 1:STEPS
        optimization_step!(Y, opt, -S * Y.A * D)
    end
    Matrix(Y.A), λ
end

function part_1()
    println("St(n, N), Adam: ‖Yₜ - Yₜ'‖_F after $STEPS steps:")
    Y₁, λ₁ = stiefel_path(4, 4, 1)
    Y₂, λ₂ = stiefel_path(4, 4, 2)
    d = norm(Y₁ - Y₂)
    @printf("  St(4, 4), N - n = 0, seed 1 against seed 2   %.2e   (structural: the complement is 4 × %d, nothing is drawn)\n",
        d, size(λ₁, 2))
    check(size(λ₁, 2) == 0 && size(λ₂, 2) == 0, "St(4, 4) has a non-empty complement")
    check(d ≤ TOL, "St(4, 4) moved: $d")

    Y₁, λ₁ = stiefel_path(4, 3, 1)
    Y₄, λ₄ = stiefel_path(4, 3, 4)
    d = norm(Y₁ - Y₄)
    @printf("  St(3, 4), N - n = 1, seed 1 against seed 4   %.2e   (completions: ‖λ₁ + λ₄‖ = %.2e)\n",
        d, norm(λ₁ + λ₄))
    check(norm(λ₁ + λ₄) ≤ TOL, "St(3, 4): seeds 1 and 4 do not draw opposite completions")
    check(d ≤ TOL, "St(3, 4) moved: $d")

    Y₁, λ₁ = stiefel_path(4, 2, 1)
    Y₂, λ₂ = stiefel_path(4, 2, 2)
    d = norm(Y₁ - Y₂)
    @printf("  St(2, 4), N - n = 2, seed 1 against seed 2   %.2e   (completions: ‖λ₁ - λ₂‖ = %.2e)\n",
        d, norm(λ₁ - λ₂))
    check(norm(λ₁ - λ₂) > TOL, "St(2, 4): seeds 1 and 2 draw the same completion")
    check(d > TOL, "St(2, 4) did not move: $d")

    Y₁, λ₁ = stiefel_path(4, 3, 1)
    Y₋, λ₋ = stiefel_path(4, 3, 1; flip = true)
    d = norm(Y₁ - Y₋)
    @printf("  St(3, 4), completion Y⊥ against -Y⊥          %.2e   (completions: ‖λ₁ + λ₋‖ = %.2e, ‖λ₁‖ = %.2f)\n",
        d, norm(λ₁ + λ₋), norm(λ₁))
    check(norm(λ₁ + λ₋) ≤ TOL && norm(λ₁) > TOL, "St(3, 4): the flip did not negate the completion")
    check(d ≤ TOL, "St(3, 4) moved under -Y⊥: $d")
end

function grassmann_projection(method, Y₀, Y⊥, S)
    Y = GrassmannManifold(copy(Y₀))
    opt = TrainingOptimizer(Y; algorithm = method, linesearch = η)
    section(opt.state).λ .= Y⊥
    for _ in 1:STEPS
        optimization_step!(Y, opt, -S * Y.A)
    end
    Y.A * Y.A'
end

function part_2()
    rng = Xoshiro(7)
    println("Gr(n, N): ‖YₜYₜᵀ - Yₜ'Yₜ'ᵀ‖_F after $STEPS steps, completion Y⊥ fixed:")
    for (N, n) in ((2, 1), (3, 2), (4, 3), (6, 5))
        A = randn(rng, N, N)
        S = A + A'
        Y₀ = Matrix(qr(randn(rng, N, n)).Q)[:, 1:n]
        Y⊥ = nullspace(Y₀')
        Q = Matrix(qr(randn(rng, n, n)).Q)
        # `qr` of a 1 × 1 matrix gives `Q = 1`; `O(1) = {±1}`, so take the element that moves `Y`.
        isone(Q) && (Q = -Q)
        check(norm(Y₀ * Q - Y₀) > TOL, "Gr($n, $N): Y₀Q equals Y₀")
        check(norm(Y⊥) > TOL, "Gr($n, $N): the completion is empty")
        for (name, method) in (("Adam", Adam), ("GradientMethod", GradientMethod))
            P = grassmann_projection(method(), Y₀, Y⊥, S)
            representative = norm(P - grassmann_projection(method(), Y₀ * Q, Y⊥, S))
            sign = norm(P - grassmann_projection(method(), Y₀, -Y⊥, S))
            @printf("  Gr(%d, %d) %-15s representative YQ %.2e   completion -Y⊥ %.2e\n",
                n, N, name, representative, sign)
            check(sign ≤ TOL, "Gr($n, $N) $name moved under -Y⊥: $sign")
            if name == "Adam" && n ≥ 2
                check(representative > TOL, "Gr($n, $N) Adam did not move under YQ")
            else
                check(representative ≤ TOL,
                    "Gr($n, $N) $name moved under YQ: $representative")
            end
        end
    end
end

part_1()
part_2()

if isempty(failures)
    println("all stated properties hold")
else
    println("FAILED:\n  ", join(failures, "\n  "))
    exit(1)
end
