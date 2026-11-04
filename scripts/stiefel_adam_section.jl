# What the random complement of a `GlobalSection` decides for the Stiefel optimizers, and what
# separates `ScalarMomentAdam` from `DirectScalarMomentAdam`. Every figure of `KNOWN_ISSUES.md` that
# names this script comes from a run of it.
#
#   julia --startup-file=no --project=. scripts/stiefel_adam_section.jl
#
# The problem is `f(Y) = -½ tr(YᵀSYD)` on `St(3, 8)` with the spectrum below, 60 training steps at
# step size 0.01 and a fixed complement written into the section. `Xoshiro(1401)` draws `S`, `Y₀`,
# the complement and the changes of it, in that order.
#
# Part 1 changes the complement by a signed permutation, which must leave entrywise `Adam` alone, and
# by Haar matrices, which must not; the other three methods must not move under either. It then
# seeds the global generator instead, which is how a caller meets the draw.
#
# Part 2 writes the scalar method once in the tangent space, with a switch for each of the norm of
# the second moment, the place of `δ`, the cap and the transport. Set one way the loop must reproduce
# `ScalarMomentAdam`, set the other way `DirectScalarMomentAdam`, and these two controls are what
# make the differences below it mean anything. Each switch is then thrown alone.

using GeometricOptimizers
using GeometricOptimizers: section
using LinearAlgebra
using Printf
using Random

const N, n = 8, 3
const η, STEPS = 0.01, 60
const β₁, β₂, δ = 0.9, 0.99, 1.0e-8

randstiefel(rng, N, k) = Matrix(qr(randn(rng, N, k)).Q)[:, 1:k]

function complement(rng, Y)
    Z = randn(rng, N, N - n)
    Z -= Y * (Y' * Z)
    Z = Matrix(qr(Z).Q)[:, 1:(N - n)]
    Z -= Y * (Y' * Z)
    Matrix(qr(Z).Q)[:, 1:(N - n)]
end

function haar(rng, m)
    F = qr(randn(rng, m, m))
    Matrix(F.Q) * Diagonal(sign.(diag(F.R)))
end

function signed_permutation(rng, m)
    Matrix(1.0I, m, m)[:, randperm(rng, m)] * Diagonal(rand(rng, [-1.0, 1.0], m))
end

const RNG = Xoshiro(1401)
const S = let Q = randstiefel(RNG, N, N)
    Q * Diagonal([8.0, 7.0, 6.0, 3.0, 2.5, 2.0, 1.5, 1.0]) * Q'
end
const D = Diagonal([3.0, 2.0, 1.0])
∇f(Y) = -S * Y * D
const Y₀ = randstiefel(RNG, N, n)
const Y⊥ = complement(RNG, Y₀)

function train(method; steps = STEPS, step_size = η, λ = nothing, gradient = ∇f)
    Y = StiefelManifold(copy(Y₀))
    opt = TrainingOptimizer(Y; algorithm = method, linesearch = step_size)
    λ === nothing || (section(opt.state).λ .= λ)
    for _ in 1:steps
        optimization_step!(Y, opt, gradient(Y.A))
    end
    Matrix(Y.A)
end

const SECTION_METHODS = (
    "Adam" => Adam, "GradientMethod" => GradientMethod,
    "MomentumMethod" => () -> MomentumMethod(; α = 0.8), "ScalarMomentAdam" =>
        ScalarMomentAdam)

function part_1()
    P = signed_permutation(RNG, N - n)
    Q = haar(RNG, N - n)
    moved(m, λ) = norm(train(m(); λ = Y⊥) - train(m(); λ))
    println("‖Yₜ - Yₜ'‖_F after $STEPS steps, for one change of the complement:")
    for (name, m) in SECTION_METHODS
        @printf("  %-17s signed permutation %.2e   Haar %.2e\n", name, moved(m, Y⊥ * P),
            moved(m, Y⊥ * Q))
    end

    haars = [norm(train(Adam(); λ = Y⊥) - train(Adam(); λ = Y⊥ * haar(RNG, N - n)))
             for _ in 1:32]
    @printf("Adam over 32 Haar changes: min %.3f, median %.3f, max %.3f\n",
        minimum(haars), sort(haars)[16], maximum(haars))

    seeded(m, s) = (Random.seed!(s); train(m()))
    println("‖Yₜ - Yₜ'‖_F after $STEPS steps, for Random.seed!(1) against Random.seed!(2), no complement written:")
    for (name, m) in SECTION_METHODS
        @printf("  %-17s %.2e\n", name, norm(seeded(m, 1) - seeded(m, 2)))
    end
end

# The canonical gradient and the left translation `Q = cay(Ω)` of the step, `Y₊ = QY`.
canonical_gradient(Y, G) = G - Y * (G' * Y)
function translation(Y, Δ)
    P = I - Y * Y' / 2
    Ω = P * Δ * Y' - Y * Δ' * P
    (I - Ω / 2) \ (I + Ω / 2)
end
canonical_square(Y, Δ) = tr(Δ' * Δ) - tr((Δ' * Y) * (Y' * Δ)) / 2
frobenius_square(Y, Δ) = tr(Δ' * Δ)
project(Y, ξ) = ξ - Y * (Y' * ξ + ξ' * Y) / 2
weights(β, t) = ((β - β^t) / (1 - β^t), (1 - β) / (1 - β^t))

function scalar_loop(; square, δ_inside, capped, section_transport, step_size = η,
        gradient = ∇f)
    Y, B₁, v = Y₀, zero(Y₀), 0.0
    for t in 1:STEPS
        G_R = canonical_gradient(Y, gradient(Y))
        c₁₁, c₁₂ = weights(β₁, t)
        c₂₁, c₂₂ = weights(β₂, t)
        B₁ = c₁₁ * B₁ + c₁₂ * G_R
        v = c₂₁ * v + c₂₂ * square(Y, G_R)
        direction = B₁ / (δ_inside ? sqrt(v + δ) : sqrt(v) + δ)
        s = capped ? min(step_size, 1.0 / (norm(direction) + δ)) : step_size
        Q = translation(Y, -s * direction)
        Y = Q * Y
        B₁ = section_transport ? Q * B₁ : project(Y, B₁)
    end
    Y
end

const SCALAR = (square = canonical_square, δ_inside = true, capped = false,
    section_transport = true)
const DIRECT = (square = frobenius_square, δ_inside = false, capped = true,
    section_transport = false)

function part_2()
    direct = train(DirectScalarMomentAdam())
    println("the loop against the package after $STEPS steps (controls, both at round-off):")
    @printf("  ScalarMomentAdam       %.2e, and %.2e with the complement Y⊥Q\n",
        norm(scalar_loop(; SCALAR...) - train(ScalarMomentAdam(); λ = Y⊥)),
        norm(scalar_loop(; SCALAR...) -
             train(ScalarMomentAdam(); λ = Y⊥ * haar(RNG, N - n))))
    @printf("  DirectScalarMomentAdam %.2e\n", norm(scalar_loop(; DIRECT...) - direct))

    @printf("ScalarMomentAdam against DirectScalarMomentAdam: %.4f\n",
        norm(scalar_loop(; SCALAR...) - direct))
    @printf("ScalarMomentAdam(; ambient_norm = true) against DirectScalarMomentAdam: %.4f\n",
        norm(train(ScalarMomentAdam(; ambient_norm = true)) - direct))

    println("each switch alone, against DirectScalarMomentAdam, step size $η:")
    for (name, change) in ("the canonical norm" => (; square = canonical_square),
        "δ inside the root" => (; δ_inside = true),
        "no cap" => (; capped = false),
        "the section transport" => (; section_transport = true))
        @printf("  %-22s %.2e\n", name, norm(scalar_loop(; DIRECT..., change...) - direct))
    end
    large = 4.0
    @printf("  no cap, step size %.0f    %.2f\n", large,
        norm(scalar_loop(; DIRECT..., capped = false, step_size = large) -
             scalar_loop(; DIRECT..., step_size = large)))

    # `A = skew(YᵀΔ)` vanishes for this gradient, so the two norms agree and this switch must not move.
    invariant(Y) = -S * Y
    @printf("control, the canonical norm for an invariant loss: %.2e\n",
        norm(scalar_loop(; DIRECT..., square = canonical_square, gradient = invariant) -
             scalar_loop(; DIRECT..., gradient = invariant)))
end

part_1()
part_2()
