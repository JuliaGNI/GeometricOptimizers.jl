# What it costs `Adam` to hold `global_rep` of the Riemannian gradient instead of `gradient_rep!`, the
# derivative of the loss with respect to the stored blocks, on a `BiorthogonalManifold` and a
# `SymplecticStiefelManifold`. `PULLBACK[]` switches the gradient Adam holds on these two lifts:
# `gradient_rep!` for `true`, which is the package's choice on the biorthogonal lift, and `global_rep`
# for `false`, which is its choice on the symplectic one. Nothing else changes.
#
# Three measurements, each over 50 seeds:
#
# 1. the test problem of `test/integration/biorthogonal_optimizer_tests.jl`: `Optimizer`, 500
#    steps at `η = 1e-3` under `Geodesic`, and `TrainingOptimizer`, 200 steps at `1e-3`; the seeds
#    select the completion `K` of the section;
# 2. `L = ‖Y - Y₀‖²` on `BiorthogonalManifold` for `(N, n) = (2, 1), (6, 2), (20, 3)`, from a random
#    point to a random target, 1000 steps of a `TrainingOptimizer` at `η = 1e-2`;
# 3. the same on `SymplecticStiefelManifold` for `(2N, 2n) = (4, 2), (12, 4), (40, 6)`.
#
# Adam runs at its defaults, `β₁ = 0.9`, `β₂ = 0.99`, `δ = 1e-8`.
#
#   julia --startup-file=no --project=test scripts/adam_pullback_severity.jl

using GeometricOptimizers
using GeometricOptimizers: Geodesic, check, gradient_rep!, global_rep!
using SimpleSolvers: Static
using Statistics: median
import Random

const GO = GeometricOptimizers
const T = Float64
const PULLBACK = Ref(true)

function GO.gradient_rep!(D::GO.BiorthogonalLieAlgHorMatrix, λY, Δ, ::Adam)
    PULLBACK[] ? gradient_rep!(D, λY, Δ) : global_rep!(D, λY, Δ)
end
function GO.gradient_rep!(D::GO.SymplecticLieAlgHorMatrix, λY, Δ, ::Adam)
    PULLBACK[] ? gradient_rep!(D, λY, Δ) : global_rep!(D, λY, Δ)
end

variant() = PULLBACK[] ? "gradient_rep!" : "global_rep"

const N₁, n₁ = 10, 3
const X = randn(Random.Xoshiro(3), T, N₁, 30) .* (1:N₁) ./ N₁
objective(Y) = sum(abs2, X - Y[:, 1:n₁] * (Y[:, (n₁ + 1):(2n₁)]' * X)) / 2
function euclidean_gradient(Y)
    Φ, Ψ = Y[:, 1:n₁], Y[:, (n₁ + 1):(2n₁)]
    R = Φ * (Ψ' * X) - X
    hcat(R * (X' * Ψ), X * (R' * Φ))
end
start() = rand(Random.Xoshiro(4), BiorthogonalManifold{T}, N₁, 2n₁)

function test_problem()
    f₀ = objective(start())
    opt_finals = map(1:50) do seed
        Random.seed!(seed)
        Y = start()
        opt = Optimizer(Y, objective; algorithm = Adam(), linesearch = Static(T(1e-3)),
            retraction = Geodesic(), max_iterations = 500, warn_iterations = 1000)
        solve!(Y, OptimizerState(Adam(), Y), opt)
        objective(Y)
    end
    tr_finals = map(1:50) do seed
        Random.seed!(seed)
        Y = start()
        opt = TrainingOptimizer(Y; algorithm = Adam(), linesearch = T(1e-3))
        for _ in 1:200
            optimization_step!(Y, opt, euclidean_gradient(Y))
        end
        objective(Y)
    end
    for (name, finals) in (("Optimizer, 500 steps", opt_finals),
        ("TrainingOptimizer, 200 steps", tr_finals))
        println("test problem, ", variant(), ", ", name, ": start ", round(f₀; digits = 1),
            ", final between ", round(minimum(finals); digits = 1), " and ",
            round(maximum(finals); digits = 1), ", median ", round(median(finals); digits = 1),
            "; above the start ", count(>(f₀), finals), "/50, above a fifth of it ",
            count(>(f₀ / 5), finals), "/50, seeds ", findall(>(f₀ / 5), finals))
    end
end

function model(M, N, n; η = T(1e-2), steps = 1000, runs = 50)
    ascents, above, drifted, ratios = 0, 0, 0, T[]
    for seed in 1:runs
        Random.seed!(1000seed + N)
        Y = rand(M{T}, N, n)
        Y₀ = Matrix(rand(M{T}, N, n))
        L(Y) = sum(abs2, Matrix(Y) - Y₀)
        L₀ = L(Y)
        opt = TrainingOptimizer(Y; algorithm = Adam(), linesearch = η)
        for k in 1:steps
            optimization_step!(Y, opt, 2 * (Matrix(Y) - Y₀))
            k == 1 && L(Y) > L₀ && (ascents += 1)
        end
        drifted += !(check(Y) < 1e-8)   # the constraint violated by more than 1e-8, or NaN
        above += !(L(Y) <= L₀)
        push!(ratios, L(Y) / L₀)
    end
    println(nameof(M), " $((N, n)), ", variant(), ": first steps that raise L ", ascents,
        "/$runs, runs ending above the start ", above, "/$runs, off the manifold ", drifted,
        "/$runs, median L_end/L_0 ", round(median(ratios); sigdigits = 3), ", largest ",
        round(maximum(ratios); sigdigits = 3))
end

for p in (false, true)
    PULLBACK[] = p
    test_problem()
    for (N, n) in ((2, 2), (6, 4), (20, 6))   # the stored N×2n of (N, n) = (2, 1), (6, 2), (20, 3)
        model(BiorthogonalManifold, N, n)
    end
    for (N, n) in ((4, 2), (12, 4), (40, 6))
        model(SymplecticStiefelManifold, N, n)
    end
end
