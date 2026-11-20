# The result of `Adam` at a fixed step on a `BiorthogonalManifold` over 50 sections, on the test
# problem of `test/integration/biorthogonal_optimizer_tests.jl`: the reconstruction of `X` by the
# oblique projector `ΦΨᵀ`. Every run starts from the same point; `Random.seed!(seed)` selects the
# completion `K` that `global_section` draws, and with it the coordinates Adam's entrywise moments are
# taken in, the blocks of `gradient_rep!`. The test asserts that the first step descends for every
# section and that a fixed-step run decreases the loss; this script measures what the section
# changes. `scripts/adam_pullback_severity.jl` compares the result with the one of `global_rep`.
#
# Two measurements:
#
# 1. `Optimizer`, 500 steps at `η = 1e-3` under `Geodesic` and `Cayley`;
# 2. `TrainingOptimizer`, 200 steps at `η = 1e-3`, on the leaf alone and on a parameter set that holds
#    the leaf beside an ordinary vector.
#
#   julia --startup-file=no --project=test scripts/biorthogonal_adam_seeds.jl

using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, check
using SimpleSolvers: Static
using NeuralNetworkParameters: NetworkParameters
using LinearAlgebra: svdvals
import Random

const T = Float64
const N, n, K = 10, 3, 30
const X = randn(Random.Xoshiro(3), T, N, K) .* (1:N) ./ N
const FMIN = sum(abs2, svdvals(X)[(n + 1):end]) / 2
const SEEDS = 1:50

objective(Y) = sum(abs2, X - Y[:, 1:n] * (Y[:, (n + 1):(2n)]' * X)) / 2
function euclidean_gradient(Y)
    Φ, Ψ = Y[:, 1:n], Y[:, (n + 1):(2n)]
    R = Φ * (Ψ' * X) - X
    hcat(R * (X' * Ψ), X * (R' * Φ))
end
start() = rand(Random.Xoshiro(4), BiorthogonalManifold{T}, N, 2n)

function summary(name, finals, f₀)
    println(name)
    println("  start ", round(f₀; digits = 1), ", minimum ", round(FMIN; digits = 2),
        "; final between ", round(minimum(finals); digits = 1), " and ",
        round(maximum(finals); digits = 1))
    for fraction in (1, 1 / 2, 1 / 5)
        println("  above ", fraction, " of the start: ", count(>(fraction * f₀), finals),
            " of ", length(finals))
    end
    println("  seeds 1 to 5: ", round.(finals[1:5] ./ f₀; digits = 3), " of the start")
end

for R in (Geodesic(), Cayley())
    finals = map(SEEDS) do seed
        Random.seed!(seed)
        Y = start()
        opt = Optimizer(Y, objective; algorithm = Adam(), linesearch = Static(T(1e-3)),
            retraction = R, max_iterations = 500, warn_iterations = 1000)
        solve!(Y, OptimizerState(Adam(), Y), opt)
        @assert check(Y) < 1e-12
        objective(Y)
    end
    summary("Optimizer, Adam, 500 steps at 1e-3, $(nameof(typeof(R)))", finals,
        objective(start()))
end

leaf = map(SEEDS) do seed
    Random.seed!(seed)
    Y = start()
    opt = TrainingOptimizer(Y; algorithm = Adam(), linesearch = T(1e-3))
    for _ in 1:200
        optimization_step!(Y, opt, euclidean_gradient(Y))
    end
    objective(Y)
end
summary("TrainingOptimizer, Adam, 200 steps at 1e-3, the leaf", leaf, objective(start()))

F(p) = objective(p.Y) + sum(abs2, p.b)
const b₀ = T[1, -2, 3]
set = map(SEEDS) do seed
    Random.seed!(seed)
    ps = NetworkParameters((Y = start(), b = copy(b₀)))
    opt = TrainingOptimizer(ps; algorithm = Adam(), linesearch = T(1e-3))
    for _ in 1:200
        optimization_step!(ps, opt,
            NetworkParameters((Y = euclidean_gradient(ps.Y), b = 2 * ps.b)))
    end
    F(ps)
end
summary("TrainingOptimizer, Adam, 200 steps at 1e-3, the parameter set", set,
    F(NetworkParameters((Y = start(), b = b₀))))
