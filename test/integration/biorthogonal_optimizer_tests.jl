# The optimizers on the manifold of biorthogonal pairs, through `BiorthogonalLieAlgHorMatrix`.
#
# The problem is the reconstruction of a data matrix `X` by the oblique projector `ΦΨᵀ`: minimize
# `½‖X - ΦΨᵀX‖²` over `(Φ, Ψ) ∈ 𝓑(n, N)`. `ΦΨᵀ` is a projector of rank `n`, so by the Eckart-Young
# theorem the minimum is half the sum of the squared singular values of `X` after the `n`-th, which the
# orthogonal projector onto the leading left singular vectors attains. This is the loss of a linear
# biorthogonal autoencoder, the one-layer case of the biorthogonal lifts this manifold is for.
#
# The minimum is attained on the noncompact set of pairs `(Φg, Ψg⁻ᵀ)`, `g ∈ GL(n)`, and the loss is
# flat along it, so first-order methods approach it slowly; BFGS is asked for the minimum, the others
# for a decrease.
using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, check, cache, gradient, problem, initialize!,
                           update!, linesearch_problem, isconverged, section
using SimpleSolvers: Static, Backtracking
using NeuralNetworkParameters: NetworkParameters, flatten, unflatten
using LinearAlgebra: norm, svdvals
using Test
import Random

const T = Float64
const N, n, K = 10, 3, 30
const X = randn(Random.Xoshiro(3), T, N, K) .* (1:N) ./ N
const FMIN = sum(abs2, svdvals(X)[(n + 1):end]) / 2

objective(Y) = sum(abs2, X - Y[:, 1:n] * (Y[:, (n + 1):(2n)]' * X)) / 2
function euclidean_gradient(Y)
    Φ, Ψ = Y[:, 1:n], Y[:, (n + 1):(2n)]
    R = Φ * (Ψ' * X) - X
    hcat(R * (X' * Ψ), X * (R' * Φ))
end

start() = rand(Random.Xoshiro(4), BiorthogonalManifold{T}, N, 2n)

@testset "the Euclidean gradient of the test problem" begin
    Y = start()
    V = randn(Random.Xoshiro(5), N, 2n)
    h = cbrt(eps(T))
    difference = (objective(Y.A + h * V) - objective(Y.A - h * V)) / 2h
    @test sum(euclidean_gradient(Y) .* V) ≈ difference rtol = 1e-8
end

# `φ'(α)` against a central difference of `φ(α)`, as for the symplectic Stiefel manifold. This is
# what `gradient_rep!` is for.
@testset "φ' is the derivative of φ on the biorthogonal manifold, $(nameof(typeof(R)))" for R in (
    Geodesic(), Cayley())
    Y = start()
    opt = Optimizer(
        Y, objective; algorithm = BFGS(), retraction = R, linesearch = Backtracking(T))
    state = OptimizerState(BFGS(), Y)
    c = cache(opt)
    initialize!(c, Y)
    update!(c, state, gradient(opt), GeometricOptimizers._direction_rule(opt), Y)
    ls = linesearch_problem(problem(opt), gradient(opt), c, R)
    params = (x = Y, state = state)

    # the first direction is the whole gradient, whose norm is of the order of the objective, so the
    # step of the difference is scaled down by it
    h = cbrt(eps(T)) / norm(euclidean_gradient(Y))
    for α in T.((0.0, 1e-5, 1e-4, 5e-4))
        φ′ = ls.D(α, params)
        difference = (ls.F(α + h, params) - ls.F(α - h, params)) / (2h)
        @test abs(φ′ - difference) / abs(difference) < 10 * sqrt(eps(T))
    end
    @test ls.D(zero(T), params) < 0
end

@testset "BFGS finds the Eckart-Young minimum, $(nameof(typeof(R)))" for R in (
    Geodesic(), Cayley())
    Y = start()
    Y₀ = copy(Y.A)
    opt = Optimizer(Y, objective; algorithm = BFGS(), linesearch = Backtracking(T),
        retraction = R, max_iterations = 5000)
    result = solve!(Y, OptimizerState(BFGS(), Y), opt)
    @test Y isa BiorthogonalManifold{T}
    @test Y.A != Y₀
    @test check(Y) < 1e-12
    @test isconverged(result.status)
    @test abs(objective(Y) - FMIN) < 1e-10 * FMIN
end

@testset "$(algorithm) with a line search decreases the objective, $(nameof(typeof(R)))" for algorithm in (
        GradientMethod(), GradientMethod(; descent = SectionDescent()), DFP()),
    R in (Geodesic(), Cayley())

    Y = start()
    f₀ = objective(Y)
    opt = Optimizer(Y, objective; algorithm = algorithm, linesearch = Backtracking(T),
        retraction = R, max_iterations = 200, warn_iterations = 1000)
    solve!(Y, OptimizerState(algorithm, Y), opt)
    @test check(Y) < 1e-12
    @test objective(Y) - FMIN < (f₀ - FMIN) / 20
end

# The gradient stop reads the Riemannian norm of the gradient at the iterate, the same number for
# every method.
@testset "rg is the Riemannian norm of the gradient, $(algorithm)" for algorithm in (
    GradientMethod(), GradientMethod(; descent = SectionDescent()), Adam(), BFGS())
    Y = start()
    opt = Optimizer(Y, objective; algorithm = algorithm, linesearch = Backtracking(T),
        max_iterations = 1, warn_iterations = 0)
    result = solve!(Y, OptimizerState(algorithm, Y), opt)
    Δ = rgrad(Y, euclidean_gradient(Y))
    @test GeometricOptimizers.g_residual(result.status) ≈ sqrt(metric(Y, Δ, Δ)) rtol = 64eps(T)
end

# At a fixed step the sign-like direction of Adam is a descent direction in the coordinates its
# gradient is held in. With `SectionDescent` those are the coordinates the derivative of the loss
# pairs with, and the result does not depend on the section drawn at the start; with `MetricDescent`
# they are the dual frame's, and it does. Momentum is not normalised and takes a smaller step, and a
# smaller one still with `SectionDescent`, whose coordinates weigh the gradient by the frame: at
# `1e-4` the objective oscillates above where it started under `Geodesic`, and under `Cayley` the
# lift reaches an eigenvalue near `±2`, where the Cayley transform is singular.
@testset "Adam and Momentum at a fixed step, $(algorithm), $(nameof(typeof(R)))" for (algorithm, η) in (
        (Adam(; descent = SectionDescent()), 1e-3), (MomentumMethod(), 1e-4),
        (MomentumMethod(; descent = SectionDescent()), 1e-5)),
    R in (Geodesic(), Cayley())

    finals = map(1:5) do seed
        Random.seed!(seed)
        Y = start()
        opt = Optimizer(Y, objective; algorithm = algorithm, linesearch = Static(T(η)),
            retraction = R, max_iterations = 500, warn_iterations = 1000)
        solve!(Y, OptimizerState(algorithm, Y), opt)
        @test check(Y) < 1e-12
        objective(Y)
    end
    @test maximum(finals) < objective(start()) / 5
end

@testset "Adam with the metric descent stays on the manifold" begin
    Y = start()
    opt = Optimizer(Y, objective; algorithm = Adam(), linesearch = Static(T(1e-3)),
        max_iterations = 100, warn_iterations = 1000)
    solve!(Y, OptimizerState(Adam(), Y), opt)
    @test check(Y) < 1e-12
end

# The training step takes the Euclidean gradient of the leaf, and a parameter set may hold a
# biorthogonal leaf beside an ordinary one.
@testset "a TrainingOptimizer steps a biorthogonal leaf, $(algorithm)" for (algorithm, η) in (
    (GradientMethod(), 1e-4), (MomentumMethod(), 1e-4), (Adam(; descent = SectionDescent()), 1e-3))
    Y = start()
    f₀ = objective(Y)
    opt = TrainingOptimizer(Y; algorithm = algorithm, linesearch = T(η))
    for _ in 1:200
        optimization_step!(Y, opt, euclidean_gradient(Y))
    end
    @test check(Y) < 1e-12
    @test objective(Y) < f₀ / 2
    @test section(opt.state).Y == Y

    ps = NetworkParameters((Y = start(), b = T[1, -2, 3]))
    F(p) = objective(p.Y) + sum(abs2, p.b)
    F₀ = F(ps)
    opt = TrainingOptimizer(ps; algorithm = algorithm, linesearch = T(η))
    for _ in 1:200
        optimization_step!(ps, opt,
            NetworkParameters((Y = euclidean_gradient(ps.Y), b = 2 * ps.b)))
    end
    @test ps.Y isa BiorthogonalManifold{T}
    @test check(ps.Y) < 1e-12
    @test F(ps) < F₀ / 2
end

@testset "a CompositeMethod steps a biorthogonal leaf" begin
    ps = NetworkParameters((Y = start(), b = T[1, -2, 3]))
    F(p) = objective(p.Y) + sum(abs2, p.b)
    F₀ = F(ps)
    opt = TrainingOptimizer(ps;
        algorithm = CompositeMethod(; manifold = Adam(; descent = SectionDescent()),
            array = Adam()),
        linesearch = T(1e-3))
    for _ in 1:200
        optimization_step!(ps, opt,
            NetworkParameters((Y = euclidean_gradient(ps.Y), b = 2 * ps.b)))
    end
    @test check(ps.Y) < 1e-12
    @test F(ps) < F₀ / 2
end

@testset "AdamWithEuclideanDecay refuses a biorthogonal point" begin
    Y = start()
    @test_throws ErrorException Optimizer(
        Y, objective; algorithm = AdamWithEuclideanDecay(; λ = T(0.1)))
end
