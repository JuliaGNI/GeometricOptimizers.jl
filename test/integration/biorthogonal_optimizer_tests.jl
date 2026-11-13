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
                           update!, linesearch_problem, isconverged, section, gradient_rep!,
                           zero_tangent, step_αmax, l2norm
using SimpleSolvers: Static, Backtracking
using NeuralNetworkParameters: NetworkParameters, flatten, unflatten
using LinearAlgebra: Diagonal, I, SingularException, cond, det, eigvals, norm, svdvals
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
        GradientMethod(), DFP()),
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
    GradientMethod(), Adam(), BFGS())
    Y = start()
    opt = Optimizer(Y, objective; algorithm = algorithm, linesearch = Backtracking(T),
        max_iterations = 1, warn_iterations = 0)
    result = solve!(Y, OptimizerState(algorithm, Y), opt)
    Δ = rgrad(Y, euclidean_gradient(Y))
    @test GeometricOptimizers.g_residual(result.status) ≈ sqrt(metric(Y, Δ, Δ)) rtol = 64eps(T)
end

# The change of the section `λ ↦ λ diag(𝕀, S)`, which leaves the point where it is: the frame
# `[Φ K]` becomes `[Φ KS]` and the dual frame `[Ψ K̃]` becomes `[Ψ K̃S⁻ᵀ]`. `S` is diagonal with
# `κ(S) = 32`, so not orthogonal.
const SECTION_SCALES = exp2.(range(-2.5, 2.5; length = N - n))
function change_section!(λY, s)
    K, K̃ = λY.λ[:, 1:(N - n)], λY.λ[:, (N - n + 1):end]
    λY.λ .= hcat(K * Diagonal(s), K̃ * Diagonal(inv.(s)))
    λY
end

# BFGS and DFP hold `gradient_rep!`, which is `M` times the lift `global_rep` of the Riemannian
# gradient for the Gram matrix `M` of the metric in the stored blocks, and start `Q` at `M⁻¹`. So
# their first direction, and the direction after a restart or a failed descent test, is
# `-global_rep(Δ)`, and its tangent vector is `-Δ` for every section. `M` is not the identity here,
# so the control is the direction `-gradient_rep(Δ)` that `Q = 𝕀` would give, whose tangent vector is
# not `-Δ`, at `global_section`'s section and after a non-orthogonal change of it.
@testset "$(nameof(typeof(algorithm))) starts with the GradientMethod direction" for algorithm in (
    BFGS(), DFP())
    flat(B) = first(GeometricOptimizers.flatten(B))
    tangent = GeometricOptimizers._section_tangent
    Random.seed!(1)
    Y = start()
    Δ = rgrad(Y, euclidean_gradient(Y))
    opt = Optimizer(Y, objective; algorithm = algorithm, linesearch = Backtracking(T))
    c = cache(opt)

    first_direction(state) = begin
        initialize!(c, Y)
        update!(c, state, gradient(opt), GeometricOptimizers._direction_rule(opt), Y)
        GeometricOptimizers.direction(c)
    end
    held_direction(λY) = tangent(λY, gradient_rep!(zero_tangent(Y), λY, Δ))

    state = OptimizerState(algorithm, Y)
    δ = first_direction(state)
    @test tangent(section(state), δ) ≈ -Δ rtol = 1e-10
    @test flat(δ) ≈ -flat(global_rep(section(state), Δ)) rtol = 1e-10
    @test !(held_direction(section(state)) ≈ Δ)

    # a new state, whose secant pair is `NaN`: the one above would pair its first direction with the
    # change of the gradient's coordinates and update `Q`
    state = OptimizerState(algorithm, Y)
    change_section!(section(state), SECTION_SCALES)
    GeometricOptimizers.restart!(state)
    δ = first_direction(state)
    @test tangent(section(state), δ) ≈ -Δ rtol = 1e-9
    @test flat(δ) ≈ -flat(global_rep(section(state), Δ)) rtol = 1e-9
    @test !(held_direction(section(state)) ≈ Δ)

    # a restart, and the substitute for a direction that does not descend
    GeometricOptimizers.restart!(state)
    GeometricOptimizers.steepest_descent!(c, state)
    @test tangent(section(state), GeometricOptimizers.direction(c)) ≈ -Δ rtol = 1e-9
    GeometricOptimizers._copyto!(GeometricOptimizers.direction(c), GeometricOptimizers.rhs(c))
    GeometricOptimizers._rmul!(GeometricOptimizers.direction(c), -1)
    GeometricOptimizers.ensure_descent!(c, state, algorithm, GeometricOptimizers.config(opt))
    @test tangent(section(state), GeometricOptimizers.direction(c)) ≈ -Δ rtol = 1e-9
end

# The path under a line search does not depend on the section: two solves from one point, one at
# `global_section`'s section and one after the change above. The step ceiling reads the spectral
# radius of the lift, which the change keeps, and BFGS starts at the inverse of the metric, which
# transforms with the coordinates. Measured after 15 steps, the two paths are 1.3e-14 apart for
# GradientMethod and 3.1e-11 for BFGS; under the ceiling `2πc/‖δ‖` they are 0.67 and 0.63 apart, and
# with BFGS started at `Q = 𝕀` 1.09.
@testset "the path of $(nameof(typeof(algorithm))) does not depend on the section" for algorithm in (
    GradientMethod(), BFGS())
    function path(s)
        Random.seed!(1)
        Y = start()
        opt = Optimizer(Y, objective; algorithm = algorithm, linesearch = Backtracking(T))
        state = OptimizerState(algorithm, Y)
        GeometricOptimizers._start_section!(section(state), Y)
        change_section!(section(state), s)
        GeometricOptimizers._start_inverse_hessian!(state)
        iterates = Matrix{T}[]
        for _ in 1:15
            GeometricOptimizers.increase_iteration_number!(state)
            GeometricOptimizers.solver_step!(Y, state, opt)
            update!(state, opt, Y, objective(Y))
            push!(iterates, copy(Y.A))
        end
        iterates
    end
    a, b = path(ones(T, N - n)), path(SECTION_SCALES)
    @test !(a[end] ≈ start().A)
    @test maximum(norm.(a .- b) ./ norm.(a)) < 1e-10
end

# The step ceiling of a biorthogonal lift is `c/ρ(δ)`, as for a symplectic one: `Cayley(αX)` has a
# pole at `α = 2/μ` for every real eigenvalue `μ > 0` of `X`, and its inverse `Cayley(-αX)` one at
# `2/|μ|` for `μ < 0`. The generic ceiling `2πc/‖δ‖` of a rotation lies past the pole for the
# witness `A = [1]`, `B = C = 𝕆`.
@testset "step_αmax on a BiorthogonalLieAlgHorMatrix is c/ρ(δ), inside the Cayley poles" begin
    c = T(1.3)
    δ = rand(Random.Xoshiro(1234), BiorthogonalLieAlgHorMatrix{T}, N, n)
    ρ = maximum(abs, eigvals(Matrix(δ)))
    @test step_αmax(c, δ) ≈ c / ρ rtol = 64eps(T)
    @test step_αmax(c, δ) ≥ c / l2norm(δ)
    @test step_αmax(c, zero_tangent(start())) == Inf

    # the change of the section takes the lift to `diag(𝕀, S)⁻¹ δ diag(𝕀, S)`, which keeps the
    # eigenvalues and not the norm
    S = Diagonal(SECTION_SCALES)
    δ_S = BiorthogonalLieAlgHorMatrix(δ.A, S \ δ.B, δ.C * S, N, n)
    @test step_αmax(c, δ_S) ≈ step_αmax(c, δ) rtol = 64eps(T) * cond(S)
    @test !isapprox(l2norm(δ_S), l2norm(δ); rtol = 0.1)

    δ₁ = BiorthogonalLieAlgHorMatrix(ones(T, 1, 1), zeros(T, 1, 1), zeros(T, 1, 1), 2, 1)
    @test l2norm(δ₁) == 1
    @test det(I - Matrix(δ₁)) == 0
    @test_throws SingularException cayley(2 * δ₁)
    @test invoke(step_αmax, Tuple{T, Any}, one(T), δ₁) ≈ 2π
    @test step_αmax(one(T), δ₁) == 1
    @test check(cayley(step_αmax(one(T), δ₁) * δ₁)) < 8eps(T)
end

# At a fixed step the result of Adam depends on the section drawn at the start (K60): its sign-like
# direction is taken entrywise in the coordinates of `global_rep`, which are not the ones the
# derivative of the loss pairs with. Over the seeds 1 to 50, 500 steps at `1e-3` under `Geodesic` end
# between 44.9 and 8982 from 1405, and 6 of them above a fifth of the start; these are the first five
# seeds, which end below 56. Momentum is not normalised and takes a smaller step, and its path does not depend on the
# section.
@testset "Adam and Momentum at a fixed step, $(algorithm), $(nameof(typeof(R)))" for (algorithm, η) in (
        (Adam(), 1e-3), (MomentumMethod(), 1e-4)),
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

@testset "Adam at the default settings stays on the manifold" begin
    Y = start()
    opt = Optimizer(Y, objective; algorithm = Adam(), linesearch = Static(T(1e-3)),
        max_iterations = 100, warn_iterations = 1000)
    solve!(Y, OptimizerState(Adam(), Y), opt)
    @test check(Y) < 1e-12
end

# The training step takes the Euclidean gradient of the leaf, and a parameter set may hold a
# biorthogonal leaf beside an ordinary one. The section is seeded because Adam's result depends on it
# (K60): over the seeds 1 to 50, 200 steps at `1e-3` end above half the start on 4 of them for the
# leaf and on 5 for the parameter set, and on seed 1 at 0.15 and 0.33 of it.
@testset "a TrainingOptimizer steps a biorthogonal leaf, $(algorithm)" for (algorithm, η) in (
    (GradientMethod(), 1e-4), (MomentumMethod(), 1e-4), (Adam(), 1e-3))
    Random.seed!(1)
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

# Each leaf takes the method of its kind. The manifold method is `GradientMethod`, whose path does not
# depend on the section, so the result is the same for every seed: 200 steps at `1e-4` end at 0.053
# of the start. The array leaf takes Adam, whose step under a gradient of constant sign is `η` per
# entry, which `GradientMethod` would not give.
@testset "a CompositeMethod steps a biorthogonal leaf, seed $(seed)" for seed in 1:3
    Random.seed!(seed)
    b₀ = T[1, -2, 3]
    ps = NetworkParameters((Y = start(), b = copy(b₀)))
    F(p) = objective(p.Y) + sum(abs2, p.b)
    F₀ = F(ps)
    η = T(1e-4)
    opt = TrainingOptimizer(ps;
        algorithm = CompositeMethod(; manifold = GradientMethod(), array = Adam()),
        linesearch = η)
    for _ in 1:200
        optimization_step!(ps, opt,
            NetworkParameters((Y = euclidean_gradient(ps.Y), b = 2 * ps.b)))
    end
    @test ps.Y isa BiorthogonalManifold{T}
    @test check(ps.Y) < 1e-12
    @test F(ps) < F₀ / 10
    @test ps.b ≈ b₀ .- 200η .* sign.(b₀) rtol = 1e-3
end

@testset "AdamWithEuclideanDecay refuses a biorthogonal point" begin
    Y = start()
    @test_throws ArgumentError Optimizer(
        Y, objective; algorithm = AdamWithEuclideanDecay(; λ = T(0.1)))
end
