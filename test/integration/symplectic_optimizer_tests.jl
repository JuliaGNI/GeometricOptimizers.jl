# The optimizers on the symplectic Stiefel manifold, through `SymplecticLieAlgHorMatrix`.
#
# The problem is the symplectic eigenvalue problem: minimize `tr(UᵀMU)` over `U ∈ Sp(2n, 2N)` for a
# symmetric positive definite `M`. By Williamson's theorem its minimum is twice the sum of the `n`
# smallest symplectic eigenvalues of `M`, the moduli of the eigenvalues of `𝕁M`, which gives the
# answer in closed form. The starting points are built without the symplectic SR decomposition,
# whose `Float32` draws are not reliably on the manifold (issue #152); the `Float32` solves at the end
# take the median over seeded starts, as A22 of `KNOWN_ISSUES.md` asks of a residual on this manifold.
#
# `check` is read on every accepted iterate: the objective records it when it is called on the
# solution `solve!` holds, which is the iterate and not a trial point of the line search.
using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, check, cache, gradient, problem, initialize!,
                           update!, linesearch_problem, isconverged, section
using SimpleSolvers: Static, Backtracking
using NeuralNetworkParameters: NetworkParameters
using LinearAlgebra: Diagonal, I, cond, eigvals, norm, tr
using Test
import Random

include("../helpers/symplectic_points.jl")

const T = Float64
const N2, n2 = 10, 4

function spd(rng, N2)
    A = randn(rng, T, N2, N2)
    A * A' + I
end

# the objective on `U`, which pushes `check(U)` into `checks` whenever it is called on `U` itself
function recording_objective(U, M, checks)
    V -> (V === U && push!(checks, check(V)); tr(V' * M * V))
end

function williamson_minimum(M, n2)
    d = sort(abs.(imag.(eigvals(poisson(T, size(M, 1)) * M))))[1:2:end]
    2 * sum(d[1:(n2 ÷ 2)])
end

const M = spd(Random.Xoshiro(1), N2)
const FMIN = williamson_minimum(M, n2)
objective(U) = tr(U' * M * U)

start() = conditioned_point(Random.Xoshiro(2), T, N2, n2)

# `φ'(α)` against a central difference of `φ(α)`, both from the `LinesearchProblem` the optimizer
# builds, as `test/integration/manifold_linesearch_tests.jl` checks it for the Stiefel manifold. This
# is what `gradient_rep!` is for: `global_rep!` of the Riemannian gradient does not pair to the
# derivative on this manifold, which `test/global_sections/global_sections.jl` asserts.
@testset "φ' is the derivative of φ on the symplectic Stiefel manifold, $(nameof(typeof(R)))" for R in (
    Geodesic(), Cayley())
    U = start()
    opt = Optimizer(
        U, objective; algorithm = BFGS(), retraction = R, linesearch = Backtracking(T))
    state = OptimizerState(BFGS(), U)
    c = cache(opt)
    initialize!(c, U)
    update!(c, state, gradient(opt), GeometricOptimizers._direction_rule(opt), U)
    ls = linesearch_problem(problem(opt), gradient(opt), c, R)
    params = (x = U, state = state)

    h = cbrt(eps(T))
    for α in T.((0.0, 1e-3, 1e-2, 5e-2))
        φ′ = ls.D(α, params)
        difference = (ls.F(α + h, params) - ls.F(α - h, params)) / (2h)
        @test abs(φ′ - difference) / abs(difference) < 10 * sqrt(eps(T))
    end
    @test ls.D(zero(T), params) < 0
end

# BFGS and DFP hold the derivative with respect to the stored entries of the lift, `M` times the lift
# `global_rep` of the Riemannian gradient for the Gram matrix `M` of the metric, and start `Q` at `M⁻¹`.
# So their first direction, and the direction after a restart or a failed descent test, is
# `-global_rep(Δ)`, the one `GradientMethod` takes, and its tangent vector is `-Δ` for every section.
# The control is the direction `-gradient_rep(Δ)` that `Q = I` would give.
@testset "$(nameof(typeof(algorithm))) starts with the GradientMethod direction" for algorithm in (
    BFGS(), DFP())
    flat(B) = first(GeometricOptimizers.flatten(B))
    tangent = GeometricOptimizers._section_tangent
    U = start()
    Δ = rgrad(U, 2 * M * U.A)
    opt = Optimizer(U, objective; algorithm = algorithm, linesearch = Backtracking(T))
    c = cache(opt)

    first_direction(state) = begin
        initialize!(c, U)
        update!(c, state, gradient(opt), GeometricOptimizers._direction_rule(opt), U)
        GeometricOptimizers.direction(c)
    end

    state = OptimizerState(algorithm, U)
    G = global_rep(section(state), Δ)
    δ = first_direction(state)
    @test flat(δ) ≈ -flat(G) rtol = 1e-10
    @test tangent(section(state), δ) ≈ -Δ rtol = 1e-10
    @test !(flat(GeometricOptimizers.rhs(c)) ≈ -flat(G))

    # another section over the same point, `λ ↦ λ diag(𝕀, S)` for a symplectic `S` with `κ(S) ≈ 30`
    k = (N2 - n2) ÷ 2
    d = exp2.(range(-2.5, 2.5; length = k))
    S = cat(Diagonal(d), Diagonal(inv.(d)); dims = (1, 2))
    section(state).λ .= section(state).λ * S
    GeometricOptimizers.restart!(state)
    δ = first_direction(state)
    @test tangent(section(state), δ) ≈ -Δ rtol = 1e-9
    @test flat(δ) ≈ -flat(global_rep(section(state), Δ)) rtol = 1e-9

    # a restart, and the substitute for a direction that does not descend
    GeometricOptimizers.restart!(state)
    GeometricOptimizers.steepest_descent!(c, state)
    @test tangent(section(state), GeometricOptimizers.direction(c)) ≈ -Δ rtol = 1e-9
    GeometricOptimizers._copyto!(GeometricOptimizers.direction(c), GeometricOptimizers.rhs(c))
    GeometricOptimizers._rmul!(GeometricOptimizers.direction(c), -1)
    GeometricOptimizers.ensure_descent!(c, state, algorithm, GeometricOptimizers.config(opt))
    @test tangent(section(state), GeometricOptimizers.direction(c)) ≈ -Δ rtol = 1e-9
end

# The two methods with a line search find the minimum, and every iterate stays on the manifold.
@testset "$(nameof(typeof(algorithm))) finds the symplectic eigenvalues, $(nameof(typeof(R)))" for algorithm in (
        GradientMethod(), BFGS()),
    R in (Geodesic(), Cayley())

    U = start()
    U₀ = copy(U.A)
    checks = T[]
    opt = Optimizer(U, recording_objective(U, M, checks); algorithm = algorithm,
        linesearch = Backtracking(T), retraction = R, max_iterations = 5000)
    result = solve!(U, OptimizerState(algorithm, U), opt)
    @test U isa SymplecticStiefelManifold{T}
    @test U.A != U₀
    @test length(checks) > 2
    @test maximum(checks) < 1e-9
    @test isconverged(result.status)
    @test abs(objective(U) - FMIN) < 1e-8 * FMIN
end

# The gradient stop reads `rg`, the Riemannian norm `√g(Δ, Δ)` of the gradient at the iterate, which
# is the same for every method. The control is the norm of the array each method holds: the flat
# lift of BFGS and DFP and `global_rep` of the first-order methods have different norms on this
# manifold, and the latter differs from `rg` by orders of magnitude.
@testset "rg is the Riemannian norm of the gradient, $(nameof(typeof(algorithm))), $(nameof(typeof(R)))" for algorithm in (
        GradientMethod(), MomentumMethod(), Adam(), BFGS(), DFP()),
    R in (Geodesic(), Cayley())

    U = start()
    opt = Optimizer(U, objective; algorithm = algorithm, linesearch = Backtracking(T),
        retraction = R, max_iterations = 1, warn_iterations = 0)
    result = solve!(U, OptimizerState(algorithm, U), opt)
    Δ = rgrad(U, 2 * M * U.A)
    rg = GeometricOptimizers.g_residual(result.status)
    @test rg ≈ sqrt(metric(U, Δ, Δ)) rtol = 64eps(T)
    algorithm isa Union{GradientMethod, MomentumMethod, Adam} &&
        @test !isapprox(
            GeometricOptimizers.l2norm(GeometricOptimizers.latest_gradient(cache(opt))),
            rg; rtol = T(0.1))
end

# A fixed step only crawls, so these are asked to decrease the objective and stay on the manifold.
@testset "$(nameof(typeof(algorithm))) at a fixed step decreases the objective, $(nameof(typeof(R)))" for algorithm in (
        Adam(), MomentumMethod()),
    R in (Geodesic(), Cayley())

    U = start()
    f₀ = objective(U)
    checks = T[]
    opt = Optimizer(U, recording_objective(U, M, checks); algorithm = algorithm,
        linesearch = Static(T(1e-3)), retraction = R, max_iterations = 500)
    solve!(U, OptimizerState(algorithm, U), opt)
    @test length(checks) == 502
    @test maximum(checks) < 1e-9
    @test objective(U) < f₀ - (f₀ - FMIN) / 2
end

# `Float32`, over nine seeded starts with `κ ≤ 4`. The bound on `check` is the one the tests of
# `test/manifolds/symplectic_stiefel_manifold.jl` give a point, `8eps(T)κ²` for the `κ` of the start,
# on the median over the starts of the largest `check` over the iterates of a solve. Measured
# medians, Julia 1.13: 16eps(T) for `GradientMethod` and 28eps(T) for `Adam`.
# `GradientMethod` converges to the `Float64` minimum to `1e-5` relative on every start.
@testset "$(nameof(algorithm)) on a Float32 point" for algorithm in (GradientMethod, Adam)
    S = Float32
    M₃₂ = S.(M)
    bounds, maxima, decreases, errors = S[], S[], Bool[], S[]
    for seed in 1:9
        U = conditioned_point(Random.Xoshiro(100 + seed), S, N2, n2)
        κ = cond(U.A)
        f₀ = tr(U' * M₃₂ * U)
        checks = S[]
        method = algorithm()
        linesearch, iterations = algorithm === GradientMethod ? (Backtracking(S), 2000) :
                                 (Static(S(1e-3)), 500)
        opt = Optimizer(U, recording_objective(U, M₃₂, checks); algorithm = method,
            linesearch = linesearch, max_iterations = iterations)
        solve!(U, OptimizerState(method, U), opt)
        f = tr(U' * M₃₂ * U)
        push!(bounds, 8eps(S) * κ^2)
        push!(maxima, maximum(checks))
        push!(decreases, f < f₀ - (f₀ - FMIN) / 2)
        push!(errors, abs(f - FMIN) / FMIN)
    end
    @test sort(maxima)[5] < sort(bounds)[5]
    @test all(isfinite, maxima)
    @test all(decreases)
    algorithm === GradientMethod && @test maximum(errors) < 1e-5
end

# The training step takes the Euclidean gradient of the leaf, and a parameter set may hold a
# symplectic leaf beside an ordinary one.
@testset "a TrainingOptimizer steps a symplectic leaf, $(nameof(typeof(algorithm)))" for algorithm in (
    GradientMethod(), MomentumMethod(), Adam())
    U = start()
    f₀ = objective(U)
    opt = TrainingOptimizer(U; algorithm = algorithm, linesearch = T(1e-3))
    for _ in 1:200
        optimization_step!(U, opt, 2 * M * U.A)
    end
    @test check(U) < 1e-9
    @test objective(U) < f₀
    @test section(opt.state).Y == U

    ps = NetworkParameters((U = start(), b = T[1, -2, 3]))
    F(p) = objective(p.U) + sum(abs2, p.b)
    F₀ = F(ps)
    opt = TrainingOptimizer(ps; algorithm = algorithm, linesearch = T(1e-3))
    for _ in 1:200
        optimization_step!(ps, opt, NetworkParameters((U = 2 * M * ps.U.A, b = 2 * ps.b)))
    end
    @test ps.U isa SymplecticStiefelManifold{T}
    @test check(ps.U) < 1e-9
    @test F(ps) < F₀
end

@testset "a CompositeMethod steps a symplectic leaf" begin
    ps = NetworkParameters((U = start(), b = T[1, -2, 3]))
    F(p) = objective(p.U) + sum(abs2, p.b)
    F₀ = F(ps)
    opt = TrainingOptimizer(
        ps; algorithm = CompositeMethod(; manifold = Adam(), array = Adam()),
        linesearch = T(1e-3))
    for _ in 1:200
        optimization_step!(ps, opt, NetworkParameters((U = 2 * M * ps.U.A, b = 2 * ps.b)))
    end
    @test check(ps.U) < 1e-9
    @test F(ps) < F₀
end
