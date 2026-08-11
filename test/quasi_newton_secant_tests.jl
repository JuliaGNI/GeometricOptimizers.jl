using GeometricOptimizers
using GeometricOptimizers: _BFGS, _DFP, cache, solver_step!, initialize_state!, inverse_hessian,
                           increase_iteration_number!, iteration_number, update!
using LinearAlgebra: norm, dot, eigvals
using Test

# `_BFGS` and `_DFP` build their inverse Hessian `Q` from the secant pair
#
#     δ = x⁽ᵏ⁾ - x⁽ᵏ⁻¹⁾,    γ = ∇f(x⁽ᵏ⁾) - ∇f(x⁽ᵏ⁻¹⁾),
#
# so `state.ḡ` has to hold the gradient at the *previous* iterate when the cache forms `γ`. It used
# to be refreshed at the end of the iteration, i.e. at the same iterate the next `γ` was computed at,
# which made `γ` identically zero; `ΔxΔg` was then zero too and the guard around the `Q` update
# skipped it on every iteration. Both methods silently ran as steepest descent with `Q ≡ I`.

F(x) = sum(sin.(x) .^ 2)

# a genuinely non-separable, non-quadratic objective, so that a wrong `Q` cannot go unnoticed the way
# it does on a problem an exact line search solves in one step
rosenbrock(x) = sum((1 - x[i])^2 + 100 * (x[i+1] - x[i]^2)^2 for i in 1:(length(x)-1))

@testset "the secant pair is formed from consecutive iterates" begin
    for algorithm in (_BFGS(), _DFP())
        x = fill(0.5, 3)
        state = OptimizerState(algorithm, x)
        opt = Optimizer(x, F; algorithm=algorithm, linesearch=Backtracking())

        initialize_state!(state)
        updates = 0

        for _ in 1:20
            increase_iteration_number!(state)
            Q_before = copy(inverse_hessian(state))
            solver_step!(x, state, opt)
            norm(inverse_hessian(state) - Q_before) > 0 && (updates += 1)
            update!(state, opt, x)
        end

        # the first iteration is skipped on purpose (`state.s` is `NaN`, so there is no step to build
        # a secant pair from yet, and BFGS is supposed to start from `Q = I`); every one after it has
        # a valid pair. Before the fix this was 0.
        @test updates ≥ 15
        @test inverse_hessian(state) != one(inverse_hessian(state))
    end
end

@testset "the quasi-Newton methods beat gradient descent on Rosenbrock" begin
    # `Q ≡ I` makes `_BFGS`/`_DFP` identical to `GradientMethod`, which is what this separates. On
    # Rosenbrock, gradient descent is famously slow while a working quasi-Newton method is not.
    x₀ = [-1.2, 1.0]

    for algorithm in (_BFGS(), _DFP())
        x = copy(x₀)
        state = OptimizerState(algorithm, x)
        opt = Optimizer(x, rosenbrock; algorithm=algorithm, linesearch=Backtracking())

        solve!(x, state, opt)

        @test rosenbrock(x) < 1e-12
        @test x ≈ [1.0, 1.0] atol = 1e-5
        # gradient descent does not come close to this within `max_iterations`
        @test iteration_number(state) < 200
    end
end

@testset "the DFP update matches the textbook formula" begin
    # DFP is `Q ← Q - Qγγᵀ Q/(γᵀQγ) + δδᵀ/(δᵀγ)` (nocedal2006numerical, eq. 6.15). The middle term
    # used to be built from `cache.ΔxΔx`, i.e. `δδᵀ`, which left `cache.ΔgΔg` computed and unused.
    # Driving one `update!` with a chosen secant pair pins the formula directly, without depending on
    # what a line search happens to do.
    δ = [1.0, 2.0, -1.0]
    ḡ = [0.5, -1.0, 2.0]
    γ = [2.0, 1.0, 1.0]           # δᵀγ = 3 > 0, so the update is well defined
    x = [0.1, 0.2, 0.3]

    cache = GeometricOptimizers.DFPCache(x)
    state = OptimizerState(_DFP(), x)
    inverse_hessian(state) .= one(inverse_hessian(state))
    state.s .= δ
    state.ḡ .= ḡ

    update!(cache, state, x, ḡ .+ γ)

    Q = one(zeros(3, 3))
    expected = Q - (Q * γ * γ' * Q) / (γ' * Q * γ) + (δ * δ') / dot(δ, γ)

    @test inverse_hessian(state) ≈ expected

    # the δδᵀ variant this replaces is a different matrix, so the test would catch a revert
    wrong = Q - (Q * δ * δ' * Q) / (γ' * Q * γ) + (δ * δ') / dot(δ, γ)
    @test !isapprox(inverse_hessian(state), wrong)

    # `ḡ` was advanced to the gradient the cache was just called with
    @test state.ḡ ≈ ḡ .+ γ
end

@testset "the BFGS update matches the textbook formula" begin
    δ = [1.0, 2.0, -1.0]
    ḡ = [0.5, -1.0, 2.0]
    γ = [2.0, 1.0, 1.0]
    x = [0.1, 0.2, 0.3]

    cache = GeometricOptimizers.BFGSCache(x)
    state = OptimizerState(_BFGS(), x)
    inverse_hessian(state) .= one(inverse_hessian(state))
    state.s .= δ
    state.ḡ .= ḡ

    update!(cache, state, x, ḡ .+ γ)

    Q = one(zeros(3, 3))
    δγ = dot(δ, γ)
    expected = Q - (δ * γ' * Q + Q * γ * δ' - (1 + (γ' * Q * γ) / δγ) * (δ * δ')) / δγ

    @test inverse_hessian(state) ≈ expected
    @test state.ḡ ≈ ḡ .+ γ
end
