using GeometricOptimizers
using GeometricOptimizers: _BFGS, _DFP, cache, solver_step!, initialize_state!, inverse_hessian,
                           increase_iteration_number!, iteration_number, update!
using LinearAlgebra: norm, dot
using Test

# `_BFGS` and `_DFP` build their inverse Hessian `Q` from the secant pair
#
#     δ = x⁽ᵏ⁾ - x⁽ᵏ⁻¹⁾,    γ = ∇f(x⁽ᵏ⁾) - ∇f(x⁽ᵏ⁻¹⁾),
#
# so `state.ḡ` has to hold the gradient at the *previous* iterate when the cache forms `γ`. It used
# to be refreshed at the end of the iteration, i.e. at the same iterate the next `γ` was computed at,
# which made `γ` identically zero; `ΔxΔg` was then zero too and the guard around the `Q` update
# skipped it on every iteration. Both methods silently ran as steepest descent with `Q ≡ I`.

# a genuinely non-separable, non-quadratic objective, so that a wrong `Q` cannot go unnoticed the way
# it does on a problem an exact line search solves in one step
rosenbrock(x) = sum((1 - x[i])^2 + 100 * (x[i+1] - x[i]^2)^2 for i in 1:(length(x)-1))

@testset "the secant pair is formed from consecutive iterates" begin
    # Rosenbrock rather than `F`, and only ten iterations, so that the whole window stays in the
    # pre-convergence regime: `f` is still of order 1e-2 at the end of it. Once a solve reaches
    # machine precision, `δ` and `γ` underflow to zero and the guard around the `Q` update *correctly*
    # skips, and how soon that happens is a floating-point detail that differs between platforms --
    # so counting updates over a window that runs past convergence pins nothing. Every iteration in
    # this window has a genuine secant pair, and `Q` has to move on each of them.
    ITERATIONS = 10

    for algorithm in (_BFGS(), _DFP())
        x = [-1.2, 1.0]
        state = OptimizerState(algorithm, x)
        opt = Optimizer(x, rosenbrock; algorithm=algorithm, linesearch=Backtracking())

        initialize_state!(state)
        updates = 0

        for _ in 1:ITERATIONS
            increase_iteration_number!(state)
            Q_before = copy(inverse_hessian(state))
            solver_step!(x, state, opt)
            norm(inverse_hessian(state) - Q_before) > 0 && (updates += 1)
            update!(state, opt, x)
        end

        # the first iteration is skipped on purpose: `state.s` is `NaN`, so there is no step to build
        # a secant pair from yet, and BFGS is supposed to start from `Q = I`. Every one after it has a
        # valid pair. Before the fix this was 0 -- `Q` was never updated at all, on any iteration.
        @test updates == ITERATIONS - 1
        @test inverse_hessian(state) != one(inverse_hessian(state))

        # guards the premise above: if a future change makes this converge inside the window, the
        # update count would drop legitimately and the assertion above would be measuring the wrong
        # thing. This says so rather than leaving it to look like a regression.
        @test rosenbrock(x) > 1e-8
    end
end

# How many iterations either method may take on Rosenbrock from `(-1.2, 1)` with the `Backtracking`
# search that `default_linesearch` returns for both of them, i.e. the expanding one. Measured: 20 for
# `_BFGS` and 34 for `_DFP`, and both are *invariant* over 900 starting points one ulp either side of
# `(-1.2, 1)` in each coordinate. One bound covers both with a factor of three to spare.
#
# The search has to be the expanding one. A shrink-only `Backtracking` starts at `α = 1` and can
# never exceed it, which is right for a direction already scaled like a Newton step but wrong for
# `_DFP`, whose direction is systematically under-scaled and which wants a median `α` of 8 (see
# `default_linesearch`). Starved of the step length it needs, `_DFP` picks its way to the minimizer
# along a path that is *chaotic* in the last bits of the arithmetic: over 400 starting points one ulp
# apart the count ranges over `108 .. 591_735`, median 1 784, while `f < 1e-12` holds in every one of
# them. That is not a quantity a test can bound — an earlier version of this file bounded it at 2 000
# on the strength of the 851 this machine produces, and CI's ubuntu/1.13 and windows/nightly runners,
# whose BLAS rounds differently, returned 2 941.
#
# The bound still separates a working quasi-Newton method from `Q ≡ I`, which is the regression this
# whole file exists to catch: plain gradient descent needs 42 608 iterations to reach `f < 1e-12`
# here (`GradientMethod` with the best fixed step found for it, `Static(0.001)`), three orders of
# magnitude above this bound.
#
# What the shrink-only search costs `_DFP` — and that it costs `_BFGS` nothing — is documented where
# it belongs, in `curvature_is_usable` and `default_linesearch`, both of which carry the measured
# figures. `max_iterations` is set past the bound so that a solve which did drift cannot be truncated
# by the cap and fail the `f < 1e-12` assertion for a different reason than the one being tested.
const ROSENBROCK_MAX_ITERATIONS = 100

@testset "the quasi-Newton methods beat gradient descent on Rosenbrock" begin
    # `Q ≡ I` makes `_BFGS`/`_DFP` identical to `GradientMethod`, which is what this separates. On
    # Rosenbrock, gradient descent is famously slow while a working quasi-Newton method is not.
    x₀ = [-1.2, 1.0]

    for algorithm in (_BFGS(), _DFP())
        x = copy(x₀)
        state = OptimizerState(algorithm, x)
        opt = Optimizer(x, rosenbrock; algorithm=algorithm, linesearch=Backtracking(expand=true),
            max_iterations=3 * ROSENBROCK_MAX_ITERATIONS, warn_iterations=0)

        solve!(x, state, opt)

        @test rosenbrock(x) < 1e-12
        @test x ≈ [1.0, 1.0] atol = 1e-5
        @test iteration_number(state) < ROSENBROCK_MAX_ITERATIONS
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
