using GeometricOptimizers
using GeometricOptimizers: BFGS, DFP, cache, solver_step!, initialize_state!,
                           inverse_hessian,
                           increase_iteration_number!, iteration_number, update!,
                           isconverged, status
using LinearAlgebra: norm, dot
using Test
import Random

include("../helpers/eltypes.jl")

# `BFGS` and `DFP` build their inverse Hessian `Q` from the secant pair
#
#     δ = x⁽ᵏ⁾ - x⁽ᵏ⁻¹⁾,    γ = ∇f(x⁽ᵏ⁾) - ∇f(x⁽ᵏ⁻¹⁾),
#
# so `state.ḡ` has to hold the gradient at the *previous* iterate when the cache forms `γ`. It used
# to be refreshed at the end of the iteration, i.e. at the same iterate the next `γ` was computed at,
# which made `γ` identically zero; `ΔxΔg` was then zero too and the guard around the `Q` update
# skipped it on every iteration. Both methods silently ran as steepest descent with `Q ≡ I`.

# a genuinely non-separable, non-quadratic objective, so that a wrong `Q` cannot go unnoticed the way
# it does on a problem an exact line search solves in one step
rosenbrock(x) = sum((1 - x[i])^2 + 100 * (x[i + 1] - x[i]^2)^2 for i in 1:(length(x) - 1))

@testset "the secant pair is formed from consecutive iterates, $T" for T in REAL_ELTYPES
    # Rosenbrock rather than `F`, and only ten iterations, so that the whole window stays in the
    # pre-convergence regime: `f` is still of order 1e-2 at the end of it, in both precisions
    # (measured: 0.024 for `BFGS` and 0.15 for `DFP`, in `Float32` and in `Float64`). Once a solve reaches
    # machine precision, `δ` and `γ` underflow to zero and the guard around the `Q` update *correctly*
    # skips, and how soon that happens is a floating-point detail that differs between platforms --
    # so counting updates over a window that runs past convergence pins nothing. Every iteration in
    # this window has a genuine secant pair, and `Q` has to move on each of them.
    ITERATIONS = 10

    for algorithm in (BFGS(), DFP())
        # the standard Rosenbrock start; `-1.2` is not exact in either precision
        x = T[-1.2, 1.0]
        state = OptimizerState(algorithm, x)
        opt = Optimizer(x, rosenbrock; algorithm = algorithm, linesearch = Backtracking())

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
        @test eltype(inverse_hessian(state)) == T
        @test eltype(x) == T

        # guards the premise above: if a future change makes this converge inside the window, the
        # update count would drop legitimately and the assertion above would be measuring the wrong
        # thing. This says so rather than leaving it to look like a regression. A converged iterate
        # has `f` of order `eps(T)`; `√eps(T)` is far above that and far below the measured 0.024.
        @test rosenbrock(x) > √eps(T)
    end
end

# How many iterations either method may take on Rosenbrock from `(-1.2, 1)` with the `Backtracking`
# search that `default_linesearch` returns for both of them, i.e. the expanding one. Measured: 20 for
# `BFGS` and 34 for `DFP`, and both are *invariant* over 900 starting points one ulp either side of
# `(-1.2, 1)` in each coordinate. One bound covers both with a factor of three to spare.
#
# The search has to be the expanding one. A shrink-only `Backtracking` starts at `α = 1` and can
# never exceed it, which is right for a direction already scaled like a Newton step but wrong for
# `DFP`, whose direction is systematically under-scaled and which wants a median `α` of 8 (see
# `default_linesearch`). Starved of the step length it needs, `DFP` picks its way to the minimizer
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
# What the shrink-only search costs `DFP` — and that it costs `BFGS` nothing — is documented where
# it belongs, in `curvature_is_usable` and `default_linesearch`, both of which carry the measured
# figures. `max_iterations` is set past the bound so that a solve which did drift cannot be truncated
# by the cap and fail the `f < eps(T)` assertion for a different reason than the one being tested.
#
# In `Float32` the same start takes 17 iterations for `BFGS` and 35 for `DFP`. The invariance above
# is a `Float64` measurement and does not carry over: over the 49 `Float32` starts up to three ulps
# either side of `(-1.2, 1)`, `BFGS` takes at most 18, but `DFP` from the start two ulps below in each
# coordinate does not converge within 300 iterations (it stops at `f = 5e-4`). This file runs the
# one start, where both converge.
const ROSENBROCK_MAX_ITERATIONS = 100

@testset "the quasi-Newton methods beat gradient descent on Rosenbrock, $T" for T in REAL_ELTYPES
    # `Q ≡ I` makes `BFGS`/`DFP` identical to `GradientMethod`, which is what this separates. On
    # Rosenbrock, gradient descent is famously slow while a working quasi-Newton method is not.
    x₀ = T[-1.2, 1.0]
    minimizer = ones(T, 2)

    for algorithm in (BFGS(), DFP())
        x = copy(x₀)
        state = OptimizerState(algorithm, x)
        opt = Optimizer(
            x, rosenbrock; algorithm = algorithm, linesearch = Backtracking(expand = true),
            max_iterations = 3 * ROSENBROCK_MAX_ITERATIONS, warn_iterations = 0)

        result = solve!(x, state, opt)

        @test isconverged(status(result))
        @test eltype(x) == T
        # `f` is quadratic at the minimizer, so an iterate accurate to `√eps(T)` has `f` of order
        # `eps(T)`; measured at most `0.017eps(T)` (`DFP`, `Float32`) over both methods and precisions
        @test rosenbrock(x) < eps(T)
        # a minimizer is accurate to the root of the objective's precision; measured at most
        # `0.29√eps(T)` (`DFP`, `Float32`) over both methods and precisions
        @test norm(x - minimizer) ≤ √eps(T)
        @test iteration_number(state) < ROSENBROCK_MAX_ITERATIONS
    end
end

# A secant pair in `T` drawn from `rng`: `γ = Aδ` with `A = I + MMᵀ` symmetric positive definite, so
# that `δᵀγ ≥ ‖δ‖² > 0` and the update is well defined, as it is for a pair taken on a convex
# objective. Drawn rather than integer-valued, so that the update rounds in `T`.
function secant_pair(rng, ::Type{T}) where {T}
    δ = randn(rng, T, 3)
    M = randn(rng, T, 3, 3)
    γ = (one(M) + M * M') * δ
    (δ = δ, γ = γ, ḡ = randn(rng, T, 3), x = randn(rng, T, 3))
end

@testset "the DFP update matches the textbook formula, $T" for T in REAL_ELTYPES
    # DFP is `Q ← Q - Qγγᵀ Q/(γᵀQγ) + δδᵀ/(δᵀγ)` (nocedal2006numerical, eq. 6.15). The middle term
    # used to be built from `cache.ΔxΔx`, i.e. `δδᵀ`, which left `cache.ΔgΔg` computed and unused.
    # Driving one `update!` with a chosen secant pair pins the formula directly, without depending on
    # what a line search happens to do.
    (; δ, γ, ḡ, x) = secant_pair(Random.Xoshiro(105), T)

    cache = GeometricOptimizers.QuasiNewtonCache(DFP(), x)
    state = OptimizerState(DFP(), x)
    inverse_hessian(state) .= one(inverse_hessian(state))
    state.s .= δ
    state.ḡ .= ḡ

    update!(cache, state, x, ḡ .+ γ)

    Q = one(zeros(T, 3, 3))
    expected = Q - (Q * γ * γ' * Q) / (γ' * Q * γ) + (δ * δ') / dot(δ, γ)

    @test eltype(inverse_hessian(state)) == T
    @test inverse_hessian(state) ≈ expected

    # the δδᵀ variant this replaces is a different matrix, so the test would catch a revert
    wrong = Q - (Q * δ * δ' * Q) / (γ' * Q * γ) + (δ * δ') / dot(δ, γ)
    @test !isapprox(inverse_hessian(state), wrong)

    # `ḡ` was advanced to the gradient the cache was just called with
    @test state.ḡ ≈ ḡ .+ γ
end

@testset "the BFGS update matches the textbook formula, $T" for T in REAL_ELTYPES
    (; δ, γ, ḡ, x) = secant_pair(Random.Xoshiro(136), T)

    cache = GeometricOptimizers.QuasiNewtonCache(BFGS(), x)
    state = OptimizerState(BFGS(), x)
    inverse_hessian(state) .= one(inverse_hessian(state))
    state.s .= δ
    state.ḡ .= ḡ

    update!(cache, state, x, ḡ .+ γ)

    Q = one(zeros(T, 3, 3))
    δγ = dot(δ, γ)
    expected = Q - (δ * γ' * Q + Q * γ * δ' - (1 + (γ' * Q * γ) / δγ) * (δ * δ')) / δγ

    @test eltype(inverse_hessian(state)) == T
    @test inverse_hessian(state) ≈ expected
    @test state.ḡ ≈ ḡ .+ γ
end
