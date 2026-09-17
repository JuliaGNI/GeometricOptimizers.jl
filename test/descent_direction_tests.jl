using GeometricOptimizers
using GeometricOptimizers: ensure_descent!, NewtonOptimizerCache, OptimizerCache, direction,
                           rhs, iteration_number, _dot
using SimpleSolvers: Options
using LinearAlgebra: dot
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Random
using Test

# `sin²` has second derivative `2cos(2x)`, which is negative on `(π/4, 3π/4)`. Started from a point in
# that interval the Newton direction ascends, and up to SimpleSolvers 0.8 the bracketing line searches
# hid it by returning a negative step. They no longer do, so without `ensure_descent!` the solve
# converges to `π/2`, where `F` is *maximal* (`F = 3`) rather than minimal (`F = 0`).
F(x) = sum(sin.(x) .^ 2)

@testset "the (quasi-)Newton methods descend from an indefinite Hessian" begin
    for algorithm in (Newton(), BFGS(), DFP())
        for linesearch in (Bisection(), Backtracking(), Quadratic(), BierlaireQuadratic())
            for x₀ in (0.5, 1.0, 2.0, 3.0)
                x = fill(x₀, 3)
                state = OptimizerState(algorithm, x)
                opt = Optimizer(x, F; algorithm = algorithm, linesearch = linesearch)

                solve!(x, state, opt)

                # Every minimum of `F` has `F = 0`, every maximum has `F = 3`. Without
                # `ensure_descent!` the `x₀ = 1.0` cases converge to `π/2` and land on `F = 3`; all
                # 48 combinations here reach the minimizer.
                #
                # `1e-15` and not the `1e-27` this used to assert. That value was measured before
                # `rg` was the residual at the iterate the solve returns (issue A8): the
                # (quasi-)Newton caches reported `‖∇F‖` at whatever point the line search had last
                # probed, which overestimates it near a minimiser, so `g_converged` fired late and
                # these solves *overshot* their own stopping criterion. They now stop when they meet
                # it, one iteration earlier in 20 of the 48 cases.
                #
                # What the criterion guarantees is the tolerance to use: it stops at
                # `‖∇F‖ ≤ f_reltol = √eps ≈ 1.5e-8`, and `sin²` has curvature `2` at its minima, so
                # `F ≈ ‖∇F‖²/4 ≈ 5.6e-17` is where that lands. The worst of the 48 is `4.2e-17`,
                # right at that bound; `1e-15` is a factor of 24 above it and still fifteen orders
                # of magnitude below the `F = 3` this test exists to exclude.
                @test F(x) < 1e-15

                # and they get there fast -- the slowest is six iterations
                @test iteration_number(state) ≤ 10
            end
        end
    end
end

@testset "ensure_descent! leaves a descent direction alone" begin
    # `rhs` is `-∇f`, so `_dot(rhs, δ) > 0` is the descent test
    cache = NewtonOptimizerCache([1.0, 2.0])
    rhs(cache) .= [1.0, 1.0]
    direction(cache) .= [2.0, 3.0]         # dot = 5 > 0, descends

    ensure_descent!(cache, Newton(), Options(Float64))

    @test direction(cache) == [2.0, 3.0]
    @test dot(rhs(cache), direction(cache)) > 0
end

@testset "ensure_descent! replaces an ascent direction by the steepest-descent one" begin
    cache = NewtonOptimizerCache([1.0, 2.0])
    rhs(cache) .= [1.0, 1.0]
    direction(cache) .= [-2.0, -3.0]       # dot = -5 < 0, ascends

    ensure_descent!(cache, Newton(), Options(Float64))

    @test direction(cache) == rhs(cache)
    @test dot(rhs(cache), direction(cache)) > 0
end

@testset "ensure_descent! catches an orthogonal and a NaN direction" begin
    # `dot == 0` makes no progress, and every comparison against `NaN` is `false`; both have to be
    # replaced, which is why the test is written as `!(_dot(r, δ) > 0)`
    for δ in ([1.0, -1.0], [NaN, NaN])
        cache = NewtonOptimizerCache([1.0, 2.0])
        rhs(cache) .= [1.0, 1.0]
        direction(cache) .= δ

        ensure_descent!(cache, Newton(), Options(Float64))

        @test direction(cache) == rhs(cache)
    end
end

# `dot(B1, B2) == 2 * _dot(B1, B2)` for a lift, exactly as `_dot`'s docstring says: `dot` is the
# *ambient* Frobenius product and `_dot` the intrinsic one. The factor does not flip the sign of the
# descent test on its own, so the test below is about the other consequence, not this one.
@testset "dot and _dot disagree by the documented factor of two on a lift" begin
    Random.seed!(321)
    T = Float64
    N, n = 6, 3
    B1 = StiefelLieAlgHorMatrix(rand(SkewSymMatrix{T}, n), rand(T, N - n, n), N, n)
    B2 = StiefelLieAlgHorMatrix(rand(SkewSymMatrix{T}, n), rand(T, N - n, n), N, n)

    @test dot(B1, B2) ≈ 2 * _dot(B1, B2)
end

# A minimal `OptimizerCache` that exercises only what `ensure_descent!` needs -- `direction`, `rhs`,
# and, on the non-descending branch, `_copyto!`. On a manifold both are `AbstractLieAlgHorMatrix`,
# which is what makes the ambient `dot`'s scalar indexing reachable at all; a real (BFGS or DFP) cache
# cannot be built on a device here, because its `GlobalSection` needs a `qr` `JLArray` does not have
# (see `similar_backend.jl`).
struct _LiftCache{T, GT} <: OptimizerCache{T}
    δ::GT
    r::GT
end
GeometricOptimizers.direction(cache::_LiftCache) = cache.δ
GeometricOptimizers.rhs(cache::_LiftCache) = cache.r

@testset "ensure_descent! does not take the ambient scalar-indexed dot product on a device" begin
    Random.seed!(654)
    T = Float32
    N, n = 6, 3
    # `allowscalar(false)` is what makes this a test rather than a description: outside a
    # non-interactive session the default is `ScalarAllowed`, and a scalar index would merely warn.
    # It is task-global and left set, as at `retractions/exponential_accuracy.jl`, which `runtests.jl`
    # runs earlier and which therefore already puts every later test file under the same setting.
    allowscalar(false)

    r = StiefelLieAlgHorMatrix(
        SkewSymMatrix(JLArray(rand(T, n, n))), JLArray(rand(T, N - n, n)), N, n)
    # a copy of `r`, so `_dot(r, δ) = ‖r‖² > 0` and the direction already descends -- the
    # `_copyto!` fallback has a device defect of its own and is not what this test is about
    δ = copy(r)

    cache = _LiftCache{T, typeof(δ)}(δ, r)

    # The ambient `dot` scalar-indexes the lift, which the setting above turns into a
    # `Scalar indexing is disallowed` error. Asserting that here gives the assertion below its teeth:
    # were the setting ever lost, the call would complete whichever product it paired with.
    @test_throws ErrorException dot(r, δ)

    # `ensure_descent!` pairs with `_dot`, which reads the free parameters directly and never reaches
    # the rejected path. That the call returns at all is the assertion this testset exists for.
    @test (ensure_descent!(cache, BFGS(), Options(T)); true)
end
