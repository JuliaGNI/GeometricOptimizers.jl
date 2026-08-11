using GeometricOptimizers
using GeometricOptimizers: ensure_descent!, NewtonOptimizerCache, direction, rhs, iteration_number
using SimpleSolvers: Options
using LinearAlgebra: dot
using Test

# `sin²` has second derivative `2cos(2x)`, which is negative on `(π/4, 3π/4)`. Started from a point in
# that interval the Newton direction ascends, and up to SimpleSolvers 0.8 the bracketing line searches
# hid it by returning a negative step. They no longer do, so without `ensure_descent!` the solve
# converges to `π/2`, where `F` is *maximal* (`F = 3`) rather than minimal (`F = 0`).
F(x) = sum(sin.(x) .^ 2)

@testset "the (quasi-)Newton methods descend from an indefinite Hessian" begin
    for algorithm in (Newton(), GeometricOptimizers._BFGS(), GeometricOptimizers._DFP())
        for linesearch in (Bisection(), Backtracking(), Quadratic(), BierlaireQuadratic())
            for x₀ in (0.5, 1.0, 2.0, 3.0)
                x = fill(x₀, 3)
                state = OptimizerState(algorithm, x)
                opt = Optimizer(x, F; algorithm=algorithm, linesearch=linesearch)

                solve!(x, state, opt)

                # Every minimum of `F` has `F = 0`, every maximum has `F = 3`. Without
                # `ensure_descent!` the `x₀ = 1.0` cases converge to `π/2` and land on `F = 3`; all
                # 48 combinations here reach the minimizer, and the worst of them gets to 1.1e-28,
                # so this is a machine-precision tolerance and not a "close enough" one.
                @test F(x) < 1e-27

                # and they get there fast -- the slowest is seven iterations
                @test iteration_number(state) ≤ 10
            end
        end
    end
end

@testset "ensure_descent! leaves a descent direction alone" begin
    # `rhs` is `-∇f`, so `dot(rhs, δ) > 0` is the descent test
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
    # replaced, which is why the test is written as `!(dot(r, δ) > 0)`
    for δ in ([1.0, -1.0], [NaN, NaN])
        cache = NewtonOptimizerCache([1.0, 2.0])
        rhs(cache) .= [1.0, 1.0]
        direction(cache) .= δ

        ensure_descent!(cache, Newton(), Options(Float64))

        @test direction(cache) == rhs(cache)
    end
end
