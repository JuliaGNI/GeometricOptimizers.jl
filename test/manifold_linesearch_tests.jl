using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, _BFGS, StiefelManifold, check, iteration_number,
                           status, DecayingStatic, step_size
using SimpleSolvers: Static, Backtracking, Bisection, l2norm, Options, f_reltol
using LinearAlgebra: norm
using Test
import Random

# Until this branch, `linesearch_problem` built its merit with `SimpleSolvers.compute_new_iterate!`,
# i.e. `xₖ + α·pₖ`. On a manifold that is undefined -- and would be wrong anyway, since a step has to
# go through the retraction and the direction is a horizontal lift of a different shape than the
# point. So `Static`, the one line search that never evaluates the merit, was the only one that
# worked, and with a fixed step the first-order methods could only crawl.

# Minimise the distance to `[0, 0, 1.2]` over `St(3, 1)`, i.e. the unit sphere in R³.
const TARGET = [0.0, 0.0, 1.2]
const MINIMIZER = StiefelManifold([0.0; 0.0; 1.0;;])
f(x::StiefelManifold) = l2norm(vec(x), TARGET)
x₀() = StiefelManifold([0.0; sqrt(0.5); sqrt(0.5);;])

@testset "a searching line search runs on a Manifold at all" begin
    # every one of these threw `Not implemented for StiefelManifold{...}` from
    # `SimpleSolvers.compute_new_iterate!` before
    for linesearch in (Backtracking(Float64), Bisection(Float64)), retraction in (Geodesic(), Cayley())
        x = x₀()
        opt = Optimizer(x, f; algorithm=GradientMethod(), linesearch=linesearch, retraction=retraction)

        solve!(x, OptimizerState(GradientMethod(), x), opt)

        @test x isa StiefelManifold{Float64}       # the type survives ...
        # ... and so does the manifold. This is a round-off tolerance: `check` measures the deviation
        # from `St(3, 1)`, and a line search puts several retractions into every iteration, so it
        # accumulates a little more of it than the one-retraction-per-step loop does. The values here
        # are 0 (Geodesic) and 2.4e-15 (Cayley).
        @test check(x) < 100eps()
        @test isapprox(x, MINIMIZER; atol=1e-7)
    end
end

@testset "a searching line search converges where a fixed step only crawls" begin
    # `Static(0.1)` needs 28 iterations here and stops just under the gradient gate; `Bisection`
    # solves this one-dimensional problem essentially exactly, in two.
    results = map((Static(0.1), Backtracking(Float64), Bisection(Float64))) do linesearch
        x = x₀()
        state = OptimizerState(GradientMethod(), x)
        opt = Optimizer(x, f; algorithm=GradientMethod(), linesearch=linesearch, retraction=Geodesic())
        result = solve!(x, state, opt)
        (its=iteration_number(state), g=status(result).rg)
    end

    static, backtracking, bisection = results

    # all three terminate on a criterion, none on `max_iterations`
    @test all(r -> r.its < 1000, results)

    # `Bisection` is an exact line search on this one-dimensional problem, so it lands in a couple of
    # iterations and drives the gradient far below what the fixed step reaches
    @test bisection.its < static.its
    @test bisection.g < static.g

    # `Backtracking` is not exact, so it takes a comparable number of iterations to the fixed step
    # here; the point is that it runs at all, and reaches the gate
    @test backtracking.g ≤ f_reltol(Options(Float64))
end

@testset "the quasi-Newton methods converge on a manifold NamedTuple" begin
    # This is the SVD problem of `optimizer_convergence/svd_optim.jl`, which no algorithm could
    # converge before: with `Static(0.01)` the three first-order methods exhaust 1000 iterations at
    # a relative error of 1e-2 and a gradient of 8e-2, seven orders of magnitude off the gate.
    # `_BFGS` needs a searching line search, so it could not be used on a manifold at all.
    Random.seed!(1234)
    A = rand(10, 10)
    n = 3
    err(ps::NamedTuple) = norm(A - ps.w₁ * ps.w₂' * A)

    for linesearch in (Backtracking(Float64), Bisection(Float64))
        Random.seed!(1234)
        ps = (w₁=rand(StiefelManifold, 10, n), w₂=rand(StiefelManifold, 10, n))
        state = OptimizerState(_BFGS(), ps)
        opt = Optimizer(ps, err; algorithm=_BFGS(), linesearch=linesearch, retraction=Cayley())

        result = solve!(ps, state, opt)

        @test iteration_number(state) < 1000                    # it terminates on a criterion ...
        @test status(result).rg < 1e-6                          # ... with a small gradient
        for Y in values(ps)
            @test check(Y) < 1e-10                              # and still on the manifold
        end
    end
end

@testset "DecayingStatic decays the step geometrically" begin
    ls = DecayingStatic(; η₁=1.0e-2, η₂=1.0e-6, n=1000)

    @test step_size(ls, 0) ≈ 1.0e-2                 # starts at η₁ ...
    @test step_size(ls, 1000) ≈ 1.0e-6              # ... reaches η₂ at the horizon ...
    @test step_size(ls, 2000) < 1.0e-6              # ... and keeps going, which is what converges
    @test step_size(ls, 500) ≈ sqrt(1.0e-2 * 1.0e-6)  # geometric, so the midpoint is the geometric mean

    @test eltype(DecayingStatic(Float32)) == Float32
    @test_throws AssertionError DecayingStatic(; η₁=1.0e-6, η₂=1.0e-2)   # η₂ ≤ η₁
    @test_throws AssertionError DecayingStatic(; η₁=-1.0)
    @test_throws AssertionError DecayingStatic(; n=0)
end

@testset "DecayingStatic drives the step of a solve to zero" begin
    # `Adam`'s direction has magnitude ≈1 per component whatever the gradient is, so with a constant
    # step it circles the minimizer at that distance and never terminates on a criterion. This is
    # the `Float64`/`Cayley` case that used to run out its 1000 iterations.
    x = x₀()
    state = OptimizerState(Adam(Float64), x)
    opt = Optimizer(x, f; algorithm=Adam(Float64), retraction=Cayley(),
        linesearch=DecayingStatic(; η₁=0.1, η₂=1.0e-8, n=400))

    result = solve!(x, state, opt)

    @test iteration_number(state) < 1000            # terminates on a criterion rather than the cap
    @test status(result).rxₐ < 1e-10                # the step really has gone to zero
    @test isapprox(x, MINIMIZER; atol=1e-3)
end
