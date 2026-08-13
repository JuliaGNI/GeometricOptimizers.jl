using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, _BFGS, _DFP, StiefelManifold, check, iteration_number,
                           status, DecayingStatic, step_size
using SimpleSolvers: Static, Backtracking, Bisection, Quadratic, BierlaireQuadratic, l2norm
using LinearAlgebra: norm
using Test
import Random

# Until this branch, `linesearch_problem` built its merit with `SimpleSolvers.compute_new_iterate!`,
# i.e. `xₖ + α·pₖ`. On a manifold that is undefined -- and would be wrong anyway, since a step has to
# go through the retraction and the direction is a horizontal lift of a different shape than the
# point. So `Static`, the one line search that never evaluates the merit, was the only one that
# worked, and with a fixed step the first-order methods could only crawl.

# Every `GlobalSection` -- one per `OptimizerState`, one per cache -- completes the frame with
# `global_section`, which draws from the *global* RNG. So a solve on a manifold is only reproducible if
# the RNG is seeded before the state and the optimizer are built: unseeded, `_BFGS` + `Backtracking`
# below takes 17 or 18 iterations from run to run and `check(x)` wanders between 2e-16 and 4e-14.
# `x₀` therefore seeds, and is called immediately before every state/optimizer pair in this file.
# `manifold_optimizers_with_new_interface.jl` seeds inside its `optimize` for the same reason.

# Minimise the distance to `[0, 0, 1.2]` over `St(3, 1)`, i.e. the unit sphere in R³.
const TARGET = [0.0, 0.0, 1.2]
const MINIMIZER = StiefelManifold([0.0; 0.0; 1.0;;])
f(x::StiefelManifold) = l2norm(vec(x), TARGET)

function x₀()
    Random.seed!(1234)
    StiefelManifold([0.0; sqrt(0.5); sqrt(0.5);;])
end

# `check` measures the deviation from `St(3, 1)`. A line search puts several retractions into every
# iteration, so it accumulates more round-off than the one-retraction-per-step loop does, and a
# quasi-Newton run of 17-27 iterations accumulates more again. This is the tolerance
# `optimizer_convergence/svd_optim.jl` uses for the same reason.
const MANIFOLD_TOLERANCE = 1e-12

@testset "a searching line search runs on a Manifold at all" begin
    # every one of these threw `Not implemented for StiefelManifold{...}` from
    # `SimpleSolvers.compute_new_iterate!` before. All four searching methods this package exports are
    # covered, not just the two the rest of the file uses.
    searching = (Backtracking(Float64), Backtracking(Float64; expand=true), Bisection(Float64),
        Quadratic(Float64), BierlaireQuadratic(Float64))
    for linesearch in searching, retraction in (Geodesic(), Cayley())
        x = x₀()
        opt = Optimizer(x, f; algorithm=GradientMethod(), linesearch=linesearch, retraction=retraction)

        solve!(x, OptimizerState(GradientMethod(), x), opt)

        @test x isa StiefelManifold{Float64}       # the type survives ...
        @test check(x) < MANIFOLD_TOLERANCE        # ... and so does the manifold
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
        (its=iteration_number(state), g=status(result).rg, g_converged=status(result).g_converged)
    end

    static, backtracking, bisection = results

    # all three terminate on a criterion, none on `max_iterations`
    @test all(r -> r.its < 1000, results)

    # `Bisection` is an exact line search on this one-dimensional problem, so it lands in a couple of
    # iterations and drives the gradient far below what the fixed step reaches
    @test bisection.its < static.its
    @test bisection.g < static.g

    # `Backtracking` accepts α = 1 on every step of this problem, so it behaves like `Static(1.0)` and
    # takes a comparable number of iterations to the fixed step (31 against 28). The point is that it
    # runs at all, and that it still meets the gradient criterion rather than the iteration cap.
    @test backtracking.g_converged
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

@testset "the quasi-Newton methods run on a bare Manifold" begin
    # `Q` is sized by the *intrinsic* dimension -- the length of the flattening, 2 for `St(3, 1)` --
    # while the gradient and the direction are horizontal lifts of the ambient shape, `3 × 3`. Four
    # methods that the `NamedTuple` case had and the bare case did not (`outer!`, `_mul!`, `alloc_h`
    # and `_copyto!` for a section) sat on that boundary; without them `_BFGS` on a bare `Manifold`
    # died in `outer!` with `AssertionError: axes(O, 1) == axes(x, 1)`.
    for algorithm in (_BFGS(), _DFP()), linesearch in (Backtracking(Float64), Bisection(Float64))
        x = x₀()
        state = OptimizerState(algorithm, x)
        opt = Optimizer(x, f; algorithm=algorithm, linesearch=linesearch)

        result = solve!(x, state, opt)

        @test x isa StiefelManifold{Float64}
        @test check(x) < MANIFOLD_TOLERANCE
        @test iteration_number(state) < 100          # 2 with `Bisection`, 17 and 26 with `Backtracking`
        @test status(result).rg < 1e-7
        @test isapprox(x, MINIMIZER; atol=1e-7)
    end
end
