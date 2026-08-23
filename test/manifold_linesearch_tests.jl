using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, StiefelManifold, check, iteration_number,
                           status, DecayingStatic, step_size, increase_iteration_number!,
                           solver_step!, update!
using GeometricOptimizers: ScaledSquaring, NativePade, AugmentedPade, ProjectedSkew
using GeometricOptimizers: linesearch_problem, retraction_differential, retraction, initialize!,
                           cache, gradient, hessian, problem, StiefelProjection
using GeometricOptimizers: step_αmax, _manifold_αmax, linesearch_parameters, _caller_αmax,
                           step_ceiling, DEFAULT_STEP_CEILING, linesearch_rejected,
                           OptimizerCache, GradientMethod, direction, StiefelLieAlgHorMatrix
using SimpleSolvers: Static, Backtracking, Bisection, Quadratic, BierlaireQuadratic, StrongWolfe, l2norm
using SimpleSolvers: LinesearchStatus, LINESEARCH_FLOOR, LINESEARCH_EXHAUSTED, LINESEARCH_NO_DESCENT,
                     LINESEARCH_DECREASED
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
# the RNG is seeded before the state and the optimizer are built: unseeded, `BFGS` + `Backtracking`
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

# `trial_slope` used to pair the gradient with the direction `B` itself. That is `φ'(α)` only where
# `α ↦ retract(αB)` is a one-parameter subgroup -- `Geodesic` is and `Cayley` is not -- so under
# `Cayley` the slope was exact at `α = 0` and drifted from there. Measured on the `St(6, 3)` problem
# below, against a central difference of the merit the search itself evaluates:
#
#     α                 0.25     0.5      1.0      2.0
#     paired with B     2.2%    8.9%      36%     143%
#     with D(α)        4e-10   1e-9     4e-9    4e-10
#
# `Backtracking`, the default, never saw it: it evaluates `φ'` at `α = 0` only, where the two agree.
# `Bisection` uses the sign and `StrongWolfe` compares against `φ'(0)`, so both merely paid a few
# iterations. `Quadratic` and `BierlaireQuadratic` fit a polynomial to it *quantitatively*, and on the
# SVD problem that took `BFGS` off the manifold on two of eight starting points -- open issue A1b.
# `retraction_differential` supplies the generator that turns with the step.
#
# The tolerance is set by the central difference, not by the slope: `h = 1e-6` leaves a truncation
# error of order `1e-9`, which is what the "with D(α)" row above is.
const SLOPE_TOLERANCE = 1e-7

"""
    slope_errors(ps, F, retraction, αs)

The relative error of `φ'(α)` against a central difference of `φ(α)`, both taken from the
`LinesearchProblem` the optimizer actually builds, at each `α`.
"""
function slope_errors(ps, F, retraction, αs; h=1e-6)
    Random.seed!(7)
    opt = Optimizer(ps, F; algorithm=BFGS(), retraction=retraction, linesearch=Backtracking(Float64))
    state = OptimizerState(BFGS(), ps)
    c = cache(opt)
    initialize!(c, ps)
    update!(c, state, gradient(opt), hessian(opt), ps)

    ls = linesearch_problem(problem(opt), gradient(opt), c, retraction)
    params = (x=ps, state=state)

    map(αs) do α
        φ′ = ls.D(α, params)
        difference = (ls.F(α + h, params) - ls.F(α - h, params)) / (2h)
        abs(φ′ - difference) / abs(difference)
    end
end

@testset "φ' is the derivative of φ, under both retractions" begin
    αs = (0.0, 0.25, 0.5, 1.0, 2.0)

    Random.seed!(1234)
    Y = rand(StiefelManifold, 6, 3)
    # a `NamedTuple` mixing a manifold with an ordinary array, so that the pass-through for a
    # Euclidean block is covered too -- its retraction is addition, so `D(α) = B` there
    mixed = (w=rand(StiefelManifold, 6, 3), b=randn(4))

    for retraction in (Geodesic(), Cayley())
        for e in slope_errors(Y, Z -> sum(abs2, Z .- 0.3) + sum(sin.(Z)), retraction, αs)
            @test e < SLOPE_TOLERANCE
        end

        objective(p) = sum(abs2, p.w .- 0.3) + sum(sin.(p.w)) + sum(abs2, p.b) + sum(p.b)
        for e in slope_errors(mixed, objective, retraction, αs)
            @test e < SLOPE_TOLERANCE
        end
    end
end

# The Grassmann branch of `retraction_differential` is checked directly rather than through a merit:
# `D(α)` against a central difference of the retraction it is the derivative of. This is the only
# place the identity is checked against `retract` rather than against `f ∘ retract`, which is what
# makes it independent of `global_rep` and `_dot`.
#
# It was written this way because a `GrassmannManifold` could not be driven through an `Optimizer` at
# all (issue A11, the concrete content of issue #27). That is fixed, and
# `test/grassmann_optimizer_tests.jl` now covers the end-to-end solve — but the direct check is worth
# keeping for the independence above, so it stays.
struct UncoveredRetraction <: GeometricOptimizers.AbstractRetraction end

@testset "retraction_differential is d/dα retract(αB), for both lifts" begin
    for LT in (StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix)
        Random.seed!(1234)
        B = rand(LT{Float64}, 6, 3)
        E = StiefelProjection(B)
        frame(retr, α) = Matrix(retraction(retr, α * B))

        for retr in (Geodesic(), Cayley()), α in (0.25, 0.5, 1.0, 2.0)
            h = 1e-6
            difference = (frame(retr, α + h) * E - frame(retr, α - h) * E) / (2h)
            velocity = frame(retr, α) * Matrix(retraction_differential(retr, B, α)) * E

            @test norm(velocity - difference) / norm(difference) < SLOPE_TOLERANCE
        end

        # `α = 0` returns the direction untouched, which is what keeps `Backtracking` free
        @test retraction_differential(Cayley(), B, 0.0) === B

        # A retraction with no differential of its own has to say so here rather than fail with a
        # `MethodError` from inside a merit evaluation, three frames down. Same reason `retraction`
        # carries an explicit error for the combinations it does not cover.
        @test_throws ErrorException retraction_differential(UncoveredRetraction(), B, 1.0)
    end
end

@testset "a searching line search runs on a Manifold at all" begin
    # every one of these threw `Not implemented for StiefelManifold{...}` from
    # `SimpleSolvers.compute_new_iterate!` before. All four searching methods this package exports are
    # covered, not just the two the rest of the file uses.
    searching = (Backtracking(Float64), Backtracking(Float64; expand=true), Bisection(Float64),
        Quadratic(Float64), BierlaireQuadratic(Float64))
    # All four exponential algorithms are put through a real solve here, not just through
    # `geodesic` in isolation: they have to agree on where the optimizer ends up, not merely on the
    # value of one retraction. `TaylorSeries` is the one left out, and deliberately — it is not a
    # retraction at a large lift and this asserts convergence.
    retractions = (Geodesic(ScaledSquaring()), Geodesic(NativePade()), Geodesic(AugmentedPade()),
        Geodesic(ProjectedSkew()), Cayley())
    for linesearch in searching, retraction in retractions
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
    # `BFGS` needs a searching line search, so it could not be used on a manifold at all.
    Random.seed!(1234)
    A = rand(10, 10)
    n = 3
    err(ps::NamedTuple) = norm(A - ps.w₁ * ps.w₂' * A)

    for linesearch in (Backtracking(Float64), Bisection(Float64))
        Random.seed!(1234)
        ps = (w₁=rand(StiefelManifold, 10, n), w₂=rand(StiefelManifold, 10, n))
        state = OptimizerState(BFGS(), ps)
        opt = Optimizer(ps, err; algorithm=BFGS(), linesearch=linesearch, retraction=Cayley())

        result = solve!(ps, state, opt)

        @test iteration_number(state) < 1000                    # it terminates on a criterion ...
        @test status(result).rg < 1e-6                          # ... with a small gradient
        for Y in values(ps)
            @test check(Y) < 1e-10                              # and still on the manifold
        end
    end
end

# The `NamedTuple` counterpart of `optimizer_tests.jl`'s Euclidean loop. The manifold branch of
# `trial_slope` allocates rather than evaluating into the cache, so it never hit the
# `gradient(cache)` `MethodError` that made that loop's Euclidean twin throw -- but nothing solved
# here with a first-order method and a searching line search either, and that is the gap that let the
# defect through.
#
# Two spheres rather than the SVD problem of the testset above: the first-order methods do not
# converge on that one at all (1000 iterations at `err = 1.71`), and `Quadratic` and
# `BierlaireQuadratic` leave the manifold there under `Cayley` -- `check` of `7.1e-4` and `4.8e-3`
# for `GradientMethod` against `5.8e-14` for `Backtracking` -- which is open issue A1b and not this
# file's subject. The exact `Cayley` differential improves the `Quadratic` column by about 2.5x
# (`1.8e-3` to `7.1e-4`) and leaves `BierlaireQuadratic` bit-identical, so it does not close that
# issue; see the CHANGELOG.
const TARGET₂ = [0.0, 1.5, 0.0]
const MINIMIZER₂ = StiefelManifold([0.0; 1.0; 0.0;;])
two_spheres(ps::NamedTuple) = l2norm(vec(ps.w₁), TARGET) + l2norm(vec(ps.w₂), TARGET₂)

function ps₀()
    Random.seed!(1234)
    (w₁=StiefelManifold([0.0; sqrt(0.5); sqrt(0.5);;]),
        w₂=StiefelManifold([sqrt(0.5); 0.0; sqrt(0.5);;]))
end

const NT_LINESEARCHES = (Static(0.1), Backtracking(Float64), Backtracking(Float64; expand=true),
    Bisection(Float64), Quadratic(Float64), BierlaireQuadratic(Float64), StrongWolfe(Float64; c₂=0.1))

@testset "the first-order methods solve on a manifold NamedTuple, on every line search" begin
    for method in (GradientMethod(), MomentumMethod(0.1)),
        linesearch in NT_LINESEARCHES,
        retraction in (Geodesic(), Cayley())

        ps = ps₀()
        state = OptimizerState(method, ps)
        opt = Optimizer(ps, two_spheres; algorithm=method, linesearch=linesearch,
            retraction=retraction, max_iterations=1000)

        solve!(ps, state, opt)

        # 64 is the worst of the 28, and it is a `Static` one: every searching line search here is
        # under 25
        @test iteration_number(state) < 100                     # terminates on a criterion ...
        @test isapprox(ps.w₁, MINIMIZER; atol=1e-6)             # ... at the minimiser ...
        @test isapprox(ps.w₂, MINIMIZER₂; atol=1e-6)
        for Y in values(ps)
            @test check(Y) < MANIFOLD_TOLERANCE                 # ... and still on the manifold
        end
    end
end

@testset "Adam runs on a manifold NamedTuple under a searching line search" begin
    # `Adam` is in this file's coverage but not in the loop above, and the reason is the one the
    # `DecayingStatic` testset states: its direction has magnitude ≈1 per component whatever the
    # gradient is, so with a step that does not shrink it circles the minimiser at that distance. It
    # needs 251-331 iterations here where the two methods above need 9-64, which is why
    # `default_linesearch` keeps `Static` for `AdamFamily` -- the searching alternatives cost an
    # order of magnitude and buy nothing.
    #
    # It does now *terminate* under all fourteen, which is new: `Adam` + `BierlaireQuadratic` used to
    # run out all 1000 iterations under both retractions while sitting 6.8e-6 from the minimiser,
    # with no criterion it could meet. That was recorded as issue A9, and it was the same defect as
    # A7 -- a rejected search returns `α = 1` and `solver_step!` exempted the `AdamFamily` methods
    # from doing anything about it. With the exemption gone the worst of the fourteen is 331
    # iterations, so the iteration bound below is a real one rather than a restatement of the cap.
    #
    # The distance tolerance moves with it, 1e-4 to 1e-6. The worst of the fourteen is now 5.0e-7 and
    # that one is `Static`, i.e. the orbit of radius ∝ α this comment opens with, which no line-search
    # change can touch; every *searching* one is at 1.8e-8 or better, against the 6.8e-6 that
    # `BierlaireQuadratic` used to sit at while exhausting the cap.
    for linesearch in NT_LINESEARCHES, retraction in (Geodesic(), Cayley())
        ps = ps₀()
        state = OptimizerState(Adam(Float64), ps)
        opt = Optimizer(ps, two_spheres; algorithm=Adam(Float64), linesearch=linesearch,
            retraction=retraction, max_iterations=1000, warn_iterations=0)

        solve!(ps, state, opt)

        @test iteration_number(state) < 1000                    # terminates on a criterion ...
        @test isapprox(ps.w₁, MINIMIZER; atol=1e-6)             # ... 5.0e-7 is the worst of the 14
        @test isapprox(ps.w₂, MINIMIZER₂; atol=1e-6)
        for Y in values(ps)
            @test check(Y) < MANIFOLD_TOLERANCE
        end
    end
end

@testset "the reused gradient is the one a fresh evaluation would give" begin
    # `solver_step!` refreshes `latest_gradient` at the accepted iterate and the next
    # `update!(cache, ...)` reuses it rather than evaluating `∇f` again at the same point; see
    # `store_gradient!`. The manifold case is the one where that could go wrong quietly, because the
    # gradient is expressed in the frame of a `GlobalSection` and the cache's and the state's frames
    # are advanced by two different calls. They are the same call underneath -- `update_section!`'s
    # three-argument method has the body the two-argument one uses -- and this asserts it, bit for
    # bit, rather than taking it on trust.
    for method in (GradientMethod(), MomentumMethod(0.1), Adam(Float64)),
        retraction in (Geodesic(), Cayley())

        ps = ps₀()
        state = OptimizerState(method, ps)
        opt = Optimizer(ps, two_spheres; algorithm=method, linesearch=Bisection(Float64),
            retraction=retraction)
        grad = GeometricOptimizers.gradient(opt)

        for k in 1:6
            @test GeometricOptimizers.latest_gradient_is_current(GeometricOptimizers.cache(opt), state, ps) == (k > 1)

            fresh = GeometricOptimizers.global_rep(GeometricOptimizers.section(state), grad(ps))
            increase_iteration_number!(state)
            solver_step!(ps, state, opt)
            update!(state, opt, ps)

            stored = GeometricOptimizers.gradient_array(GeometricOptimizers.cache(opt))
            for key in keys(fresh)
                @test stored[key] == fresh[key]
            end
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
    # and `_copyto!` for a section) sat on that boundary; without them `BFGS` on a bare `Manifold`
    # died in `outer!` with `AssertionError: axes(O, 1) == axes(x, 1)`.
    for algorithm in (BFGS(), DFP()), linesearch in (Backtracking(Float64), Bisection(Float64))
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

# The step ceiling of issue A1b. A line search bounds its step by the merit, and on a compact manifold
# `φ` is bounded, so that test never fires -- `Quadratic` returned `α = 4.3e7` on a direction of norm
# 5.54 and reported a genuine decrease. The bound that does exist is geometric (`2π` for a rotation,
# over `‖δ‖`) and changes at every step, which is why SimpleSolvers 0.12 takes it per call through
# `params.αmax` and leaves the value to this package. See `DEFAULT_STEP_CEILING`.

@testset "step_αmax is c⋅2π/‖δ‖, and Inf where there is no scale" begin
    @test step_αmax(1.0, [3.0, 4.0]) ≈ 2π / 5
    @test step_αmax(2.0, [3.0, 4.0]) ≈ 2 * 2π / 5

    # `Inf` is what `SimpleSolvers.linesearch_αmax` reads as "the caller has no scale of its own", and
    # it leaves the method's own ceiling standing. These three all have to produce it rather than a
    # `NaN` or a non-positive value, which upstream rejects with an `ArgumentError` -- correctly, since
    # ignoring one would hand back exactly the unbounded step the ceiling exists to rule out.
    @test step_αmax(1.0, [0.0, 0.0]) == Inf        # a vanishing direction
    @test step_αmax(1.0, [NaN, 1.0]) == Inf        # a direction that has already gone wrong
    @test step_αmax(1.0, [Inf, 1.0]) == Inf
    @test step_αmax(Inf, [3.0, 4.0]) == Inf        # the ceiling switched off

    # in `T`, so that a `Float32` solve does not silently widen
    @test step_αmax(1.0f0, Float32[3, 4]) isa Float32
    @test step_αmax(1.0f0, Float32[3, 4]) ≈ 2.0f0π / 5.0f0
end

# `_manifold_αmax` is the block-wise half, and issue A15. One `α` is applied to every block of a
# `NamedTuple`, so each manifold block needs `‖αδᵢ‖ ≤ 2πc` and the binding one is the largest `‖δᵢ‖`.
# A block that is an ordinary array contributes nothing: the `2π` is the turn of a rotation, and a
# vector space has no such scale to impose or to inflate.
@testset "the ceiling is derived per block, over the manifold blocks only" begin
    lift(scale) = scale * rand(Random.Xoshiro(1234), StiefelLieAlgHorMatrix{Float64}, 6, 3)

    # two manifold blocks: the *smallest* per-block ceiling, i.e. the largest direction, and not the
    # quadrature combination of the two that `l2norm` over the whole `NamedTuple` would give
    let sol = (a=rand(StiefelManifold, 6, 3), b=rand(StiefelManifold, 6, 3)),
        δ = (a=lift(1.0), b=lift(3.0))

        @test _manifold_αmax(values(sol), values(δ), 1.0) ==
              min(step_αmax(1.0, δ.a), step_αmax(1.0, δ.b))
        @test _manifold_αmax(values(sol), values(δ), 1.0) == step_αmax(1.0, δ.b)
        # looser than the quadrature version, and necessarily so: `‖δ‖ ≥ maxᵢ‖δᵢ‖`, so bounding by
        # the total tightened every block by the presence of its neighbours
        @test _manifold_αmax(values(sol), values(δ), 1.0) > step_αmax(1.0, δ)
    end

    # mixed: the Euclidean block neither imposes a ceiling nor tightens the manifold block's. This is
    # the direction of the A15 error -- a Euclidean block of large norm used to drag the whole
    # ceiling down with it through the quadrature norm.
    let sol = (Y=rand(StiefelManifold, 6, 3), W=zeros(3, 4)),
        δ = (Y=lift(1.0), W=fill(1.0e3, 3, 4))

        @test _manifold_αmax(values(sol), values(δ), 1.0) == step_αmax(1.0, δ.Y)
        @test _manifold_αmax(values(sol), values(δ), 1.0) > step_αmax(1.0, δ)
    end

    # no manifold block at all: no scale exists, so there is no ceiling. `Inf` and not a number:
    # `SimpleSolvers.linesearch_αmax` reads it as "the caller has no scale of its own".
    let sol = (W=zeros(3, 4), b=zeros(3)), δ = (W=fill(2.0, 3, 4), b=fill(5.0, 3))
        @test _manifold_αmax(values(sol), values(δ), 1.0) == Inf
    end

    # in `T`, as `step_αmax` is
    let sol = (W=zeros(Float32, 2, 2),), δ = (W=fill(2.0f0, 2, 2),)
        @test _manifold_αmax(values(sol), values(δ), 1.0f0) isa Float32
    end
end

@testset "linesearch_parameters supplies αmax where a manifold supplies a scale" begin
    # Euclidean: no geometric scale exists and none is needed, since `f(x + αp)` grows with `α` and
    # the search's own decrease test rejects an over-long step unaided. Omitting the field (rather
    # than passing `Inf`) is also what keeps upstream's `hasproperty` guard constant-folded.
    let x = [1.0, 2.0]
        algorithm = GradientMethod()
        state = OptimizerState(algorithm, x)
        c = OptimizerCache(algorithm, x)
        params = linesearch_parameters(c, x, state, DEFAULT_STEP_CEILING)
        @test !hasproperty(params, :αmax)
        @test params.x === x && params.state === state
    end

    # Manifold: the ceiling is there, and it is the one `step_αmax` computes from the direction the
    # cache holds at that moment.
    let x = x₀()
        algorithm = BFGS()
        state = OptimizerState(algorithm, x)
        opt = Optimizer(x, f; algorithm=algorithm)
        # the same two calls `slope_errors` above makes to get a cache holding a real direction
        initialize!(cache(opt), x)
        update!(cache(opt), state, gradient(opt), hessian(opt), x)

        params = linesearch_parameters(cache(opt), x, state, DEFAULT_STEP_CEILING)
        @test hasproperty(params, :αmax)
        @test params.αmax == step_αmax(DEFAULT_STEP_CEILING, direction(cache(opt)))
        @test 0 < params.αmax < Inf
    end

    # A `NamedTuple` of manifolds: block-wise, and finite.
    let ps = ps₀()
        algorithm = BFGS()
        state = OptimizerState(algorithm, ps)
        opt = Optimizer(ps, two_spheres; algorithm=algorithm)
        initialize!(cache(opt), ps)
        update!(cache(opt), state, gradient(opt), hessian(opt), ps)

        params = linesearch_parameters(cache(opt), ps, state, DEFAULT_STEP_CEILING)
        @test params.αmax ==
              _manifold_αmax(values(ps), values(direction(cache(opt))), DEFAULT_STEP_CEILING)
        @test 0 < params.αmax < Inf
    end

    # A `NamedTuple` with *no* manifold block is Euclidean, whatever `ArrayNamedTuple` says: it is
    # any `NamedTuple` of arrays, so this used to take the manifold branch and be handed a ceiling
    # derived from a rotation the problem does not have. See the solve below.
    let ps = (W=[1.0 2.0; 3.0 4.0], b=[5.0, 6.0])
        algorithm = GradientMethod()
        state = OptimizerState(algorithm, ps)
        c = OptimizerCache(algorithm, ps)
        params = linesearch_parameters(c, ps, state, DEFAULT_STEP_CEILING)
        @test params.αmax == Inf     # equivalent to the omission in the `AbstractVector` branch
    end
end

# The regression test for that last case. `‖αδ‖ ≤ 2π` on a problem whose optimum is 10⁴ away is a
# step-size cap of `6e-4`, so the solve crawls where it should converge outright -- measured at 3 184
# iterations against 1 before the ceiling became block-wise. The assertion is *relative*, between the
# ceiling and no ceiling on the same problem, so it pins no figure of its own.
@testset "a NamedTuple with no manifold block is not bounded by a manifold's geometry" begin
    target = fill(1.0e4, 4)
    far_away(ps::NamedTuple) = sum(abs2, ps.w .- target) / 2
    far_away(x::AbstractVector) = sum(abs2, x .- target) / 2

    iterations(x, ceiling) = let state = OptimizerState(BFGS(), x)
        solve!(x, state, Optimizer(x, far_away; algorithm=BFGS(), max_iterations=20_000,
            warn_iterations=0, step_ceiling=ceiling))
        iteration_number(state)
    end

    @test iterations((w=zeros(4),), DEFAULT_STEP_CEILING) == iterations((w=zeros(4),), Inf)
    # ... and the same problem written as a vector, which never had a ceiling, agrees with both
    @test iterations((w=zeros(4),), DEFAULT_STEP_CEILING) == iterations(zeros(4), DEFAULT_STEP_CEILING)
end

# Issue B3. A search stopped *at* the ceiling `solver_step!` itself imposed, with the merit still
# falling, is classified by the same round-off rule as any other step and so can come back as
# `LINESEARCH_FLOOR` -- which is a claim about the *direction*, and which `linesearch_rejected`
# answers by throwing `Q` away. What was established is only that no *permitted* step decreases the
# merit measurably, so the ceiling case is exempt and the step is taken.
@testset "a step at the caller's own ceiling is not a rejected direction" begin
    αmax = 2.5

    @test !linesearch_rejected(LinesearchStatus(αmax, LINESEARCH_FLOOR), αmax)
    @test linesearch_rejected(LinesearchStatus(1.0, LINESEARCH_FLOOR), αmax)

    # the other two outcomes are not confusable with a bound step and stay rejections: the budget
    # running out and `φ'(0) ≥ 0` are true of the direction whatever ceiling was in force
    @test linesearch_rejected(LinesearchStatus(αmax, LINESEARCH_EXHAUSTED), αmax)
    @test linesearch_rejected(LinesearchStatus(αmax, LINESEARCH_NO_DESCENT), αmax)

    # a successful search is not a rejection either way
    @test !linesearch_rejected(LinesearchStatus(αmax, LINESEARCH_DECREASED), αmax)

    # with no ceiling -- every Euclidean solve, since `linesearch_parameters` omits the field there
    # and `_caller_αmax` reads that as `Inf` -- the two forms agree on every outcome
    for oc in (LINESEARCH_FLOOR, LINESEARCH_EXHAUSTED, LINESEARCH_NO_DESCENT, LINESEARCH_DECREASED)
        status = LinesearchStatus(1.0e9, oc)
        @test linesearch_rejected(status, Inf) == linesearch_rejected(status)
    end

    @test _caller_αmax(Float64, (x=1, state=2, αmax=3.0)) == 3.0
    @test _caller_αmax(Float64, (x=1, state=2)) == Inf
end

@testset "the ceiling is a per-Optimizer knob, and Inf switches it off" begin
    x = x₀()
    @test step_ceiling(Optimizer(x, f; algorithm=BFGS())) == DEFAULT_STEP_CEILING
    @test step_ceiling(Optimizer(x, f; algorithm=BFGS(), step_ceiling=0.5)) == 0.5
    @test step_ceiling(Optimizer(x, f; algorithm=BFGS(), step_ceiling=Inf)) == Inf

    # carried in the element type of the parameters, not in whatever the keyword was written as
    @test step_ceiling(Optimizer(x, f; algorithm=BFGS(), step_ceiling=1)) isa Float64
end

# The regression test for A1b itself. `svd_optim.jl` runs the two polynomial searches on seed `1234`
# only, where they always passed; the failure lives on other starting points. This is the smallest
# problem that reproduces the mechanism -- a bounded merit on a compact manifold -- rather than the
# `St(20, 3)²` SVD problem, whose eight-seed sweep is in `scripts/retraction_accuracy.jl`.
@testset "a bounded merit does not produce an unbounded step" begin
    for retraction in (Geodesic(), Cayley()), linesearch in (Quadratic(Float64), BierlaireQuadratic(Float64))
        x = x₀()
        state = OptimizerState(BFGS(), x)
        opt = Optimizer(x, f; algorithm=BFGS(), linesearch=linesearch, retraction=retraction)

        solve!(x, state, opt)

        @test check(x) < MANIFOLD_TOLERANCE
        @test isapprox(x, MINIMIZER; atol=1e-6)
    end
end
