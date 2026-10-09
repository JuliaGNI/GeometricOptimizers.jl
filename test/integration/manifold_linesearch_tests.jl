using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, StiefelManifold, check, iteration_number,
                           status, DecayingStatic, step_size, increase_iteration_number!,
                           solver_step!, update!
using GeometricOptimizers: ScaledSquaring, NativePade, AugmentedPade, ProjectedSkew
using GeometricOptimizers: linesearch_problem, retraction_differential, retraction,
                           initialize!,
                           cache, gradient, problem, StiefelProjection
using GeometricOptimizers: step_αmax, _manifold_αmax, linesearch_parameters, _caller_αmax,
                           step_ceiling, DEFAULT_STEP_CEILING, linesearch_rejected,
                           OptimizerCache, GradientMethod, direction, StiefelLieAlgHorMatrix
using SimpleSolvers: Static, Backtracking, Bisection, Quadratic, BierlaireQuadratic,
                     StrongWolfe, l2norm
using SimpleSolvers: LinesearchStatus, LINESEARCH_FLOOR, LINESEARCH_EXHAUSTED,
                     LINESEARCH_NO_DESCENT,
                     LINESEARCH_DECREASED
using GeometricOptimizers: isconverged
using LinearAlgebra: norm, svd
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/manifold_tolerance.jl")

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

# Minimise the distance to `[0, 0, 1.2]` over `St(3, 1)`, i.e. the unit sphere in R³. The minimiser
# is `[0, 0, 1]` in closed form; its integer entries are the exact answer, not data.
target(::Type{T}) where {T} = T[0, 0, 1.2]
minimizer(::Type{T}) where {T} = StiefelManifold(T[0; 0; 1;;])
f(x::StiefelManifold{T}) where {T} = l2norm(vec(x), target(T))

# The start is the point at 45° between the second and third axes, whose entries `√½` are not exact
# in binary.
function x₀(::Type{T}) where {T}
    Random.seed!(1234)
    StiefelManifold(T[0; sqrt(T(0.5)); sqrt(T(0.5));;])
end

# `check` measures the deviation from `St(3, 1)`. A line search puts several retractions into every
# iteration, so it accumulates more round-off than the one-retraction-per-step loop does, and a
# quasi-Newton run of 17-27 iterations accumulates more again. `manifold_tolerance(T)` is the
# tolerance `verification/svd_optim.jl` uses for the same reason.

# The distance of a solve on the sphere problems of this file to their minimiser. The objective is
# quadratic at the minimiser, so a solve stops within a multiple of `√eps(T)` of it. The worst
# measured over every solve to convergence on `x₀` and `ps₀` below, with three seeds of the global
# RNG, is 1.8√eps(T) in `Float64` and 4.5√eps(T) in `Float32` (`Adam`).
sphere_tolerance(::Type{T}) where {T} = 10 * sqrt(eps(T))

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
# The tolerance is set by the central difference, not by the slope. At its optimal step
# `h = ∛eps(T)` the difference is accurate to the order of `eps(T)^(2/3)`, which is below `√eps(T)`:
# measured over five seeds, the worst relative error of the slope and of the differential below is
# 2.5√eps(T) in `Float32` and 0.078√eps(T) in `Float64`. The defects this catches are of order 1e-2.
slope_step(::Type{T}) where {T} = cbrt(eps(T))
slope_tolerance(::Type{T}) where {T} = 10 * sqrt(eps(T))

"""
    slope_errors(T, ps, F, retraction, αs)

The relative error of `φ'(α)` against a central difference of `φ(α)`, both taken from the
`LinesearchProblem` the optimizer actually builds, at each `α`.
"""
function slope_errors(::Type{T}, ps, F, retraction, αs; h = slope_step(T)) where {T}
    Random.seed!(7)
    opt = Optimizer(ps, F; algorithm = BFGS(), retraction = retraction,
        linesearch = Backtracking(T))
    state = OptimizerState(BFGS(), ps)
    c = cache(opt)
    initialize!(c, ps)
    update!(c, state, gradient(opt), GeometricOptimizers._direction_rule(opt), ps)

    ls = linesearch_problem(problem(opt), gradient(opt), c, retraction)
    params = (x = ps, state = state)

    map(αs) do α
        φ′ = ls.D(α, params)
        difference = (ls.F(α + h, params) - ls.F(α - h, params)) / (2h)
        abs(φ′ - difference) / abs(difference)
    end
end

@testset "φ' is the derivative of φ, under both retractions, $T" for T in REAL_ELTYPES
    αs = T.((0.0, 0.25, 0.5, 1.0, 2.0))
    c = T(0.3)

    rng = Random.Xoshiro(1234)
    Y = rand(rng, StiefelManifold{T}, 6, 3)
    # a `NamedTuple` mixing a manifold with an ordinary array, so that the pass-through for a
    # Euclidean block is covered too -- its retraction is addition, so `D(α) = B` there
    mixed = NetworkParameters((
        w = rand(rng, StiefelManifold{T}, 6, 3), b = randn(rng, T, 4)))

    for retraction in (Geodesic(), Cayley())
        errors = slope_errors(T, Y, Z -> sum(abs2, Z .- c) + sum(sin.(Z)), retraction, αs)
        @test eltype(errors) == T
        for e in errors
            @test e < slope_tolerance(T)
        end

        objective(p) = sum(abs2, p.w .- c) + sum(sin.(p.w)) + sum(abs2, p.b) + sum(p.b)
        errors = slope_errors(T, mixed, objective, retraction, αs)
        @test eltype(errors) == T
        for e in errors
            @test e < slope_tolerance(T)
        end
    end
end

# The Grassmann branch of `retraction_differential` is checked directly rather than through a merit:
# `D(α)` against a central difference of the retraction it is the derivative of. This is the only
# place the identity is checked against `retract` rather than against `f ∘ retract`, which is what
# makes it independent of `global_rep` and `_dot`.
#
# `test/integration/grassmann_optimizer_tests.jl` covers the end-to-end solve; the direct check stays
# for the independence above.
struct UncoveredRetraction <: GeometricOptimizers.AbstractRetraction end

@testset "retraction_differential is d/dα retract(αB), for both lifts, $T" for T in REAL_ELTYPES
    for LT in (StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix)
        B = rand(Random.Xoshiro(1234), LT{T}, 6, 3)
        E = StiefelProjection(B)
        frame(retr, α) = Matrix(retraction(retr, α * B))

        for retr in (Geodesic(), Cayley()), α in T.((0.25, 0.5, 1.0, 2.0))

            h = slope_step(T)
            difference = (frame(retr, α + h) * E - frame(retr, α - h) * E) / (2h)
            velocity = frame(retr, α) * Matrix(retraction_differential(retr, B, α)) * E

            @test eltype(velocity) == T
            @test norm(velocity - difference) / norm(difference) < slope_tolerance(T)
        end

        # `α = 0` returns the direction untouched, which is what keeps `Backtracking` free
        @test retraction_differential(Cayley(), B, zero(T)) === B

        # A retraction with no differential of its own has to say so here rather than fail with a
        # `MethodError` from inside a merit evaluation, three frames down. Same reason `retraction`
        # carries an explicit error for the combinations it does not cover.
        @test_throws ErrorException retraction_differential(UncoveredRetraction(), B, one(T))
    end
end

@testset "a searching line search runs on a Manifold at all, $T" for T in REAL_ELTYPES
    # every one of these threw `Not implemented for StiefelManifold{...}` from
    # `SimpleSolvers.compute_new_iterate!` before. All four searching methods this package exports are
    # covered, not just the two the rest of the file uses.
    searching = (
        Backtracking(T), Backtracking(T; expand = true), Bisection(T),
        Quadratic(T), BierlaireQuadratic(T))
    # All four exponential algorithms are put through a real solve here, not just through
    # `geodesic` in isolation: they have to agree on where the optimizer ends up, not merely on the
    # value of one retraction. `TaylorSeries` is the one left out, and deliberately — it is not a
    # retraction at a large lift and this asserts convergence.
    retractions = (
        Geodesic(ScaledSquaring()), Geodesic(NativePade()), Geodesic(AugmentedPade()),
        Geodesic(ProjectedSkew()), Cayley())
    for linesearch in searching, retraction in retractions

        x = x₀(T)
        opt = Optimizer(x, f; algorithm = GradientMethod(),
            linesearch = linesearch, retraction = retraction)

        result = solve!(x, OptimizerState(GradientMethod(), x), opt)

        @test x isa StiefelManifold{T}             # the type survives ...
        @test eltype(x) == T
        @test check(x) < manifold_tolerance(T)     # ... and so does the manifold
        @test isconverged(status(result))
        @test isapprox(x, minimizer(T); atol = sphere_tolerance(T))
    end
end

@testset "a searching line search converges where a fixed step only crawls, $T" for T in REAL_ELTYPES
    # `Static(0.1)` needs 27 iterations here in `Float64` and stops just under the gradient gate;
    # `Bisection` solves this one-dimensional problem essentially exactly, in one.
    results = map((Static(T(0.1)), Backtracking(T), Bisection(T))) do linesearch
        x = x₀(T)
        state = OptimizerState(GradientMethod(), x)
        opt = Optimizer(x, f; algorithm = GradientMethod(),
            linesearch = linesearch, retraction = Geodesic())
        result = solve!(x, state, opt)
        (its = iteration_number(state), g = status(result).rg,
            g_converged = status(result).g_converged, converged = isconverged(status(result)),
            x = x)
    end

    static, backtracking, bisection = results

    @test all(r -> eltype(r.x) == T, results)
    # all three terminate on a criterion, none on `max_iterations`
    @test all(r -> r.its < 1000, results)
    @test all(r -> r.converged, results)
    @test all(r -> isapprox(r.x, minimizer(T); atol = sphere_tolerance(T)), results)

    # `Bisection` is an exact line search on this one-dimensional problem, so it lands in a couple of
    # iterations and drives the gradient far below what the fixed step reaches
    @test bisection.its < static.its
    @test bisection.g < static.g

    # `Backtracking` accepts α = 1 on every step of this problem, so it behaves like `Static(1.0)`.
    # The point is that it runs at all, and that it still meets the gradient criterion rather than the
    # iteration cap.
    #
    # A `Float32` branch, and the assertion it marks is still made, as broken. In `Float32` the solve
    # stops after 5 iterations on the change of `x` and `f`, at a gradient of 6.4e-4 = 1.9√eps(Float32),
    # above the gate `f_reltol = √eps(T)` that `g_converged` compares it with
    # (`convergence_measures`, `src/optimizers/optimizer_status.jl`): at the round-off floor of `f`
    # the gradient of this objective is of the order of `√eps(T)` itself, so the gate is reached in
    # `Float64` (0.72√eps(Float64)) by a margin that `Float32` does not have.
    if T === Float32
        @test_broken backtracking.g_converged  # issue #155, K29 in KNOWN_ISSUES.md
    else
        @test backtracking.g_converged
    end
end

@testset "the quasi-Newton methods converge on a manifold NamedTuple, $T" for T in REAL_ELTYPES
    # This is the SVD problem of `verification/svd_optim.jl`, which no algorithm could
    # converge before: with `Static(0.01)` the three first-order methods exhaust 1000 iterations at
    # a relative error of 1e-2 and a gradient of 8e-2, seven orders of magnitude off the gate.
    # `BFGS` needs a searching line search, so it could not be used on a manifold at all.
    #
    # The optimum comes from an SVD, which does not go through the optimizer: `w₁w₂ᵀA` has rank at
    # most `n`, so `err` is at least the error of the best rank-`n` approximation of `A`, which
    # `w₁ = w₂ = Uₙ` attains. `A` is invertible, so `w₁w₂ᵀ = UₙUₙᵀ` at every minimiser, and the
    # distance is taken between those projectors. The SVD is in `Float64`, so the reference carries
    # no error of the precision under test.
    rng = Random.Xoshiro(1234)
    A = rand(rng, T, 10, 10)
    n = 3
    err(ps::NetworkParameters) = norm(A - ps.w₁ * ps.w₂' * A)
    U = svd(Float64.(A)).U[:, 1:n]
    projector = U * U'

    # The distance tolerance is `300√eps(T)` and not the sphere's: `σ₃/σ₄ = 1.2` for this `A`, and a
    # small gap between the singular values that the minimiser separates makes the minimiser badly
    # conditioned. Measured over three seeds of `A` and of the start: 155√eps(T) in `Float32`,
    # 100√eps(T) in `Float64`. The gradient tolerance is `30√eps(T)`, against a worst of 10√eps(T)
    # in `Float64` and 5.9√eps(T) in `Float32`.
    for linesearch in (Backtracking(T), Bisection(T))
        start = Random.Xoshiro(1234)
        ps = NetworkParameters((
            w₁ = rand(start, StiefelManifold{T}, 10, n),
            w₂ = rand(start, StiefelManifold{T}, 10, n)))
        Random.seed!(1234)
        state = OptimizerState(BFGS(), ps)
        opt = Optimizer(
            ps, err; algorithm = BFGS(), linesearch = linesearch, retraction = Cayley())

        result = solve!(ps, state, opt)

        @test eltype(ps.w₁) == T
        @test iteration_number(state) < 1000                    # it terminates on a criterion ...
        @test isconverged(status(result))
        @test status(result).rg < 30 * sqrt(eps(T))             # ... with a small gradient
        @test norm(ps.w₁ * ps.w₂' - projector) < 300 * sqrt(eps(T))   # ... at the minimiser
        for Y in values(ps)
            @test check(Y) < manifold_tolerance(T)              # and still on the manifold
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
target₂(::Type{T}) where {T} = T[0, 1.5, 0]
minimizer₂(::Type{T}) where {T} = StiefelManifold(T[0; 1; 0;;])
function two_spheres(ps::NetworkParameters)
    T = eltype(ps.w₁)
    l2norm(vec(ps.w₁), target(T)) + l2norm(vec(ps.w₂), target₂(T))
end

function ps₀(::Type{T}) where {T}
    Random.seed!(1234)
    NetworkParameters((w₁ = StiefelManifold(T[0; sqrt(T(0.5)); sqrt(T(0.5));;]),
        w₂ = StiefelManifold(T[sqrt(T(0.5)); 0; sqrt(T(0.5));;])))
end

function nt_linesearches(::Type{T}) where {T}
    (
        Static(T(0.1)), Backtracking(T), Backtracking(T; expand = true),
        Bisection(T), Quadratic(T), BierlaireQuadratic(T), StrongWolfe(T; c₂ = T(0.1)))
end

@testset "the first-order methods solve on a manifold NamedTuple, on every line search, $T" for T in REAL_ELTYPES
    for method in (GradientMethod(), MomentumMethod(; α = T(0.1))),
        linesearch in nt_linesearches(T),
        retraction in (Geodesic(), Cayley())

        ps = ps₀(T)
        state = OptimizerState(method, ps)
        opt = Optimizer(ps, two_spheres; algorithm = method, linesearch = linesearch,
            retraction = retraction, max_iterations = 1000)

        result = solve!(ps, state, opt)

        @test eltype(ps.w₁) == T
        # 63 is the worst of the 28, and it is a `Static` one: every searching line search here
        # takes at most 19
        @test iteration_number(state) < 100                     # terminates on a criterion ...
        @test isconverged(status(result))
        @test isapprox(ps.w₁, minimizer(T); atol = sphere_tolerance(T))   # ... at the minimiser ...
        @test isapprox(ps.w₂, minimizer₂(T); atol = sphere_tolerance(T))
        for Y in values(ps)
            @test check(Y) < manifold_tolerance(T)              # ... and still on the manifold
        end
    end
end

@testset "Adam runs on a manifold NamedTuple under a searching line search, $T" for T in REAL_ELTYPES
    # `Adam` is in this file's coverage but not in the loop above, and the reason is the one the
    # `DecayingStatic` testset states: its direction has magnitude ≈1 per component whatever the
    # gradient is, so with a step that does not shrink it circles the minimiser at that distance. It
    # needs 251-331 iterations here where the two methods above need 8-60, which is why
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
    #
    # The distance tolerance is now `sphere_tolerance(T)`, 1.5e-7 in `Float64`. On this seed of the
    # global RNG every pair converges. With `Random.seed!(1)` or `(2)` instead, `Adam` + `Static`
    # runs out the 1000 iterations in `Float64` under both retractions, at 2.3e-3 to 4.5e-3 from
    # the minimiser; `Float32` converges on all three seeds.
    for linesearch in nt_linesearches(T), retraction in (Geodesic(), Cayley())

        ps = ps₀(T)
        state = OptimizerState(Adam(), ps)
        opt = Optimizer(
            ps, two_spheres; algorithm = Adam(), linesearch = linesearch,
            retraction = retraction, max_iterations = 1000, warn_iterations = 0)

        result = solve!(ps, state, opt)

        @test eltype(ps.w₁) == T
        @test iteration_number(state) < 1000                    # terminates on a criterion ...
        @test isconverged(status(result))
        @test isapprox(ps.w₁, minimizer(T); atol = sphere_tolerance(T))   # ... at the minimiser
        @test isapprox(ps.w₂, minimizer₂(T); atol = sphere_tolerance(T))
        for Y in values(ps)
            @test check(Y) < manifold_tolerance(T)
        end
    end
end

@testset "the reused gradient is the one a fresh evaluation would give, $T" for T in REAL_ELTYPES
    # `solver_step!` refreshes `latest_gradient` at the accepted iterate and the next
    # `update!(cache, ...)` reuses it rather than evaluating `∇f` again at the same point; see
    # `store_gradient!`. The manifold case is the one where that could go wrong quietly, because the
    # gradient is expressed in the frame of a `GlobalSection`. The state's frame is the copy of the
    # cache's that `advance_state!` makes, and this asserts, bit for bit, that the reused gradient
    # is the fresh one, rather than taking it on trust.
    for method in (GradientMethod(), MomentumMethod(; α = T(0.1)), Adam()),
        retraction in (Geodesic(), Cayley())

        ps = ps₀(T)
        state = OptimizerState(method, ps)
        opt = Optimizer(
            ps, two_spheres; algorithm = method, linesearch = Bisection(T),
            retraction = retraction)
        grad = GeometricOptimizers.gradient(opt)

        for k in 1:6
            @test GeometricOptimizers.latest_gradient_is_current(GeometricOptimizers.cache(opt), state, ps) ==
                  (k > 1)

            fresh = GeometricOptimizers.global_rep(GeometricOptimizers.section(state), grad(ps))
            increase_iteration_number!(state)
            solver_step!(ps, state, opt)
            update!(state, opt, ps)

            stored = GeometricOptimizers.gradient_array(GeometricOptimizers.cache(opt))
            @test eltype(stored.w₁) == T
            for key in keys(fresh)
                @test stored[key] == fresh[key]
            end
        end
    end
end

@testset "DecayingStatic decays the step geometrically, $T" for T in REAL_ELTYPES
    η₁, η₂ = T(1.0e-2), T(1.0e-6)
    ls = DecayingStatic(; η₁, η₂, n = 1000)

    @test eltype(ls) == T
    @test eltype(step_size(ls, 500)) == T
    @test step_size(ls, 0) ≈ η₁                     # starts at η₁ ...
    @test step_size(ls, 1000) ≈ η₂                  # ... reaches η₂ at the horizon ...
    @test step_size(ls, 2000) < η₂                  # ... and keeps going, which is what converges
    @test step_size(ls, 500) ≈ sqrt(η₁ * η₂)        # geometric, so the midpoint is the geometric mean

    @test_throws AssertionError DecayingStatic(; η₁ = η₂, η₂ = η₁)   # η₂ ≤ η₁
    @test_throws AssertionError DecayingStatic(; η₁ = -one(T))
    @test_throws AssertionError DecayingStatic(; η₁, η₂, n = 0)
end

# the default, without an element type to take it from
@test eltype(DecayingStatic()) == Float64

@testset "DecayingStatic drives the step of a solve to zero, $T" for T in REAL_ELTYPES
    # `Adam`'s direction has magnitude ≈1 per component whatever the gradient is, so with a constant
    # step it circles the minimizer at that distance and never terminates on a criterion. This is
    # the `Float64`/`Cayley` case that used to run out its 1000 iterations.
    x = x₀(T)
    state = OptimizerState(Adam(), x)
    opt = Optimizer(x, f; algorithm = Adam(), retraction = Cayley(),
        linesearch = DecayingStatic(; η₁ = T(0.1), η₂ = T(1.0e-8), n = 400))

    result = solve!(x, state, opt)

    @test eltype(x) == T
    @test iteration_number(state) < 1000            # terminates on a criterion rather than the cap
    @test isconverged(status(result))
    # The step really has gone to zero: below `√eps(T)`, the smallest change of `x` that the change of
    # an objective quadratic at its minimiser resolves. `Float32` stops on that after about 92
    # iterations, at a step of 8.6e-5 = 0.25√eps(Float32); `Float64` runs the schedule down to 9.6e-14.
    # A constant step stays at the order of `η₁`.
    @test status(result).rxₐ < sqrt(eps(T))
    # The one tolerance of this file with a part that is not in `T`. In `Float64` where the run ends is
    # set by the schedule, which shrinks the step faster than `Adam`'s moving average closes on the
    # minimiser: 1.9e-4 to 2.3e-4 over three seeds of the global RNG, far above `√eps(Float64)`, so
    # that part is the `1e-3` this test always had. In `Float32` the run stops on the change of `f`
    # first, at the `√eps(T)` scale of every other solve here: 0.14 to 3.1√eps(Float32) over four
    # seeds, 3.1 on this one, hence `10√eps(T)`.
    @test isapprox(x, minimizer(T); atol = max(T(1.0e-3), 10 * sqrt(eps(T))))
end

@testset "the quasi-Newton methods run on a bare Manifold, $T" for T in REAL_ELTYPES
    # `Q` is sized by the *intrinsic* dimension -- the length of the flattening, 2 for `St(3, 1)` --
    # while the gradient and the direction are horizontal lifts of the ambient shape, `3 × 3`. Four
    # methods that the `NamedTuple` case had and the bare case did not (`outer!`, the product with
    # `Q`, `alloc_h` and `_copyto!` for a section) sat on that boundary; without them `BFGS` on a bare `Manifold`
    # died in `outer!` with `AssertionError: axes(O, 1) == axes(x, 1)`.
    for algorithm in (BFGS(), DFP()),
        linesearch in (Backtracking(T), Bisection(T))

        x = x₀(T)
        state = OptimizerState(algorithm, x)
        opt = Optimizer(x, f; algorithm = algorithm, linesearch = linesearch)

        result = solve!(x, state, opt)

        @test x isa StiefelManifold{T}
        @test eltype(x) == T
        @test check(x) < manifold_tolerance(T)
        @test iteration_number(state) < 100          # 1 with `Bisection`, 4 or 5 with `Backtracking`
        @test isconverged(status(result))
        # below the gradient gate `√eps(T)` by a factor of two; the worst measured over three seeds of
        # the global RNG is 0.043√eps(T), in both precisions
        @test status(result).rg < sqrt(eps(T)) / 2
        @test isapprox(x, minimizer(T); atol = sphere_tolerance(T))
    end
end

# The step ceiling of issue A1b. A line search bounds its step by the merit, and on a compact manifold
# `φ` is bounded, so that test never fires -- `Quadratic` returned `α = 4.3e7` on a direction of norm
# 5.54 and reported a genuine decrease. The bound that does exist is geometric (`2π` for a rotation,
# over `‖δ‖`) and changes at every step, which is why SimpleSolvers 0.12 takes it per call through
# `params.αmax` and leaves the value to this package. See `DEFAULT_STEP_CEILING`.

@testset "step_αmax is c⋅2π/‖δ‖, and Inf where there is no scale, $T" for T in REAL_ELTYPES
    # `[3, 4]` is kept integer-valued for the exact identity `‖[3, 4]‖ = 5`; the ceiling `c = 1.3` is
    # not a power of two
    δ = T[3, 4]
    c = T(1.3)
    @test eltype(step_αmax(c, δ)) == T              # in `T`, so that a `Float32` solve does not silently widen
    @test step_αmax(c, δ) ≈ c * 2 * T(π) / 5
    @test step_αmax(2c, δ) ≈ 2c * 2 * T(π) / 5

    # `Inf` is what `SimpleSolvers.linesearch_αmax` reads as "the caller has no scale of its own", and
    # it leaves the method's own ceiling standing. These three all have to produce it rather than a
    # `NaN` or a non-positive value, which upstream rejects with an `ArgumentError` -- correctly, since
    # ignoring one would hand back exactly the unbounded step the ceiling exists to rule out.
    @test step_αmax(c, zeros(T, 2)) == Inf          # a vanishing direction
    @test step_αmax(c, T[NaN, 1]) == Inf            # a direction that has already gone wrong
    @test step_αmax(c, T[Inf, 1]) == Inf
    @test step_αmax(T(Inf), δ) == Inf               # the ceiling switched off
end

# `_manifold_αmax` is the block-wise half, and issue A15. One `α` is applied to every block of a
# `NamedTuple`, so each manifold block needs `‖αδᵢ‖ ≤ 2πc` and the binding one is the largest `‖δᵢ‖`.
# A block that is an ordinary array contributes nothing: the `2π` is the turn of a rotation, and a
# vector space has no such scale to impose or to inflate.
@testset "the ceiling is derived per block, over the manifold blocks only, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1234)
    lift(scale) = scale * rand(Random.Xoshiro(1234), StiefelLieAlgHorMatrix{T}, 6, 3)
    c = T(1.3)

    # two manifold blocks: the *smallest* per-block ceiling, i.e. the largest direction, and not the
    # quadrature combination of the two that `l2norm` over the whole set would give
    let sol = NetworkParameters((
            a = rand(rng, StiefelManifold{T}, 6, 3), b = rand(rng, StiefelManifold{T}, 6, 3))),
        δ = NetworkParameters((a = lift(one(T)), b = lift(T(3))))

        @test eltype(_manifold_αmax(sol, δ, c)) == T
        @test _manifold_αmax(sol, δ, c) ==
              min(step_αmax(c, δ.a), step_αmax(c, δ.b))
        @test _manifold_αmax(sol, δ, c) == step_αmax(c, δ.b)
        # looser than the quadrature version, and necessarily so: `‖δ‖ ≥ maxᵢ‖δᵢ‖`, so bounding by
        # the total tightened every block by the presence of its neighbours
        @test _manifold_αmax(sol, δ, c) > step_αmax(c, δ)
    end

    # mixed: the Euclidean block neither imposes a ceiling nor tightens the manifold block's. This is
    # the direction of the A15 error -- a Euclidean block of large norm used to drag the whole
    # ceiling down with it through the quadrature norm.
    let sol = NetworkParameters((
            Y = rand(rng, StiefelManifold{T}, 6, 3), W = zeros(T, 3, 4))),
        δ = NetworkParameters((Y = lift(one(T)), W = T(1.0e3) .* randn(rng, T, 3, 4)))

        @test eltype(_manifold_αmax(sol, δ, c)) == T
        @test _manifold_αmax(sol, δ, c) == step_αmax(c, δ.Y)
        @test _manifold_αmax(sol, δ, c) > step_αmax(c, δ)
    end

    # no manifold block at all: no scale exists, so there is no ceiling. `Inf` and not a number:
    # `SimpleSolvers.linesearch_αmax` reads it as "the caller has no scale of its own".
    let sol = NetworkParameters((W = zeros(T, 3, 4), b = zeros(T, 3))),
        δ = NetworkParameters((W = randn(rng, T, 3, 4), b = randn(rng, T, 3)))

        @test eltype(_manifold_αmax(sol, δ, c)) == T     # in `T`, as `step_αmax` is
        @test _manifold_αmax(sol, δ, c) == Inf
    end
end

@testset "linesearch_parameters supplies αmax where a manifold supplies a scale, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1234)
    ceiling = T(DEFAULT_STEP_CEILING)

    # Euclidean: no geometric scale exists and none is needed, since `f(x + αp)` grows with `α` and
    # the search's own decrease test rejects an over-long step unaided. Omitting the field (rather
    # than passing `Inf`) is also what keeps upstream's `hasproperty` guard constant-folded.
    let x = randn(rng, T, 2)
        algorithm = GradientMethod()
        state = OptimizerState(algorithm, x)
        c = OptimizerCache(algorithm, x)
        params = linesearch_parameters(c, x, state, ceiling)
        @test !hasproperty(params, :αmax)
        @test params.x === x && params.state === state
    end

    # Manifold: the ceiling is there, and it is the one `step_αmax` computes from the direction the
    # cache holds at that moment.
    let x = x₀(T)
        algorithm = BFGS()
        state = OptimizerState(algorithm, x)
        opt = Optimizer(x, f; algorithm = algorithm)
        # the same two calls `slope_errors` above makes to get a cache holding a real direction
        initialize!(cache(opt), x)
        update!(
            cache(opt), state, gradient(opt), GeometricOptimizers._direction_rule(opt), x)

        params = linesearch_parameters(cache(opt), x, state, ceiling)
        @test hasproperty(params, :αmax)
        @test eltype(params.αmax) == T
        @test params.αmax == step_αmax(ceiling, direction(cache(opt)))
        @test 0 < params.αmax < Inf
    end

    # A `NamedTuple` of manifolds: block-wise, and finite.
    let ps = ps₀(T)
        algorithm = BFGS()
        state = OptimizerState(algorithm, ps)
        opt = Optimizer(ps, two_spheres; algorithm = algorithm)
        initialize!(cache(opt), ps)
        update!(
            cache(opt), state, gradient(opt), GeometricOptimizers._direction_rule(opt), ps)

        params = linesearch_parameters(cache(opt), ps, state, ceiling)
        @test eltype(params.αmax) == T
        @test params.αmax ==
              _manifold_αmax(ps, direction(cache(opt)), ceiling)
        @test 0 < params.αmax < Inf
    end

    # A set of parameters with *no* manifold block is Euclidean, and telling the two apart is not
    # free: taking the manifold branch here hands the problem a ceiling derived from a rotation it
    # does not have. See the solve below for what that costs.
    let ps = NetworkParameters((W = randn(rng, T, 2, 2), b = randn(rng, T, 2)))
        algorithm = GradientMethod()
        state = OptimizerState(algorithm, ps)
        c = OptimizerCache(algorithm, ps)
        params = linesearch_parameters(c, ps, state, ceiling)
        @test eltype(params.αmax) == T
        @test params.αmax == Inf     # equivalent to the omission in the `AbstractVector` branch
    end
end

# The regression test for that last case. `‖αδ‖ ≤ 2π` on a problem whose optimum is 10⁴ away is a
# step-size cap of `6e-4`, so the solve crawls where it should converge outright -- measured at 3 184
# iterations against 1 before the ceiling became block-wise. The assertion is *relative*, between the
# ceiling and no ceiling on the same problem, so it pins no figure of its own.
@testset "a parameter set with no manifold block is not bounded by a manifold's geometry, $T" for T in REAL_ELTYPES
    goal = T(1.0e4) .* (1 .+ rand(Random.Xoshiro(1234), T, 4))
    far_away(ps::NetworkParameters) = sum(abs2, ps.w .- goal) / 2
    far_away(x::AbstractVector) = sum(abs2, x .- goal) / 2
    ceiling = T(DEFAULT_STEP_CEILING)

    iterations(x, ceiling) =
        let state = OptimizerState(BFGS(), x)
            solve!(x,
                state,
                Optimizer(x, far_away; algorithm = BFGS(), max_iterations = 20_000,
                    warn_iterations = 0, step_ceiling = ceiling))
            iteration_number(state)
        end

    ps = NetworkParameters((w = zeros(T, 4),))
    with_ceiling = iterations(ps, ceiling)
    @test eltype(ps.w) == T
    @test with_ceiling == iterations(NetworkParameters((w = zeros(T, 4),)), T(Inf))
    # ... and the same problem written as a vector, which never had a ceiling, agrees with both
    @test with_ceiling == iterations(zeros(T, 4), ceiling)
end

# Issue B3. A search stopped *at* the ceiling `solver_step!` itself imposed, with the merit still
# falling, is classified by the same round-off rule as any other step and so can come back as
# `LINESEARCH_FLOOR` -- which is a claim about the *direction*, and which `linesearch_rejected`
# answers by throwing `Q` away. What was established is only that no *permitted* step decreases the
# merit measurably, so the ceiling case is exempt and the step is taken.
@testset "a step at the caller's own ceiling is not a rejected direction, $T" for T in REAL_ELTYPES
    αmax = T(2.3)

    @test !linesearch_rejected(LinesearchStatus(αmax, LINESEARCH_FLOOR), αmax)
    @test linesearch_rejected(LinesearchStatus(T(0.9), LINESEARCH_FLOOR), αmax)

    # the other two outcomes are not confusable with a bound step and stay rejections: the budget
    # running out and `φ'(0) ≥ 0` are true of the direction whatever ceiling was in force
    @test linesearch_rejected(LinesearchStatus(αmax, LINESEARCH_EXHAUSTED), αmax)
    @test linesearch_rejected(LinesearchStatus(αmax, LINESEARCH_NO_DESCENT), αmax)

    # a successful search is not a rejection either way
    @test !linesearch_rejected(LinesearchStatus(αmax, LINESEARCH_DECREASED), αmax)

    # with no ceiling -- every Euclidean solve, since `linesearch_parameters` omits the field there
    # and `_caller_αmax` reads that as `Inf` -- the two forms agree on every outcome
    for oc in (LINESEARCH_FLOOR, LINESEARCH_EXHAUSTED, LINESEARCH_NO_DESCENT, LINESEARCH_DECREASED)
        status = LinesearchStatus(T(1.0e9), oc)
        @test linesearch_rejected(status, T(Inf)) == linesearch_rejected(status)
    end

    @test eltype(_caller_αmax(T, (x = 1, state = 2, αmax = T(3.1)))) == T
    @test _caller_αmax(T, (x = 1, state = 2, αmax = T(3.1))) == T(3.1)
    @test _caller_αmax(T, (x = 1, state = 2)) == Inf
end

@testset "the ceiling is a per-Optimizer knob, and Inf switches it off, $T" for T in REAL_ELTYPES
    x = x₀(T)
    @test step_ceiling(Optimizer(x, f; algorithm = BFGS())) == T(DEFAULT_STEP_CEILING)
    @test step_ceiling(Optimizer(x, f; algorithm = BFGS(), step_ceiling = 0.3)) == T(0.3)
    @test step_ceiling(Optimizer(x, f; algorithm = BFGS(), step_ceiling = Inf)) == Inf

    # carried in the element type of the parameters, not in whatever the keyword was written as
    @test eltype(step_ceiling(Optimizer(x, f; algorithm = BFGS(), step_ceiling = 1))) == T
end

# The regression test for A1b itself. `svd_optim.jl` runs the two polynomial searches on seed `1234`
# only, where they always passed; the failure lives on other starting points. This is the smallest
# problem that reproduces the mechanism -- a bounded merit on a compact manifold -- rather than the
# `St(20, 3)²` SVD problem, whose eight-seed sweep is in `scripts/retraction_accuracy.jl`.
@testset "a bounded merit does not produce an unbounded step, $T" for T in REAL_ELTYPES
    for retraction in (Geodesic(), Cayley()),
        linesearch in (Quadratic(T), BierlaireQuadratic(T))

        x = x₀(T)
        state = OptimizerState(BFGS(), x)
        opt = Optimizer(
            x, f; algorithm = BFGS(), linesearch = linesearch, retraction = retraction)

        result = solve!(x, state, opt)

        @test eltype(x) == T
        @test check(x) < manifold_tolerance(T)
        @test isconverged(status(result))
        @test isapprox(x, minimizer(T); atol = sphere_tolerance(T))
    end
end
