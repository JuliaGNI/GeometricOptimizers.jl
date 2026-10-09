using GeometricOptimizers
using GeometricOptimizers: Adam, AdamOptimizerWithDecay, AdamWithEuclideanDecay, Cayley,
                           DecayingStatic, StiefelManifold, check, default_linesearch,
                           increase_iteration_number!, iteration_number, linesearch, status,
                           isconverged,
                           step_size
using SimpleSolvers: Static, l2norm, method
using Test
import Random

include("../helpers/eltypes.jl")

# `AdamOptimizerWithDecay` is a convenience pairing, not a method: everything below therefore checks
# that it *is* `Adam` plus `DecayingStatic` and nothing else. The schedule itself is tested in
# `manifold_linesearch_tests.jl`; what is new here is the pairing, the argument forwarding, and the
# claim that it reproduces the method of the same name in `GeometricMachineLearning`.

# same sphere problem as `manifold_linesearch_tests.jl`; see there for why the RNG is seeded. The
# target is converted to the element type of the point, so that a `Float32` objective stays `Float32`.
# The start `e₁` is integer-valued, and the minimizer `e₃` is the exact answer; the iterates between
# them are not exact in either precision.
const TARGET = [0.0, 0.0, 1.2]
minimizer(::Type{T}) where {T} = StiefelManifold(T[0.0; 0.0; 1.0;;])
f(x::StiefelManifold{T}) where {T} = l2norm(vec(x), T.(TARGET))
function x₀(::Type{T} = Float64) where {T}
    Random.seed!(1234)
    StiefelManifold(T[1.0; 0.0; 0.0;;])
end

@testset "AdamOptimizerWithDecay pairs Adam with DecayingStatic" begin
    o = AdamOptimizerWithDecay(1000)

    @test keys(o) == (:algorithm, :linesearch)
    @test o.algorithm isa Adam
    @test o.linesearch isa DecayingStatic

    # it adds no schedule of its own -- this is the same object the two arguments would build, all
    # four fields of it, and `Adam` likewise gets nothing but its own defaults
    @test o.linesearch == DecayingStatic(; η₁ = 1.0e-2, η₂ = 1.0e-6, n = 1000)
    @test o.algorithm == Adam()
end

@testset "AdamOptimizerWithDecay forwards every argument" begin
    o = AdamOptimizerWithDecay(
        500; η₁ = 1.0e-1, η₂ = 1.0e-4, β₁ = 8.0e-1, β₂ = 9.0e-1, δ = 1.0e-6)

    @test o.algorithm.β₁ == 8.0e-1                  # β₁, β₂, δ go to Adam ...
    @test o.algorithm.β₂ == 9.0e-1
    @test o.algorithm.δ == 1.0e-6
    @test o.linesearch.η₁ == 1.0e-1                 # ... η₁, η₂, n_epochs to the line search
    @test o.linesearch.η₂ == 1.0e-4
    @test o.linesearch.n == 500

    # everything that is not the schedule goes to `Adam`, so `Adam`'s defaults are not copied here
    # and cannot drift from it -- and a name `Adam` does not know is an error rather than a silent
    # no-op, which is what a call migrated from GML's `ρ₁`/`ρ₂` runs into
    @test_throws MethodError AdamOptimizerWithDecay(10; ρ₁ = 8.0e-1)

    # the assertions belong to `DecayingStatic` and have to survive the forwarding
    @test_throws AssertionError AdamOptimizerWithDecay(10; η₁ = 1.0e-6, η₂ = 1.0e-2)
    @test_throws AssertionError AdamOptimizerWithDecay(0)
end

@testset "change_precision computes γ in the element type, $T" for T in REAL_ELTYPES
    # neither half carries the element type of the parameters; the optimizer converts both, and
    # `change_precision` computes `γ` in `T` rather than rounding the `Float64` one
    o₁₀ = AdamOptimizerWithDecay(10)
    ls = GeometricOptimizers.change_precision(T, o₁₀.linesearch)
    @test ls isa DecayingStatic{T}
    @test eltype(ls.γ) == T
    @test ls.γ === T(exp(log(T(1.0e-6) / T(1.0e-2)) / 10))
end

@testset "AdamOptimizerWithDecay reproduces GML's schedule, $T" for T in REAL_ELTYPES
    # `GeometricMachineLearning`'s method of this name stored γ = exp(log(η₂/η₁)/n_epochs) and took
    # the step η₁·γ^t in iteration t. This is the claim that let GML delete it, which it did in 0.5;
    # these assertions are what that deletion rests on, so they stay.
    n_epochs, η₁, η₂ = 100, 1.0e-2, 1.0e-6
    γ_gml = exp(log(η₂ / η₁) / n_epochs)
    o = AdamOptimizerWithDecay(n_epochs; η₁ = η₁, η₂ = η₂)

    # the pairing itself is built in `Float64` whatever `T` is; `T` enters through the optimizer below
    for t in (0, 1, 7, 50, 99, 100, 250)
        @test step_size(o.linesearch, t) ≈ η₁ * γ_gml^t
    end

    # The formula is only half of the claim: the two also have to agree on *which* `t` the first step
    # uses, and neither of them uses `t = 0`. GML increments `o.step` before `update!`, so its first
    # step is η₁γ¹; this one's is too, because `solve!` calls `increase_iteration_number!` before
    # `solver_step!`. The line below is how `solver_step!` asks for `α`.
    x = x₀(T)
    state = OptimizerState(o.algorithm, x)
    opt = Optimizer(x, f; retraction = Cayley(), o...)

    for t in 1:3
        increase_iteration_number!(state)
        @test iteration_number(state) == t
        α = solve(linesearch(opt), one(T), (x = x, state = state))
        @test eltype(α) == T
        # the default `≈`, whose `rtol` is `√eps(T)`: the schedule is converted once to `T`
        @test α ≈ η₁ * γ_gml^t
    end

    # GML's defaults are `Float32` literals ρ₁ = 9f-1, ρ₂ = 9.9f-1, δ = 1f-8, which are Adam's; the
    # optimizer converts the pairing to the element type of the parameters, to the literal in `T`
    @test opt.algorithm.β₁ === T(9.0e-1)
    @test opt.algorithm.β₂ === T(9.9e-1)
    @test opt.algorithm.δ === T(1.0e-8)
    @test method(linesearch(opt)).η₁ === T(1.0e-2)
    @test method(linesearch(opt)).η₂ === T(1.0e-6)
end

@testset "AdamOptimizerWithDecay splats into Optimizer and converges, $T" for T in REAL_ELTYPES
    # `manifold_linesearch_tests.jl` already runs this solve with the line search built by hand; what
    # it does not cover, and this does, is that the pairing reaches `Optimizer` through a splat and
    # that `OptimizerState` accepts the `algorithm` half of it.
    o = AdamOptimizerWithDecay(400; η₁ = 0.1, η₂ = 1.0e-8)
    x = x₀(T)
    state = OptimizerState(o.algorithm, x)
    opt = Optimizer(x, f; retraction = Cayley(), o...)

    result = solve!(x, state, opt)

    @test eltype(x) == T
    @test isconverged(status(result))
    @test iteration_number(state) < 1000            # the decaying step terminates on a criterion
    # It stops on `f_converged`, once a step changes `f` by under `f_suctol |f| = 2eps(T) |f|`: with
    # `|∇f| ≈ 6d` at a distance `d ≈ 1e-4` and `f ≈ 0.2`, that is a step of a few hundred `eps(T)`.
    # Measured `928eps(T)` in `Float64` and `432eps(T)` in `Float32`.
    @test status(result).rxₐ < 10_000eps(T)
    # In `Float64` the distance at the stop is set by the schedule and not by `T`: once `η` has
    # decayed, Adam's second moment (`β₂ = 0.999`, a memory of about 1000 iterations) still holds the
    # early gradients, so the step shrinks faster than the distance and the solve stops short of the
    # minimizer, at 1.2e-4 (`7900√eps(Float64)`); that part is the `1e-3` this test always had. In
    # `Float32` it is 5.3e-4 (`1.5√eps(Float32)`), hence `10√eps(T)`. The assertion holds at the start
    # of this file and is not a bound over starts: over 20 perturbed starts the distance reaches
    # 1.4e-3 in `Float64` and 1.8e-3 in `Float32`.
    @test isapprox(x, minimizer(T); atol = max(T(1.0e-3), 10 * sqrt(eps(T))))
    # and stays on the manifold: the round-off of a few hundred Cayley steps, measured `9.5eps(T)`
    # in `Float64` and `1.0eps(T)` in `Float32`, and at most `20eps(T)` over 20 nearby starts
    @test check(x) < 100eps(T)
end

@testset "learning-rate decay is not weight decay" begin
    # The two decays share a word and nothing else: this one leaves the weights alone and the other
    # leaves the learning rate alone. See `docs/src/weight_decay.md`.
    @test AdamOptimizerWithDecay(100).algorithm isa Adam
    @test !(AdamOptimizerWithDecay(100).algorithm isa AdamWithEuclideanDecay)
    @test !hasproperty(AdamOptimizerWithDecay(100).algorithm, :λ)

    # `AdamWithEuclideanDecay` has a fixed learning rate, i.e. no schedule at all
    @test default_linesearch(Float64, AdamWithEuclideanDecay()) isa Static

    # and they compose: the decayed schedule can drive a weight-decaying method
    o = Optimizer(x₀(), f; algorithm = AdamWithEuclideanDecay(; λ = 0.0),
        linesearch = AdamOptimizerWithDecay(400).linesearch)
    @test method(linesearch(o)) isa DecayingStatic
end
