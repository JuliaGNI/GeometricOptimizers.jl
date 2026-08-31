using GeometricOptimizers
using GeometricOptimizers: Adam, AdamOptimizerWithDecay, AdamWithEuclideanDecay, Cayley,
                           DecayingStatic, StiefelManifold, check, default_linesearch,
                           increase_iteration_number!, iteration_number, linesearch, status,
                           step_size
using SimpleSolvers: Static, l2norm, method
using Test
import Random

# `AdamOptimizerWithDecay` is a convenience pairing, not a method: everything below therefore checks
# that it *is* `Adam` plus `DecayingStatic` and nothing else. The schedule itself is tested in
# `manifold_linesearch_tests.jl`; what is new here is the pairing, the argument forwarding, and the
# claim that it reproduces the method of the same name in `GeometricMachineLearning`.

# same sphere problem as `manifold_linesearch_tests.jl`; see there for why the RNG is seeded
const TARGET = [0.0, 0.0, 1.2]
const MINIMIZER = StiefelManifold([0.0; 0.0; 1.0;;])
f(x::StiefelManifold) = l2norm(vec(x), TARGET)
x₀() = (Random.seed!(1234); StiefelManifold([1.0; 0.0; 0.0;;]))

@testset "AdamOptimizerWithDecay pairs Adam with DecayingStatic" begin
    o = AdamOptimizerWithDecay(1000)

    @test keys(o) == (:algorithm, :linesearch)
    @test o.algorithm isa Adam
    @test o.linesearch isa DecayingStatic

    # it adds no schedule of its own -- this is the same object the two arguments would build, all
    # four fields of it, and `Adam` likewise gets nothing but its own defaults
    @test o.linesearch == DecayingStatic(; η₁ = 1.0e-2, η₂ = 1.0e-6, n = 1000)
    @test o.algorithm == Adam(Float64)
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

    # `T` is positional, as it is for `Adam` and `DecayingStatic`, and reaches both halves
    @test eltype(AdamOptimizerWithDecay(10, Float32).linesearch) == Float32
    @test AdamOptimizerWithDecay(10, Float32).algorithm isa Adam{Float32}

    # `γ` is computed in `T` and not in `Float64` and then rounded, so the pairing is the same object
    # `DecayingStatic(T; …)` builds in `Float32` too, not merely the same to within an ulp
    @test AdamOptimizerWithDecay(10, Float32).linesearch === DecayingStatic(Float32; n = 10)

    # everything that is not the schedule goes to `Adam`, so `Adam`'s defaults are not copied here
    # and cannot drift from it -- and a name `Adam` does not know is an error rather than a silent
    # no-op, which is what a call migrated from GML's `ρ₁`/`ρ₂` runs into
    @test_throws MethodError AdamOptimizerWithDecay(10; ρ₁ = 8.0e-1)

    # the assertions belong to `DecayingStatic` and have to survive the forwarding
    @test_throws AssertionError AdamOptimizerWithDecay(10; η₁ = 1.0e-6, η₂ = 1.0e-2)
    @test_throws AssertionError AdamOptimizerWithDecay(0)
end

@testset "AdamOptimizerWithDecay reproduces GML's schedule" begin
    # `GeometricMachineLearning`'s method of this name stored γ = exp(log(η₂/η₁)/n_epochs) and took
    # the step η₁·γ^t in iteration t. This is the claim that let GML delete it, which it did in 0.5;
    # these assertions are what that deletion rests on, so they stay.
    n_epochs, η₁, η₂ = 100, 1.0e-2, 1.0e-6
    γ_gml = exp(log(η₂ / η₁) / n_epochs)
    o = AdamOptimizerWithDecay(n_epochs; η₁ = η₁, η₂ = η₂)

    for t in (0, 1, 7, 50, 99, 100, 250)
        @test step_size(o.linesearch, t) ≈ η₁ * γ_gml^t
    end

    # The formula is only half of the claim: the two also have to agree on *which* `t` the first step
    # uses, and neither of them uses `t = 0`. GML increments `o.step` before `update!`, so its first
    # step is η₁γ¹; this one's is too, because `solve!` calls `increase_iteration_number!` before
    # `solver_step!`. The line below is how `solver_step!` asks for `α`.
    x = x₀()
    state = OptimizerState(o.algorithm, x)
    opt = Optimizer(x, f; retraction = Cayley(), o...)

    for t in 1:3
        increase_iteration_number!(state)
        @test iteration_number(state) == t
        @test solve(linesearch(opt), 1.0, (x = x, state = state)) ≈ η₁ * γ_gml^t
    end

    # GML's defaults are `Float32` literals ρ₁ = 9f-1, ρ₂ = 9.9f-1, δ = 1f-8, which are Adam's --
    # but GML takes `T` from `η₁ = 1f-2` and so defaults to `Float32`, where this defaults to
    # `Float64`; a migrated call has to pass the type
    o₃₂ = AdamOptimizerWithDecay(n_epochs, Float32)
    @test o₃₂.algorithm.β₁ == 9.0f-1
    @test o₃₂.algorithm.β₂ == 9.9f-1
    @test o₃₂.algorithm.δ == 1.0f-8
    @test o₃₂.linesearch.η₁ == 1.0f-2
    @test o₃₂.linesearch.η₂ == 1.0f-6
end

@testset "AdamOptimizerWithDecay splats into Optimizer and converges" begin
    # `manifold_linesearch_tests.jl` already runs this solve with the line search built by hand; what
    # it does not cover, and this does, is that the pairing reaches `Optimizer` through a splat and
    # that `OptimizerState` accepts the `algorithm` half of it.
    o = AdamOptimizerWithDecay(400; η₁ = 0.1, η₂ = 1.0e-8)
    x = x₀()
    state = OptimizerState(o.algorithm, x)
    opt = Optimizer(x, f; retraction = Cayley(), o...)

    result = solve!(x, state, opt)

    @test iteration_number(state) < 1000            # the decaying step terminates on a criterion
    @test status(result).rxₐ < 1e-10
    @test isapprox(x, MINIMIZER; atol = 1e-3)
    @test check(x) < 1e-12                          # and stays on the manifold
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
    o = Optimizer(x₀(), f; algorithm = AdamWithEuclideanDecay(Float64; λ = 0.0),
        linesearch = AdamOptimizerWithDecay(400).linesearch)
    @test method(linesearch(o)) isa DecayingStatic
end
