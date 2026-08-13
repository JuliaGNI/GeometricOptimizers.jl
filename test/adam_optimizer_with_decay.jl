using GeometricOptimizers
using GeometricOptimizers: Adam, AdamOptimizerWithDecay, AdamWithEuclideanDecay, Cayley,
                           DecayingStatic, StiefelManifold, check, iteration_number, status,
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

    # it adds no schedule of its own -- this is the same object the two arguments would build
    @test o.linesearch.γ == DecayingStatic(; η₁=1.0e-2, η₂=1.0e-6, n=1000).γ
end

@testset "AdamOptimizerWithDecay forwards every argument" begin
    o = AdamOptimizerWithDecay(500; η₁=1.0e-1, η₂=1.0e-4, β₁=8.0e-1, β₂=9.0e-1, δ=1.0e-6)

    @test o.algorithm.β₁ == 8.0e-1                  # β₁, β₂, δ go to Adam ...
    @test o.algorithm.β₂ == 9.0e-1
    @test o.algorithm.δ == 1.0e-6
    @test o.linesearch.η₁ == 1.0e-1                 # ... η₁, η₂, n_epochs to the line search
    @test o.linesearch.η₂ == 1.0e-4
    @test o.linesearch.n == 500

    @test eltype(AdamOptimizerWithDecay(10; T=Float32).linesearch) == Float32
    @test AdamOptimizerWithDecay(10; T=Float32).algorithm isa Adam{Float32}

    # the assertions belong to `DecayingStatic` and have to survive the forwarding
    @test_throws AssertionError AdamOptimizerWithDecay(10; η₁=1.0e-6, η₂=1.0e-2)
    @test_throws AssertionError AdamOptimizerWithDecay(0)
end

@testset "AdamOptimizerWithDecay reproduces GML's schedule" begin
    # `GeometricMachineLearning`'s method of this name stores γ = exp(log(η₂/η₁)/n_epochs) and takes
    # the step η₁·γ^t in iteration t (`src/utils.jl`). This is the claim that lets GML delete it.
    n_epochs, η₁, η₂ = 100, 1.0e-2, 1.0e-6
    γ_gml = exp(log(η₂ / η₁) / n_epochs)
    ls = AdamOptimizerWithDecay(n_epochs; η₁=η₁, η₂=η₂).linesearch

    for t in (0, 1, 7, 50, 99, 100, 250)
        @test step_size(ls, t) ≈ η₁ * γ_gml^t
    end

    # GML's defaults are `Float32` literals ρ₁ = 9f-1, ρ₂ = 9.9f-1, δ = 1f-8, which are Adam's
    o = AdamOptimizerWithDecay(n_epochs; T=Float32)
    @test o.algorithm.β₁ == Float32(9.0e-1)
    @test o.algorithm.β₂ == Float32(9.9e-1)
    @test o.algorithm.δ == Float32(1.0e-8)
end

@testset "AdamOptimizerWithDecay splats into Optimizer and converges" begin
    x = x₀()
    state = OptimizerState(Adam(Float64), x)
    opt = Optimizer(x, f; retraction=Cayley(), AdamOptimizerWithDecay(400; η₁=0.1, η₂=1.0e-8)...)

    result = solve!(x, state, opt)

    @test iteration_number(state) < 1000            # the decaying step terminates on a criterion
    @test status(result).rxₐ < 1e-10
    @test isapprox(x, MINIMIZER; atol=1e-3)
    @test check(x) < 1e-12                          # and stays on the manifold
end

@testset "learning-rate decay is not weight decay" begin
    # The two decays share a word and nothing else: this one leaves the weights alone and the other
    # leaves the learning rate alone. See `docs/src/weight_decay.md`.
    @test AdamOptimizerWithDecay(100).algorithm isa Adam
    @test !(AdamOptimizerWithDecay(100).algorithm isa AdamWithEuclideanDecay)
    @test !hasproperty(AdamOptimizerWithDecay(100).algorithm, :λ)

    # `AdamWithEuclideanDecay` has a fixed learning rate, i.e. no schedule at all
    @test GeometricOptimizers.default_linesearch(Float64, AdamWithEuclideanDecay()) isa Static

    # and they compose: the decayed schedule can drive a weight-decaying method
    o = Optimizer(x₀(), f; algorithm=AdamWithEuclideanDecay(Float64; λ=0.0),
        linesearch=AdamOptimizerWithDecay(400).linesearch)
    @test method(GeometricOptimizers.linesearch(o)) isa DecayingStatic
end
