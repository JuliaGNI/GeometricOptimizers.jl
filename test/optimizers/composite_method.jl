using GeometricOptimizers
using GeometricOptimizers: CompositeCache, FirstOrderMethod, LeafTypeSelector, check,
                           default_step_size, leafmethod,
                           iteration_number, DEFAULT_LEARNING_RATE
using Test
import Random

# A `CompositeMethod` on a parameter set gives every leaf the cache and the state of the method it
# selects for that leaf, and steps every leaf with them. What is pinned here is that the selection is
# right, that each leaf then moves exactly as it would under a `TrainingOptimizer` of its own, and that
# what the composite cannot do is refused by name rather than by a `MethodError` several frames in.
#
# The seed is fixed for the reason given in `test/manifold_optimizers/scalar_moment_adam_optimizer.jl`:
# every point below is drawn at random, and an unseeded file is a different test on every run.
Random.seed!(1234)

# A Stiefel weight beside an ordinary layer, nested one level as a network's parameters are. Drawn
# from a fixed seed, so that two calls give two copies of one point with the same section completions.
function mixed_parameters(::Type{T} = Float64) where {T}
    Random.seed!(7)
    NetworkParameters((L1 = (weight = rand(StiefelManifold{T}, 6, 2),),
        L2 = (W = rand(T, 3, 2), b = rand(T, 3))))
end

function minibatch_gradient(::Type{T}, k) where {T}
    Random.seed!(100 + k)
    NetworkParameters((
        L1 = (weight = randn(T, 6, 2),), L2 = (W = randn(T, 3, 2), b = randn(T, 3))))
end

@testset "the leaf-type selector" begin
    stiefel = ScalarMomentAdam()
    euclidean = Adam()
    method = CompositeMethod(; manifold = stiefel, array = euclidean)

    @test method isa FirstOrderMethod
    @test method.select isa LeafTypeSelector
    @test method.select.manifold === stiefel
    @test method.select.array === euclidean

    @test leafmethod(method, rand(StiefelManifold{Float32}, 4, 2)) === stiefel
    @test leafmethod(method, rand(GrassmannManifold{Float32}, 4, 2)) === stiefel
    @test leafmethod(method, rand(Float32, 3)) === euclidean
    @test leafmethod(method, rand(Float32, 3, 2)) === euclidean

    # The identity arm, for an ordinary method.
    @test leafmethod(euclidean, rand(3)) === euclidean
    @test leafmethod(GradientMethod(), rand(3)) === GradientMethod()

    # An arbitrary selector, which is the general form the keyword constructor is sugar for.
    by_size = CompositeMethod(x -> length(x) > 4 ? euclidean : MomentumMethod(; α = 0.5))
    @test leafmethod(by_size, rand(Float32, 6)) === euclidean
    @test leafmethod(by_size, rand(Float32, 2)) == MomentumMethod(; α = 0.5)
end

@testset "what a composite cannot select or be asked is refused by name" begin
    # `GradientMethod` is a first-order method like the others and may be selected.
    @test leafmethod(CompositeMethod(_ -> GradientMethod()), rand(3)) === GradientMethod()

    # A method that needs a Hessian or a line search, and a composite inside a composite, are not.
    @test_throws ArgumentError leafmethod(CompositeMethod(_ -> BFGS()), rand(3))
    @test_throws ArgumentError leafmethod(CompositeMethod(_ -> Newton()), rand(3))
    inner = CompositeMethod(; manifold = Adam(), array = Adam())
    @test_throws ArgumentError leafmethod(CompositeMethod(_ -> inner), rand(3))
    @test_throws TypeError CompositeMethod(; manifold = BFGS(), array = Adam())

    # A parameter set is not a leaf: that it needs two methods is why the composite exists, so no one
    # arm is the answer for it. This used to return the `array` arm for a nested set, silently.
    method = CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam())
    @test_throws ArgumentError leafmethod(method, mixed_parameters())
    @test_throws ArgumentError leafmethod(method, (weight = rand(StiefelManifold, 4, 2),))

    # `solve!` searches one step length for the whole set, which a composite does not have.
    F = ps -> sum(ps.L2.b)
    @test_throws ArgumentError Optimizer(mixed_parameters(), F; algorithm = method)
    @test_throws ArgumentError Optimizer(rand(3), sum; algorithm = method)
end

@testset "one cache and one state per leaf, for the selected method" begin
    ps = mixed_parameters(Float32)
    method = CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam())
    opt = TrainingOptimizer(ps; algorithm = method)

    @test opt.cache isa CompositeCache
    @test opt.state isa CompositeState{Float32}
    @test opt.state.states.L1.weight isa ScalarMomentAdamState{Float32}
    @test opt.state.states.L2.W isa AdamState{Float32}
    @test opt.state.states.L2.b isa AdamState{Float32}
    @test opt.cache.caches.L1.weight isa GeometricOptimizers.ScalarMomentAdamCache{Float32}
    @test opt.cache.caches.L2.W isa GeometricOptimizers.AdamCache{Float32}

    # Each arm is converted to the element type of the set, which `change_precision` on the
    # composite as a whole could not have done for a selector function.
    @test opt.cache.methods.L1.weight isa ScalarMomentAdam{Float32}
    @test opt.cache.methods.L2.W isa Adam{Float32}
    by_type = CompositeMethod(x -> x isa Manifold ? ScalarMomentAdam() :
                                   MomentumMethod(; α = 0.5))
    opt = TrainingOptimizer(
        mixed_parameters(Float32); algorithm = by_type, linesearch = 1.0e-2)
    @test opt.cache.methods.L2.b isa MomentumMethod{Float32}
    @test opt.state.states.L2.b isa MomentumState{Float32}

    # On a single leaf the composite is the method it selects for it.
    Y = rand(StiefelManifold{Float32}, 4, 2)
    @test TrainingOptimizer(Y; algorithm = method).method isa ScalarMomentAdam{Float32}
    @test TrainingOptimizer(rand(Float32, 3); algorithm = method).method isa Adam{Float32}
end

@testset "a composite of one method is that method over the whole set, exactly" begin
    # Nothing is pooled across leaves, and `Adam` pools nothing either, so `Adam` on every leaf has to
    # reproduce `Adam` on the set to the bit -- the random section completions included, which the
    # composite draws in the same order: every cache, then every state. Each optimizer is built
    # straight after its parameters, so that both draw them from the same position of the seed.
    whole = mixed_parameters()
    whole_opt = TrainingOptimizer(whole; algorithm = Adam())
    per_leaf = mixed_parameters()
    per_leaf_opt = TrainingOptimizer(per_leaf;
        algorithm = CompositeMethod(; manifold = Adam(), array = Adam()))
    for k in 1:5
        optimization_step!(whole, whole_opt, minibatch_gradient(Float64, k))
        optimization_step!(per_leaf, per_leaf_opt, minibatch_gradient(Float64, k))
    end
    @test per_leaf.L1.weight.A == whole.L1.weight.A
    @test per_leaf.L2.W == whole.L2.W
    @test per_leaf.L2.b == whole.L2.b
    @test iteration_number(per_leaf_opt.state) == 5
    @test iteration_number(per_leaf_opt.state.states.L2.b) == 5
end

@testset "each leaf moves as under a TrainingOptimizer of its own" begin
    ps = mixed_parameters()
    opt = TrainingOptimizer(ps;
        algorithm = CompositeMethod(;
            manifold = ScalarMomentAdam(), array = GradientMethod()),
        linesearch = 1.0e-2)

    # The same leaves, drawn from the same seed, each with an optimizer of its own.
    alone = mixed_parameters()
    Y, W, b = alone.L1.weight, alone.L2.W, alone.L2.b
    Y_opt = TrainingOptimizer(Y; algorithm = ScalarMomentAdam(), linesearch = 1.0e-2)
    b_opt = TrainingOptimizer(b; algorithm = GradientMethod(), linesearch = 1.0e-2)

    for k in 1:5
        dp = minibatch_gradient(Float64, k)
        optimization_step!(ps, opt, dp)
        optimization_step!(Y, Y_opt, dp.L1.weight)
        optimization_step!(b, b_opt, dp.L2.b)
        W .-= 1.0e-2 .* dp.L2.W
    end
    @test ps.L1.weight.A == Y.A
    @test ps.L2.b == b
    @test ps.L2.W ≈ W
    @test check(ps.L1.weight) ≤ 1.0e-10

    # The scalar second moment is the Stiefel leaf's own, and not one pooled over the set.
    @test opt.state.states.L1.weight.m₂ == Y_opt.state.m₂
end

@testset "the step size is the composite's, read once per step" begin
    method = CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam())
    @test default_step_size(method) == DEFAULT_LEARNING_RATE

    # One schedule steps every leaf, so a composite whose methods disagree on a default has none.
    @test_throws ArgumentError default_step_size(
        CompositeMethod(; manifold = ScalarMomentAdam(), array = GradientMethod()))
    @test_throws ArgumentError TrainingOptimizer(mixed_parameters();
        algorithm = CompositeMethod(;
            manifold = ScalarMomentAdam(), array = GradientMethod()))
    @test_throws ArgumentError default_step_size(CompositeMethod(_ -> Adam()))

    # A schedule is read at the composite's iteration number, and every leaf takes that step: here
    # the `GradientMethod` leaf moves by exactly `step_size(schedule, t)` times its gradient.
    schedule = DecayingStatic(; η₁ = 0.5, η₂ = 0.05, n = 3)
    ps = mixed_parameters()
    opt = TrainingOptimizer(ps; linesearch = schedule,
        algorithm = CompositeMethod(;
            manifold = ScalarMomentAdam(), array = GradientMethod()))
    for t in 1:3
        dp = minibatch_gradient(Float64, t)
        b = copy(ps.L2.b)
        optimization_step!(ps, opt, dp)
        @test ps.L2.b ≈ b - GeometricOptimizers.step_size(opt.linesearch, t) * dp.L2.b
    end
    @test iteration_number(opt.state) == 3
end

@testset "a refused step leaves the state as it was" begin
    ps = mixed_parameters()
    opt = TrainingOptimizer(
        ps; algorithm = CompositeMethod(; manifold = ScalarMomentAdam(),
            array = Adam()))
    wrong_size = NetworkParameters((L1 = (weight = randn(6, 2),),
        L2 = (W = randn(2, 2), b = randn(3))))
    @test_throws DimensionMismatch optimization_step!(ps, opt, wrong_size)
    @test_throws ArgumentError optimization_step!(ps, opt, minibatch_gradient(Float32, 1))
    @test iteration_number(opt.state) == 0
    @test iteration_number(opt.state.states.L1.weight) == 0
end

@testset "a composite step allocates what its leaves' steps allocate" begin
    # The walk over the leaves is `foreachparameters`, which allocates nothing of its own: the
    # composite of `Adam` on every leaf costs what `Adam` on the set costs.
    whole = mixed_parameters()
    per_leaf = mixed_parameters()
    whole_opt = TrainingOptimizer(whole; algorithm = Adam())
    per_leaf_opt = TrainingOptimizer(per_leaf;
        algorithm = CompositeMethod(; manifold = Adam(), array = Adam()))
    dp = minibatch_gradient(Float64, 1)
    optimization_step!(whole, whole_opt, dp)
    optimization_step!(per_leaf, per_leaf_opt, dp)
    @test (@allocated optimization_step!(per_leaf, per_leaf_opt, dp)) ≤
          (@allocated optimization_step!(whole, whole_opt, dp))
end

@testset "a composite step reports one retraction per leaf to the observer" begin
    ps = mixed_parameters(Float32)
    recorder = EventLog()
    opt = TrainingOptimizer(ps; algorithm = CompositeMethod(; manifold = ScalarMomentAdam(),
            array = Adam()), linesearch = 1.0f-3, observer = recorder)
    optimization_step!(ps, opt, minibatch_gradient(Float32, 1))
    @test recorder.events == repeat(
        [(:retraction_application, :enter), (:retraction_application, :exit)], 3)  # weight, W and b
end
