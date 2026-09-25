using GeometricOptimizers
using GeometricOptimizers: cache, default_linesearch, direction, first_moment,
                           second_moment, gradient_array, global_rep,
                           increase_iteration_number!, momentum, section, solution,
                           solver_step!, update!, DEFAULT_LEARNING_RATE,
                           FirstOrderMethodWithState
using LinearAlgebra: norm
using SimpleSolvers: Static
using Test
import Random

# `CompositeMethod` is a *choice* of method per leaf and deliberately not a second implementation of
# anything. So what is pinned here is that it chooses correctly, that every forwarded call is
# bit-for-bit the call the chosen method would have received, and that a selection outside the scope
# the composite step supports is rejected by name rather than by a `MethodError` several frames in.
#
# The seed is fixed for the reason given in `test/scalar_moment_adam.jl`: every point below is drawn
# at random, and an unseeded file is a different test on every run.
Random.seed!(1234)

linear_stiefel_objective(C) = Y -> sum(C .* Y.A)

@testset "the leaf-type selector" begin
    stiefel = ScalarMomentAdam(Float32)
    euclidean = Adam(Float32)
    method = CompositeMethod(; manifold = stiefel, array = euclidean)

    @test method isa CompositeMethod
    @test method.select isa LeafTypeSelector
    @test method.select.manifold === stiefel
    @test method.select.array === euclidean

    Y = rand(StiefelManifold{Float32}, 4, 2)
    @test leafmethod(method, Y) === stiefel
    @test leafmethod(method, rand(GrassmannManifold{Float32}, 4, 2)) === stiefel
    @test leafmethod(method, rand(Float32, 3)) === euclidean
    @test leafmethod(method, rand(Float32, 3, 2)) === euclidean

    # A group of weights -- the shape a host package hands over when it steps a network one layer at
    # a time -- is answered for as a whole, because one cache is what it gets.
    @test leafmethod(method, (weight = Y,)) === stiefel
    @test leafmethod(method, NetworkParameters((weight = Y,))) === stiefel
    @test leafmethod(method, (W = rand(Float32, 3, 2), b = rand(Float32, 3))) === euclidean
    @test leafmethod(method, NetworkParameters((w = rand(Float32, 3, 2),))) === euclidean

    # A group that mixes the two has no answer: that it needs two methods is the whole reason a
    # composite exists, so it has to be split before it is given one cache.
    @test_throws ArgumentError leafmethod(method, (weight = Y, b = rand(Float32, 2)))
    @test_throws ArgumentError leafmethod(method, NamedTuple())

    # The identity arm: a caller that asks this unconditionally, once per leaf, needs no test of its
    # own for whether a composite is in play.
    @test leafmethod(euclidean, Y) === euclidean
    @test leafmethod(GradientMethod(), Y) === GradientMethod()

    # An arbitrary selector, which is the general form the keyword constructor is sugar for.
    by_size = CompositeMethod(x -> length(x) > 4 ? euclidean : MomentumMethod(0.5f0))
    @test leafmethod(by_size, rand(Float32, 6)) === euclidean
    @test leafmethod(by_size, rand(Float32, 2)) == MomentumMethod(0.5f0)
end

@testset "a selection outside the composite step's scope is rejected" begin
    # `solver_step!` hands `update!` the method rather than a Hessian for everything in
    # `FirstOrderMethodWithState`, and a composite is in that union so that it can forward. A
    # selector returning a method from the *other* branch would reach an `update!` that does not
    # exist, several frames below the mistake.
    @test CompositeMethod(; manifold = Adam(), array = Adam()) isa FirstOrderMethodWithState
    quasi_newton = CompositeMethod(_ -> BFGS())
    @test_throws ArgumentError leafmethod(quasi_newton, rand(3))
    @test_throws ArgumentError leafmethod(CompositeMethod(_ -> GradientMethod()), rand(3))
end

@testset "cache, state and Hessian forward to the selected method" begin
    method = CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam())
    Y = rand(StiefelManifold, 4, 2)
    v = rand(3)

    @test GeometricOptimizers.OptimizerCache(method, Y) isa
          GeometricOptimizers.ScalarMomentAdamCache
    @test GeometricOptimizers.OptimizerCache(method, v) isa GeometricOptimizers.AdamCache
    @test OptimizerState(method, Y) isa ScalarMomentAdamState
    @test OptimizerState(method, v) isa AdamState

    # The gradient-supplying form of `OptimizerState`, which both selected methods have.
    Ḡ = global_rep(GlobalSection(Y), rgrad(Y, randn(4, 2)))
    @test OptimizerState(method, Y, Ḡ) isa ScalarMomentAdamState

    # An unsupported leaf is still the selected method's error and not a composite one: the
    # composite adds no scope of its own.
    @test_throws ArgumentError GeometricOptimizers.OptimizerCache(
        CompositeMethod(_ -> ScalarMomentAdam()), rand(3))

    @test default_linesearch(Float64, method) isa Static
    @test default_linesearch(Float64, method).α == DEFAULT_LEARNING_RATE
end

@testset "a composite step is the selected method's step, exactly" begin
    # Two `Optimizer`s over the same point and the same objective, one built with the method and one
    # with a composite that selects it. Every number either produces has to agree exactly: a
    # composite that is only a choice cannot move an iterate to a different place.
    C = randn(4, 2)
    F = linear_stiefel_objective(C)

    function run(algorithm, steps)
        Random.seed!(99)
        Y = rand(StiefelManifold, 4, 2)
        state = OptimizerState(algorithm, Y)
        optimizer = Optimizer(Y, F; algorithm = algorithm, linesearch = Static(1.0e-2))
        for _ in 1:steps
            increase_iteration_number!(state)
            solver_step!(Y, state, optimizer)
            update!(state, optimizer, Y)
        end
        (Y = Matrix(Y.A), m₂ = second_moment(state), drift = check(Y))
    end

    method = ScalarMomentAdam()
    composite = CompositeMethod(; manifold = method, array = Adam())
    plain_run = run(method, 3)
    composite_run = run(composite, 3)
    @test plain_run.Y == composite_run.Y
    @test plain_run.m₂ == composite_run.m₂
    @test plain_run.drift == composite_run.drift
    @test composite_run.drift ≤ 1.0e-10
end

@testset "sync_state! carries what a method carries" begin
    # The hook a training loop that drives a cache directly needs: without it the moments restart
    # from zero on every step and the method silently degrades to its first iteration forever.
    Y = rand(StiefelManifold, 4, 2)
    Ḡ = global_rep(GlobalSection(Y), rgrad(Y, randn(4, 2)))
    F = linear_stiefel_objective(randn(4, 2))

    for (method, is_scalar) in ((Adam(), false), (ScalarMomentAdam(), true))
        state = OptimizerState(method, Y)
        optimizer = Optimizer(Y, F; algorithm = method, linesearch = Static(1.0e-2))
        increase_iteration_number!(state)
        update!(cache(optimizer), state, gradient(optimizer), method, Y)

        @test sync_state!(state, cache(optimizer), method) === state
        @test Matrix(first_moment(state)) == Matrix(first_moment(cache(optimizer)))
        if is_scalar
            @test second_moment(state) isa Real
            @test second_moment(state) == second_moment(cache(optimizer))
        else
            @test Matrix(second_moment(state)) == Matrix(second_moment(cache(optimizer)))
        end

        # ... and reached through a composite, which resolves to the same body.
        composite = CompositeMethod(; manifold = method, array = Adam())
        @test sync_state!(state, cache(optimizer), composite) === state
    end

    # `MomentumMethod` advances rather than copies -- the cache never holds the new momentum -- and
    # is the one arm that reads the method object, so it is the one the composite has to resolve.
    momentum_method = MomentumMethod(0.5)
    state = OptimizerState(momentum_method, Y)
    optimizer = Optimizer(Y, F; algorithm = momentum_method, linesearch = Static(1.0e-2))
    increase_iteration_number!(state)
    update!(cache(optimizer), state, gradient(optimizer), momentum_method, Y)
    expected = 0.5 * Matrix(momentum(state)) + Matrix(gradient_array(cache(optimizer)))
    sync_state!(state, cache(optimizer),
        CompositeMethod(; manifold = momentum_method, array = momentum_method))
    @test Matrix(momentum(state)) ≈ expected

    # A method that carries nothing is a no-op rather than an error, so a caller may ask
    # unconditionally.
    gradient_state = OptimizerState(GradientMethod(), Y)
    gradient_optimizer = Optimizer(Y, F; algorithm = GradientMethod())
    @test sync_state!(gradient_state, cache(gradient_optimizer), GradientMethod()) === gradient_state
end

@testset "accepts_parameter_set states ScalarMomentAdam's scope once" begin
    @test accepts_parameter_set(Adam())
    @test accepts_parameter_set(MomentumMethod(0.5))
    @test accepts_parameter_set(GradientMethod())
    @test accepts_parameter_set(BFGS())
    @test !accepts_parameter_set(ScalarMomentAdam())

    # The answer is a property of a leaf's method, so a composite has to be resolved first: asking a
    # composite is a caller error and says so.
    composite = CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam())
    @test_throws ArgumentError accepts_parameter_set(composite)
    @test !accepts_parameter_set(leafmethod(composite, rand(StiefelManifold, 4, 2)))
    @test accepts_parameter_set(leafmethod(composite, rand(3)))

    # And it is the truth about the method: a container holding exactly one Stiefel weight is what
    # `ScalarMomentAdam` rejects and `Adam` takes.
    ps = NetworkParameters((weight = rand(StiefelManifold, 4, 2),))
    @test_throws ArgumentError OptimizerState(ScalarMomentAdam(), ps)
    @test OptimizerState(Adam(), ps) isa AdamState
end
