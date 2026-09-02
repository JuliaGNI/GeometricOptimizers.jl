using GeometricOptimizers
using GeometricOptimizers: gradient, increase_iteration_number!, initialize_state!,
                           solver_step!
using Test

const EXPECTED_STEP_EVENTS = [
    (:optimizer_state_direction, :enter),
    (:gradient, :enter), (:gradient, :exit),
    (:retraction_application, :enter), (:retraction_application, :exit),
    (:objective, :enter), (:objective, :exit),
    (:retraction_application, :enter), (:retraction_application, :exit),
    (:gradient, :enter), (:gradient, :exit),
    (:objective, :enter), (:objective, :exit),
    (:retraction_application, :enter), (:retraction_application, :exit),
    (:optimizer_state_direction, :exit)
]

function observed_step_events(x, objective, method; linesearch = Static(0.01))
    observer = EventLog()
    optimizer = Optimizer(x, objective; algorithm = method, linesearch = linesearch,
        observer = observer)
    state = OptimizerState(method, x)
    initialize_state!(state)

    observe_optimizer_phase(observer, :optimizer_state_direction) do
        increase_iteration_number!(state)
        solver_step!(x, state, optimizer)
        GeometricOptimizers.update!(state, optimizer, x)
    end

    observer.events
end

@testset "step phase events are complete and nested" begin
    objective(x) = sum(abs2, x)
    for method in (GradientMethod(), MomentumMethod(0.1), Adam(Float64), BFGS(), DFP())
        @test observed_step_events([1.0, -2.0], objective, method) == EXPECTED_STEP_EVENTS
    end

    Y = StiefelManifold([1.0 0.0; 0.0 1.0; 0.0 0.0])
    stiefel_objective(Y) = sum(abs2, Y.A .- 1)
    @test observed_step_events(Y, stiefel_objective, ScalarMomentAdam()) ==
          EXPECTED_STEP_EVENTS
end

@testset "default backtracking observes its slope evaluation" begin
    events = observed_step_events(
        [1.0, -2.0], x -> sum(abs2, x), GradientMethod(); linesearch = Backtracking())
    anchor_events = [
        (:retraction_application, :enter),
        (:retraction_application, :exit),
        (:objective, :enter),
        (:objective, :exit),
        (:retraction_application, :enter),
        (:retraction_application, :exit),
        (:retraction_application, :enter),
        (:gradient, :enter),
        (:gradient, :exit),
        (:retraction_application, :exit)
    ]

    @test any(==(anchor_events), (events[i:(i + 9)] for i in 1:(length(events) - 9)))
end

@testset "solve! observes every objective evaluation" begin
    for store_trace in (false, true)
        observer = EventLog()
        objective_calls = Ref(0)
        objective = x -> (objective_calls[] += 1; sum(abs2, x))
        x = [1.0, -2.0]
        method = GradientMethod()
        optimizer = Optimizer(x, objective; (∇F!) = (g, x) -> (g .= 2 .* x),
            algorithm = method, linesearch = Static(0.01), max_iterations = 1,
            store_trace = store_trace, observer = observer)
        objective_calls[] = 0

        solve!(x, OptimizerState(method, x), optimizer)

        @test count(==((:objective, :enter)), observer.events) == objective_calls[]
        @test count(==((:objective, :exit)), observer.events) == objective_calls[]
    end
end

@testset "parameter-set gradients retain the Riemannian wrapper" begin
    observer = EventLog()
    ps = NetworkParameters((w = [1.0, -2.0],))
    objective(ps) = sum(abs2, ps.w)
    gradient!(g, x) = (g .= 2 .* x; g)
    method = GradientMethod()
    optimizer = Optimizer(ps, objective; (∇F!) = gradient!, algorithm = method,
        linesearch = Static(0.1), observer = observer)
    observed_gradient = gradient(optimizer)
    @test observed_gradient isa RiemannianGradient
    @test observed_gradient.gradient isa GeometricOptimizers.ObservedGradient
    state = OptimizerState(method, ps)
    initialize_state!(state)

    observe_optimizer_phase(observer, :optimizer_state_direction) do
        increase_iteration_number!(state)
        solver_step!(ps, state, optimizer)
        GeometricOptimizers.update!(state, optimizer, ps)
    end

    @test count(==((:gradient, :enter)), observer.events) == 2
    @test count(==((:gradient, :exit)), observer.events) == 2
    @test ps.w ≈ [0.8, -1.6]
end

@testset "observer exit is exception-safe and the default is a no-op" begin
    observer = EventLog()
    @test_throws ErrorException observe_optimizer_phase(observer, :gradient) do
        error("expected failure")
    end
    @test observer.events == [(:gradient, :enter), (:gradient, :exit)]
    @test observe_optimizer_phase(() -> 42, NoStepObserver(), :gradient) == 42
end

@testset "built-in observer recorders" begin
    log = EventLog(phases = :gradient)
    observe_optimizer_phase(log, :optimizer_state_direction) do
        observe_optimizer_phase(log, :gradient) do
            nothing
        end
    end
    @test log.events == [(:gradient, :enter), (:gradient, :exit)]
    @test empty!(log) === log
    @test isempty(log.events)

    ticks = Ref(UInt64(0))
    synchronizations = Ref(0)
    clock = () -> (ticks[] += UInt64(10))
    synchronize = () -> (synchronizations[] += 1; nothing)
    timer = PhaseTimer(clock = clock, synchronize = synchronize)
    observe_optimizer_phase(timer, :optimizer_state_direction) do
        observe_optimizer_phase(timer, :gradient) do
            nothing
        end
    end
    @test timer.exclusive == Dict(:optimizer_state_direction => UInt64(20),
        :gradient => UInt64(10))
    @test timer.calls == Dict(:optimizer_state_direction => 1, :gradient => 1)
    @test synchronizations[] == 4
    @test isempty(timer.open)

    filtered_timer = PhaseTimer(phases = (:gradient,), clock = clock)
    observe_optimizer_phase(filtered_timer, :optimizer_state_direction) do
        observe_optimizer_phase(filtered_timer, :gradient) do
            nothing
        end
    end
    @test filtered_timer.exclusive == Dict(:gradient => UInt64(10))
    @test filtered_timer.calls == Dict(:gradient => 1)
    @test empty!(filtered_timer) === filtered_timer
    @test isempty(filtered_timer.exclusive)
    @test isempty(filtered_timer.calls)
end
