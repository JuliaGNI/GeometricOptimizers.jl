using GeometricOptimizers
using GeometricOptimizers: gradient, increase_iteration_number!, initialize_state!, solver_step!
using Test

mutable struct EventObserver
    events::Vector{Tuple{Symbol,Symbol}}
end

(observer::EventObserver)(phase, event) = (push!(observer.events, (phase, event)); nothing)

const EXPECTED_STEP_EVENTS = [
    (:optimizer_state_direction, :enter),
    (:gradient, :enter), (:gradient, :exit),
    (:retraction_application, :enter), (:retraction_application, :exit),
    (:objective, :enter), (:objective, :exit),
    (:retraction_application, :enter), (:retraction_application, :exit),
    (:gradient, :enter), (:gradient, :exit),
    (:objective, :enter), (:objective, :exit),
    (:retraction_application, :enter), (:retraction_application, :exit),
    (:optimizer_state_direction, :exit),
]

function observed_step_events(x, objective, method)
    observer = EventObserver(Tuple{Symbol,Symbol}[])
    optimizer = Optimizer(x, objective; algorithm=method, linesearch=Static(0.01),
        observer=observer)
    state = OptimizerState(method, x)
    initialize_state!(state)

    observe_optimizer_phase(observer, :optimizer_state_direction) do
        increase_iteration_number!(state)
        solver_step!(x, state, optimizer)
        GeometricOptimizers.update!(state, optimizer, x)
    end

    observer.events
end

@testset "first-order step phase events are complete and nested" begin
    objective(x) = sum(abs2, x)
    for method in (GradientMethod(), MomentumMethod(0.1), Adam(Float64))
        @test observed_step_events([1.0, -2.0], objective, method) == EXPECTED_STEP_EVENTS
    end

    Y = StiefelManifold([1.0 0.0; 0.0 1.0; 0.0 0.0])
    stiefel_objective(Y) = sum(abs2, Y.A .- 1)
    @test observed_step_events(Y, stiefel_objective, ScalarMomentAdam()) == EXPECTED_STEP_EVENTS
end

@testset "parameter-set gradients retain the Riemannian wrapper" begin
    observer = EventObserver(Tuple{Symbol,Symbol}[])
    ps = NetworkParameters((w=[1.0, -2.0],))
    objective(ps) = sum(abs2, ps.w)
    gradient!(g, x) = (g .= 2 .* x; g)
    method = GradientMethod()
    optimizer = Optimizer(ps, objective; (∇F!)=gradient!, algorithm=method,
        linesearch=Static(0.1), observer=observer)
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
    observer = EventObserver(Tuple{Symbol,Symbol}[])
    @test_throws ErrorException observe_optimizer_phase(observer, :gradient) do
        error("expected failure")
    end
    @test observer.events == [(:gradient, :enter), (:gradient, :exit)]
    @test observe_optimizer_phase(() -> 42, NoStepObserver(), :gradient) == 42
end
