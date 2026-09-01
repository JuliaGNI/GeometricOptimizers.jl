using GeometricOptimizers
using GeometricOptimizers: increase_iteration_number!, initialize_state!, solver_step!
using Test

mutable struct EventObserver
    events::Vector{Tuple{Symbol,Symbol}}
end

(observer::EventObserver)(phase, event) = (push!(observer.events, (phase, event)); nothing)

@testset "first-order step phase events are complete and nested" begin
    observer = EventObserver(Tuple{Symbol,Symbol}[])
    x = [1.0, -2.0]
    objective(x) = sum(abs2, x)
    gradient!(g, x) = (g .= 2 .* x; g)
    method = GradientMethod()
    optimizer = Optimizer(x, objective; (∇F!)=gradient!, algorithm=method,
        linesearch=Static(0.1), observer=observer)
    state = OptimizerState(method, x)
    initialize_state!(state)

    observe_optimizer_phase(observer, :optimizer_state_direction) do
        increase_iteration_number!(state)
        solver_step!(x, state, optimizer)
        GeometricOptimizers.update!(state, optimizer, x)
    end

    @test observer.events == [
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
end

@testset "parameter-set gradients retain the Riemannian wrapper" begin
    observer = EventObserver(Tuple{Symbol,Symbol}[])
    ps = NetworkParameters((w=[1.0, -2.0],))
    objective(ps) = sum(abs2, ps.w)
    gradient!(g, x) = (g .= 2 .* x; g)
    method = GradientMethod()
    optimizer = Optimizer(ps, objective; (∇F!)=gradient!, algorithm=method,
        linesearch=Static(0.1), observer=observer)
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
