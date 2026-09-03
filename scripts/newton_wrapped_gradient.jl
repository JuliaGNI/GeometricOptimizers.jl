# Whether a wrapped gradient survives the Newton path, and what the default observer costs per step.
#
# Run with the repository as the active project:
#
#     julia --project=. --check-bounds=auto scripts/newton_wrapped_gradient.jl
#
# `--check-bounds=auto` and not the `Pkg.test()` default of `=yes`, which inflates the figures and is
# why an allocation number taken from a test run says nothing.
#
# Two claims, and the reason they belong in one harness. `NewtonOptimizerState`'s `update!` used to
# obtain the objective as `gradient.F(x)`. `F` is a field the three concrete `SimpleSolvers.Gradient`
# subtypes happen to share; the `Gradient` interface guarantees a functor and nothing else. So every
# wrapper threw a `FieldError` on the first step — the `ObservedGradient` that `Optimizer` installs
# whenever an `observer` is passed, and a caller-supplied `RiemannianGradient`, which failed before
# the observer existed at all. The objective is now requested through `_objective`, which unwraps.
#
# The second table is the regression guard on that repair. Closing it added a phase bracket around
# Newton's `update_section!` and a closure that reports the objective the wrapper mediates, and the
# observer's whole premise is that none of it costs anything when no observer is installed. This
# script measures one revision; the claim is a comparison, so run it on both sides of a change and
# diff the tables. The figures are per step rather than per `solve!` so that a difference lands in
# the row that caused it.
#
# `test/optimizer_observer.jl` is the complement. Its "second-order methods observe their state
# update" testset pins the wrapper matrix below at *works at all* for `Newton`, `BFGS` and `DFP`,
# which is the assertion that was missing when nine green CI jobs said nothing about the defect;
# this script says what the repair is worth per step and covers the caller-supplied wrapper too.
# `scripts/optimizer_allocations.jl` is the end-to-end view over shapes of solution.

using GeometricOptimizers
using GeometricOptimizers: increase_iteration_number!, initialize_state!, solver_step!
using SimpleSolvers: GradientAutodiff, Static

const STEP_LENGTH = 0.05
const REPEATS = 11
const WARMUPS = 3

# Away from the minimum, so the line search and the NaN guard do the work they would do in a run.
starting_point() = [0.9, -1.8]
objective(x) = sum(abs2, x)

# One step plus the state update, which is the pair the observer brackets. `update!(state, opt, x)`
# is Newton's own method and is where both halves of the repair sit.
function step_once!(x, algorithm, optimizer)
    state = OptimizerState(algorithm, x)
    initialize_state!(state)
    increase_iteration_number!(state)
    solver_step!(x, state, optimizer)
    GeometricOptimizers.update!(state, optimizer, x)
    x
end

function step_allocations(algorithm)
    x = starting_point()
    optimizer = Optimizer(x, objective; algorithm, linesearch = Static(STEP_LENGTH))

    # A fresh copy per call: `step_once!` writes into its argument, so reusing one would measure a
    # step taken from the previous step's answer rather than from the starting point.
    for _ in 1:WARMUPS
        step_once!(starting_point(), algorithm, optimizer)
    end
    samples = [@allocated(step_once!(starting_point(), algorithm, optimizer))
               for _ in 1:REPEATS]
    minimum(samples)
end

# `solve!` rather than a single step, so that every consumer of `gradient(opt)` is exercised: the
# defect was reached from `solver_step!` and from Newton's `update!` alike.
function newton_outcome(; observer = nothing, gradient = nothing)
    x = starting_point()
    algorithm = Newton()
    optimizer = if isnothing(gradient)
        Optimizer(x, objective; algorithm, linesearch = Static(STEP_LENGTH), observer)
    else
        # The `gradient` keyword lives on the `OptimizerProblem` method. The `Function` method builds
        # the gradient itself and would pass `gradient` on to the solver options, which reject it.
        Optimizer(x, OptimizerProblem(objective, x); algorithm,
            linesearch = Static(STEP_LENGTH), gradient)
    end

    try
        solve!(x, OptimizerState(algorithm, x), optimizer)
        "ok"
    catch e
        string(nameof(typeof(e)))
    end
end

function main()
    println("one step plus update!, default observer, bytes allocated")
    for algorithm in (Newton(), BFGS(), DFP(), GradientMethod())
        println("  ", rpad(string(nameof(typeof(algorithm))), 18), step_allocations(algorithm))
    end

    println("\nsolve! with Newton and a wrapped gradient")
    for (label, kwargs) in (("no observer", (; observer = NoStepObserver())),
        ("EventLog", (; observer = EventLog())),
        ("PhaseTimer", (; observer = PhaseTimer())),
        ("RiemannianGradient",
        (; gradient = RiemannianGradient(GradientAutodiff(objective, starting_point())))))
        println("  ", rpad(label, 20), newton_outcome(; kwargs...))
    end
end

main()
