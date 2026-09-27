"""
An `OptimizerState` is a data structure that is used to dispatch on different algorithms.

It needs to implement three methods,
```
initialize!(alg::OptimizerState, ::AbstractVector)
update!(alg::OptimizerState, ::AbstractVector)
solver_step!(::AbstractVector, alg::OptimizerState)
```
that initialize and update the state of the algorithm and perform an actual optimization step.

Further the following convenience methods should be implemented,
```
problem(alg::OptimizerState)
gradient(alg::OptimizerState)
hessian(alg::OptimizerState)
linesearch(alg::OptimizerState)
```
which return the problem to optimize, its gradient and (approximate) Hessian as well as the
linesearch algorithm used in conjunction with the optimization algorithm if any.

See [`NewtonState`](@ref) for a `struct` that was derived from `OptimizerState`.

!!! info
    Note that a `OptimizerState` is not necessarily a `NewtonState` as we can also have other optimizers, *Adam* for example.
"""
abstract type OptimizerState{T} <: AbstractSolverState end

function OptimizerState(alg::OptimizerMethod, args...; kwargs...)
    error("OptimizerState not implemented for $(typeof(alg))")
end

"""
    isaOptimizerState(alg)

Verify if an object implements the [`OptimizerState`](@ref) interface.
"""
function isaOptimizerState(alg)
    x = rand(3)

    applicable(gradient, alg) &&
        applicable(hessian, alg) &&
        applicable(linesearch, alg) &&
        applicable(problem, alg) &&
        applicable(initialize!, alg, x) &&
        applicable(update!, alg, x) &&
        applicable(solver_step!, x, alg)
end

iteration_number(state::OptimizerState) = state.iterations

# The accessors every state in this package answers, on the fields they all name alike: the unbarred
# field is the current iterate's and the barred one the previous iterate's. `value` is public, the
# other five internal. `gradient` is not among them: `BFGSState` holds no current gradient, so it is
# defined on the states that do, in `scalar_moment_adam_optimizer.jl`.
solution(state::OptimizerState) = state.x
previous_solution(state::OptimizerState) = state.x̄
previous_gradient(state::OptimizerState) = state.ḡ
value(state::OptimizerState) = state.f
previous_value(state::OptimizerState) = state.f̄
section(state::OptimizerState) = state.section

function increase_iteration_number!(state::OptimizerState)
    state.iterations = iteration_number(state) + 1
end
