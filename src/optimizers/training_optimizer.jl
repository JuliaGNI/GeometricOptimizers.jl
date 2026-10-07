@doc raw"""
    PrecomputedGradient(x, dp) <: SimpleSolvers.Gradient

A [`SimpleSolvers.Gradient`](@extref) that holds an already computed Euclidean gradient `dp` of the
parameters `x`, and returns its Riemannian gradient [`rgrad`](@ref)`(x, dp)` when it is called on
`x`.

This is the gradient of a training step: a training loop computes the gradient of its loss on one
minibatch, and [`optimization_step!`](@ref) wraps it in this type, so the optimizer caches build the
direction from it exactly as they build it from a gradient they evaluate themselves. For a parameter
set, `dp` is a [`NeuralNetworkParameters.NetworkParameters`](@extref) of the same shape as `x`.
"""
struct PrecomputedGradient{T, DT} <: Gradient{T}
    dp::DT
end

function PrecomputedGradient(::OptimizerSolution{T}, dp) where {T}
    PrecomputedGradient{T, typeof(dp)}(dp)
end

# One method per kind of parameter, and not one on `OptimizerSolution{T}`: `(::Gradient{T})(::Manifold{T})`
# and `SimpleSolvers`' `(::Gradient{T})(::AbstractVector{T})` are each more specific in the second
# argument than a `Union`, so a single method would be ambiguous with both.
(g::PrecomputedGradient{T})(x::AbstractVector{T}) where {T} = rgrad(x, g.dp)
(g::PrecomputedGradient{T})(x::Manifold{T}) where {T} = rgrad(x, g.dp)
(g::PrecomputedGradient{T})(x::NetworkParameters{T}) where {T} = rgrad(x, g.dp)

@doc raw"""
    TrainingOptimizer(x; algorithm = Adam(), linesearch = default_step_size(algorithm), retraction = Cayley(), observer = NoStepObserver())

The optimizer of a training loop: one step per minibatch, with [`optimization_step!`](@ref).

It holds the first-order `algorithm`, converted once to the element type of `x` with
`change_precision`, its cache and its state over the whole of `x`, the step-size schedule, the
retraction, and the buffers the retraction works in.

`algorithm` is a [`GradientMethod`](@ref), a [`MomentumMethod`](@ref), a member of the
[`AdamFamily`](@ref) or a [`CompositeMethod`](@ref); [`ScalarMomentAdam`](@ref) steps a single
`StiefelManifold` only, and raises a `MethodError` for any other `x`. A composite on a parameter set
gets one cache and one state per leaf, each for the method it chooses for that leaf and each
converted to the element type of `x`; see [`CompositeMethod`](@ref). `linesearch` is a finite
positive number, which is the fixed step size `Static(η)`, a [`SimpleSolvers.Static`](@extref) or a
[`DecayingStatic`](@ref); the default is [`default_step_size`](@ref)`(algorithm)`. `retraction` is
an instance of an [`AbstractRetraction`](@ref), `Cayley()` or `Geodesic()`.

`observer` is notified of the `:retraction_application` phase of every step, as the observer of an
[`Optimizer`](@ref) is; see [Observing the Phases of an Optimizer Step](@ref). The gradient is the caller's, so a
caller who times it brackets its own reverse pass, and one who wants the cost of the direction
brackets [`optimization_step!`](@ref) in a phase of its own, from which the nested retraction
subtracts itself.

# Why a training step is not `solve!`

[`solve!`](@ref) minimizes an objective it can evaluate: its line search compares the objective along
the direction, and it stops when a convergence test on the objective and its gradient passes. A
training step has neither. It has the gradient of the loss on one minibatch, which is a stochastic
estimate of the gradient of the loss it minimizes; the loss on that minibatch is not the objective, so
there is nothing for a line search to search along and nothing for a convergence test to test. So
`linesearch` takes a fixed or a scheduled step size and nothing else, and the loop that calls
[`optimization_step!`](@ref) decides when to stop.

# Examples

```jldoctest; setup = :(using GeometricOptimizers)
x = Float32[1, -2, 3]
opt = TrainingOptimizer(x; algorithm = GradientMethod(), linesearch = 0.5)
optimization_step!(x, opt, Float32[2, 2, 2])

# output

3-element Vector{Float32}:
  0.0
 -3.0
  2.0
```
"""
struct TrainingOptimizer{
    MT <: FirstOrderMethod, CT <: OptimizerCache, ST <: OptimizerState,
    LT <: Union{Static, DecayingStatic}, RT <: AbstractRetraction, WT, OT}
    method::MT
    cache::CT
    state::ST
    linesearch::LT
    retraction::RT
    workspace::WT
    observer::OT
end

function TrainingOptimizer(x::OptimizerSolution{T}; algorithm::FirstOrderMethod = Adam(),
        linesearch = default_step_size(algorithm),
        retraction::AbstractRetraction = Cayley(), observer = NoStepObserver()) where {T}
    method = _training_method(T, algorithm, x)
    # the cache first and the state second: each draws the random completion of its `GlobalSection`
    cache = OptimizerCache(method, x)
    state = OptimizerState(method, x)
    TrainingOptimizer(method, cache, state, _training_step_size(T, linesearch), retraction,
        _training_workspace(method, x, retraction), observer)
end

step_observer(opt::TrainingOptimizer) = opt.observer

# The method and the retraction buffers a `TrainingOptimizer` keeps for `x`. A composite on a
# parameter set keeps them per leaf; see `optimizers/composite_method.jl`.
function _training_method(::Type{T}, algorithm::OptimizerMethod, x) where {T}
    change_precision(T, algorithm)
end
_training_workspace(::OptimizerMethod, x, retraction) = retraction_workspace(x, retraction)

@doc raw"""
    optimization_step!(x, opt::TrainingOptimizer, dp)

Take one step of the training optimizer `opt` with the Euclidean gradient `dp` of the loss at `x`,
and write the new parameters into `x`, which it returns.

`x` is the object `opt` was built on, and nothing else changes it between steps: the step starts
from the point `opt.state` carries, which is `x` after the previous step. `dp` has the shape and the
element type of `x`. An `x` or a `dp` of another element type than `opt` is an `ArgumentError`. A
`dp` with arrays of other sizes is a `DimensionMismatch`, and a parameter set with other keys or
another number of leaves is an `ArgumentError`. Each is raised before the iteration number moves.

The step increments the iteration number `t` of `opt.state`, reads the step size
[`step_size`](@ref)`(opt.linesearch, t)`, forms the direction of `opt.method` from the
[`PrecomputedGradient`](@ref) of `dp`, scales it, retracts it with `opt.retraction` and copies the
result into `x`. [`advance_state!`](@ref) then carries the state over to the new parameters. A
[`CompositeMethod`](@ref) on a parameter set does this for every leaf with the leaf's own method,
cache and state, at the one step size `α`.

For a parameter set, `dp` is a [`NeuralNetworkParameters.NetworkParameters`](@extref) of the same
shape as `x`. See [`TrainingOptimizer`](@ref) for why this is not [`solve!`](@ref).
"""
function optimization_step!(x::OptimizerSolution{T},
        opt::TrainingOptimizer{<:FirstOrderMethod, <:OptimizerCache, <:OptimizerState{T}},
        dp::Union{AbstractArray{T}, NetworkParameters{T}}) where {T}
    # before the count moves, so that a refused step leaves the state as it was
    _same_shape(x, dp) || throw(DimensionMismatch(
        "the gradient does not have the shape of the parameters it is a gradient of"))
    increase_iteration_number!(opt.state)
    α = step_size(opt.linesearch, iteration_number(opt.state))
    _training_step!(x, opt.cache, opt.state, PrecomputedGradient(x, dp), opt.method, α,
        opt.retraction, opt.workspace, step_observer(opt))
end

# The step itself, once the count has moved and the step size is read: the whole of `x` for one
# method, and one leaf at a time for a `CompositeMethod`.
# The phase is the one `solver_step!` reports for the same work, the retraction, the copies of the
# retracted point and the state update after it; see the table of phases in the observers chapter.
function _training_step!(
        x, cache, state, gradient, method, α, retraction, workspace, observer)
    update!(cache, state, gradient, method, x)
    _rmul!(direction(cache), α)
    observe_optimizer_phase(observer, :retraction_application) do
        update_section!(section(cache), section(state), direction(cache), retraction, workspace)
        _copyto!(solution(cache), section(cache))
        _copyto!(x, solution(cache))
        advance_state!(state, cache, method)
    end
    x
end

_same_shape(x::AbstractArray, dp::AbstractArray) = size(x) == size(dp)
function _same_shape(x::NetworkParameters, dp::NetworkParameters)
    foldparameters((same, xᵢ, dpᵢ) -> same && size(xᵢ) == size(dpᵢ), true, x, dp)
end
_same_shape(_, _) = false

# The element-type check is dispatch, so the step above pays nothing for it.
function optimization_step!(x::OptimizerSolution, opt::TrainingOptimizer, dp)
    throw(ArgumentError("the gradient is a $(typeof(dp)), the parameters are a $(typeof(x)) and " *
                        "the optimizer state is a $(typeof(opt.state)): a training step takes " *
                        "parameters and a gradient of the element type `opt` was built for."))
end
