@doc raw"""
    PrecomputedGradient(x, dp) <: SimpleSolvers.Gradient

A [`SimpleSolvers.Gradient`](@extref) that holds an already computed Euclidean gradient `dp` of the
parameters `x`, and returns its Riemannian gradient [`rgrad`](@ref)`(x, dp)` when it is called on
`x`.

This is the gradient of a training step: a training loop computes the gradient of its loss on one
minibatch, and [`optimization_step!`](@ref) wraps it in this type, so the optimizer caches build the
direction from it exactly as they build it from a gradient they evaluate themselves. For a parameter
set, `dp` is a [`NeuralNetworkParameters.NetworkParameters`](@extref) of the same shape as `x`.

# Implementation

[`optimization_step!`](@ref) passes the `RiemannianGradientBuffer`s of its optimizer as a third
argument, and the Riemannian gradient of a Stiefel or a Grassmann leaf is then written into them
rather than allocated. The result is overwritten by the next call, which `store_gradient!` makes
only after it has read the one before.
"""
struct PrecomputedGradient{T, DT, BT} <: Gradient{T}
    dp::DT
    buffer::BT
end

function PrecomputedGradient(::OptimizerSolution{T}, dp, buffer = NoWorkspace()) where {T}
    PrecomputedGradient{T, typeof(dp), typeof(buffer)}(dp, buffer)
end

# One method per kind of parameter, and not one on `OptimizerSolution{T}`: `(::Gradient{T})(::Manifold{T})`
# and `SimpleSolvers`' `(::Gradient{T})(::AbstractVector{T})` are each more specific in the second
# argument than a `Union`, so a single method would be ambiguous with both.
(g::PrecomputedGradient{T})(x::AbstractVector{T}) where {T} = rgrad(x, g.dp)
(g::PrecomputedGradient{T})(x::Manifold{T}) where {T} = _rgrad(g.buffer, x, g.dp)
(g::PrecomputedGradient{T})(x::NetworkParameters{T}) where {T} = _rgrad(g.buffer, x, g.dp)

# `ScalarMomentAdam(; ambient_norm = true)` squares the ambient gradient, which this one holds
_squared_ambient_norm(g::PrecomputedGradient, ::StiefelManifold) = sum(abs2, g.dp)

# The `N × n` Riemannian gradient of a Stiefel or a Grassmann leaf and the `n × n` product it is
# formed from, for `rgrad!`.
struct RiemannianGradientBuffer{AT <: AbstractMatrix}
    Δ::AT
    scratch::AT
end

# A `NoWorkspace` and not `nothing` at every other leaf, for the reason `NoWorkspace` gives.
function rgrad_buffer(Y::Union{StiefelManifold, GrassmannManifold})
    n = size(Y, 2)
    RiemannianGradientBuffer(similar(Y.A), similar(Y.A, n, n))
end
rgrad_buffer(::AbstractArray) = NoWorkspace()
rgrad_buffer(ps::NetworkParameters) = mapparameters(rgrad_buffer, params(ps))

_rgrad(::NoWorkspace, x, dp) = rgrad(x, dp)
_rgrad(buffer::RiemannianGradientBuffer, Y, ∇L) = rgrad!(buffer.Δ, buffer.scratch, Y, ∇L)
# a named function and not a closure, so that the walk specialises on it
function _rgrad(buffer::NamedTuple, ps::NetworkParameters, dp)
    mapparameters(_rgrad_leaf, ps, dp, buffer)
end
_rgrad_leaf(x, dp, buffer) = _rgrad(buffer, x, dp)

@doc raw"""
    TrainingOptimizer(x; algorithm = Adam(), linesearch = default_step_size(algorithm), retraction = Cayley(), observer = NoStepObserver())

The optimizer of a training loop: one step per minibatch, with [`optimization_step!`](@ref).

It holds the first-order `algorithm`, converted once to the element type of `x` with
`change_precision`, its cache and its state over the whole of `x`, the step-size schedule, the
retraction, the buffers the retraction works in, and those the Riemannian gradient of a Stiefel or
a Grassmann leaf is written into.

`algorithm` is a [`GradientMethod`](@ref), a [`MomentumMethod`](@ref) or a member of the
[`AdamFamily`](@ref); [`ScalarMomentAdam`](@ref) steps a single `StiefelManifold` only, and
raises a `MethodError` for any other `x`. For a parameter set it may also be a
[`CompositeMethod`](@ref): then `cache`, `state`, `workspace` and `gradient_buffer` are trees of
the shape of `x`, with the cache, the state and the buffers of each leaf for the method of that leaf.
`linesearch` is a finite positive number, which is the fixed step size `Static(η)`, a
[`SimpleSolvers.Static`](@extref) or a [`DecayingStatic`](@ref); the default is
[`default_step_size`](@ref)`(algorithm)`. `retraction` is an instance of an
[`AbstractRetraction`](@ref), `Cayley()` or `Geodesic()`.

`observer` is notified of the `:retraction_application` phase of every step, once per leaf for a
`CompositeMethod`; see [Observing the Phases of an Optimizer Step](@ref).

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
struct TrainingOptimizer{MT <: Union{FirstOrderMethod, CompositeMethod}, CT, ST,
    LT <: Union{Static, DecayingStatic}, RT <: AbstractRetraction, WT, BT, OT}
    method::MT
    # `CT` and `ST` are unbounded because a composite holds trees of caches and states.
    cache::CT
    state::ST
    linesearch::LT
    retraction::RT
    workspace::WT
    # the `RiemannianGradientBuffer`s of `PrecomputedGradient`, a tree of the shape of `workspace`
    gradient_buffer::BT
    observer::OT
end

function TrainingOptimizer(x::OptimizerSolution{T};
        algorithm::Union{FirstOrderMethod, CompositeMethod} = Adam(),
        linesearch = default_step_size(algorithm),
        retraction::AbstractRetraction = Cayley(), observer = NoStepObserver()) where {T}
    _training_optimizer(
        x, change_precision(T, algorithm), _training_step_size(T, linesearch),
        retraction, observer)
end

# The composite method of this is in `optimizers/composite_method.jl`.
function _training_optimizer(x, method::FirstOrderMethod, linesearch, retraction, observer)
    # the cache first and the state second: each draws the random completion of its `GlobalSection`
    cache = OptimizerCache(method, x)
    state = OptimizerState(method, x)
    TrainingOptimizer(method, cache, state, linesearch, retraction,
        retraction_workspace(x, retraction), rgrad_buffer(x), observer)
end

step_observer(opt::TrainingOptimizer) = opt.observer

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
[`CompositeMethod`](@ref) takes this step on every leaf, with the method, the cache and the state of
that leaf.

For a parameter set, `dp` is a [`NeuralNetworkParameters.NetworkParameters`](@extref) of the same
shape as `x`. See [`TrainingOptimizer`](@ref) for why this is not [`solve!`](@ref).
"""
function optimization_step!(x::OptimizerSolution{T},
        opt::TrainingOptimizer{<:FirstOrderMethod, <:OptimizerCache, <:OptimizerState{T}},
        dp::Union{AbstractArray{T}, NetworkParameters{T}}) where {T}
    _check_shape(x, dp)
    increase_iteration_number!(opt.state)
    α = step_size(opt.linesearch, iteration_number(opt.state))
    update!(opt.cache, opt.state,
        PrecomputedGradient(x, dp, opt.gradient_buffer), opt.method, x)
    _rmul!(direction(opt.cache), α)
    # the phase `solve!` reports around its retraction and `advance_state!`
    observe_optimizer_phase(step_observer(opt), :retraction_application) do
        update_section!(section(opt.cache), section(opt.state), direction(opt.cache),
            opt.retraction, opt.workspace)
        _copyto!(solution(opt.cache), section(opt.cache))
        _copyto!(x, solution(opt.cache))
        advance_state!(opt.state, opt.cache, opt.method)
    end
    x
end

# before the count moves, so that a refused step leaves the state as it was
function _check_shape(x, dp)
    _same_shape(x, dp) || throw(DimensionMismatch(
        "the gradient does not have the shape of the parameters it is a gradient of"))
    nothing
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
