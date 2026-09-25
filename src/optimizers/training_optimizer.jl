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
    TrainingOptimizer(x; algorithm = Adam(), linesearch = default_step_size(algorithm), retraction = Cayley())

The optimizer of a training loop: one step per minibatch, with [`optimization_step!`](@ref).

It holds the first-order `algorithm`, converted once to the element type of `x` with
`change_precision`, its cache and its state over the whole of `x`, the step-size schedule, the
retraction, and the buffers the retraction works in.

`algorithm` is a [`GradientMethod`](@ref), a [`MomentumMethod`](@ref) or a member of the
[`AdamFamily`](@ref); [`ScalarMomentAdam`](@ref) steps a single `StiefelManifold` only, and
raises an `ArgumentError` for any other `x`. `linesearch` is a finite positive number, which is
the fixed step size `Static(η)`, a [`SimpleSolvers.Static`](@extref) or a
[`DecayingStatic`](@ref); the default is [`default_step_size`](@ref)`(algorithm)`. `retraction` is an [`AbstractRetraction`](@ref) type,
`Cayley()` or `Geodesic()`.

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
    LT <: Union{Static, DecayingStatic}, RT <: AbstractRetraction, WT}
    method::MT
    cache::CT
    state::ST
    linesearch::LT
    retraction::RT
    workspace::WT
end

function TrainingOptimizer(x::OptimizerSolution{T}; algorithm::FirstOrderMethod = Adam(),
        linesearch = default_step_size(algorithm),
        retraction::AbstractRetraction = Cayley()) where {T}
    method = change_precision(T, algorithm)
    # the cache first and the state second: each draws the random completion of its `GlobalSection`
    cache = OptimizerCache(method, x)
    state = OptimizerState(method, x)
    TrainingOptimizer(method, cache, state, _training_step_size(T, linesearch), retraction,
        retraction_workspace(x))
end

@doc raw"""
    optimization_step!(x, opt::TrainingOptimizer, dp)

Take one step of the training optimizer `opt` with the Euclidean gradient `dp` of the loss at `x`,
and write the new parameters into `x`, which it returns.

`x` is the object `opt` was built on, and nothing else changes it between steps: the step starts
from the point `opt.state` carries, which is `x` after the previous step. `dp` has the shape and the
element type of `x`; a `dp` of another element type is an `ArgumentError`.

The step increments the iteration number `t` of `opt.state`, reads the step size
[`step_size`](@ref)`(opt.linesearch, t)`, forms the direction of `opt.method` from the
[`PrecomputedGradient`](@ref) of `dp`, scales it, retracts it with `opt.retraction` and copies the
result into `x`. [`advance_state!`](@ref) then carries the state over to the new parameters.

For a parameter set, `dp` is a [`NeuralNetworkParameters.NetworkParameters`](@extref) of the same
shape as `x`. See [`TrainingOptimizer`](@ref) for why this is not [`solve!`](@ref).
"""
function optimization_step!(x::OptimizerSolution{T}, opt::TrainingOptimizer,
        dp::Union{AbstractArray{T}, NetworkParameters{T}}) where {T}
    increase_iteration_number!(opt.state)
    α = step_size(opt.linesearch, iteration_number(opt.state))
    update!(opt.cache, opt.state, PrecomputedGradient(x, dp), opt.method, x)
    _rmul!(direction(opt.cache), α)
    update_section!(section(opt.cache), section(opt.state), direction(opt.cache),
        opt.retraction, opt.workspace)
    _copyto!(solution(opt.cache), section(opt.cache))
    _copyto!(x, solution(opt.cache))
    advance_state!(opt.state, opt.cache, opt.method)
    x
end

# The element-type check is dispatch, so the step above pays nothing for it.
function optimization_step!(x::OptimizerSolution, ::TrainingOptimizer, dp)
    throw(ArgumentError("the gradient is a $(typeof(dp)) and the parameters are a $(typeof(x)): " *
                        "a training step takes a gradient of the shape and element type of `x`."))
end
