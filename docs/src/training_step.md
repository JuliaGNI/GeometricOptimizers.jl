```@meta
CurrentModule = GeometricOptimizers
```

# The Training Step

A training loop minimizes a loss over a data set one minibatch at a time. Each step has the gradient
of the loss on one minibatch, and a step size that is fixed or follows a schedule.
[`TrainingOptimizer`](@ref) holds what such a step needs, and [`optimization_step!`](@ref) takes
one:

```@example training_step
using GeometricOptimizers

x = NetworkParameters((weight = rand(StiefelManifold{Float32}, 5, 2), bias = zeros(Float32, 5)))
opt = TrainingOptimizer(x; algorithm = Adam(), linesearch = 1e-3)

dp = NetworkParameters((weight = ones(Float32, 5, 2), bias = ones(Float32, 5)))
optimization_step!(x, opt, dp)

GeometricOptimizers.check(x.weight) < 1f-5
```

`dp` is the Euclidean gradient of the loss at `x`, in the shape of `x`. The step projects it onto the
tangent space of each manifold leaf, forms the direction of the method, scales it by the step size,
retracts it, and writes the new parameters into `x`. The state then advances, so the next call
continues with the same momentum or moments.

## The methods and the step size

`algorithm` is one of the first-order methods: [`GradientMethod`](@ref), [`MomentumMethod`](@ref),
[`Adam`](@ref), [`AdamWithEuclideanDecay`](@ref) or [`ScalarMomentAdam`](@ref). A method carries no
element type. [`TrainingOptimizer`](@ref) converts it once, with `change_precision`, to the element
type of `x`, so `Adam()` trains `Float32` and `Float64` parameters alike:

```@example training_step
opt.method
```

`linesearch` is the step size. A number is a fixed step size, `Static(η)`. A
[`DecayingStatic`](@ref) is a step size that decays geometrically with the iteration number, and
[`AdamOptimizerWithDecay`](@ref) pairs it with [`Adam`](@ref):

```@example training_step
y = rand(StiefelManifold{Float32}, 5, 2)
opt = TrainingOptimizer(y; AdamOptimizerWithDecay(100; η₁ = 1e-2, η₂ = 1e-4)...)
GeometricOptimizers.step_size(opt.linesearch, 1)
```

Without `linesearch`, the step size is [`default_step_size`](@ref)`(algorithm)`: `1e-2` for the
gradient and the momentum method, and `1e-3` for the Adam methods.

`retraction` is [`Cayley`](@ref), the default, or [`Geodesic`](@ref).

## Why the training step is not `solve!`

[`solve!`](@ref) minimizes an objective it can evaluate. Its line search evaluates the objective
along the direction and takes a step that decreases it. Its stopping criteria test the change of the
objective and the size of its gradient.

A training step has neither of these. It has the gradient of the loss on one minibatch, which is a
stochastic estimate of the gradient of the loss that the training minimizes. The loss on that
minibatch is not the objective, so a line search has nothing to search along, and a convergence
test has nothing to test. So [`TrainingOptimizer`](@ref) takes a fixed or a scheduled step size and
refuses a searching line search, and the training loop decides when to stop.

The two share what does not depend on an objective. [`optimization_step!`](@ref) builds the
direction with the same `update!` of the optimizer cache that [`solver_step!`](@ref) uses, and
[`advance_state!`](@ref) is the part of the state update that both run.
