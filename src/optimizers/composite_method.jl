# The per-leaf half of [`CompositeMethod`](@ref): the cache and the state of a composite on a
# parameter set, and the training step over them. The type and [`leafmethod`](@ref) are in
# `optimizers/optimizer_methods.jl`, where the method types live; everything below needs the
# first-order caches and states and `TrainingOptimizer`, so it is included after them.
#
# A composite pools nothing. Every leaf gets the cache and the state of the method chosen for it, and
# the step walks the leaves and takes, for each, the step a `TrainingOptimizer` built on that leaf
# alone would take. What the leaves share is the iteration number and the step size read from it.

# What a leaf is stepped as. A vector and a manifold element are `OptimizerSolution`s of their own, and
# `ScalarMomentAdam` takes a `StiefelManifold` and nothing else, so those stay bare. Every other leaf --
# a matrix, a `SymmetricMatrix` -- is not an `OptimizerSolution`, and is wrapped in a parameter set of
# one leaf, which shares its storage, so writing the step into the wrap writes it into the leaf.
# `_leaf_like(xᵢ, yᵢ)` wraps `yᵢ` exactly when `xᵢ` is wrapped, which keeps a leaf's gradient and its
# parameters in one shape.
_leaf_like(::Union{AbstractVector, Manifold}, y) = y
_leaf_like(_, y) = NetworkParameters((leaf = y,))
_leaf_solution(x) = _leaf_like(x, x)

# The method for every leaf, converted to the element type of the set. Asked once, here, so that a
# selector function is not called at every step and cannot answer differently on a later one.
function _leaf_methods(method::CompositeMethod, x::NetworkParameters{T}) where {T}
    mapparameters(leaf -> change_precision(T, leafmethod(method, leaf)), params(x))
end

# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
"""
    CompositeCache <: OptimizerCache

The cache of a [`CompositeMethod`](@ref) on a parameter set: the method chosen for every leaf and
that method's cache for the leaf, each in a tree of the shape of the set.

# Fields
- `methods`: the method of each leaf, converted to the element type of the set,
- `caches`: the cache of each leaf, built by `OptimizerCache(method, leaf)`.
"""
struct CompositeCache{T, MT, CT} <: OptimizerCache{T}
    methods::MT
    caches::CT
end

function OptimizerCache(method::CompositeMethod, x::NetworkParameters{T}) where {T}
    methods = _leaf_methods(method, x)
    caches = mapparameters(
        (leaf, m) -> OptimizerCache(m, _leaf_solution(leaf)), params(x), methods)
    CompositeCache{T, typeof(methods), typeof(caches)}(methods, caches)
end

# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
"""
    CompositeState <: OptimizerState

The state of a [`CompositeMethod`](@ref) on a parameter set: the state of every leaf, for the method
chosen for it, in a tree of the shape of the set, and the iteration number the step size is read at.

Every leaf state counts its own iterations as well, since the bias correction of an
[`AdamFamily`](@ref) method reads it; all of them advance with the composite's at every
[`optimization_step!`](@ref).

# Examples

```jldoctest; setup = :(using GeometricOptimizers)
ps = NetworkParameters((weight = rand(StiefelManifold, 4, 2), bias = zeros(4)))
composite = CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam())
state = OptimizerState(composite, ps)
(state.states.weight isa ScalarMomentAdamState, state.states.bias isa AdamState)

# output

(true, true)
```
"""
mutable struct CompositeState{T, ST} <: OptimizerState{T}
    states::ST
    iterations::Int
end

function OptimizerState(method::CompositeMethod, x::NetworkParameters{T}) where {T}
    states = mapparameters((leaf, m) -> OptimizerState(m, _leaf_solution(leaf)), params(x),
        _leaf_methods(method, x))
    CompositeState{T, typeof(states)}(states, 0)
end

# On a single leaf a composite is the method it chooses for it, and the optimizer keeps that method.
# On a parameter set it keeps the composite, whose leaves `OptimizerCache` converts one by one.
function _training_method(::Type{T}, method::CompositeMethod,
        x::Union{AbstractVector, Manifold}) where {T}
    change_precision(T, leafmethod(method, x))
end
_training_method(::Type, method::CompositeMethod, ::NetworkParameters) = method

function _training_workspace(::CompositeMethod, x::NetworkParameters, retraction)
    mapparameters(leaf -> retraction_workspace(_leaf_solution(leaf), retraction), params(x))
end

function optimization_step!(x::NetworkParameters{T},
        opt::TrainingOptimizer{<:CompositeMethod, <:CompositeCache, <:CompositeState{T}},
        dp::NetworkParameters{T}) where {T}
    # before the count moves, so that a refused step leaves the state as it was
    _same_shape(x, dp) || throw(DimensionMismatch(
        "the gradient does not have the shape of the parameters it is a gradient of"))
    increase_iteration_number!(opt.state)
    α = step_size(opt.linesearch, iteration_number(opt.state))
    retraction = opt.retraction
    foreachparameters(
        opt.cache.caches, opt.state.states, opt.cache.methods, opt.workspace, x,
        dp) do cache, state, method, workspace, xᵢ, dpᵢ
        x̂ = _leaf_solution(xᵢ)
        increase_iteration_number!(state)
        _training_step!(
            x̂, cache, state, PrecomputedGradient(x̂, _leaf_like(xᵢ, dpᵢ)), method, α,
            retraction, workspace)
    end
    x
end

# `solve!` searches one step length along the direction of the whole set, which a composite does not
# have: its leaves take their steps one at a time. Refused where `Optimizer` first looks at the
# method, so that the mistake is named instead of being a `MethodError` below `OptimizerCache`.
function _check_method(::CompositeMethod, x)
    throw(ArgumentError(
        "`Optimizer` and `solve!` take no CompositeMethod, which steps one leaf at a time and has " *
        "no direction of the whole set for a line search; use `TrainingOptimizer(x; algorithm)`, " *
        "or the method the composite selects for a single leaf"))
end
