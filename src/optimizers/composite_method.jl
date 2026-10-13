# The per-leaf half of [`CompositeMethod`](@ref): the cache and the state of a composite on a
# parameter set, and the training step over them. The type and [`leafmethod`](@ref) are in
# `optimizers/optimizer_methods.jl`, where the method types live; everything below needs the
# first-order caches and states and `TrainingOptimizer`, so it is included after them.
#
# A composite pools nothing. Every leaf gets the cache and the state of its method, and
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

# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
"""
    CompositeCache <: OptimizerCache

The cache of a [`CompositeMethod`](@ref) on a parameter set: in `caches`, the cache of every leaf for
the method [`leafmethod`](@ref) gives for it, in a tree of the shape of the set.
"""
struct CompositeCache{T, CT} <: OptimizerCache{T}
    caches::CT
end

function OptimizerCache(method::CompositeMethod, x::NetworkParameters{T}) where {T}
    caches = mapparameters(
        leaf -> OptimizerCache(leafmethod(method, leaf), _leaf_solution(leaf)), params(x))
    CompositeCache{T, typeof(caches)}(caches)
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
    states = mapparameters(
        leaf -> OptimizerState(leafmethod(method, leaf), _leaf_solution(leaf)), params(x))
    CompositeState{T, typeof(states)}(states, 0)
end

# On a single leaf a composite is the method it gives for that leaf, and the optimizer keeps that
# method. On a parameter set it keeps the composite, converted as a whole by `change_precision`.
function _training_method(::Type{T}, method::CompositeMethod,
        x::Union{AbstractVector, Manifold}) where {T}
    change_precision(T, leafmethod(method, x))
end

function _training_workspace(::CompositeMethod, x::NetworkParameters, retraction)
    mapparameters(leaf -> retraction_workspace(_leaf_solution(leaf), retraction), params(x))
end

# The step of a composite on a parameter set: every leaf with its own method, cache and state.
function _training_step!(
        x::NetworkParameters, cache::CompositeCache, state::CompositeState, dp,
        method::CompositeMethod, α, retraction, workspace)
    foreachparameters(cache.caches, state.states, workspace, x,
        dp) do cacheᵢ, stateᵢ, workspaceᵢ, xᵢ, dpᵢ
        increase_iteration_number!(stateᵢ)
        _training_step!(
            _leaf_solution(xᵢ), cacheᵢ, stateᵢ, _leaf_like(xᵢ, dpᵢ), leafmethod(method, xᵢ), α,
            retraction, workspaceᵢ)
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
        "or the composite's method for a single leaf"))
end
