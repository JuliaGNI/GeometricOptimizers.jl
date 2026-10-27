# The training optimizer of a [`CompositeMethod`](@ref) on a parameter set. The type is in
# `optimizers/optimizer_methods.jl`, beside the other methods; this file needs the first-order
# caches and states and `TrainingOptimizer`, so it is included after them.
#
# The composite optimizer holds the cache, the state and the buffers of every leaf in
# trees of the shape of the set. Its step takes, at every leaf, the step of a `TrainingOptimizer`
# built from that leaf's method, cache, state and buffers, so the leaves share the code of the
# ordinary step and its step-size schedule, and nothing else.

# What a leaf is stepped as. A vector and a manifold element are `OptimizerSolution`s of their own, and
# `ScalarMomentAdam` takes a `StiefelManifold` and nothing else, so those stay bare. Every other leaf
# -- a matrix, a `SymmetricMatrix` -- is wrapped in a parameter set of one leaf, which shares its
# storage, so writing the step into the wrap writes it into the leaf. `_leaf_like(xᵢ, yᵢ)` wraps `yᵢ`
# exactly when `xᵢ` is wrapped, which keeps a leaf's gradient and its parameters in one shape.
_leaf_like(::Union{AbstractVector, Manifold}, y) = y
_leaf_like(_, y) = NetworkParameters((leaf = y,))
_leaf_solution(x) = _leaf_like(x, x)

function _training_optimizer(
        x::NetworkParameters, method::CompositeMethod, linesearch, retraction, observer)
    leaves(f) = mapparameters(xᵢ -> f(_leaf_solution(xᵢ), leafmethod(method, xᵢ)), params(x))
    # every cache first and every state second, the order of one method on the whole set, so that
    # the random completions of the sections are the same draws
    cache = leaves((xᵢ, mᵢ) -> OptimizerCache(mᵢ, xᵢ))
    state = leaves((xᵢ, mᵢ) -> OptimizerState(mᵢ, xᵢ))
    workspace = leaves((xᵢ, mᵢ) -> _training_workspace(xᵢ, mᵢ, retraction))
    gradient_buffer = leaves((xᵢ, mᵢ) -> _gradient_buffer(xᵢ, mᵢ))
    TrainingOptimizer(
        method, cache, state, linesearch, retraction, workspace, gradient_buffer, observer)
end

function optimization_step!(
        x::NetworkParameters{T}, opt::TrainingOptimizer{<:CompositeMethod},
        dp::NetworkParameters{T}) where {T}
    # the whole set first, so that a refused step moves no leaf
    _check_shape(x, dp)
    foreachparameters((xᵢ, dpᵢ) -> _check_gradient(leafmethod(opt.method, xᵢ), dpᵢ), x, dp)
    # `opt.cache` leads the walk: a wrapped leaf's workspace and gradient buffer are trees of one
    # leaf, handed over whole
    foreachparameters(opt.cache, opt.state, opt.workspace, opt.gradient_buffer, x,
        dp) do cacheᵢ, stateᵢ, wsᵢ, bufferᵢ, xᵢ, dpᵢ
        leaf = TrainingOptimizer(
            leafmethod(opt.method, xᵢ), cacheᵢ, stateᵢ, opt.linesearch,
            opt.retraction, wsᵢ, bufferᵢ, opt.observer)
        optimization_step!(_leaf_solution(xᵢ), leaf, _leaf_like(xᵢ, dpᵢ))
    end
    x
end
