# The forwarding half of [`CompositeMethod`](@ref). The type and [`leafmethod`](@ref) are in
# `optimizers/optimizer_methods.jl`, where the method types live and where
# [`FirstOrderMethodWithState`](@ref) can name it; everything below needs `OptimizerCache`,
# `OptimizerState` and the three first-order states, so it is included after them.
#
# There is no composite cache and no composite state. Every method here resolves the leaf's method
# and hands the call on unchanged, so a leaf optimized under a composite is bit-for-bit a leaf
# optimized under the method the selector returned for it. That is the property the whole design
# rests on: a composite is a *choice*, and a choice must not also be a second implementation.

function OptimizerCache(method::CompositeMethod, x::OptimizerSolution)
    OptimizerCache(leafmethod(method, x), x)
end

# `x, args...` and not just `x`, so that the gradient-supplying form of `OptimizerState` -- which
# `Adam` and `ScalarMomentAdam` both have -- forwards too. Without it that call reached the
# `OptimizerState(::OptimizerMethod, args...)` fallback and was told that `OptimizerState` is not
# implemented for `CompositeMethod`, which would be untrue of every leaf it selects for.
OptimizerState(method::CompositeMethod, x, args...) = OptimizerState(leafmethod(method, x), x, args...)

# Every method a composite may select is a `FirstOrderMethodWithState`, and every one of those
# answers `NoHessian`, so this is that answer rather than a forward: forwarding would need the
# `Union{Callable, OptimizerProblem}` shape `Newton` uses and would be ambiguous against it.
Hessian(::CompositeMethod, ::OptimizerProblem, ::OptimizerSolution{T}) where {T} = NoHessian{T}()

function update!(cache::OptimizerCache{T}, state::OptimizerState{T}, gradient::Gradient{T},
        method::CompositeMethod, x::OptimizerSolution{T}) where {T}
    update!(cache, state, gradient, leafmethod(method, x), x)
end

default_linesearch(::Type{T}, ::CompositeMethod) where {T} = Static(T(DEFAULT_LEARNING_RATE))

@doc raw"""
    sync_state!(state, cache, method)

Copy the per-method carry -- a momentum, a pair of moments -- out of `cache` and into `state`, after
a caller has taken a step through the cache without going through [`solver_step!`](@ref).

This is the half of `update!(state, opt, x)` that is *not* about the iterate. A training loop that
drives a cache directly, as `GeometricMachineLearning`'s does for one leaf of a network per
minibatch, still has to leave the state holding what the next step's `update!` will read back --
otherwise the moments restart from zero on every step and the method silently degrades to its
first-iteration behaviour forever.

The default is a no-op, for the methods that carry nothing. The three that carry something implement
it, and they are the reason this is a function here rather than a chain of `isa` tests at the call
site: which quantity a method carries, and whether the cache's copy is the one to keep, is knowledge
that belongs to the method.

!!! info "`MomentumMethod` recomputes rather than copies"
    `p ← αp + ∇L` is the recursion `update!(::MomentumCache, ...)` anticipated when it formed the
    direction, so the state's momentum is advanced here rather than copied from the cache, which
    never holds the *new* momentum on its own.
"""
sync_state!(state::OptimizerState, ::OptimizerCache, ::OptimizerMethod) = state

function sync_state!(state::AdamState{T}, cache::AdamCache{T}, ::OptimizerMethod) where {T}
    _copyto!(first_moment(state), first_moment(cache))
    _copyto!(second_moment(state), second_moment(cache))
    state
end

# `m₂` is a number here and not an element of `𝔤ʰᵒʳ`, so it is assigned rather than copied into --
# which is what `update!(::ScalarMomentAdamState, ...)` does with it on the `solver_step!` path
# (`state.m₂ = T(_second_moment)`).
function sync_state!(state::ScalarMomentAdamState{T}, cache::ScalarMomentAdamCache{T},
        ::OptimizerMethod) where {T}
    _copyto!(first_moment(state), first_moment(cache))
    state.m₂ = T(second_moment(cache))
    state
end

function sync_state!(state::MomentumState, cache::MomentumCache, method::MomentumMethod)
    _rmul!(momentum(state), method.α)
    _add!(momentum(state), gradient_array(cache))
    state
end

# `MomentumMethod` is the one arm that reads the method object, so it is the one arm that needs the
# composite resolved for it. The other two dispatch on the cache and state alone and reach the
# right body with the `CompositeMethod` in hand -- which is also why there is no general
# `(::OptimizerState, ::OptimizerCache, ::CompositeMethod)` method here: it would be ambiguous
# against each of them.
function sync_state!(state::MomentumState, cache::MomentumCache, method::CompositeMethod)
    sync_state!(state, cache, leafmethod(method, solution(cache)))
end
