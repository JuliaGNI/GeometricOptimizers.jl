"""
    OptimizerCache

See e.g. [`NewtonOptimizerCache`](@ref) and [`BFGSCache`](@ref).

# Extended help

!!! todo
    `OptimizerCache`s are only used during [`solver_step!`](@ref)s. Outside of these, [`OptimizerState`](@ref)s are used to communicate information between different iterations. This may still have to be enforced consistently.
"""
abstract type OptimizerCache{T} end

@doc raw"""
    latest_gradient(cache)

The array holding the *most recently evaluated* gradient: a line search trial point while
[`linesearch_problem`](@ref)'s ``\varphi'`` is being evaluated, and the accepted iterate once
[`solver_step!`](@ref) has called [`refresh_latest_gradient!`](@ref) on it.

This is the array [`OptimizerStatus`](@ref) reports as `rg`, and it is deliberately *not*
[`gradient`](@ref): the latter is ``\nabla{}f(x_k)`` at the iterate the step is built from, which
the direction and the state updates need to keep reading unchanged.

# Implementation

The default aliases the two together, which is what the quasi-Newton caches did before there was a
name for the distinction. Nothing refreshes it for them — `refresh_latest_gradient!` is a no-op on
`NewtonOptimizerCache`, `BFGSCache` and `DFPCache` — so for those `rg` is still ``\|\nabla{}f(x_k)\|``
and every measurement in this package that involves them is unchanged.

The three first-order caches override it with a scratch array of their own; see
[`GradientCache`](@ref).
"""
latest_gradient(cache::OptimizerCache) = gradient(cache)

@doc raw"""
    refresh_latest_gradient!(cache, gradient_instance)

Evaluate the gradient at the iterate `cache` currently holds and store it in
[`latest_gradient`](@ref).

[`solver_step!`](@ref) calls this once the accepted step has been taken, so that `rg` is a statement
about the point the solve is about to report rather than about the one it started the step from. The
default is a no-op; see [`latest_gradient`](@ref) for which caches implement it and why the others
do not.
"""
refresh_latest_gradient!(cache::OptimizerCache, ::Gradient) = cache

# The shared body of the three first-order methods' `refresh_latest_gradient!`. This is the same
# expression `update!(::GradientCache, ...)` builds `cache.g` from, and it covers all three kinds of
# parameters at once: `global_rep` maps an ambient gradient to the horizontal lift on a `Manifold`
# and is the identity on a `GlobalSection{T,AT,Nothing}`, i.e. on a plain array. `section(cache)` and
# `solution(cache)` are both at the accepted iterate by the time `solver_step!` gets here.
_refresh_latest_gradient!(cache::OptimizerCache, g::Gradient) =
    (_copyto!(latest_gradient(cache), global_rep(section(cache), g(solution(cache)))); cache)
