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

Every cache in this package carries a scratch array of its own for this and implements the three
methods that go with it ([`refresh_latest_gradient!`](@ref), [`latest_gradient_is_current`](@ref),
[`invalidate_latest_gradient!`](@ref)); see [`GradientCache`](@ref) for the field and for why it may
not be shared with `gradient`.

The default here aliases the two together, which is what all six caches did before there was a name
for the distinction, and is what a cache that does not refresh anything should keep doing: nothing
sets the pairing [`store_gradient!`](@ref) relies on, [`latest_gradient_is_current`](@ref) defaults to
`false`, and `rg` then means ``\|\nabla{}f(x_k)\|`` at the iterate the step started from.
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

It is also what establishes the pairing [`store_gradient!`](@ref) relies on, so a cache that
implements this has to implement [`latest_gradient_is_current`](@ref) with it.
"""
refresh_latest_gradient!(cache::OptimizerCache, ::Gradient) = cache

# The shared body of the three first-order methods' `refresh_latest_gradient!`. This is the same
# expression `update!(::GradientCache, ...)` builds `cache.g` from: `section(cache)` and
# `solution(cache)` are both at the accepted iterate by the time `solver_step!` gets here.
#
# It splits on the parameters for the same reason `trial_slope` does, and the split is the same one:
# `global_rep` maps an ambient gradient to the horizontal lift on a `Manifold`, and on a plain array
# it is the identity (`global_rep(::GlobalSection{T,AT,Nothing}, gx) = gx`), so there the gradient can
# go straight into `latest_gradient` and the allocation the manifold branch needs is pure waste. At
# `n = 500` that is 4 160 bytes an iteration against none.
function _refresh_latest_gradient!(cache::OptimizerCache, g::Gradient)
    _refresh_latest_gradient!(solution(cache), cache, g)
    cache.g̃_is_current[] = true

    cache
end

_refresh_latest_gradient!(::AbstractVector, cache::OptimizerCache, g::Gradient) =
    g(latest_gradient(cache), solution(cache))

_refresh_latest_gradient!(::Union{Manifold,ArrayNamedTuple}, cache::OptimizerCache, g::Gradient) =
    _copyto!(latest_gradient(cache), global_rep(section(cache), g(solution(cache))))

@doc raw"""
    latest_gradient_is_current(cache, state, x)

Whether [`latest_gradient`](@ref) already holds ``\mathrm{global\_rep}(\mathrm{section}(state),
\nabla{}f(x))``, i.e. exactly what [`store_gradient!`](@ref) would otherwise evaluate.

The default is `false`: a cache that never refreshes `latest_gradient` has nothing to reuse.
"""
latest_gradient_is_current(::OptimizerCache, ::OptimizerState, ::OptimizerSolution) = false

# The shared body for the three first-order caches. `g̃_is_current` says the pairing "`latest_gradient`
# is `∇f` at `solution(cache)`, in the frame of `section(cache)`" was established -- only
# `refresh_latest_gradient!` sets it and only `store_gradient!` clears it, so every intermediate move
# of `solution(cache)`, in `solver_step!`'s `NaN` loop and throughout the line search, is covered. The
# two comparisons then say that the pairing is about the `x` and the frame *this* `update!` is being
# asked for, which is what makes a caller that moves the iterate between steps fall back to a fresh
# evaluation instead of silently reusing a stale one. Both are `O(n)` against a gradient evaluation
# that is not, and `update_section!` on a manifold is `O(N³)` where the comparison is `O(N²)`.
#
# The flag is not redundant with the comparisons: on Euclidean parameters the cache's and the state's
# sections both start life as a copy of `x₀` with `λ = nothing`, so before the first step they compare
# *equal*, and without the flag the `NaN`-filled scratch would be reused.
_latest_gradient_is_current(cache::OptimizerCache, state::OptimizerState, x::OptimizerSolution) =
    cache.g̃_is_current[] && solution(cache) == x && section(cache) == section(state)

"""
    invalidate_latest_gradient!(cache)

Declare that [`latest_gradient`](@ref) is no longer the gradient at `solution(cache)`.

The default is a no-op, for the caches that never claim the pairing in the first place.
"""
invalidate_latest_gradient!(cache::OptimizerCache) = cache

_invalidate_latest_gradient!(cache::OptimizerCache) = (cache.g̃_is_current[] = false; cache)

@doc raw"""
    store_gradient!(cache, state, gradient_instance, x)

Put ``\mathrm{global\_rep}(\mathrm{section}(state), \nabla{}f(x))`` into `gradient_array(cache)`,
which is what the three first-order `update!(cache, ...)` methods build their direction from.

# Implementation

This reuses [`latest_gradient`](@ref) when [`latest_gradient_is_current`](@ref) says it already holds
that value, which in a [`solve!`](@ref) loop is every iteration but the first: `solver_step!` refreshes
it at the accepted iterate, and the next `update!` is asked for the gradient at that same iterate in
the same frame. The two are not merely close, they are the same computation --
`update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B, retraction)` has the body the two-argument form
`update!(::MomentumState, ...)` uses, so `section(cache)` after `solver_step!` is bit-for-bit
`section(state)` after `update!(state, opt, x)`.

Without the reuse the refresh doubles the gradient evaluations of a first-order step: on the SVD
problem of `test/optimizer_convergence/svd_optim.jl`, `Adam` + `Static` over 2 000 iterations costs
124 ms without the refresh, 167 ms with it and 128 ms with it reused — one trajectory throughout. With
the reuse, the refresh *is* the step's gradient evaluation.

!!! warning "This must run before the cache's `section` and `solution` are overwritten"
    `latest_gradient_is_current` compares `solution(cache)` against `x` and `section(cache)` against
    `section(state)`, so calling it after `update!`'s `_copyto!(section(cache), section(state))` would
    compare each with itself and report `true` unconditionally.
"""
function store_gradient!(cache::OptimizerCache, state::OptimizerState, g::Gradient, x::OptimizerSolution)
    if latest_gradient_is_current(cache, state, x)
        _copyto!(gradient_array(cache), latest_gradient(cache))
    else
        _copyto!(gradient_array(cache), global_rep(section(state), g(x)))
    end
    # consumed: from here until the next `refresh_latest_gradient!`, `solver_step!` and the line
    # search move `solution(cache)` around and nothing keeps `latest_gradient` in step with it
    invalidate_latest_gradient!(cache)

    cache
end

# The shared body of the three first-order caches' `gradient_difference!`. Both gradients are in the
# cache once `solver_step!` has refreshed `latest_gradient`, so unlike the generic method this needs
# no `state.ḡ` -- see `gradient_difference!` for why that matters here.
_latest_gradient_difference!(cache::OptimizerCache) =
    _difference!(cache.Δg, latest_gradient(cache), gradient(cache))
