"""
    OptimizerCache

See e.g. [`NewtonOptimizerCache`](@ref) and [`QuasiNewtonCache`](@ref).

# Extended help

!!! todo
    `OptimizerCache`s are only used during [`solver_step!`](@ref)s. Outside of these, [`OptimizerState`](@ref)s are used to communicate information between different iterations. This may still have to be enforced consistently.
"""
abstract type OptimizerCache{T} end

# The accessors every cache in this package answers, on the fields they all name alike. `direction`
# and `rhs` are not among them: the caches store those under different names.
solution(cache::OptimizerCache) = cache.x

"""
    gradient(cache)

Return the stored gradient (array) of an [`OptimizerCache`](@ref): ``\\nabla{}f(x_k)`` at the iterate
the step is built from.
"""
gradient(cache::OptimizerCache) = cache.g
section(cache::OptimizerCache) = cache.section
# the array `store_gradient!` writes the gradient the direction is built from into
gradient_array(cache::OptimizerCache) = gradient(cache)

@doc raw"""
    latest_gradient(cache)

The array holding the *most recently evaluated* gradient: a line search trial point while
[`linesearch_problem`](@ref)'s ``\varphi'`` is being evaluated, and the accepted iterate once
[`solver_step!`](@ref) has called [`refresh_latest_gradient!`](@ref) on it.

This is the array [`OptimizerStatus`](@ref) reports as `rg`, and it is deliberately *not*
[`gradient`](@ref): the latter is ``\nabla{}f(x_k)`` at the iterate the step is built from, which
the direction and the state updates need to keep reading unchanged.

# Implementation

Every cache in this package carries a scratch array of its own for this, the field `g̃`, and a flag
`g̃_is_current` for the pairing [`store_gradient!`](@ref) relies on. The defaults of this method and
of the three that go with it ([`refresh_latest_gradient!`](@ref), [`latest_gradient_is_current`](@ref),
[`invalidate_latest_gradient!`](@ref)) read those two fields; see [`GradientCache`](@ref) for why `g̃`
may not be shared with `gradient`.
"""
latest_gradient(cache::OptimizerCache) = cache.g̃

@doc raw"""
    refresh_latest_gradient!(cache, gradient_instance)

Evaluate the gradient at the iterate `cache` currently holds and store it in
[`latest_gradient`](@ref).

[`solver_step!`](@ref) calls this once the accepted step has been taken, so that `rg` is a statement
about the point the solve is about to report rather than about the one it started the step from.

It is also what establishes the pairing [`store_gradient!`](@ref) relies on. See
[`latest_gradient`](@ref).
"""
function refresh_latest_gradient!(cache::OptimizerCache, g::Gradient)
    _refresh_latest_gradient!(solution(cache), cache, g)
    cache.g̃_is_current[] = true

    cache
end

# This is the same expression `update!(::GradientCache, ...)` builds `cache.g` from: `section(cache)` and
# `solution(cache)` are both at the accepted iterate by the time `solver_step!` gets here.
#
# It splits on the parameters for the same reason `trial_slope` does, and the split is the same one:
# `global_rep` maps an ambient gradient to the horizontal lift on a `Manifold`, and on a plain array
# it is the identity (`global_rep(::GlobalSection{T}, gx::AbstractVecOrMat{T}) = gx`), so there the gradient can
# go straight into `latest_gradient` and the allocation the manifold branch needs is pure waste. At
# `n = 500` that is 4 160 bytes an iteration against none.
function _refresh_latest_gradient!(::AbstractVector, cache::OptimizerCache, g::Gradient)
    g(latest_gradient(cache), solution(cache))
end

function _refresh_latest_gradient!(::Union{Manifold, NetworkParameters}, cache::OptimizerCache, g::Gradient)
    _copyto!(latest_gradient(cache), global_rep(section(cache), g(solution(cache))))
end

@doc raw"""
    latest_gradient_is_current(cache, state, x)

Whether [`latest_gradient`](@ref) already holds ``\mathrm{global\_rep}(\mathrm{section}(state),
\nabla{}f(x))``, i.e. exactly what [`store_gradient!`](@ref) would otherwise evaluate.
"""
function latest_gradient_is_current(cache::OptimizerCache, state::OptimizerState, x::OptimizerSolution)
    cache.g̃_is_current[] && solution(cache) == x && section(cache) == section(state)
end

# `g̃_is_current` says the pairing "`latest_gradient`
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

"""
    invalidate_latest_gradient!(cache)

Declare that [`latest_gradient`](@ref) is no longer the gradient at `solution(cache)`.
"""
invalidate_latest_gradient!(cache::OptimizerCache) = (cache.g̃_is_current[] = false; cache)

@doc raw"""
    store_gradient!(cache, state, gradient_instance, x)

Put ``\mathrm{global\_rep}(\mathrm{section}(state), \nabla{}f(x))`` into `gradient_array(cache)`,
which is what the three first-order `update!(cache, ...)` methods build their direction from.

# Implementation

This reuses [`latest_gradient`](@ref) when [`latest_gradient_is_current`](@ref) says it already holds
that value, which in a [`solve!`](@ref) loop is every iteration but the first: `solver_step!` refreshes
it at the accepted iterate, and the next `update!` is asked for the gradient at that same iterate in
the same frame. The two are not merely close, they are the same values -- for the first-order
states, `update!(state, opt, x)` copies `section(cache)` into the state through
[`advance_state!`](@ref), so `section(state)` after it is bit-for-bit `section(cache)` after
`solver_step!`.

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
