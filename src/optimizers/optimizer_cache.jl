"""
    OptimizerCache

See e.g. [`NewtonOptimizerCache`](@ref) and [`QuasiNewtonCache`](@ref).

# Extended help

!!! todo
    `OptimizerCache`s are only used during [`solver_step!`](@ref)s. Outside of these, [`OptimizerState`](@ref)s carry information between different iterations. This may still have to be enforced consistently.
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

[`OptimizerStatus`](@ref) reports its norm as `rg`, through [`latest_gradient_norm`](@ref). It is
deliberately *not* [`gradient`](@ref): the latter is ``\nabla{}f(x_k)`` at the iterate the step is built from, which
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
    latest_gradient_norm(cache)

The Riemannian norm ``\sqrt{g_x(\Delta, \Delta)}`` of the gradient ``\Delta`` at the iterate `cache`
holds, as [`refresh_latest_gradient!`](@ref) last stored it: the Euclidean norm on a vector, the norm
of [`metric`](@ref) on a [`Manifold`](@ref), and the two combined in quadrature over the leaves of a
parameter set. [`OptimizerStatus`](@ref) reports it as `rg`, and the gradient stop reads it.

It is the same number for every method. [`latest_gradient`](@ref) holds the gradient in the form the
method takes it, see [`GeometricOptimizers.gradient_rep!`](@ref), and on a
[`SymplecticStiefelManifold`](@ref) those forms have different norms, so a norm of that array would
stop each method at a different point. On a vector, a [`StiefelManifold`](@ref) and a
[`GrassmannManifold`](@ref) every form is [`global_rep`](@ref) of ``\Delta``, whose norm over its free
parameters is the Riemannian one.
"""
latest_gradient_norm(cache::OptimizerCache) = cache.g̃_norm[]

@doc raw"""
    refresh_latest_gradient!(cache, gradient_instance, method)

Evaluate the gradient at the iterate `cache` currently holds and store it in
[`latest_gradient`](@ref), in the form `method` holds it, and its Riemannian norm in
[`latest_gradient_norm`](@ref); see [`GeometricOptimizers.gradient_rep!`](@ref).

[`solver_step!`](@ref) calls this once the accepted step has been taken, so that `rg` is a statement
about the point the solve is about to report rather than about the one it started the step from.

It is also what establishes the pairing [`store_gradient!`](@ref) relies on. See
[`latest_gradient`](@ref).
"""
function refresh_latest_gradient!(cache::OptimizerCache, g::Gradient, method::OptimizerMethod)
    _refresh_latest_gradient!(solution(cache), cache, g, method)
    cache.g̃_is_current[] = true

    cache
end

# `_refresh_latest_gradient!` evaluates the same expression `update!(::GradientCache, ...)` builds
# `cache.g` from: `section(cache)` and `solution(cache)` are both at the accepted iterate by the time
# `solver_step!` gets here.
#
# It splits on the parameters for the same reason `trial_slope` does, and the split is the same one:
# `global_rep` maps an ambient gradient to the horizontal lift on a `Manifold`, and on a plain array
# it is the identity (`global_rep(::GlobalSection{T}, gx::AbstractVecOrMat{T}) = gx`), so there the gradient can
# go straight into `latest_gradient` and the allocation the manifold branch needs is pure waste.
function _refresh_latest_gradient!(::AbstractVector, cache::OptimizerCache, g::Gradient,
        ::OptimizerMethod)
    g(latest_gradient(cache), solution(cache))
    cache.g̃_norm[] = l2norm(latest_gradient(cache))
end

function _refresh_latest_gradient!(
        ::Union{Manifold, NetworkParameters}, cache::OptimizerCache,
        g::Gradient, method::OptimizerMethod)
    Δ = g(solution(cache))
    gradient_rep!(latest_gradient(cache), section(cache), Δ, method)
    cache.g̃_norm[] = _riemannian_norm(solution(cache), Δ)
end

# `√g_x(Δ, Δ)` of a Riemannian gradient `Δ` at `x`. A leaf of a parameter set that is not a manifold
# is a vector space, where the gradient is Euclidean.
_riemannian_norm(::AbstractArray, Δ) = l2norm(Δ)
_riemannian_norm(x::Manifold, Δ) = √metric(x, Δ, Δ)

function _riemannian_norm(ps::NetworkParameters{T}, Δ::NetworkParameters{T}) where {T}
    sumsq = Ref(zero(T))
    foreachparameters((x, d) -> (sumsq[] += _riemannian_norm(x, d)^2), ps, Δ)
    √sumsq[]
end

# The gradient each method holds; see `gradient_rep!`. Every method without a `descent` holds the
# derivative with respect to the stored entries, which its algebra pairs under `_dot`.
gradient_rep!(D, λY, Δ, ::OptimizerMethod) = gradient_rep!(D, λY, Δ)

function gradient_rep!(D, λY, Δ, method::Union{GradientMethod, MomentumMethod, Adam})
    _descent_rep!(D, λY, Δ, method.descent)
end

_descent_rep!(D, λY, Δ, ::MetricDescent) = global_rep!(D, λY, Δ)
_descent_rep!(D, λY, Δ, ::SectionDescent) = _section_sharp!(gradient_rep!(D, λY, Δ))

@doc raw"""
    latest_gradient_is_current(cache, state, x)

Whether [`latest_gradient`](@ref) already holds ``\mathrm{global\_rep}(\mathrm{section}(state),
\nabla{}f(x))``, i.e. exactly what [`store_gradient!`](@ref) would otherwise evaluate.
"""
function latest_gradient_is_current(cache::OptimizerCache, state::OptimizerState, x::OptimizerSolution)
    # `g̃_is_current` says the pairing "`latest_gradient` is `∇f` at `solution(cache)`, in the frame
    # of `section(cache)`" was established -- only `refresh_latest_gradient!` sets it and only
    # `store_gradient!` clears it, so every intermediate move of `solution(cache)`, in
    # `solver_step!`'s `NaN` loop and throughout the line search, is covered. The two comparisons
    # then say that the pairing is about the `x` and the frame *this* `update!` is being asked for,
    # which is what makes a caller that moves the iterate between steps fall back to a fresh
    # evaluation instead of silently reusing a stale one. Both are `O(n)` against a gradient
    # evaluation that is not, and `update_section!` on a manifold is `O(N³)` where the comparison is
    # `O(N²)`.
    #
    # The flag is not redundant with the comparisons: on Euclidean parameters the cache's and the
    # state's sections both start life as a copy of `x₀` with `λ = nothing`, so before the first
    # step they compare *equal*, and without the flag the `NaN`-filled scratch would be reused.
    cache.g̃_is_current[] && solution(cache) == x && section(cache) == section(state)
end

"""
    invalidate_latest_gradient!(cache)

Declare that [`latest_gradient`](@ref) is no longer the gradient at `solution(cache)`.
"""
invalidate_latest_gradient!(cache::OptimizerCache) = (cache.g̃_is_current[] = false; cache)

@doc raw"""
    store_gradient!(cache, state, gradient_instance, x, method)

Put the gradient at `x` in the form `method` holds it,
[`GeometricOptimizers.gradient_rep!`](@ref)`(·, section(state), ∇f(x), method)`, into
`gradient_array(cache)`, which is what the `update!(cache, ...)` methods build their direction from.

# Implementation

This reuses [`latest_gradient`](@ref) when [`latest_gradient_is_current`](@ref) says it already holds
that value, which in a [`solve!`](@ref) loop is every iteration but the first: `solver_step!` refreshes
it at the accepted iterate, and the next `update!` is asked for the gradient at that same iterate in
the same frame. The two are not merely close, they are the same values -- for the first-order
states, `update!(state, opt, x)` copies `section(cache)` into the state through
[`advance_state!`](@ref), so `section(state)` after it is bit-for-bit `section(cache)` after
`solver_step!`.

Without the reuse the refresh doubles the gradient evaluations of a first-order step. With the reuse,
the refresh *is* the step's gradient evaluation.

!!! warning "This must run before the cache's `section` and `solution` are overwritten"
    `latest_gradient_is_current` compares `solution(cache)` against `x` and `section(cache)` against
    `section(state)`, so calling it after `update!`'s `_copyto!(section(cache), section(state))` would
    compare each with itself and report `true` unconditionally.
"""
function store_gradient!(cache::OptimizerCache, state::OptimizerState, g::Gradient,
        x::OptimizerSolution, method::OptimizerMethod)
    if latest_gradient_is_current(cache, state, x)
        _copyto!(gradient_array(cache), latest_gradient(cache))
    else
        # `gradient_rep!` into the cache's own array, rather than into a fresh lift: the
        # destination already has the shape, so a gradient the step builds costs the products and
        # nothing else. On a vector space the write is a copy.
        gradient_rep!(gradient_array(cache), section(state), g(x), method)
    end
    # consumed: from here to the next `refresh_latest_gradient!`, `solver_step!` and the line search
    # move `solution(cache)` around and nothing keeps `latest_gradient` in step with it
    invalidate_latest_gradient!(cache)

    cache
end
