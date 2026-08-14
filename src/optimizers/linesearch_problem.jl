@doc raw"""
    trial_iterate!(cache, params, α, retraction)

Write the iterate a step of length `α` along the current direction would produce into
`solution(cache)`, and return it. This is what the merit function of the line search evaluates.

# Implementation

There are two methods, chosen on the type of the solution.

For an `AbstractVector` the iterate is ``x_k + \alpha{}p_k``, i.e.
[`SimpleSolvers.compute_new_iterate!`](@extref), which is what this used to do unconditionally.

For a [`Manifold`](@ref) — or a `NamedTuple` that contains one — that is both undefined and wrong:
adding ``\alpha{}p_k`` to a point of ``St(N, n)`` leaves the manifold, and the direction is an
[`AbstractLieAlgHorMatrix`](@ref) of a different shape than the point to begin with. The step has to
go through the `retraction`, exactly as [`solver_step!`](@ref) does once the line search has picked
its `α`, so this reproduces that: scale the direction, push the *state's* section through the
retraction into the cache's section, and read the point back out.

!!! info "Which section is which"
    `section(params.state)` is the base and has to survive unchanged across the trial steps of one
    line search; `section(cache)` is scratch. `solver_step!` uses the same pair the same way, both in
    its `NaN` loop and for the accepted step, so the line search leaves nothing behind that it does
    not overwrite itself.
"""
trial_iterate!(cache::OptimizerCache, params, α, retraction) =
    _trial_iterate!(solution(cache), cache, params, α, retraction)

function _trial_iterate!(::AbstractVector, cache::OptimizerCache, params, α, ::AbstractRetraction)
    compute_new_iterate!(solution(cache), params.x, α, direction(cache))
end

@noinline _no_state_error() =
    error("a trial step on a manifold retracts from `section(params.state)`, so the line search " *
          "parameters have to carry the `state`; `solver_step!` passes it, a bare `(x = x,)` does not.")

function _trial_iterate!(::Union{Manifold,ArrayNamedTuple}, cache::OptimizerCache, params, α, retraction)
    # `params` is a concrete `NamedTuple` here, so this is constant-folded away rather than checked on
    # every merit evaluation. Without it a missing `state` surfaces as `has no field state`.
    hasproperty(params, :state) || _no_state_error()
    # `_mul` allocates a scaled copy because `direction(cache)` has to stay intact for the next trial
    # step; `solver_step!` can afford the in-place `_rmul!` only because it scales exactly once, by
    # the `α` the line search has already settled on.
    update_section!(section(cache), section(params.state), _mul(α, direction(cache)), retraction)
    _copyto!(solution(cache), section(cache))
end

@doc raw"""
    trial_slope(gradient_instance, cache, retraction, α)

Return ``\varphi'(\alpha) = \langle\nabla{}f, p\rangle`` at the iterate currently held in
`solution(cache)`, for the derivative of the line search's merit.

# Implementation

On a [`Manifold`](@ref) the gradient that `gradient_instance` produces lives in the tangent space of
the embedding while `direction(cache)` is a horizontal lift, so the two are not even the same shape
and have to be brought together by [`global_rep`](@ref) first — which is the same map
`update!(::GradientCache, …)` applies to the gradient before storing it.

The pairing is then [`_dot`](@ref) and not `dot`: the `α` of the line search parameterizes a curve in
the *intrinsic* coordinates of the lift, and `dot` on a lift is the ambient Frobenius product, which
is exactly twice that. With `dot` this returned `2\varphi'(\alpha)` — invisible to
[`SimpleSolvers.Bisection`](@extref), which only looks for a sign change, and swamped by the Armijo
slack in [`SimpleSolvers.Backtracking`](@extref), but wrong, and wrong in a way a curvature condition
or a quadratic fit would act on.

The direction the gradient is paired *with* is [`retraction_differential`](@ref)`(retraction, B, α)`
and not `B` itself. ``\varphi'(\alpha) = \langle\nabla{}f(x(\alpha)), B\rangle`` holds only where
``\alpha \mapsto \mathrm{retract}(\alpha{}B)`` is a one-parameter subgroup, which [`Geodesic`](@ref)
is and [`Cayley`](@ref) is not; under `Cayley` the generator of the curve's velocity turns with the
step, and the differential is what supplies it. Against a central difference of the merit this is now
exact for both retractions at every ``\alpha``.

!!! info "This changed in 0.2.0"
    The slope used to be paired against `B` under either retraction, so under `Cayley` it was exact
    at ``\alpha = 0`` and drifted with the step — on a `St(6, 3)` problem, 8.9% out at
    ``\alpha = 0.5``, 36% at ``\alpha = 1`` and 143% at ``\alpha = 2``, against a central difference
    of the merit. Searches that use ``\varphi'`` only qualitatively absorbed that: `Bisection` looks
    for a sign change, [`SimpleSolvers.StrongWolfe`](@extref) compares against ``\varphi'(0)``, and
    `Backtracking` — the default — evaluates ``\varphi'`` at ``\alpha = 0`` only, where the two agree
    exactly. The two polynomial searches fit a curve to it *quantitatively*, and on the SVD problem of
    `test/optimizer_convergence/svd_optim.jl` that took `_BFGS` off the manifold altogether on two of
    eight starting points. See the CHANGELOG entry for issue A1b.

    `α = 0` still returns `B` untouched, so the `Backtracking` default costs nothing for this.
"""
trial_slope(gradient_instance::Gradient, cache::OptimizerCache, retraction, α) =
    _trial_slope(solution(cache), gradient_instance, cache, retraction, α)

# These two differ in one respect worth knowing about: the `AbstractVector` method evaluates *into* an
# array of the cache — that is what makes it allocation-free — and so leaves the gradient at the last
# trial `α` there, while the manifold method allocates and leaves the cache alone.
#
# Which array it writes into is `latest_gradient(cache)` and deliberately not `gradient(cache)`. On
# the quasi-Newton caches the two are the same array, which is what this did unconditionally and is
# harmless there: `BFGSCache` and `DFPCache` advance `state.ḡ` and form `cache.Δg` inside
# `update!(cache, ...)`, i.e. before the line search runs, so nothing downstream reads `cache.g`
# again. On the three first-order caches they are not, because `update!(::MomentumState, ...)` re-runs
# `p ← αp + ∇f(xₖ)` from `gradient_array(cache)` *after* the search — so a shared array made the
# momentum accumulate the gradient at whatever trial step the search last probed. Measured over eight
# iterations, the state's momentum was 104% wrong under `Bisection`, `Quadratic` and `StrongWolfe` and
# 451% wrong under `BierlaireQuadratic`; `Backtracking` was exact only because it evaluates `φ'` once,
# at `α = 0`, where the trial gradient *is* `∇f(xₖ)`. See the CHANGELOG entry for issue A2.
function _trial_slope(::AbstractVector, gradient_instance::Gradient, cache::OptimizerCache, ::AbstractRetraction, α)
    gradient_instance(latest_gradient(cache), solution(cache))
    _dot(latest_gradient(cache), direction(cache))
end

function _trial_slope(::Union{Manifold,ArrayNamedTuple}, gradient_instance::Gradient, cache::OptimizerCache, retraction::AbstractRetraction, α)
    _dot(global_rep(section(cache), gradient_instance(solution(cache))),
        retraction_differential(retraction, direction(cache), α))
end

@doc raw"""
    linesearch_problem(problem, gradient, cache, retraction)

Create a [`SimpleSolvers.LinesearchProblem`](@extref) for the linesearch algorithm.

The variable on which this problem depends is ``\alpha``. The trial iterate is built by
[`trial_iterate!`](@ref), which is what makes this work for manifold parameters and not only for
`AbstractVector`s.

# Example

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: NewtonOptimizerCache, linesearch_problem, update!, Cayley)
julia> x = [1, 0., 0.]
3-element Vector{Float64}:
 1.0
 0.0
 0.0

julia> f = x -> sum(x .^ 3 / 6 + x .^ 2 / 2);

julia> obj = OptimizerProblem(f, x);

julia> grad = GradientAutodiff{Float64}(obj.F, length(x));

julia> hess = HessianAutodiff{Float64}(obj.F, length(x));

julia> cache = NewtonOptimizerCache(x);

julia> state = NewtonOptimizerState(x); update!(state, grad, x);

julia> params = (x = state.x, state = state);

julia> update!(cache, state, grad, hess, x);

julia> ls_obj = linesearch_problem(obj, grad, cache, Cayley());

julia> ls_obj.F(0., params)
0.6666666666666666

julia> ls_obj.D(0., params)
-1.125

```

!!! info
    Note that in the example above calling [`update!`](@ref) on the [`NewtonOptimizerCache`](@ref) requires a [`SimpleSolvers.Hessian`](@extref).
"""
function linesearch_problem(problem::OptimizerProblem{T}, gradient_instance::Gradient, cache::OptimizerCache{T}, retraction::AbstractRetraction) where {T}
    function f(α, params)
        trial_iterate!(cache, params, α, retraction)
        value(problem, solution(cache))
    end

    function d(α, params)
        trial_iterate!(cache, params, α, retraction)
        trial_slope(gradient_instance, cache, retraction, α)
    end

    LinesearchProblem{T}(f, d)
end
