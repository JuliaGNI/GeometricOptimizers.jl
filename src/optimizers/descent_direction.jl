@doc raw"""
    ensure_descent!(cache, method, config)

Replace the direction stored in `cache` by the steepest-descent direction if it does not descend.

A (quasi-)Newton step solves ``H\delta = -\nabla{}f`` and is a descent direction only where ``H`` is
positive definite. Where it is not — an indefinite Hessian near a saddle, or a quasi-Newton ``Q`` that
has lost positive definiteness — ``\delta`` points *uphill*. Since a line search only ever returns a
non-negative step, the iteration then walks to the nearest stationary point, which may be a maximum.

`F(x) = sum(sin.(x) .^ 2)` started from `x = ones(3)` is the smallest example: ``\sin^2`` has second
derivative ``2\cos(2x) = -0.83`` there, so the [`Newton`](@ref) direction ascends and the solve
converges to ``\pi/2``, where `F` is *maximal*. With this safeguard it descends to `0`.

Descent is tested as ``\nabla{}f\cdot\delta < 0``. [`rhs`](@ref) stores ``-\nabla{}f``, so the test is
``\mathrm{rhs}\cdot\delta > 0``; a direction that fails it — including a `NaN` one, for which every
comparison is `false` — is replaced by `rhs`, which always descends. The substitution is reported at
`config.verbosity ≥ 2`.

!!! info "Only the (quasi-)Newton methods are safeguarded"
    [`Adam`](@ref) and [`MomentumMethod`](@ref) build their direction from a moving average, which is
    *allowed* not to descend on an individual step — that is what the momentum term is for — so
    their method, on [`FirstOrderMethodWithState`](@ref), leaves the direction as it is.

!!! info "Why the safeguard is here"
    A line search returns a non-negative step, so it cannot turn an ascent direction into a descent
    one. The safeguard has to act on the direction before the search.

# Examples

```jldoctest; setup = :(using GeometricOptimizers)
F(x) = sum(sin.(x) .^ 2)
x = ones(3)
algorithm = Newton()
state = OptimizerState(algorithm, x)
optimizer = Optimizer(x, F; algorithm = algorithm, linesearch = Bisection())

solve!(x, state, optimizer)
F(x) < 1e-20

# output

true
```
"""
function ensure_descent!(cache::OptimizerCache, method::OptimizerMethod, config::Options)
    δ = direction(cache)
    r = rhs(cache)

    # `_dot`, not `⋅`: on a manifold both operands are horizontal lifts, and `⋅` on those is the
    # ambient Frobenius product. That is twice the intrinsic one, which cannot flip the sign of the
    # test on its own, but it is also taken by scalar-indexing the whole N×N matrix, which a device
    # forbids outright and which costs O(N²) where the intrinsic pairing costs O(Nn). See `_dot`.
    if !(_dot(r, δ) > 0)
        config.verbosity ≥ 2 &&
            @warn "the $(method) direction is not a descent direction, so the Hessian is not positive definite here; using the steepest-descent direction for this step." maxlog = 1
        _copyto!(δ, r)
    end

    cache
end

# The `FirstOrderMethodWithState` methods are exempt: their direction is a moving average and is
# allowed not to descend on an individual step.
ensure_descent!(cache::OptimizerCache, ::FirstOrderMethodWithState, ::Options) = cache

@doc raw"""
    linesearch_rejected(status)
    linesearch_rejected(status, αmax)

Whether a line search reported that it could not decrease the merit along the direction it was
given.

The second form takes the ceiling the caller passed as `params.αmax` and is what
[`solver_step!`](@ref) uses; see *A step the caller forbade is not a failed direction* below.

`SimpleSolvers.solve` returns only the step length, so an outcome of `LINESEARCH_FLOOR`,
`LINESEARCH_EXHAUSTED` or `LINESEARCH_NO_DESCENT` looks the same as a successful search. The three
mean, in order, that the decrease achieved was no larger than the merit's own round-off resolution,
that the budget ran out or the merit could not be bracketed, and that ``\varphi'(0) \geq 0``. In none
of them does the returned ``\alpha`` carry a guarantee, and a solve that takes it can walk *uphill*.

The outcome comes from `SimpleSolvers.solve_with_status`, which `solver_step!` calls in place of
`solve` for this reason. (Plain code and not an `@extref`: SimpleSolvers documents
`solve_with_status` per method and not as a binding, so its inventory has no entry to link to.)

!!! info "This is deliberately the outcome and not `φ > φ₀`"
    A test on the merit — "reject the step only if it made things worse" — is the narrower
    condition. It stops the divergence described in [`solver_step!`](@ref), but it leaves the
    terminal gradient residual larger on the SVD problem of `test/verification/svd_optim.jl`. A
    search that ends on the floor has stopped making progress along *this* direction, and the
    cheapest remedy is a different direction.

!!! info "It applies to every method, including `Adam` and `MomentumMethod`"
    [`ensure_descent!`](@ref) exempts those two, because a moving average is *allowed* not to
    descend on an individual step. That is a statement about the direction and not about the step.
    A rejected search returns ``\alpha = 1`` untouched, so an exemption here would take the
    *longest* step along a direction that does not descend, and multiply ``f`` by many orders of
    magnitude. `LINESEARCH_EXHAUSTED` does the same, so the test here is the whole of
    `linesearch_rejected` and not only the ascent outcome.

    The momentum recursion does not change: ``p \gets \alpha{}p + \nabla{}f`` is evaluated in
    `update!(::MomentumState, …)` from `gradient_array(cache)` after the step, so the direction of
    the step does not enter it. Only the step changes.

!!! info "A step the caller forbade is not a failed direction"
    This is the `αmax` form. [`solver_step!`](@ref) gives the search a `params.αmax` of its own (see
    [`DEFAULT_STEP_CEILING`](@ref)), and a search stopped *at* that ceiling with the merit still
    falling is classified by the same round-off rule ``\tau`` as any other returned step — so it
    comes back as `LINESEARCH_FLOOR` when the fall over the whole admissible range was smaller than
    ``\tau``. That outcome is a claim about the **direction**, and the one-argument form answers it
    by throwing ``Q`` away and searching again along steepest descent. But the search established
    only that no step *this caller permits* decreases the merit measurably: the bound was the
    caller's own.

    So `LINESEARCH_FLOOR` counts as a rejection only where the step returned is strictly below the
    ceiling that was passed. Where it *is* the ceiling, the step is taken. The remedy is here and not
    upstream, because this package supplies the ceiling and can recognise its own bound without a
    new field on the status.

    `LINESEARCH_EXHAUSTED` and `LINESEARCH_NO_DESCENT` are deliberately not exempted. Neither is
    confusable with a bound step: the first says the budget ran out or the merit could not be
    bracketed, the second that ``\varphi'(0) \geq 0``, and both are true of the direction whatever
    ceiling was in force. A Euclidean solve passes no ceiling at all, so `αmax` is `Inf` there, the
    exemption is unreachable and the two forms agree.
"""
linesearch_rejected(status::LinesearchStatus) = outcome(status) ∈ (
    LINESEARCH_FLOOR, LINESEARCH_EXHAUSTED, LINESEARCH_NO_DESCENT)

function linesearch_rejected(status::LinesearchStatus, αmax)
    linesearch_rejected(status) &&
        !(outcome(status) == LINESEARCH_FLOOR && steplength(status) ≥ αmax)
end

# Whether a search rejected its direction because the merit resolves no decrease along it, and not
# because of the ceiling `αmax`; see `solver_step!`.
function at_round_off_floor(status::LinesearchStatus, αmax)
    outcome(status) == LINESEARCH_FLOOR && linesearch_rejected(status, αmax)
end

@doc raw"""
    steepest_descent!(cache)

Replace the direction stored in `cache` by ``-\nabla{}f``, the one direction that always descends.

# Implementation

[`rhs`](@ref) *is* ``-\nabla{}f`` on the (quasi-)Newton caches, so the default copies it across.

It is not on the three first-order caches, where `rhs` is an alias for [`direction`](@ref) itself, so
copying would be a silent no-op. `GradientCache`'s direction already *is* ``-\nabla{}f`` and the
no-op is correct there; `MomentumCache` and `AdamCache` hold a moving average and have to be given
the gradient explicitly.
"""
function steepest_descent!(cache::OptimizerCache)
    _copyto!(direction(cache), rhs(cache))

    cache
end

function _steepest_descent_from_gradient!(cache::OptimizerCache)
    _copyto!(direction(cache), gradient_array(cache))
    _rmul!(direction(cache), -1)

    cache
end

"""
    restart!(state)

Discard whatever curvature information `state` has accumulated and start again from the identity.

Only the quasi-Newton states carry any, so this is a no-op for everything else. See
[`restart!(::BFGSState)`](@ref) and [`solver_step!`](@ref).
"""
restart!(state::OptimizerState) = state

"""
    CURVATURE_TOLERANCE

How positive ``\\delta^T\\gamma`` has to be, relative to ``\\|\\delta\\|\\|\\gamma\\|``, for the
secant pair to be used. See [`curvature_is_usable`](@ref).
"""
const CURVATURE_TOLERANCE = 1.0e-8

@doc raw"""
    curvature_is_usable(ΔxΔg, Δx, Δg)

Whether the secant pair ``(\delta, \gamma)`` satisfies the curvature condition, i.e. whether the
quasi-Newton update built from it keeps ``Q`` positive definite.

Both the BFGS and the DFP update divide by ``\delta^T\gamma`` and preserve positive definiteness of
``Q`` only for ``\delta^T\gamma > 0`` [nocedal2006numerical](@cite). A test of `!iszero(ΔxΔg)` admits
both signs and denominators that are zero to within round-off, and a rank-two correction divided by
such a denominator makes ``Q`` badly conditioned or indefinite.

The threshold is *relative*: an absolute one cannot tell a round-off pairing on a problem scaled to
``10^0`` from a legitimate small pairing on a problem scaled to ``10^{-15}``. Its exact value
matters much less than its existence: `1e-8` and `eps(T)` behave identically on everything measured
here, because the pairs being rejected are the ones that are non-positive rather than merely small.

!!! info "Rejecting an update is not free"
    A skipped update leaves ``Q`` where it was, so the next direction is built from staler curvature.
    `DFP` with a shrink-only `Backtracking` can take many more iterations under the guard: its
    direction is systematically under-scaled (see [`default_linesearch`](@ref)), and a
    negative-curvature update can inflate ``Q`` in a way that partly compensates. In exchange, the
    iteration count of `DFP` with an expanding `Backtracking` depends much less on the starting
    point.
"""
function curvature_is_usable(ΔxΔg::T, Δx, Δg) where {T}
    isnan(ΔxΔg) && return false
    ΔxΔg > T(CURVATURE_TOLERANCE) * l2norm(Δx) * l2norm(Δg)
end
