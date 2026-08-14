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
    [`solver_step!`](@ref) does not call this for them.

!!! info "Why this is needed now"
    Up to SimpleSolvers 0.8 the `Bisection` and `Quadratic` line searches could return a *negative*
    step, which silently turned an ascent direction into a descent one and hid the problem. They no
    longer do, so the safeguard has to live here.

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

    if !(dot(r, δ) > 0)
        config.verbosity ≥ 2 && @warn "the $(method) direction is not a descent direction, so the Hessian is not positive definite here; using the steepest-descent direction for this step." maxlog = 1
        _copyto!(δ, r)
    end

    cache
end

@doc raw"""
    linesearch_rejected(status)

Whether a line search reported that it could not decrease the merit along the direction it was
given.

`SimpleSolvers.solve` returns only the step length, so an outcome of `LINESEARCH_FLOOR`,
`LINESEARCH_EXHAUSTED` or `LINESEARCH_NO_DESCENT` used to be indistinguishable from a successful
search: the step came back and [`solver_step!`](@ref) took it. The three mean, in order, that the
decrease achieved was no larger than the merit's own round-off resolution, that the budget ran out or
the merit could not be bracketed, and that ``\varphi'(0) \geq 0``. In none of them does the returned
``\alpha`` carry a guarantee, and taking it anyway is how a solve can walk *uphill*.

The outcome comes from [`SimpleSolvers.solve_with_status`](@extref), which `solver_step!` calls in
place of `solve` for exactly this reason.

!!! info "This is deliberately the outcome and not `φ > φ₀`"
    Testing the merit directly — "reject the step only if it actually made things worse" — is the
    narrower condition, and it does fix the divergence described in [`solver_step!`](@ref). It is
    not enough, though: measured over the eight starting points of
    `test/optimizer_convergence/svd_optim.jl`, restarting only on a genuine increase leaves the
    terminal gradient residual at `1.8e-5`, where restarting on the outcome brings it to `2.9e-7`.
    A search that ends on the floor has stopped making progress along *this* direction, and the
    cheapest thing to do about it is to pick a different one.

!!! info "It applies to every method, including `Adam` and `MomentumMethod`"
    Those two used to be exempt, on the grounds that [`ensure_descent!`](@ref) exempts them: a moving
    average is *allowed* not to descend on an individual step. That is a true statement about the
    direction and the wrong conclusion about the step. A rejected search returns ``\alpha = 1``
    untouched, so the exemption did not permit a non-descent step, it took the *longest* step
    available along one.

    Issue A7 is what that cost. On Rosenbrock from ``(-1.2, 1)`` with `MomentumMethod(0.1)` under the
    expanding `Backtracking` default, the solve reaches `f = 7.8e-5` by iteration 400 and then:

    | iteration | outcome | ``\alpha`` | ``f`` |
    |---|---|---|---|
    | 441 | `LINESEARCH_NO_DESCENT` | 1.0 | 4.97e-2 → 4.65e3 |
    | 443 | `LINESEARCH_NO_DESCENT` | 1.0 | 8.16e1 → 5.33e9 |
    | 449 | `LINESEARCH_NO_DESCENT` | 1.0 | 2.87e2 → 9.12e7 |
    | 453 | `LINESEARCH_NO_DESCENT` | 1.0 | 1.46e3 → 4.41e19 |
    | 455 | `LINESEARCH_NO_DESCENT` | 1.0 | 8.20e16 → 1.51e61 |

    Thirteen such events over 457 iterations, each multiplying ``f`` by between ``10^3`` and
    ``10^{42}``; the steps in between do descend and cannot make it back. `LINESEARCH_EXHAUSTED`
    does the same thing — under `BierlaireQuadratic` it takes the same solve to `Inf` in six
    iterations, at ``\alpha = 1`` every time — which is why the test here is the whole of
    `linesearch_rejected` and not just the ascent outcome.

    The momentum recursion is untouched by this: ``p \gets \alpha{}p + \nabla{}f`` is evaluated in
    `update!(::MomentumState, …)` from `gradient_array(cache)` after the step, so which direction the
    step was taken along does not enter it. Only the step changes. `ensure_descent!`, which acts on
    the direction *before* the search, still exempts them.
"""
linesearch_rejected(status::LinesearchStatus) =
    outcome(status) ∈ (LINESEARCH_FLOOR, LINESEARCH_EXHAUSTED, LINESEARCH_NO_DESCENT)

@doc raw"""
    steepest_descent!(cache)

Replace the direction stored in `cache` by ``-\nabla{}f``, the one direction that always descends.

# Implementation

[`rhs`](@ref) *is* ``-\nabla{}f`` on the (quasi-)Newton caches, so the default copies it across —
which is what [`solver_step!`](@ref) did inline before this existed.

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
``Q`` only for ``\delta^T\gamma > 0`` [nocedal2006numerical](@cite). The guard this replaced was
`!iszero(ΔxΔg) && !isnan(ΔxΔg)`, which admits both signs and, more importantly, admits denominators
that are zero to within round-off: on the SVD problem of
`test/optimizer_convergence/svd_optim.jl`, ``\delta^T\gamma`` took the values `-12.8`, `-4.5e-16`
and `+1.5e-15` on consecutive iterations, all three of which `!iszero` accepts. Dividing a rank-two
correction by `1.5e-15` is what drove ``\lambda_\mathrm{max}(Q)`` from 3 to 442 there, and
``\lambda_\mathrm{min}(Q)`` to `-398` from another starting point.

The threshold is therefore *relative* -- an absolute one cannot tell `1.5e-15` on a problem scaled
to ``10^0`` from a legitimate small pairing on a problem scaled to ``10^{-15}``. Its exact value
matters much less than its existence: `1e-8` and `eps(T)` behave identically on everything measured
here, because the pairs being rejected are the ones that are non-positive rather than merely small.

!!! info "Rejecting an update is not free"
    A skipped update leaves ``Q`` where it was, so the next direction is built from staler curvature.
    For `_DFP` on Rosenbrock from ``x_0 = (-1.2, 1)`` with shrink-only `Backtracking` that costs a
    factor of seventeen -- 50 iterations against 851, both reaching `f ≈ 3e-24`. `_DFP` was
    exploiting the invalid updates: it produces a systematically under-scaled direction (see
    [`default_linesearch`](@ref)), and a negative-curvature update happens to inflate ``Q`` in a way
    that partly compensates. `_BFGS` is unaffected, at 22 iterations either way. In exchange, `_DFP`
    stops being wildly sensitive to its starting point -- over eight starting points of the SVD
    problem with an expanding `Backtracking` its iteration count goes from `512..77_890` to
    `512..845`.
"""
function curvature_is_usable(ΔxΔg::T, Δx, Δg) where {T}
    isnan(ΔxΔg) && return false
    ΔxΔg > T(CURVATURE_TOLERANCE) * l2norm(Δx) * l2norm(Δg)
end
