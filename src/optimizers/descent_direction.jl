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
