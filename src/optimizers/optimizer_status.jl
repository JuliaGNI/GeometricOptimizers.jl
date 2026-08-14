"""
    OptimizerStatus

Contains residuals (relative and absolute) and various convergence properties.

This is also used in [`OptimizerResult`](@ref).

# Examples

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: NewtonOptimizerCache, OptimizerStatus)
x = ones(3)
state = NewtonOptimizerState(x)
cache = NewtonOptimizerCache(x)
f = 1.
config = Options()
OptimizerStatus(state, cache, f; config = config)

# output

 * Convergence measures

    |x - x'|               = NaN
    |x - x'|/|x'|          = NaN
    |f(x) - f(x')|         = NaN
    |f(x) - f(x')|/|f(x')| = NaN
    |g(x) - g(x')|         = NaN
    |g(x)|                 = NaN

```
"""
struct OptimizerStatus{XT,YT}
    rxₐ::XT  # absolute change in x
    rxᵣ::XT  # relative change in x
    rfₐ::YT  # absolute change in f
    rfᵣ::YT  # relative change in f
    rgₐ::YT  # absolute change in g
    rg::XT   # residual of g

    Δf::YT    # change of function
    Δf̃::YT

    x_converged::Bool
    f_converged::Bool
    g_converged::Bool
    f_increased::Bool

    x_nonfinite::Bool
    f_nonfinite::Bool
    g_nonfinite::Bool
end

x_abschange(status::OptimizerStatus) = status.rxₐ
x_relchange(status::OptimizerStatus) = status.rxᵣ
f_abschange(status::OptimizerStatus) = status.rfₐ
f_relchange(status::OptimizerStatus) = status.rfᵣ
f_change(status::OptimizerStatus) = status.Δf
f_change_approx(status::OptimizerStatus) = status.Δf̃
g_abschange(status::OptimizerStatus) = status.rgₐ
g_residual(status::OptimizerStatus) = status.rg

"""
    gradient_difference!(cache, state)

Write the gradient difference `∇f(xᵏ) - ∇f(xᵏ⁻¹)` into `cache.Δg`, for [`OptimizerStatus`](@ref).

# Implementation

Every cache in this package overrides this with the difference of the two gradients it holds itself,
`latest_gradient` at ``x_{k+1}`` and `gradient` at ``x_k``, which is the successive difference the
status prints and needs no `state.ḡ`. The default -- `cache.g` against `state.ḡ` -- was wrong for
each of them in a different way:

- for the three first-order caches, `update!(::MomentumState, ...)` runs *after* the step and copies
  the cache's *pre-step* gradient into `state.g`, shifting the one before it into `state.ḡ`, so
  `state.ḡ` ends up two iterates behind `cache.g` rather than one: on
  ``f(x) = \\sum(x^2 + 0.1x^4)`` from `[1.5, -0.8, 0.4]` with `MomentumMethod` + `Bisection`,
  iteration three reported `rgₐ = 4.976` where ``\\|\\nabla{}f(x_k) - \\nabla{}f(x_{k-1})\\| = 0.295``.
  On the first iteration it differenced against the `_similar` memory `MomentumState` never writes.
- for `BFGSCache` and `DFPCache` it was the `γ` of the secant pair, ``\\nabla{}f(x_k) -
  \\nabla{}f(x_{k-1})``, which those form inside `update!(cache, ...)` and which is one step behind
  the `rg` reported next to it.
- for `NewtonOptimizerCache` it was *structurally zero*: [`solver_step!`](@ref) advances `state.ḡ` at
  the same iterate the cache takes its gradient at, so the difference could only ever be `0`.

In all three cases the two `g` rows of a status are now about one step rather than about two
different ones; see [`convergence_measures`](@ref) for which iterate `rg` belongs to.

`state.ḡ` is still two iterates behind for the first-order methods, and `Δf̃` above still reads it.
That is recorded as open issue A10 in `CHANGELOG.md`.
"""
gradient_difference!(cache::OptimizerCache, state::OptimizerState) = _difference!(cache.Δg, cache.g, state.ḡ)

function OptimizerStatus(state::OST, cache::OCT, f::T; config::Options) where {T,OST<:OptimizerState{T},OCT<:OptimizerCache{T}}
    rxₐ = l2norm(direction(cache))
    # `solution_scale`, not `l2norm`: on a manifold the norm of the iterate is a constant the geometry
    # supplies, and a measured norm that is not that constant means the iterate has left the manifold
    # rather than that the problem has a large scale. See `solution_scale` and `convergence_measures`.
    rxᵣ = rxₐ / solution_scale(cache.x)

    Δf = f - state.f̄
    # `_dot`, not `⋅`: this is the decrease in `f` the step predicts to first order, so it has to be
    # comparable with `Δf` above. On a manifold both operands are horizontal lifts, and `⋅` on those is
    # the ambient product, which is twice the intrinsic one. See `_dot`.
    Δf̃ = _dot(state.ḡ, direction(cache))

    rfₐ = norm(Δf)
    rfᵣ = rfₐ / norm(f)

    gradient_difference!(cache, state)

    rgₐ = l2norm(cache.Δg)
    # `latest_gradient` and not `cache.g`: for the caches that refresh it, this is `∇f` at the iterate
    # the step ended at rather than at the one it started from. See `convergence_measures`.
    rg = l2norm(latest_gradient(cache))

    # `f > f̄` and not `abs(f) > abs(f̄)`: the question is whether the objective went up, and for an
    # objective that takes negative values the two are different questions -- `-5 → -6` is a decrease
    # and reads as an increase through `abs`. That was tolerable while nothing acted on the flag;
    # `convergence_measures` now does.
    f_increased = f > state.f̄

    x_nonfinite = contains_nonfinite(cache.x)
    f_nonfinite = contains_nonfinite(f)
    g_nonfinite = contains_nonfinite(latest_gradient(cache))

    _status = OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, false, false, false, f_increased, x_nonfinite, f_nonfinite, g_nonfinite)

    (x_converged, f_converged, f_converged_strong, g_converged) = convergence_measures(_status, config)

    OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, x_converged, f_converged, g_converged, f_increased, x_nonfinite, f_nonfinite, g_nonfinite)
end

@doc raw"""
    solution_scale(x)

The norm the *relative* change in the iterate is measured against, i.e. the denominator of
``\|x - x'\|/\|x'\|``.

For an ordinary array this is the `l2norm` of the iterate, which is the only scale a Euclidean
problem supplies. For a [`Manifold`](@ref) it is *not*: a point of ``St(N, n)`` or
``Gr(n, N)`` satisfies ``Y^TY = \mathbb{I}_n``, so ``\|Y\|_F = \sqrt{\mathrm{tr}(Y^TY)} = \sqrt{n}``
exactly, and the scale is a constant the geometry supplies rather than something to be measured off
the iterate.

That distinction is what keeps `x_converged` honest on a solve that has diverged; see
[`convergence_measures`](@ref). While the iterate is on the manifold the two agree to round-off, so
this changes no measurement of any converging solve — it changes what is reported once the iterate is
somewhere ``\sqrt{n}`` no longer describes.

A `NamedTuple` combines its blocks in quadrature, exactly as `l2norm` does, so a `NamedTuple` holding
a manifold next to an ordinary array uses the nominal scale for the one and the measured scale for
the other.
"""
solution_scale(x::AbstractVecOrMat) = l2norm(x)
solution_scale(Y::Manifold{T}) where {T} = √T(size(Y, 2))
solution_scale(ps::ArrayNamedTuple) = √sum(abs2, values(apply_toNT(solution_scale, ps)))

l2norm(a::StiefelLieAlgHorMatrix) = √(l2norm(a.A)^2 + l2norm(a.B)^2)

l2norm(a::SkewSymMatrix) = l2norm(a.S)
# Type piracy: `l2norm` is `GeometricBase.Utils.l2norm` (SimpleSolvers only re-exports it)
# and both argument types are Base's, so every package that loads GeometricOptimizers
# inherits these. They should be upstreamed to GeometricBase. See issue #16.
l2norm(a::AbstractMatrix) = l2norm(vec(a))
l2norm(a::AbstractFloat) = norm(a)
# Type piracy as well, but only because `ArrayNamedTuple` is an alias for `NamedTuple`; a
# wrapper `struct` would fix this one locally. See issue #16.
function l2norm(a::ArrayNamedTuple)
    # the block norms combine in quadrature, as for `StiefelLieAlgHorMatrix` above: summing them
    # (which this used to do) overestimates the ℓ² norm by up to `√k` for `k` blocks and thereby
    # every stopping criterion computed from it.
    norms = apply_toNT(l2norm, a)
    √sum(abs2, values(norms))
end

@doc raw"""
    contains_nonfinite(a)

Whether `a` holds any value that is not finite.

This was `contains_nan`, and tested `isnan` only. `NaN` is the *last* thing a diverging solve
produces: it reaches `Inf` first, and before that every finite magnitude on the way. On the SVD
problem of `test/optimizer_convergence/svd_optim.jl` the diverging solve passed through
`f = 1.2e169` and `check(Y) = 1.07e200` — both perfectly ordinary `Float64`s, neither of them `NaN`
— and only went `NaN` on the iteration after that. By then it had been off the manifold for two
iterations.

`isfinite` still does not catch `1e200`, which is why it is not the only guard; see
[`convergence_measures`](@ref) for the one that does.
"""
contains_nonfinite(a::Real) = !isfinite(a)
contains_nonfinite(a) = any(contains_nonfinite, a)

function Base.show(io::IO, s::OptimizerStatus)

    @printf io " * Convergence measures\n"
    @printf io "\n"
    @printf io "    |x - x'|               = %.2e\n" x_abschange(s)
    @printf io "    |x - x'|/|x'|          = %.2e\n" x_relchange(s)
    @printf io "    |f(x) - f(x')|         = %.2e\n" f_abschange(s)
    @printf io "    |f(x) - f(x')|/|f(x')| = %.2e\n" f_relchange(s)
    @printf io "    |g(x) - g(x')|         = %.2e\n" g_abschange(s)
    @printf io "    |g(x)|                 = %.2e\n" g_residual(s)

end

"""
    isconverged(status)

Whether any of the three convergence flags [`convergence_measures`](@ref) sets is set.

The flags are a disjunction on purpose: `x_converged`, `f_converged` and `g_converged` test different
things and a solve is entitled to stop on any one of them. Note that [`solve!`](@ref) can also stop
for reasons that are *not* convergence — the iteration cap, a non-finite iterate, an increase in `f`
where one is not allowed — and none of those sets a flag here, so this is what tells the two apart;
see [`meets_stopping_criteria`](@ref).

`x_converged` is the one to be careful with: "the iterate stopped moving" is a statement about a
ratio, and a diverging solve is where the denominator stops meaning anything. See the warning under
[`convergence_measures`](@ref) for the two guards on it and for the one case they do not cover.
"""
isconverged(status::OptimizerStatus) = status.x_converged || status.f_converged || status.g_converged

@doc raw"""
    convergence_measures(status, config)

Checks if the optimizer converged.

Here `status` is an [`OptimizerStatus`](@ref) object and `config` is an [`SimpleSolvers.Options`](@extref) object.

# Extended help

!!! info "Which iterate `g_converged` is a statement about"
    `rg` is [`latest_gradient`](@ref), i.e. ``\|\nabla{}f(x_{k+1})\|`` at the iterate the step ended
    at, for every method. It used to be ``\|\nabla{}f(x_k)\|`` at the one the step started from, and
    for the (quasi-)Newton caches it was worse than that: [`trial_slope`](@ref) evaluates the trial
    gradient into the same array, so `rg` was ``\|\nabla{}f\|`` at whatever point the line search
    last probed. On Rosenbrock from ``(-1.2, 1)`` with the default `Backtracking` — which probes
    ``\alpha = 0``, so that point is ``x_k`` — `rg` came out `5.8\times10^4` times the true residual
    for `_BFGS` and `299` times it for `_DFP`. It errs high near a minimiser, so `g_converged` fired
    *late*; that was open issue A8.

    The distinction is not cosmetic for a direction that carries momentum. Under
    [`SimpleSolvers.Static`](@extref) a stale `rg` is harmless, because the direction is a scaled
    gradient and a vanishing gradient means a vanishing step. Under a line search accurate enough to
    drive ``\nabla{}f(x_{k+1}) \approx 0`` it is not: the momentum term still moves the iterate, so
    `g_converged` fired one step *past* the minimiser. `MomentumMethod` + `Backtracking` on
    ``f(x) = 1 + x^2`` from ``x = 1`` stopped at ``x = -0.2`` reporting `rg = 0`, where
    ``\|\nabla{}f(x)\| = 0.4`` and the momentum was ``2``. Over `Bisection`, `Quadratic` and
    `BierlaireQuadratic` this left `MomentumMethod` at ``\|x\| = 0.35`` and `Adam` at
    ``\|x\| = 1.16``, i.e. barely moved from `ones(3)`, both reporting convergence. Refreshing costs
    one gradient evaluation per iteration and never cost an iteration in any case measured.

!!! warning "What `x_converged` is guarded against, and what it is not"
    ``\|x - x'\|/\|x'\|`` measures "the iterate stopped moving" only while ``\|x'\|`` is bounded, and
    a diverging solve is exactly the case where it is not. On the SVD problem of
    `test/optimizer_convergence/svd_optim.jl`, `_BFGS` + `Bisection` + `Geodesic` once left the
    manifold on iteration 4 with an iterate of magnitude ``10^{100}``. The step that took it there
    had ``\|\delta\| = 345`` — not remotely a solve that has stopped moving — but the *relative*
    change was ``345/10^{100} \approx 10^{-98}``, far under `x_reltol`, so `x_converged` fired and
    the solve reported success. Two guards, neither of which invents a tolerance:

    - the denominator is [`solution_scale`](@ref) and not ``\|x'\|``. On a manifold the two are the
      same number — ``\|Y\|_F = \sqrt{n}`` exactly — right up to the point where the iterate leaves
      the manifold, and there the constant is the honest scale and the measured norm is the
      divergence. The trace above gives ``345/\sqrt{3} \approx 199`` in place of ``10^{-98}``.
    - `x_converged` also requires `!f_increased`. An iterate that has stopped moving is evidence of
      convergence only if the objective did not just go up, and in that trace it went `3.38 → 9.13 →
      1.2\times10^{169}`. This is the only guard the *Euclidean* case has, because nothing there
      bounds ``\|x\|``.

    What is still not covered: a Euclidean solve that runs away *downhill*, where ``\|x\|`` grows
    without bound and `f` decreases at every step, has no scale to be measured against and is still
    reported as converged. Closing that needs a threshold on ``\|x\|`` that no property of the
    problem supplies. Note also that the divergence above is fixed at its source
    ([`linesearch_rejected`](@ref), [`curvature_is_usable`](@ref)) and that
    [`contains_nonfinite`](@ref) catches the `Inf`/`NaN` end of the range, so nothing measured
    reaches any of this.
"""
function convergence_measures(status::OptimizerStatus, config::Options)
    # `&& !f_increased`: both branches above say "the iterate stopped moving", which is evidence of
    # convergence only if the objective did not just go up. See the warning above.
    x_converged = (x_abschange(status) ≤ x_abstol(config) ||
                   x_relchange(status) ≤ x_reltol(config)) && !status.f_increased

    # `f_relchange` is a *successive* change, so it is gated on `f_suctol`, which SimpleSolvers
    # 0.9 introduced for exactly that and gave `f_reltol`'s former default. `f_reltol` itself is
    # now anchored to the initial residual and defaults to `√eps(T)` -- seven orders of magnitude
    # looser -- so keeping it here would stop a `Static` line search long before the minimizer.
    f_converged = f_abschange(status) ≤ f_abstol(config) ||
                  f_relchange(status) ≤ f_suctol(config)

    f_converged_strong = f_change(status) ≤ f_mindec(config) * f_change_approx(status)

    # SimpleSolvers 0.9 removed `Options.g_restol` and gave its role to `f_reltol`, whose default
    # (`√eps(T)`) is the same number `g_restol` defaulted to. The residual of an optimizer is
    # `‖∇f(x)‖`, so this is the corresponding gate here.
    g_converged = g_residual(status) ≤ f_reltol(config)

    (x_converged, f_converged, f_converged_strong, g_converged)
end

@doc raw"""
    meets_stopping_criteria(status, config, iterations)

Check if the optimizer has converged.

# Implementation

`meets_stopping_criteria` checks if one of the following is true:
- `converged` (the output of [`SimpleSolvers.assess_convergence`](@extref)) is `true` and `iterations` ``\geq`` `config.min_iterations`,
- if `config.allow_f_increases` is `false`: `status.f_increased` is `true`,
- `iterations` ``\geq`` `config.max_iterations`,
- `status.rfₐ` ``>`` `config.f_abstol_break`,
- any of `status.x_nonfinite`, `status.f_nonfinite`, `status.g_nonfinite`.

# Extended help

!!! info "A non-finite iterate stops the solve"
    The last of those used to be reported and then ignored: the `@error` below fired and the loop
    carried on. Nothing an iteration does to a `NaN` iterate can recover it, so the only effect was
    to burn the whole iteration budget printing the same message. On the SVD problem of
    `test/optimizer_convergence/svd_optim.jl` one starting point spent all 100 000 iterations of a
    raised cap that way, at roughly one `@error` per iteration.

    A solve that stops here is *not* converged — [`isconverged`](@ref) reads the three convergence
    flags and none of them is set by this — so a caller that checks the status rather than only the
    return value can tell the two apart.
"""
function meets_stopping_criteria(status::OptimizerStatus, config::Options, iterations::Integer)
    converged = isconverged(status)
    nonfinite = status.x_nonfinite || status.f_nonfinite || status.g_nonfinite

    if iterations ≥ 1 && nonfinite
        @error "x, f or g in the OptimizerStatus you provided is not finite; stopping." iterations
    end

    (converged && iterations ≥ config.min_iterations) ||
        (status.f_increased && !config.allow_f_increases) ||
        iterations ≥ config.max_iterations ||
        (iterations ≥ 1 && nonfinite) ||
        # `f_abstol_break` is the only `*_break` field SimpleSolvers 0.9 kept; the four that
        # are gone (`x_abstol_break`, `x_reltol_break`, `f_reltol_break`, `g_restol_break`)
        # all defaulted to `Inf`, so dropping them changes nothing at default `Options`.
        status.rfₐ > config.f_abstol_break
end
