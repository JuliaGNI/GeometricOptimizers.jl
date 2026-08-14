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

The default forms it from the gradient the cache holds and the previous gradient the state holds. The
quasi-Newton caches have already formed exactly this difference -- it is the `γ` of their secant pair
-- and have advanced `state.ḡ` past it in doing so, so for them this is a no-op.
"""
gradient_difference!(cache::OptimizerCache, state::OptimizerState) = _difference!(cache.Δg, cache.g, state.ḡ)

function OptimizerStatus(state::OST, cache::OCT, f::T; config::Options) where {T,OST<:OptimizerState{T},OCT<:OptimizerCache{T}}
    rxₐ = l2norm(direction(cache))
    rxᵣ = rxₐ / l2norm(cache.x)

    Δf = f - state.f̄
    # `_dot`, not `⋅`: this is the decrease in `f` the step predicts to first order, so it has to be
    # comparable with `Δf` above. On a manifold both operands are horizontal lifts, and `⋅` on those is
    # the ambient product, which is twice the intrinsic one. See `_dot`.
    Δf̃ = _dot(state.ḡ, direction(cache))

    rfₐ = norm(Δf)
    rfᵣ = rfₐ / norm(f)

    gradient_difference!(cache, state)

    rgₐ = l2norm(cache.Δg)
    rg = l2norm(cache.g)

    f_increased = abs(f) > abs(state.f̄)

    x_nonfinite = contains_nonfinite(cache.x)
    f_nonfinite = contains_nonfinite(f)
    g_nonfinite = contains_nonfinite(cache.g)

    _status = OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, false, false, false, f_increased, x_nonfinite, f_nonfinite, g_nonfinite)

    (x_converged, f_converged, f_converged_strong, g_converged) = convergence_measures(_status, config)

    OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, x_converged, f_converged, g_converged, f_increased, x_nonfinite, f_nonfinite, g_nonfinite)
end

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

`x_converged` is the one to be careful with: it cannot be trusted on a solve that has diverged, for
the reason recorded under [`convergence_measures`](@ref).
"""
isconverged(status::OptimizerStatus) = status.x_converged || status.f_converged || status.g_converged

@doc raw"""
    convergence_measures(status, config)

Checks if the optimizer converged.

Here `status` is an [`OptimizerStatus`](@ref) object and `config` is an [`SimpleSolvers.Options`](@extref) object.

# Extended help

!!! warning "`x_converged` cannot be trusted on a solve that has diverged"
    ``\|x - x'\|/\|x'\|`` measures "the iterate stopped moving" only while ``\|x'\|`` is bounded, and
    a diverging solve is exactly the case where it is not. On the SVD problem of
    `test/optimizer_convergence/svd_optim.jl`, `_BFGS` + `Bisection` + `Geodesic` once left the
    manifold on iteration 4 with an iterate of magnitude ``10^{100}``. The step that took it there
    had ``\|\delta\| = 345`` — not remotely a solve that has stopped moving — but the *relative*
    change was ``345/10^{100} \approx 10^{-98}``, far under `x_reltol`, so `x_converged` fired and
    the solve reported success.

    That divergence is fixed at its source ([`linesearch_rejected`](@ref),
    [`curvature_is_usable`](@ref)) and [`contains_nonfinite`](@ref) catches the `Inf`/`NaN` end of
    the range, so nothing measured still reaches this. The hole itself is left open on purpose: every
    way of closing it here needs a threshold on ``\|x\|`` or on ``\|x - x'\|`` that no property of
    the problem supplies, and imposing one would change the stopping behaviour of every Euclidean
    solve to guard a state that can no longer be reached. On a manifold the honest test is
    `check`, which the caller has and this function does not: ``\|Y\|_F = \sqrt{n}`` exactly
    for ``Y \in St(N, n)``, so any deviation is measurable without a tolerance being invented for it.
"""
function convergence_measures(status::OptimizerStatus, config::Options)
    x_converged = x_abschange(status) ≤ x_abstol(config) ||
                  x_relchange(status) ≤ x_reltol(config)

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
