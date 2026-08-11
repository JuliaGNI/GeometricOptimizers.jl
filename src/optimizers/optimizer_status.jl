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

    x_isnan::Bool
    f_isnan::Bool
    g_isnan::Bool
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
    Δf̃ = state.ḡ ⋅ direction(cache)

    rfₐ = norm(Δf)
    rfᵣ = rfₐ / norm(f)

    gradient_difference!(cache, state)

    rgₐ = l2norm(cache.Δg)
    rg = l2norm(cache.g)

    f_increased = abs(f) > abs(state.f̄)

    x_isnan = contains_nan(cache.x)
    f_isnan = contains_nan(f)
    g_isnan = contains_nan(cache.g)

    _status = OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, false, false, false, f_increased, x_isnan, f_isnan, g_isnan)

    (x_converged, f_converged, f_converged_strong, g_converged) = convergence_measures(_status, config)

    OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, x_converged, f_converged, g_converged, f_increased, x_isnan, f_isnan, g_isnan)
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

contains_nan(a::Real) = isnan(a)
contains_nan(a) = any(contains_nan, a)

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

isconverged(status::OptimizerStatus) = status.x_converged || status.f_converged || status.g_converged

"""
    convergence_measures(status, config)

Checks if the optimizer converged.

Here `status` is an [`OptimizerStatus`](@ref) object and `config` is an [`SimpleSolvers.Options`](@extref) object.
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
- `status.rfₐ` ``>`` `config.f_abstol_break`
- `status.x_isnan`
- `status.f_isnan`
- `status.g_isnan`
"""
function meets_stopping_criteria(status::OptimizerStatus, config::Options, iterations::Integer)
    converged = isconverged(status)

    if iterations ≥ 1 && (status.x_isnan || status.f_isnan || status.g_isnan)
        @error "x, f or g in the OptimizerStatus you provided are NaNs."
    end

    (converged && iterations ≥ config.min_iterations) ||
        (status.f_increased && !config.allow_f_increases) ||
        iterations ≥ config.max_iterations ||
        # `f_abstol_break` is the only `*_break` field SimpleSolvers 0.9 kept; the four that
        # are gone (`x_abstol_break`, `x_reltol_break`, `f_reltol_break`, `g_restol_break`)
        # all defaulted to `Inf`, so dropping them changes nothing at default `Options`.
        status.rfₐ > config.f_abstol_break
end
