"""
    OptimizerStatus

Contains residuals (relative and absolute) and various convergence properties.

This is also used in [`OptimizerResult`](@ref).

# Examples

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: NewtonOptimizerCache, OptimizerStatus)
x = ones(3)
state = NewtonState(x)
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
struct OptimizerStatus{XT, YT}
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

This is the difference of the two gradients the cache holds itself, `latest_gradient` at
``x_{k+1}`` and `gradient` at ``x_k``, which is the successive difference the status prints and
needs no `state.ḡ`. It is not the `γ` of the quasi-Newton secant pair, which is one step behind the
`rg` reported next to it. So the two `g` rows of a status are about one step; see
[`convergence_measures`](@ref) for which iterate `rg` belongs to.
"""
function gradient_difference!(cache::OptimizerCache, ::OptimizerState)
    _difference!(cache.Δg, latest_gradient(cache), gradient(cache))
end

function OptimizerStatus(state::OST, cache::OCT, f::T;
        config::Options) where {T, OST <: OptimizerState{T}, OCT <: OptimizerCache{T}}
    rxₐ = l2norm(direction(cache))
    # `solution_scale`, not `l2norm`: on a manifold the norm of the iterate is a constant the geometry
    # supplies, and a measured norm that is not that constant means the iterate has left the manifold
    # rather than that the problem has a large scale. See `solution_scale` and `convergence_measures`.
    rxᵣ = rxₐ / solution_scale(cache.x)

    # `solve!` has advanced `state` to `x` before it builds this, so `previous_value(state)` is the
    # objective one step back and `previous_gradient(state)` the gradient the step was built from.
    Δf = f - previous_value(state)
    # `_dot`, not `⋅`: this is the decrease in `f` the step predicts to first order, so it has to be
    # comparable with `Δf` above. On a manifold both operands are horizontal lifts, and `⋅` on those is
    # the ambient product, which is twice the intrinsic one. See `_dot`.
    Δf̃ = _dot(previous_gradient(state), direction(cache))

    rfₐ = norm(Δf)
    rfᵣ = rfₐ / norm(f)

    gradient_difference!(cache, state)

    rgₐ = l2norm(cache.Δg)
    # `latest_gradient` and not `cache.g`: this is `∇f` at the iterate the step ended at rather than
    # at the one it started from. See `convergence_measures`.
    rg = l2norm(latest_gradient(cache))

    # `f > f̄` and not `abs(f) > abs(f̄)`: the question is whether the objective went up, and for an
    # objective that takes negative values the two are different questions -- `-5 → -6` is a decrease
    # and reads as an increase through `abs`.
    #
    # On the first iteration the previous objective is `f(x₀)`, which `initialize_state!` records.
    f_increased = f > previous_value(state)

    x_nonfinite = contains_nonfinite(cache.x)
    f_nonfinite = contains_nonfinite(f)
    g_nonfinite = contains_nonfinite(latest_gradient(cache))

    _status = OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, false, false, false,
        f_increased, x_nonfinite, f_nonfinite, g_nonfinite)

    # `f_converged_strong` is computed and discarded; see KNOWN_ISSUES C1
    # fatou-ignore unused-binding
    (x_converged, f_converged, f_converged_strong, g_converged) = convergence_measures(_status, config)

    OptimizerStatus(rxₐ, rxᵣ, rfₐ, rfᵣ, rgₐ, rg, Δf, Δf̃, x_converged, f_converged,
        g_converged, f_increased, x_nonfinite, f_nonfinite, g_nonfinite)
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
# A fold over the leaves, and not `map` + `sum` over one level: a container is a tree of layers, so the
# scales to combine are at its *leaves*, at any depth. `_sumsq_leaves` is
# `NeuralNetworkParameters.foldparameters`, which reaches a leaf at any depth and allocates nothing.
solution_scale(ps::NetworkParameters) = √_sumsq_leaves(solution_scale, ps)

# The norm of a horizontal lift is taken over its *free parameters*, i.e. over `Base.parent`, and in
# quadrature -- the same intrinsic-versus-ambient distinction `_dot` documents. Leaving it to the
# `AbstractMatrix` fallback would give the ambient Frobenius norm, which counts each off-diagonal
# block twice and so comes out `√2` times too large in `step_αmax`, `curvature_is_usable`, `rxₐ` and
# `rg`. For a `StiefelLieAlgHorMatrix` this is `√(l2norm(a.A)^2 + l2norm(a.B)^2)`.
#
# `_sumsq_leaves` over the blocks, so that every quadrature sum in the package is the same fold.
# `parent(a)` and not `a`: `foldparameters` treats a lift as a *leaf*, so the lift itself would call
# this method on it again, forever.
l2norm(a::AbstractLieAlgHorMatrix) = √_sumsq_leaves(l2norm, parent(a))

# The same over the free parameters, for the same reason, for the four types that keep theirs in one
# vector: `GeometricBase`'s `AbstractArray` method would take the norm of the `n²` entries instead.
l2norm(a::VectorStorageMatrix) = l2norm(parent(a))

# The norm of a matrix and of a number are `GeometricBase`'s, which owns `l2norm`; a method here for
# a `Base` type would be type piracy.
#
# `l2norm` of a whole set is **not** defined here: `l2norm` is `GeometricBase`'s and
# `NetworkParameters` is `NeuralNetworkParameters`', so a method here would own neither side of its
# signature. `NeuralNetworkParameters` defines `L2norm(::NetworkParameters)`, from which the generic
# `l2norm(x) = sqrt(L2norm(x))` follows. It calls `l2norm` on the *leaves*, so the two methods above
# decide the contribution of a lift and of a `VectorStorageMatrix`, and the leaves combine in
# quadrature.

@doc raw"""
    _sumsq_leaves(f, x)

``\sum_i f(x_i)^2`` over the leaves of `x`, at whatever depth they are.

The quadrature sum in which `l2norm` and [`solution_scale`](@ref) both combine their blocks; the two
differ only in `f`. (`l2norm` is `GeometricBase`'s function, so it has no docstring here to link to.)

# Implementation

`NeuralNetworkParameters.foldparameters` and not `map` + `sum` over one level, because a container is a
tree of layers and the quantities to combine are at its *leaves* — `map` would hand `f` a whole layer,
for which neither of them has a method. `foldparameters` recurses on the branches and so reaches a leaf
at any depth, and it allocates nothing. Its body is generated at literal indices, so its compile time
is linear in the number of children of a branch.

`foldparameters` and not `foldstorage`, which is the pairing `_dot` wants and would be wrong here.
`foldstorage` descends past a leaf into the numbers it stores, and
[`solution_scale`](@ref)`(Y::Manifold)` is `√size(Y, 2)` — a *nominal* value that has nothing to do with
those numbers.

`false` and not `zero(T)` for the initial value: there is no `T` in scope, and `false` is the strong
zero that takes its type from whatever it is added to, so a one-block set adds it to that block's value
and stays a `T`. A set has one element type, so every leaf gives it the same `T`.
`zero(parameter_eltype(x))` would put the generated `_promote_eltypes` body of upstream on this path,
once per branch shape, which costs compile time.

`f` is annotated `::F where {F}`, so that the method specialises on it even where it only passes `f`
on: Julia does not specialise on a function argument that it does not see called. The docstring of
`foldparameters` asks this of every caller that hands on an `op`.
"""
_sumsq_leaves(f::F, x) where {F} = foldparameters((acc, y) -> acc + abs2(f(y)), false, x)

@doc raw"""
    contains_nonfinite(a)

Whether `a` holds any value that is not finite.

`NaN` is the *last* thing a diverging solve produces: it reaches `Inf` first, and before that every
finite magnitude on the way, so this tests `isfinite` and not `isnan`.

`isfinite` does not catch `1e200`, so it is not the only guard; see
[`convergence_measures`](@ref) for the one that does.
"""
contains_nonfinite(a::Real) = !isfinite(a)
# over the storage of every leaf, which holds the same values as the dense interface does: the
# interface of a manifold point or a structured matrix is read entry by entry, which a device
# refuses, while `any` over a storage array is a reduction on its backend
contains_nonfinite(a) = foldstorage(_any_nonfinite, false, a)
_any_nonfinite(acc, storage) = acc || any(!isfinite, storage)

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
function isconverged(status::OptimizerStatus)
    status.x_converged || status.f_converged || status.g_converged
end

@doc raw"""
    convergence_measures(status, config)

Checks if the optimizer converged.

Here `status` is an [`OptimizerStatus`](@ref) object and `config` is an [`SimpleSolvers.Options`](@extref) object.

# Extended help

!!! info "Which iterate `g_converged` is a statement about"
    `rg` is [`latest_gradient`](@ref), i.e. ``\|\nabla{}f(x_{k+1})\|`` at the iterate the step ended
    at, for every method, and not at the iterate it started from or at the last point that the line
    search probed.

    The distinction matters for a direction that carries momentum. Under
    [`SimpleSolvers.Static`](@extref) a stale `rg` is harmless, because the direction is a scaled
    gradient and a vanishing gradient means a vanishing step. Under a line search accurate enough to
    drive ``\nabla{}f(x_{k+1}) \approx 0`` it is not: the momentum term still moves the iterate, and
    a stale `rg` fires `g_converged` one step *past* the minimiser. The refresh costs one gradient
    evaluation per iteration.

!!! warning "What `x_converged` is guarded against, and what it is not"
    ``\|x - x'\|/\|x'\|`` measures "the iterate stopped moving" only while ``\|x'\|`` is bounded, and
    a diverging solve is the case where it is not: a large step from an iterate of magnitude
    ``10^{100}`` is a small *relative* change. Two guards, neither of which invents a tolerance:

    - the denominator is [`solution_scale`](@ref) and not ``\|x'\|``. On a manifold the two are the
      same number — ``\|Y\|_F = \sqrt{n}`` exactly — up to the point where the iterate leaves the
      manifold, and there the constant is the true scale and the measured norm is the divergence.
    - `x_converged` also requires `!f_increased`. An iterate that has stopped moving is evidence of
      convergence only if the objective did not just go up. This is the only guard of the
      *Euclidean* case, because nothing there bounds ``\|x\|``.

    Not covered: a Euclidean solve that runs away *downhill*, where ``\|x\|`` grows without bound
    and `f` decreases at every step, has no scale to be measured against and is reported as
    converged. A threshold on ``\|x\|`` would close it, but no property of the problem supplies one:
    on a manifold the geometry supplies [`DEFAULT_STEP_CEILING`](@ref), and in ``\mathbb{R}^n``
    nothing does.

    The divergences are also stopped at their source — [`linesearch_rejected`](@ref) and
    [`curvature_is_usable`](@ref) for a search that *failed*, `DEFAULT_STEP_CEILING` for one that
    *succeeded* on a step too long for the manifold — and [`contains_nonfinite`](@ref) catches the
    `Inf`/`NaN` end of the range.
"""
function convergence_measures(status::OptimizerStatus, config::Options)
    # `&& !f_increased`: both branches above say "the iterate stopped moving", which is evidence of
    # convergence only if the objective did not just go up. See the warning above.
    x_converged = (x_abschange(status) ≤ x_abstol(config) ||
                   x_relchange(status) ≤ x_reltol(config)) && !status.f_increased

    # `f_relchange` is a *successive* change, so it is gated on `f_suctol`. `f_reltol` is anchored to
    # the initial residual and defaults to `√eps(T)`, much looser, so it would stop a `Static` line
    # search long before the minimizer.
    f_converged = f_abschange(status) ≤ f_abstol(config) ||
                  f_relchange(status) ≤ f_suctol(config)

    f_converged_strong = f_change(status) ≤ f_mindec(config) * f_change_approx(status)

    # `Options` has no gradient tolerance; `f_reltol` is the residual gate, with the default
    # `√eps(T)`. The residual of an optimizer is `‖∇f(x)‖`, so this is the corresponding gate here.
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
    No iteration can recover a `NaN` iterate, so the solve stops with an `@error` and does not spend
    the rest of its iteration budget.

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
        # `f_abstol_break` is the only `*_break` field of `Options`.
        status.rfₐ > config.f_abstol_break
end
