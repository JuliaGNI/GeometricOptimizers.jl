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

function _trial_iterate!(::Union{Manifold,ParameterContainer}, cache::OptimizerCache, params, α, retraction)
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
    step_αmax(c, δ)

The ceiling on the line search's ``\alpha`` that a step ceiling of `c` imposes on the direction `δ`:
``c\,2\pi/\|\delta\|``.

`c` is in multiples of ``2\pi`` because that is the scale the *geometry* supplies. Retracting a
horizontal lift is a rotation, so ``\alpha\mapsto\mathrm{retract}(\alpha\bar{B})`` has nothing left to
reach once ``\|\alpha\bar{B}\|`` passes ``2\pi`` — [`Cayley`](@ref) has converged to a fixed rotation
by then and [`Geodesic`](@ref) is periodic. Everything past it is round-off, and on the SVD problem
enough of it to leave the manifold; see [`DEFAULT_STEP_CEILING`](@ref) and issue A1b.

`δ` is one direction and not a `NamedTuple` of them: the ``2\pi`` is a property of *a* rotation, so a
solution built of several blocks needs one ceiling per block and the smallest of them, which is
[`_manifold_αmax`](@ref)'s job. See [`linesearch_parameters`](@ref).

# Implementation

A direction whose norm is zero or not finite yields `Inf`, which
[`SimpleSolvers.linesearch_αmax`](@extref) reads as *"the caller has no scale of its own"* and which
leaves the method's own ceiling standing. That is a guard and not a formality: upstream raises an
`ArgumentError` on a `NaN` or non-positive `params.αmax` before it evaluates the merit — correctly,
since silently ignoring one would hand back exactly the unbounded step the caller was ruling out — so
a vanishing direction has to be special-cased here rather than there.
"""
function step_αmax(c::T, δ) where {T}
    n = l2norm(δ)
    (isfinite(n) && n > zero(n)) ? c * T(2π) / T(n) : T(Inf)
end

@doc raw"""
    _manifold_αmax(solution_blocks, direction_blocks, c)

The ceiling a step ceiling of `c` imposes on a solution made of several blocks: the smallest of the
per-block [`step_αmax`](@ref) over the blocks that live on a [`Manifold`](@ref), and `Inf` where none
does.

One `\alpha` is applied to every block, so each manifold block needs ``\|\alpha\delta_i\| \leq 2\pi{}c``
and the binding one is the largest ``\|\delta_i\|``. A block that is an ordinary array contributes
nothing: the ``2\pi`` is the turn of a rotation and a vector space has no such scale, so it must
neither impose a ceiling of its own nor inflate the norm that sets one for its neighbours.

That last part is what this function exists for. The ceiling used to be
`step_αmax(c, direction(cache))` over the whole direction, whose `l2norm` combines the blocks in
quadrature — so a `NamedTuple` of *ordinary arrays* was bounded by a rotation that does not exist in
its problem (measured: a Euclidean `NamedTuple` solve took 3 184 iterations against 1 for the same
problem written as a vector), and in a mixed `NamedTuple` the Euclidean blocks tightened the manifold
blocks' bound for no reason (measured on the mixed problem of `test/named_tuple_parameters.jl`:
``\|\delta_Y\| = 2.5\times10^{-16}`` against a total of `3.9`, bounding ``\alpha`` at `1.6` where the
geometry of the manifold block permits ``2.6\times10^{16}``). That was catalogued as issue A15 and is
what this closes.

# Implementation

Recursive over the two tuples rather than a loop over `zip`, so that a *heterogeneous* `NamedTuple`
— a `StiefelManifold` beside a `Matrix` beside a `Vector` — stays inferable and the returned
parameters stay concrete.
"""
_manifold_αmax(::Tuple{}, ::Tuple{}, c::T) where {T} = T(Inf)

_manifold_αmax(ys::Tuple, δs::Tuple, c) =
    min(_block_αmax(first(ys), first(δs), c), _manifold_αmax(Base.tail(ys), Base.tail(δs), c))

_block_αmax(::Manifold, δ, c) = step_αmax(c, δ)
_block_αmax(::Any, ::Any, c::T) where {T} = T(Inf)

# A block that is itself a branch is descended into rather than being written off as "not a manifold".
# Without this a container -- which is a tree of layers, so *every* block at the top level is a branch
# -- would take the `::Any` method above for all of them and get `Inf`, i.e. no ceiling at all, and
# issue A1b would be back for exactly the parameter shape a network has. A flat `ArrayNamedTuple`
# never reaches it, its values being arrays by construction.
# `_as_walkable` — `src/parameter_walks.jl` — is `values` of whichever shape the direction arrived in.
# It tracks the solution block by block, so it is a branch wherever the solution is one, but not
# necessarily the *same* kind of branch: a container solution can be paired with the plain `NamedTuple`
# its `GlobalSection` tree is built as. This used to be a second copy of that function under the name
# `_as_blocks`.
_block_αmax(y::ParameterSet, δ, c) =
    _manifold_αmax(values(y), values(_as_walkable(δ)), c)

@doc raw"""
    linesearch_parameters(cache, x, state, c)

The parameters [`solver_step!`](@ref) hands the line search: the iterate `x`, the `state` whose
section the trial steps retract from, and — where a manifold supplies a scale — the step ceiling
`αmax`.

# Implementation

Three methods, chosen on the type of the solution, as [`trial_iterate!`](@ref)'s two are.

For a [`Manifold`](@ref), `αmax` is [`step_αmax`](@ref)`(c, direction(cache))`. This is the caller's
half of the fix for issue A1b, and it has to be rebuilt at every solver step because
``\|\delta\|`` changes at every solver step — including between the two searches of one
`solver_step!`, since the second runs on a direction [`steepest_descent!`](@ref) has just replaced.

For a `NamedTuple` it is [`_manifold_αmax`](@ref) over the blocks, i.e. the same quantity derived per
block and minimised over the manifold ones. A `NamedTuple` carrying no manifold block therefore gets
`Inf`, which is what upstream reads as *"the caller has no scale of its own"*
([`SimpleSolvers.linesearch_αmax`](@extref)) and is exactly equivalent to the omission below.

For an `AbstractVector` the field is **omitted**, which is not the same as passing `Inf` only in
spirit — upstream reads it through a `hasproperty` guard that constant-folds, so a caller that
supplies nothing pays nothing. Omitting it is also the right answer and not merely the cheap one: a
Euclidean ``f(x + \alpha{}p)`` *grows* with ``\alpha``, so the search's own sufficient-decrease test
throws an over-long step out unaided, which is why this defect went unreported upstream for so long.
There is no geometric scale to supply and the method's own
`SimpleSolvers.DEFAULT_LINESEARCH_αmax` is the whole of the bound.

Every branch returns a concrete `NamedTuple`, so the merit closures stay type-stable. That is why the
manifold-free `NamedTuple` passes `Inf` rather than joining the `AbstractVector` branch: deciding the
*shape* of the parameters on the block types makes the return type a `Union` of two `NamedTuple`s,
which the merit closures then pay for on every evaluation.
"""
linesearch_parameters(cache::OptimizerCache, x, state, c) =
    _linesearch_parameters(solution(cache), cache, x, state, c)

_linesearch_parameters(::AbstractVector, ::OptimizerCache, x, state, _) = (x=x, state=state)

_linesearch_parameters(::Manifold, cache::OptimizerCache, x, state, c) =
    (x=x, state=state, αmax=step_αmax(c, direction(cache)))

_linesearch_parameters(sol::ParameterContainer, cache::OptimizerCache, x, state, c) =
    (x=x, state=state,
     αmax=_manifold_αmax(values(sol), values(_as_walkable(direction(cache))), c))

# The ceiling a `linesearch_parameters` actually carries, read back by `solver_step!` so that it can
# recognise a step of its own making; see `linesearch_rejected` and issue B3. This mirrors
# SimpleSolvers' `caller_αmax`, which is what reads the field on the other side: `Inf` where the
# field is absent, and `hasproperty` on a concrete `NamedTuple` constant-folds either way.
_caller_αmax(::Type{T}, params) where {T} = hasproperty(params, :αmax) ? T(params.αmax) : T(Inf)

@doc raw"""
    trial_slope(gradient_instance, cache, retraction, α)

Return ``\varphi'(\alpha) = \langle\nabla{}f(x(\alpha)), D(\alpha)\rangle`` at the iterate currently
held in `solution(cache)`, for the derivative of the line search's merit. ``D(\alpha)`` is
[`retraction_differential`](@ref)`(retraction, direction(cache), α)`.

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
    `test/optimizer_convergence/svd_optim.jl` that took `BFGS` off the manifold altogether on two of
    eight starting points. The exact slope did *not* fix that — the cause was the size of the step,
    not the slope, and [`DEFAULT_STEP_CEILING`](@ref) is what closed it. See the CHANGELOG entry for
    issue A1b; this differential remains worth having on its own account.

    `α = 0` still returns `B` untouched, so the `Backtracking` default costs nothing for this.
"""
trial_slope(gradient_instance::Gradient, cache::OptimizerCache, retraction, α) =
    _trial_slope(solution(cache), gradient_instance, cache, retraction, α)

# These two differ in one respect worth knowing about: the `AbstractVector` method evaluates *into* an
# array of the cache — that is what makes it allocation-free — and so leaves the gradient at the last
# trial `α` there, while the manifold method allocates and leaves the cache alone.
#
# Which array it writes into is `latest_gradient(cache)` and deliberately not `gradient(cache)`, on
# every cache. Two reasons, and they were fixed one release apart. On the three first-order caches a
# shared array corrupts the *state*: `update!(::MomentumState, ...)` re-runs `p ← αp + ∇f(xₖ)` from
# `gradient_array(cache)` *after* the search, so the momentum accumulated the gradient at whatever
# trial step the search last probed — measured over eight iterations, 104% wrong under `Bisection`,
# `Quadratic` and `StrongWolfe` and 451% wrong under `BierlaireQuadratic` (issue A2). On the
# quasi-Newton caches nothing downstream reads `cache.g` again, so a shared array corrupted nothing —
# but it made `rg` a statement about the last point the line search probed rather than about the
# iterate the solve returns, which is issue A8. `Backtracking` is exact for both, and only because it
# evaluates `φ'` once, at `α = 0`, where the trial gradient *is* `∇f(xₖ)`.
function _trial_slope(::AbstractVector, gradient_instance::Gradient, cache::OptimizerCache, ::AbstractRetraction, α)
    gradient_instance(latest_gradient(cache), solution(cache))
    _dot(latest_gradient(cache), direction(cache))
end

function _trial_slope(::Union{Manifold,ParameterContainer}, gradient_instance::Gradient, cache::OptimizerCache, retraction::AbstractRetraction, α)
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
