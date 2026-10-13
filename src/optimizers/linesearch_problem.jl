@doc raw"""
    trial_iterate!(cache, params, α, retraction, workspace = nothing)

Write the iterate a step of length `α` along the current direction would produce into
`solution(cache)`, and return it. This is what the merit function of the line search evaluates.

# Implementation

There are two methods, chosen on the type of the solution.

For an `AbstractVector` the iterate is ``x_k + \alpha{}p_k``, i.e.
[`SimpleSolvers.compute_new_iterate!`](@extref).

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
trial_iterate!(cache::OptimizerCache, params, α, retraction, workspace = nothing;
    scaled = nothing) = _trial_iterate!(
    solution(cache), cache, params, α, retraction, workspace; scaled)

function _trial_iterate!(
        ::AbstractVector, cache::OptimizerCache, params, α, ::AbstractRetraction, _;
        scaled = nothing)
    compute_new_iterate!(solution(cache), params.x, α, direction(cache))
end

# The `Manifold`/parameter-set arm: the direction is scaled by `α` and the *state's* section is
# pushed through the retraction into the cache's section, from which the trial point is read back.
#
# `scaled` is the buffer `linesearch_problem` builds for this, so that a search of any length writes
# into one array instead of allocating a copy of the lift per trial. It is `nothing` for a caller
# that has no optimizer — a bare `trial_iterate!` from outside a solve — and then the copy is made,
# because `direction(cache)` has to stay intact for the next trial step.
function _trial_iterate!(
        ::Union{Manifold, NetworkParameters}, cache::OptimizerCache, params, α, retraction,
        workspace; scaled = nothing)
    # `params` is a concrete `NamedTuple` here, so this is constant-folded away rather than checked on
    # every merit evaluation. Without it a missing `state` surfaces as `has no field state`.
    hasproperty(params, :state) ||
        error("a trial step on a manifold retracts from `section(params.state)`, so the line " *
              "search parameters have to carry the `state`; `solver_step!` passes it, a bare " *
              "`(x = x,)` does not.")
    δ = scaled === nothing ? _mul(α, direction(cache)) : _mul!(scaled, α, direction(cache))
    update_section!(section(cache), section(params.state), δ, retraction, workspace)
    _copyto!(solution(cache), section(cache))
end

# `_mul` written into `dest`: the scaled copy the line search needs, without the copy. `dest` is the
# search's own buffer and nothing else reads it between two trial steps.
function _mul!(dest, α::Number, a)
    _copyto!(dest, a)
    _rmul!(dest, α)
    dest
end

@doc raw"""
    step_αmax(c, δ)

The ceiling on the line search's ``\alpha`` that a step ceiling of `c` imposes on the direction `δ`:
``c\,2\pi/\|\delta\|``.

`c` is in multiples of ``2\pi`` because that is the scale the *geometry* supplies. Retracting a
horizontal lift is a rotation, so ``\alpha\mapsto\mathrm{retract}(\alpha\bar{B})`` has nothing left to
reach once ``\|\alpha\bar{B}\|`` passes ``2\pi`` — [`Cayley`](@ref) has converged to a fixed rotation
by then and [`Geodesic`](@ref) is periodic. Everything past it is round-off, and on the SVD problem
enough of it to leave the manifold; see [`DEFAULT_STEP_CEILING`](@ref).

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
    _manifold_αmax(solution, direction, c)

The ceiling a step ceiling of `c` imposes on a solution made of several blocks: the smallest of the
per-block [`step_αmax`](@ref) over the blocks that live on a [`Manifold`](@ref), and `Inf` where none
does.

One `\alpha` is applied to every block, so each manifold block needs ``\|\alpha\delta_i\| \leq 2\pi{}c``
and the binding one is the largest ``\|\delta_i\|``. A block that is an ordinary array contributes
nothing: the ``2\pi`` is the turn of a rotation and a vector space has no such scale, so it must
neither impose a ceiling of its own nor inflate the norm that sets one for its neighbours. A ceiling
of `step_αmax(c, direction(cache))` over the whole direction would do both: its `l2norm` combines the
blocks in quadrature.

# Implementation

`NeuralNetworkParameters.foldparameters` over the two trees in lockstep keeps a *heterogeneous*
`NamedTuple` — a `StiefelManifold` beside a `Matrix` beside a `Vector` — inferable, and its compile
time linear in the number of blocks. This function is on the per-iteration path, through
[`linesearch_parameters`](@ref).

`foldparameters` and not `foldstorage`: `_block_αmax` dispatches on the *whole* leaf, because a
`Manifold` is the only kind that imposes a ceiling. A descent into the numbers that a leaf stores
would make every manifold block an ordinary array and give back `Inf`.

`min` is associative and commutative, so the direction of the fold does not matter. A set with no
manifold in it gives `Inf` through the `_block_αmax(::Any, …)` method, and a set with no blocks
through the initial value `T(Inf)`.

The fold pairs the blocks by key and checks the widths: `δ` follows the solution block by block, but
not always as the same kind of branch, because a container solution can be paired with the plain
`NamedTuple` of its `GlobalSection` tree.

The signature takes `y::NetworkParameters` and not an unconstrained `y`. Upstream pairs the children of
a keyed branch by key, and the children of a `Tuple` branch **positionally**, so
`_manifold_αmax(values(sol), values(δ), c)` would fold crossed keys without an error. Every caller
passes a whole set of parameters.
"""
_manifold_αmax(y::NetworkParameters, δ, c::T) where {T} = foldparameters(
    (acc, yᵢ, δᵢ) -> min(acc, _block_αmax(yᵢ, δᵢ, c)), T(Inf), y, δ)

_block_αmax(::Manifold, δ, c) = step_αmax(c, δ)
_block_αmax(::Any, ::Any, c::T) where {T} = T(Inf)

@doc raw"""
    linesearch_parameters(cache, x, state, c)

The parameters [`solver_step!`](@ref) hands the line search: the iterate `x`, the `state` whose
section the trial steps retract from, and — where a manifold supplies a scale — the step ceiling
`αmax`.

# Implementation

Three methods, chosen on the type of the solution, as [`trial_iterate!`](@ref)'s two are.

For a [`Manifold`](@ref), `αmax` is [`step_αmax`](@ref)`(c, direction(cache))`. It is built again
at every solver step, because ``\|\delta\|`` changes at every solver step — also between the two
searches of one `solver_step!`, where the second runs on the direction that
[`steepest_descent!`](@ref) has replaced.

For a `NamedTuple` it is [`_manifold_αmax`](@ref) over the blocks, i.e. the same quantity derived per
block and minimised over the manifold ones. A `NamedTuple` carrying no manifold block therefore gets
`Inf`, which is what upstream reads as *"the caller has no scale of its own"*
([`SimpleSolvers.linesearch_αmax`](@extref)) and is equivalent to the omission below.

For an `AbstractVector` the field is **omitted**. Upstream reads it through a `hasproperty` guard
that constant-folds, so a caller that supplies nothing pays nothing. A Euclidean
``f(x + \alpha{}p)`` *grows* with ``\alpha``, so the sufficient-decrease test of the search rejects
an over-long step. There is no geometric scale to supply, and the method's own
`SimpleSolvers.DEFAULT_LINESEARCH_αmax` is the whole of the bound.

Every branch returns a concrete `NamedTuple`, so the merit closures stay type-stable. That is why the
manifold-free `NamedTuple` passes `Inf` rather than joining the `AbstractVector` branch: deciding the
*shape* of the parameters on the block types makes the return type a `Union` of two `NamedTuple`s,
which the merit closures then pay for on every evaluation.
"""
linesearch_parameters(cache::OptimizerCache, x, state, c) = _linesearch_parameters(
    solution(cache), cache, x, state, c)

function _linesearch_parameters(::AbstractVector, ::OptimizerCache, x, state, _)
    (x = x, state = state)
end

function _linesearch_parameters(::Manifold, cache::OptimizerCache, x, state, c)
    (x = x, state = state, αmax = step_αmax(c, direction(cache)))
end

function _linesearch_parameters(sol::NetworkParameters, cache::OptimizerCache, x, state, c)
    (x = x, state = state,
        αmax = _manifold_αmax(sol, direction(cache), c))
end

# The ceiling a `linesearch_parameters` actually carries, read back by `solver_step!` so that it can
# recognise a step of its own making; see `linesearch_rejected`. This mirrors
# SimpleSolvers' `caller_αmax`, which is what reads the field on the other side: `Inf` where the
# field is absent, and `hasproperty` on a concrete `NamedTuple` constant-folds either way.
function _caller_αmax(::Type{T}, params) where {T}
    hasproperty(params, :αmax) ? T(params.αmax) : T(Inf)
end

@doc raw"""
    trial_slope(gradient_instance, cache, retraction, α, differential, workspace)

Return ``\varphi'(\alpha) = \langle\nabla{}f(x(\alpha)), D(\alpha)\rangle`` at the iterate currently
held in `solution(cache)`, for the derivative of the line search's merit. ``D(\alpha)`` is
[`retraction_differential`](@ref)`(retraction, direction(cache), α)`, written into `differential`
by [`retraction_differential!`](@ref) in `workspace`; [`linesearch_problem`](@ref) builds
`differential` once, in the shape of the direction. On a vector the pairing is with the direction
itself, and `differential` is not read.

# Implementation

On a [`Manifold`](@ref) the gradient that `gradient_instance` produces lives in the tangent space of
the embedding while `direction(cache)` is a horizontal lift, so the two are not even the same shape
and have to be brought together by [`global_rep`](@ref) first — which is the same map
`update!(::GradientCache, …)` applies to the gradient before storing it.

The pairing is then [`_dot`](@ref) and not `dot`: the `α` of the line search parameterizes a curve in
the *intrinsic* coordinates of the lift, and `dot` on a lift is the ambient Frobenius product, which
is twice that. With `dot` the slope is `2\varphi'(\alpha)`, which a curvature condition or a
quadratic fit acts on.

The direction the gradient is paired *with* is [`retraction_differential`](@ref)`(retraction, B, α)`
and not `B` itself. ``\varphi'(\alpha) = \langle\nabla{}f(x(\alpha)), B\rangle`` holds only where
``\alpha \mapsto \mathrm{retract}(\alpha{}B)`` is a one-parameter subgroup, which [`Geodesic`](@ref)
is and [`Cayley`](@ref) is not; under `Cayley` the generator of the curve's velocity turns with the
step, and the differential supplies it. The slope agrees with a central difference of the merit for
both retractions at every ``\alpha``.

At `α = 0` the differential is `B`, copied into `differential`, so the `Backtracking` default pays a
copy and no solve for it.
"""
trial_slope(
    gradient_instance::Gradient, cache::OptimizerCache, retraction, α, differential,
    workspace; buffers = nothing) = _trial_slope(
    solution(cache), gradient_instance, cache, retraction, α, differential, workspace; buffers)

# These two differ in one respect worth knowing about: the `AbstractVector` method evaluates *into* an
# array of the cache — that is what makes it allocation-free — and so leaves the gradient at the last
# trial `α` there, while the manifold method allocates and leaves the cache alone.
#
# The array it writes into is `latest_gradient(cache)` and not `gradient(cache)`, on every cache, for
# two reasons. On the three first-order caches a shared array corrupts the *state*:
# `update!(::MomentumState, ...)` runs `p ← αp + ∇f(xₖ)` from `gradient_array(cache)` *after* the
# search, so the momentum would take the gradient at the last trial step of the search. On the
# quasi-Newton caches a shared array would make `rg` a statement about the last point that the line
# search probed, not about the iterate that the solve returns. `Backtracking` evaluates `φ'` once, at
# `α = 0`, where the trial gradient *is* `∇f(xₖ)`, so it is exact either way.
function _trial_slope(::AbstractVector, gradient_instance::Gradient,
        cache::OptimizerCache, ::AbstractRetraction, α, _, _; buffers = nothing)
    gradient_instance(latest_gradient(cache), solution(cache))
    _dot(latest_gradient(cache), direction(cache))
end

# The buffers a search of any length writes into, so that a trial step and a trial slope cost their
# products and nothing else. The same idea as `RetractionWorkspace` for the retraction and
# `_differential_buffer` for the differential: they are built once per `Optimizer` and live in the
# merit closures.
#
# The scaled direction is a copy of `direction(cache)` because the direction has to survive the next
# trial step — `solver_step!` can scale in place only because it scales exactly once, by the `α` the
# search has settled on. The lift is the destination the ambient gradient is lifted into, which on a
# manifold is a `global_rep!` write and on a vector is nothing at all.
_trial_buffers(::AbstractVector, direction) = nothing
function _trial_buffers(::Union{Manifold, NetworkParameters}, direction)
    (scaled = _similar(direction), lift = _similar(direction))
end

function _trial_slope(::Union{Manifold, NetworkParameters}, gradient_instance::Gradient,
        cache::OptimizerCache, retraction::AbstractRetraction, α, differential, workspace;
        buffers = nothing)
    ∇ = gradient_instance(solution(cache))
    lift = buffers === nothing ? global_rep(section(cache), ∇) :
           global_rep!(buffers.lift, section(cache), ∇)
    _dot(lift, retraction_differential!(
        differential, workspace, retraction, direction(cache), α))
end

# The lift `trial_slope` writes the retraction's differential into, dispatched as `_trial_slope` is:
# the `AbstractVector` method pairs the gradient with the direction itself and reads no lift.
_differential_buffer(::AbstractVector, direction) = nothing
_differential_buffer(::Union{Manifold, NetworkParameters}, direction) = _similar(direction)

@doc raw"""
    linesearch_problem(problem, gradient, cache, retraction, observer = NoStepObserver(),
        workspace = nothing)

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

julia> state = NewtonState(x); update!(state, grad, x);

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
function linesearch_problem(problem::OptimizerProblem{T}, gradient_instance::Gradient,
        cache::OptimizerCache{T}, retraction::AbstractRetraction,
        observer = NoStepObserver(), workspace = nothing) where {T}
    # built once per problem, and `nothing` for a vector-space problem
    differential = _differential_buffer(solution(cache), direction(cache))
    buffers = _trial_buffers(solution(cache), direction(cache))

    function f(α, params)
        observe_optimizer_phase(observer, :retraction_application) do
            trial_iterate!(
                cache, params, α, retraction, workspace;
                scaled = buffers === nothing ?
                         nothing : buffers.scaled)
        end
        observe_optimizer_phase(observer, :objective) do
            value(problem, solution(cache))
        end
    end

    function d(α, params)
        observe_optimizer_phase(observer, :retraction_application) do
            trial_iterate!(
                cache, params, α, retraction, workspace;
                scaled = buffers === nothing ?
                         nothing : buffers.scaled)
        end
        observe_optimizer_phase(observer, :retraction_application) do
            trial_slope(
                gradient_instance, cache, retraction, α, differential, workspace; buffers)
        end
    end

    LinesearchProblem{T}(f, d)
end
