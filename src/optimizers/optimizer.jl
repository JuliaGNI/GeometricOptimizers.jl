
const SOLUTION_MAX_PRINT_LENGTH = 10

@doc raw"""
    DEFAULT_STEP_CEILING

How far along its direction a manifold solve may step, in multiples of ``2\pi``: the `c` of
[`step_αmax`](@ref)`(c, δ)`, and hence the `params.αmax` [`solver_step!`](@ref) hands the line search
through [`linesearch_parameters`](@ref). Set per solve with `Optimizer(x, F; step_ceiling = …)`;
`Inf` switches the ceiling off.

# Why a ceiling is needed at all, and why the caller has to set it

A line search bounds the step it returns by the merit — it stops when ``\varphi`` stops falling. On a
**compact** manifold ``\varphi`` is *bounded*, so that test never fires: at ``\alpha = 10^9``,
``\varphi`` can be genuinely lower than at ``\alpha = 0``, and a search that reports a decrease there
is telling the truth. Measured on the SVD problem of `test/optimizer_convergence/svd_optim.jl`,
[`SimpleSolvers.Quadratic`](@extref) returned ``\alpha = 4.3\times10^7`` on a direction of norm 5.54 —
a step of ``\|\alpha\delta\| = 2.4\times10^8`` — and did it again two steps later on a
*steepest-descent* direction, so the behaviour belonged to the search and not to what it was handed.
Retracting a lift of that norm loses the manifold, and the solve then reported convergence from a
point that was no longer on ``St(20, 3)``. That was issue A1b.

**The merit is not a bound on the step**, and the bound that does exist is not a property of
``\varphi`` at all: it is the ``2\pi`` of a rotation divided by ``\|\delta\|``, it changes at every
solver step, and nothing a line search can measure reveals it. SimpleSolvers 0.12 therefore splits the
ceiling in two — a method's own `DEFAULT_LINESEARCH_αmax` backstop, and a per-call `params.αmax` — and
leaves this half here deliberately. Upstream measured why the backstop alone cannot close it: at
``\|\delta\| \approx 5.5`` its `65536` still permits ``\|\alpha\delta\| = 3.6\times10^5``, five orders
above the ``2\pi`` past which a rotation stops going anywhere, and four of the eight starting points
still diverge.

# Why `1`

`c` is the ceiling in multiples of ``2\pi``, so `1` says *"never step further along the direction than
one full turn"*. That is the scale the geometry gives; the value is not tuned to the SVD problem
beyond confirming that it leaves it alone. Over the converging solve next to the diverging one — seed
2 at 90 iterations — the largest ``\|\alpha\delta\|`` is `2.03`, well inside ``2\pi``, which is what
makes a ceiling that forbids the ``10^8`` step cost the ordinary ones nothing.

!!! info "This bounds the step, not the direction and not `α`"
    The same solve accepts ``\alpha = 18.4`` on an iteration where ``\|\delta\| = 5.8\times10^{-5}``,
    i.e. a perfectly ordinary step of ``10^{-3}``, and converges. A ceiling on ``\alpha`` alone would
    reject it. The direction is not what is wrong either: at the step that loses the manifold
    ``\|\delta\| = 5.54``, in the same range as the two before it, and ``Q`` is well conditioned
    (``\lambda_\mathrm{max} = 3.86``). See [`ensure_descent!`](@ref) and [`linesearch_rejected`](@ref)
    for the two defects that *are* about the direction and the outcome.

!!! info "Euclidean parameters do not get one"
    [`linesearch_parameters`](@ref) omits `αmax` for an `AbstractVector`, and passes `Inf` — which
    says the same thing — for a parameter set carrying no manifold block. There is no geometric scale
    in either case, and none is needed: ``f(x + \alpha{}p)`` grows with ``\alpha``, so the search's
    own decrease test rejects an over-long step unaided.

    Telling the two apart is not free, and getting it wrong bounds a set of parameters carrying no
    manifold block by a rotation its problem does not have. [`_manifold_αmax`](@ref) derives the
    ceiling per block, over the manifold blocks only, which is what makes the distinction.

!!! info "One `α`, one ceiling per block"
    On a parameter set the same ``\alpha`` scales every block, so the ceiling is the *smallest* of the
    per-block ones over the blocks that live on a manifold — [`_manifold_αmax`](@ref) — and not
    ``2\pi{}c`` over the norm of the whole direction. The latter is what this was written as, and it
    made every block pay for its neighbours: on the SVD problem, where both blocks are manifolds,
    combining them in quadrature tightened the bound by up to ``\sqrt{2}`` and was the only reason
    the ceiling bound anything on the pinned seed at all. That was issue A15.
"""
const DEFAULT_STEP_CEILING = 1.0

"""
    Optimizer

The optimizer that stores all the information needed for an optimization problem.

This problem can be solved by calling [`solve!(::AbstractVector, ::Optimizer)`](@ref).

# Keys
- `algorithm::`[`OptimizerState`](@ref),
- `problem::`[`OptimizerProblem`](@ref),
- `gradient::`[`SimpleSolvers.Gradient`](@extref),
- `hessian::`[`SimpleSolvers.Hessian`](@extref),
- `config::`[`SimpleSolvers.Options`](@extref),
- `cache::`[`OptimizerCache`](@ref),
- `linesearch::`[`SimpleSolvers.Linesearch`](@extref).

# Examples

```jldoctest; setup = :(using GeometricOptimizers)
F(x) = sum(sin.(x) .^ 2)
x = ones(3)
algorithm = Newton()
state = OptimizerState(algorithm, x)
optimizer = Optimizer(x, F; algorithm = algorithm, linesearch = Bisection())

solve!(x, state, optimizer)
x

# output

3-element Vector{Float64}:
 0.0
 0.0
 0.0
```

The choice of line search does not change *which* minimum this one finds, only how close to it the
stopping criterion lets the solve get — `Backtracking` stops at `1.4e-9` per component, where
`Bisection` above happens to land on `0` exactly:

```jldoctest; setup = :(using GeometricOptimizers; F(x) = sum(sin.(x) .^ 2))
x = ones(3)
algorithm = Newton()
state = OptimizerState(algorithm, x)
optimizer = Optimizer(x, F; algorithm = algorithm, linesearch = Backtracking())

solve!(x, state, optimizer)
F(x) < 1e-15

# output

true
```

!!! info "This example needs the descent safeguard"
    ``\\sin^2`` has second derivative ``2\\cos(2x) = -0.83`` at `x = 1`, so the Newton direction there
    points *uphill*. [`ensure_descent!`](@ref) replaces it by the steepest-descent direction; without
    it both solves converge to ``\\pi/2``, where `F` is maximal.

"""
struct Optimizer{T,
    ALG <: OptimizerMethod,
    OBJ <: OptimizerProblem{T},
    GT <: Gradient{T},
    HT <: Hessian{T},
    OCT <: Union{OptimizerCache, NamedTuple},
    LST <: Linesearch,
    RT <: AbstractRetraction,
    OT} <: AbstractSolver
    algorithm::ALG
    problem::OBJ
    gradient::GT
    hessian::HT
    config::Options{T}
    cache::OCT
    linesearch::LST
    retraction::RT
    step_ceiling::T
    observer::OT

    # Everything positional, and in particular the `Options` already built. See the note on Julia 1.12
    # below `Optimizer(x, F)`.
    function Optimizer(algorithm::OptimizerMethod, problem::OptimizerProblem{T},
            hessian::Hessian{T}, cache::OptimizerCache, linesearch::LinesearchMethod,
            config::Options{T}, gradient::Gradient{T}, retraction::AbstractRetraction,
            step_ceiling::Real = DEFAULT_STEP_CEILING,
            observer = NoStepObserver()) where {T}
        observed_gradient = _observed_gradient(gradient, observer)
        ls_problem = linesearch_problem(
            problem, observed_gradient, cache, retraction, observer)
        ls = Linesearch(ls_problem, linesearch)
        new{T, typeof(algorithm), typeof(problem), typeof(observed_gradient),
            typeof(hessian), typeof(cache), typeof(ls), typeof(retraction), typeof(observer)}(
            algorithm, problem, observed_gradient, hessian, config,
            cache, ls, retraction, T(step_ceiling), observer)
    end
end

function Optimizer(algorithm::OptimizerMethod, problem::OptimizerProblem{T},
        hessian::Hessian{T}, cache::OptimizerCache, linesearch::LinesearchMethod;
        gradient = default_gradient(problem, cache.x), retraction = Cayley(),
        step_ceiling = DEFAULT_STEP_CEILING, observer = NoStepObserver(),
        options_kwargs...) where {T}
    # `_riemannian_gradient` here as in the two methods below, so that every route into the inner
    # constructor projects; see the note there.
    Optimizer(
        algorithm, problem, hessian, cache, linesearch, Options(T; options_kwargs...),
        _riemannian_gradient(gradient, cache.x), retraction, step_ceiling, observer)
end

"""
    default_gradient(problem, x)

Return the [`SimpleSolvers.Gradient`](@extref) that [`Optimizer`](@ref) uses if none is
supplied.

# Implementation

The [`NeuralNetworkParameters.NetworkParameters`](@extref) method is not just a matter of the
*length*: a `Gradient` built for a parameter set is called on the flattened parameters, so it has to
be constructed from `x` itself — `SimpleSolvers`' `GradientAutodiff(F, ::NetworkParameters)`, which
composes `problem.F` with the `unflatten` that belongs to `x`. Sizing it with `length(x)` — the
number of *layers* rather than the length of the flattening — used to make the first step fail with a
`DimensionMismatch`.

It wraps in a [`RiemannianGradient`](@ref), which is what projects the result leaf by leaf. `Optimizer`
would wrap it anyway, and the wrap is idempotent; doing it here as well keeps this function's own
return value the thing `Optimizer` will actually store.
"""
function default_gradient(problem::OptimizerProblem{T}, x::AbstractArray) where {T}
    GradientAutodiff{T}(problem.F, length(x))
end
function default_gradient(problem::OptimizerProblem, x::NetworkParameters)
    RiemannianGradient(GradientAutodiff(problem.F, x))
end

"""
    _optimizer(x, problem, algorithm, linesearch, gradient, retraction, config)

Build the cache and the Hessian for `algorithm` and hand everything to [`Optimizer`](@ref)'s inner
constructor.

Takes every argument positionally on purpose; see the note on Julia 1.12 below
[`Optimizer(x, F)`](@ref).
"""
function _optimizer(
        x::OptimizerSolution{T}, problem::OptimizerProblem{T}, algorithm::OptimizerMethod,
        linesearch::LinesearchMethod, gradient::Gradient{T}, retraction::AbstractRetraction,
        config::Options{T}, step_ceiling::Real, observer) where {T}
    # translate to the correct type if we use the momentum method
    algorithm = typeof(algorithm) <: MomentumMethod ? MomentumMethod(T(algorithm.α)) :
                algorithm
    cache = OptimizerCache(algorithm, x)
    hes = Hessian(algorithm, problem, x)
    Optimizer(algorithm, problem, hes, cache, linesearch,
        config, gradient, retraction, step_ceiling, observer)
end

function Optimizer(x::VT, problem::OptimizerProblem; algorithm::OptimizerMethod = BFGS(),
        linesearch::LinesearchMethod = default_linesearch(T, algorithm),
        gradient::Union{Gradient, Nothing} = nothing, retraction::AbstractRetraction = Cayley(),
        step_ceiling = DEFAULT_STEP_CEILING, observer = NoStepObserver(),
        options_kwargs...) where {
        T, VT <: OptimizerSolution{T}}
    # `_riemannian_gradient` on the caller's gradient too, and not only on the default: a parameter
    # set's leaves are projected one at a time, and a `SimpleSolvers` gradient built for the flat
    # vector has no method that reaches them. It is the identity on everything else.
    G = _riemannian_gradient(isnothing(gradient) ? default_gradient(problem, x) : gradient, x)
    _optimizer(x, problem, algorithm, linesearch, G, retraction,
        Options(T; options_kwargs...), step_ceiling, observer)
end

@doc raw"""
    Optimizer(x, F; ∇F!, mode, algorithm, linesearch, retraction, observer, options_kwargs...)

Build an [`Optimizer`](@ref) for the objective `F` at the parameters `x`.

# Implementation

!!! warning "Do not reintroduce a level of `kwargs...` here"
    This used to reach the inner constructor through three nested levels of `kwargs...` splatting, and
    on Julia 1.12 that made a *single* method compilation take fifteen minutes. Inference has to
    resolve the whole `Core.kwcall` chain before it knows the type the constructor returns, and on
    1.12 threading that into a [`solve!`](@ref) call in the same inferred body goes superlinear.
    Measured on this package's SVD problem, with the construction and the solve in one function body:

    | | Julia 1.13 | Julia 1.12 |
    |---|---|---|
    | three levels of `kwargs...` | 4.35 s | **940.86 s** |
    | three levels, behind a `@noinline` boundary | 4.57 s | **925.27 s** |
    | one level | 4.40 s | 6.53 s |
    | as it stands now | 4.15 s | **6.71 s** |

    Note the second row: putting the construction behind a barrier does *not* help, and neither does
    `@nospecialize`ing the enclosing function. Only flattening the chain does. `Options` is built once
    here and passed *positionally* from then on, which is what makes the constructor's return type
    independent of which keywords were given — see `_optimizer` and the inner constructor.

    Neither half of the pair is slow alone on 1.12: the constructor by itself costs 0.99 s and
    `solve!` by itself 2.35 s. 1.10, 1.13 and nightly are unaffected throughout, so the regression is
    upstream and already fixed there — but a 140× compile-time cliff on a released Julia is worth one
    flat call chain.
"""
function Optimizer(x::VT, F::Function; (∇F!) = nothing, mode = :autodiff,
        algorithm::OptimizerMethod = BFGS(), linesearch::Union{LinesearchMethod, Nothing} = nothing,
        retraction::AbstractRetraction = Cayley(), step_ceiling = DEFAULT_STEP_CEILING,
        observer = NoStepObserver(), options_kwargs...) where {
        T, VT <: OptimizerSolution{T}}
    # `T` comes from the `OptimizerSolution{T}` bound and not from `eltype(x)`: for a `NamedTuple` of
    # manifolds the latter is `StiefelManifold{Float64, Matrix{Float64}}` rather than `Float64`.
    _G = if (ismissing(∇F!) | isnothing(∇F!))
        if mode == :autodiff
            GradientAutodiff(F, x)
        else
            GradientFiniteDifferences(F, x)
        end
    else
        GradientFunction(F, ∇F!, x)
    end
    # See the note on the other `Optimizer` method: the wrapper is what projects a parameter set's
    # leaves, and it is the identity on a vector or a `Manifold`.
    G = _riemannian_gradient(_G, x)
    problem = (ismissing(∇F!) | isnothing(∇F!)) ? OptimizerProblem(F, x) :
              OptimizerProblem(F, ∇F!, x)
    ls = isnothing(linesearch) ? default_linesearch(T, algorithm) : linesearch
    _optimizer(x, problem, algorithm, ls, G, retraction,
        Options(T; options_kwargs...), step_ceiling, observer)
end

config(opt::Optimizer) = opt.config
problem(opt::Optimizer) = opt.problem
algorithm(opt::Optimizer) = opt.algorithm
linesearch(opt::Optimizer) = opt.linesearch
hessian(opt::Optimizer) = opt.hessian
direction(opt::Optimizer) = direction(cache(opt))
rhs(opt::Optimizer) = rhs(cache(opt))
cache(opt::Optimizer) = opt.cache
gradient(opt::Optimizer) = opt.gradient
# The step ceiling in multiples of 2π, not the `αmax` derived from it: that one depends on `‖δ‖` and
# so changes at every step. See `DEFAULT_STEP_CEILING` and `step_αmax`.
step_ceiling(opt::Optimizer) = opt.step_ceiling

"""
    step_observer(opt::Optimizer)

Return the phase observer installed on `opt`. This is a [`NoStepObserver`](@ref) when the caller did
not supply the `observer` keyword to [`Optimizer`](@ref).
"""
step_observer(opt::Optimizer) = opt.observer

check_gradient(opt::Optimizer) = check_gradient(gradient(problem(opt)))
print_gradient(opt::Optimizer) = print_gradient(gradient(problem(opt)))

function meets_stopping_criteria(status::OptimizerStatus, opt::Optimizer, state::OptimizerState)
    meets_stopping_criteria(status, config(opt), iteration_number(state))
end

function initialize!(opt::Optimizer, x::OptimizerSolution)
    initialize!(cache(opt), x)

    opt
end

"""
    solver_step!(x, state, opt)

Compute a full iterate for an [`Optimizer`](@ref).

!!! info
    This also performs a line search.

# Examples

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: solver_step!, NewtonOptimizerState)
julia> f(x) = sum(x .^ 2 + x .^ 3 / 3);

julia> x = [1f0, 2f0]
2-element Vector{Float32}:
 1.0
 2.0

julia> opt = Optimizer(x, f; algorithm = Newton());

julia> state = NewtonOptimizerState(x);

julia> update!(state, gradient(opt), x);

julia> solver_step!(x, state, opt)
2-element Vector{Float32}:
 0.25
 0.6666666
```

# Extended help

!!! info "A line search that fails does not get its step taken"
    `SimpleSolvers.solve` returns a step length whether or not the search succeeded, so taking it
    unconditionally lets a failed search drive the iteration. On the SVD problem of
    `test/optimizer_convergence/svd_optim.jl`, `BFGS` + `Bisection` + `Geodesic` used to diverge
    outright on one of eight starting points, and this is the mechanism:

    | iteration | outcome | ``\\alpha`` | ``f`` |
    |---|---|---|---|
    | 3 | `LINESEARCH_FLOOR` | 1.0 | 3.38 → 9.13 |
    | 4 | `LINESEARCH_FLOOR` | 1.0 | 9.13 → 1.2e169 |

    `Bisection` bisects ``\\varphi'``, so on a non-convex ray it can converge on a stationary point
    that is a *maximum*. It says so — `LINESEARCH_FLOOR`, ``\\varphi(1) = \\varphi(0)`` exactly — and
    the step was taken regardless. That one uphill move corrupts the secant pair, the direction the
    next iteration builds from it has ``\\|\\delta\\| = 345``, and retracting a lift that large left
    the manifold completely: `check(Y) = 1.07e200`. The solve then reported *convergence*, because
    ``\\|\\delta\\|/\\|x\\|`` is tiny once ``\\|x\\|`` is at `1e100`.

    `SimpleSolvers.solve_with_status` reports the outcome alongside the step length, so
    the two cases can be told apart; see [`linesearch_rejected`](@ref) and [`restart!`](@ref). With
    the restart the same starting point converges in 121 iterations at `check(Y) = 6e-14`.

!!! info "A line search that succeeds can still return a step too long for the manifold"
    The safeguard above is about a search that *failed* and said so. The other half of the problem is
    a search that succeeded and was right to: on a compact manifold ``\\varphi`` is bounded, so a step
    nine orders of magnitude too long can genuinely decrease the merit and no test the search has will
    reject it. That is issue A1b, and it is why this function hands the search a `params.αmax` through
    [`linesearch_parameters`](@ref) — a bound the *geometry* supplies, since ``\\varphi`` does not. See
    [`DEFAULT_STEP_CEILING`](@ref).

    The two are independent and both are needed. The rejected-step case is a claim about the
    direction, answered by changing it; the ceiling is a claim about the step, answered by
    shortening it. Nothing about the direction is wrong at the step A1b is about.

    They do meet in one place, and that is why the rejection test takes the ceiling as an argument.
    A search stopped *at* the ceiling with the merit still falling is classified by the same
    round-off rule as any other step, so it can be reported as `LINESEARCH_FLOOR` — a claim about the
    direction — when all that was established is that no step *this function permits* decreases the
    merit measurably. Discarding ``Q`` over a bound this function imposed itself is wasted work at
    best, so that case is exempt and the step is taken. See [`linesearch_rejected`](@ref) and issue
    B3.
"""
function solver_step!(x::OptimizerSolution{T}, state::OptimizerState{T}, opt::Optimizer{
        T, MT}) where {T, MT}
    # update cache
    # solve H δx = - ∇f
    # rhs is -g
    # The `FirstOrderMethodWithState` methods -- `MomentumMethod` and the `AdamFamily` -- need their
    # own parameters to form the direction and have no Hessian; the other methods need the Hessian and
    # have no parameters. Named through the aliases and not member by member, so that a new method
    # joining one of the unions does not leave a stale list behind here.
    if MT <: FirstOrderMethodWithState
        update!(cache(opt), state, gradient(opt), algorithm(opt), x)
    else
        update!(cache(opt), state, gradient(opt), hessian(opt), x)
        # A (quasi-)Newton direction only descends where the Hessian is positive definite; see
        # `ensure_descent!`. The `FirstOrderMethodWithState` methods are excluded on purpose -- their
        # direction is a moving average and is allowed not to descend on an individual step.
        ensure_descent!(cache(opt), algorithm(opt), config(opt))
    end
    typeof(algorithm(opt)) <: Newton && update!(state, gradient(opt), x) # this will have to be removed later

    for _ in 1:config(opt).nan_max_iterations
        observe_optimizer_phase(step_observer(opt), :retraction_application) do
            update_section!(
                section(cache(opt)), section(state), direction(cache(opt)), opt.retraction)
            _copyto!(solution(cache(opt)), section(cache(opt)))
        end
        # compute_new_iterate!(solution(cache(opt)), x, one(T), direction(cache(opt)), cache(opt), opt.retraction)
        f = observe_optimizer_phase(step_observer(opt), :objective) do
            value(problem(opt), solution(cache(opt)))
        end
        if isnan(f) || isinf(f)
            (opt.config.verbosity ≥ 2 &&
             @warn "NaN or Inf detected in optimizer. Reducing length of direction vector.")
            _rmul!(direction(cache(opt)), T(config(opt).nan_factor))
        else
            break
        end
    end

    # apply line search
    # `state` is passed because the merit needs `section(state)` as the base its trial steps retract
    # from, and because a step-size schedule needs `iteration_number(state)`; see
    # `trial_iterate!` and `DecayingStatic`.
    #
    # `solve_with_status` and not `solve`: the latter returns only the step length, which is
    # indistinguishable between a search that found a decrease and one that gave up. See
    # `linesearch_rejected` for what the difference costs. Since SimpleSolvers 0.12 it is also what
    # keeps the line search's warnings off the user's terminal -- `solve` is `solve_with_status`
    # plus `linesearch_warnings`, and a solve that cannot progress asks for an impossible decrease
    # at every one of its iterations. The outcome is this function's to act on, not the user's to
    # read about.
    #
    # `linesearch_parameters` and not a literal `(x = x, state = state)`: on a manifold it adds the
    # `αmax` of issue A1b. It is built *here*, below the `NaN` loop above, because that loop shrinks
    # the direction and the ceiling is a function of `‖δ‖`.
    #
    # Bound to a name rather than inlined because the ceiling is needed again below: a step that came
    # back *at* it is not evidence against the direction. See `linesearch_rejected` and issue B3.
    ls_params = linesearch_parameters(cache(opt), x, state, step_ceiling(opt))
    ls_status = solve_with_status(linesearch(opt), one(T), ls_params)

    # A rejected search means "no step along *this* direction decreases the merit". For a
    # quasi-Newton method that is a statement about `Q`, so throw `Q` away and try the one direction
    # that always descends.
    #
    # `Adam` and `MomentumMethod` used to be exempt here, on the grounds that a moving average is
    # allowed not to descend on an individual step. That is true of the *direction* and was the wrong
    # conclusion about the *step*: a rejected search returns `α = 1` untouched, so the exemption did
    # not let those methods take a non-descent step, it made them take the longest one available
    # along it. See `linesearch_rejected` and issue A7.
    #
    # The `αmax` argument is the exemption of issue B3: a `LINESEARCH_FLOOR` returned *at* the
    # ceiling this function itself imposed says nothing about the direction, only that no step the
    # caller permits decreases the merit measurably, so that step is taken rather than answered with
    # a restart. Euclidean parameters carry no ceiling, so `αmax` is `Inf` and nothing changes there.
    if linesearch_rejected(ls_status, _caller_αmax(T, ls_params))
        config(opt).verbosity ≥ 2 &&
            @warn "the line search returned $(outcome(ls_status)), i.e. no step along the $(algorithm(opt)) direction decreased the merit; restarting the inverse Hessian and searching along the steepest-descent direction instead." maxlog = 1
        restart!(state)
        steepest_descent!(cache(opt))
        observe_optimizer_phase(step_observer(opt), :retraction_application) do
            update_section!(
                section(cache(opt)), section(state), direction(cache(opt)), opt.retraction)
            _copyto!(solution(cache(opt)), section(cache(opt)))
        end
        # rebuilt rather than reused: `steepest_descent!` has just replaced the direction, so `‖δ‖`
        # and with it the ceiling are not what they were for the first search.
        ls_status = solve_with_status(linesearch(opt), one(T),
            linesearch_parameters(cache(opt), x, state, step_ceiling(opt)))
    end

    # Whatever the second search reports, its step is taken. Substituting `α = 0` instead -- "if even
    # steepest descent cannot decrease the merit, do not move" -- reads as the safer choice and is
    # not: a zero step makes `rxₐ` vanish, `x_converged` fires, and the solve reports convergence at
    # a point where `‖∇f‖` is still of order one. That is what `test/descent_direction_tests.jl`
    # catches, on three of its forty-eight combinations.
    α = steplength(ls_status)
    _rmul!(direction(cache(opt)), α)

    # compute new minimizer
    observe_optimizer_phase(step_observer(opt), :retraction_application) do
        update_section!(
            section(cache(opt)), section(state), direction(cache(opt)), opt.retraction)
        _copyto!(solution(cache(opt)), section(cache(opt)))
        _copyto!(x, solution(cache(opt)))
    end

    # `rg` is measured at the iterate this step *ended* at, not at the one it started from; see
    # `refresh_latest_gradient!` and the note on `convergence_measures`. Costs one gradient
    # evaluation per iteration, and only for the caches that implement it.
    refresh_latest_gradient!(cache(opt), gradient(opt))

    x
end

"""
    solve!(x, state, opt)

Solve the optimization problem described by `opt::`[`Optimizer`](@ref) and store the result in `x`.

# Examples

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: solve!, NewtonOptimizerState, update!, iteration_number; using Random: seed!; seed!(123))
julia> f(x) = sum(x .^ 2 + x .^ 3 / 3);

julia> x = [1f0, 2f0]
2-element Vector{Float32}:
 1.0
 2.0

julia> opt = Optimizer(x, f; algorithm = Newton());

julia> state = NewtonOptimizerState(x);

julia> solve!(x, state, opt)
GeometricOptimizers.OptimizerResult{Float32, Float32, Vector{Float32}, GeometricOptimizers.OptimizerStatus{Float32, Float32}}( * Convergence measures

    |x - x'|               = 7.82e-03
    |x - x'|/|x'|          = 2.56e+02
    |f(x) - f(x')|         = 6.18e-05
    |f(x) - f(x')|/|f(x')| = 6.63e+04
    |g(x) - g(x')|         = 1.57e-02
    |g(x)|                 = 6.10e-05
, Float32[4.6478817f-8, 3.0517578f-5], 9.313341f-10, GeometricOptimizers.OptimizerTraceEntry{Float32, Float32}[])

julia> x
2-element Vector{Float32}:
 4.6478817f-8
 3.0517578f-5

julia> iteration_number(state)
4
```

The trailing empty vector is the per-iteration [`trace`](@ref), which is only filled if
`Options(store_trace = true)` asked for it.

Two rows of that status moved in 0.2.0 and both are the same fix (issue A8). `|g(x)|` is now
``\\|\\nabla{}f\\|`` at the iterate the solve returns, so `g_converged` fires when the residual has
actually reached `f_reltol` — this solve stops on iteration 4 where it used to run a fifth and report
the residual of the fourth. And `|g(x) - g(x')|` used to be `0.00e+00` *structurally* for `Newton`:
`solver_step!` advanced `state.ḡ` at the same iterate the cache took its gradient at, so the
difference it printed could not be anything else. See [`gradient_difference!`](@ref).

Also see [`solver_step!`](@ref).
"""
function solve!(x::OptimizerSolution{T}, state::OptimizerState, opt::Optimizer{T}) where {T}
    initialize_state!(state)
    observer = step_observer(opt)

    # `status` below is computed on every iteration regardless, so recording it costs one `Bool` test
    # per iteration when `store_trace` is unset. See `trace`.
    f = observe_optimizer_phase(observer, :objective) do
        value(problem(opt), x)
    end
    tracing = config(opt).store_trace
    _trace = OptimizerTraceEntry{typeof(f), T}[]

    while true
        increase_iteration_number!(state)
        solver_step!(x, state, opt)
        # One objective evaluation per iterate, reused for the status and the trace entry: both read
        # the same `x`, and with an observer installed a second call would also emit a second
        # `:objective` pair for a step that only ever evaluated once.
        f = observe_optimizer_phase(observer, :objective) do
            value(problem(opt), x)
        end
        status = OptimizerStatus(state, cache(opt), f; config = config(opt))
        tracing && push!(_trace,
            OptimizerTraceEntry(iteration_number(state), f, g_residual(status)))
        meets_stopping_criteria(status, opt, state) && break
        update!(state, opt, x)
    end

    f = observe_optimizer_phase(observer, :objective) do
        value(problem(opt), x)
    end
    status = OptimizerStatus(state, cache(opt), f; config = config(opt))
    warn_iteration_number(state, config(opt))
    # `warn_iteration_number` does not touch `x`, so the value above is still the one at the final
    # iterate and the result reuses it rather than evaluating the objective a second time.
    OptimizerResult(status, x, f, _trace)
end

function update!(state::OptimizerState, opt::Optimizer, x::OptimizerSolution)
    update!(state, gradient(opt), x)
end

function initialize_state!(state::OptimizerState)
    state
end

const INITIAL_BFGS_X = 0.12345
const INITIAL_BFGS_G = 0.54321
const INITIAL_BFGS_F = 0.23456

function initialize_state!(state::Union{BFGSState{T}, DFPState{T}}) where {T}
    _fill!(state.x̄, T(INITIAL_BFGS_X))
    _fill!(state.ḡ, T(INITIAL_BFGS_G))
    state.f̄ = T(INITIAL_BFGS_F)
    state.Q .= one(state.Q)

    state
end

function warn_iteration_number(state::OptimizerState, config::Options)
    if config.warn_iterations > 0 && iteration_number(state) ≥ config.warn_iterations
        println("WARNING: Optimizer took ", iteration_number(state), " iterations.")
    end
end

# put this somewhere else eventually!
function update!(state::NewtonOptimizerState, opt::Optimizer, x::AbstractVector)
    update!(state, gradient(opt), x)
    observe_optimizer_phase(step_observer(opt), :retraction_application) do
        update_section!(
            state.section, gradient_array(cache(opt)), x -> retraction(opt.retraction, x))
    end
    state
end
