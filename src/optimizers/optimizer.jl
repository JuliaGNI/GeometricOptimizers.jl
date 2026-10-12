
const SOLUTION_MAX_PRINT_LENGTH = 10

@doc raw"""
    DEFAULT_STEP_CEILING

How far along its direction a manifold solve may step, in multiples of ``2\pi``: the `c` of
[`step_αmax`](@ref)`(c, δ)`, and so the `params.αmax` that [`solver_step!`](@ref) gives the line
search through [`linesearch_parameters`](@ref). Set it per solve with
`Optimizer(x, F; step_ceiling = …)`; `Inf` switches the ceiling off.

# Why the solver sets a ceiling

A line search bounds its step by the merit: it stops when ``\varphi`` stops falling. On a compact
manifold ``\varphi`` is bounded, so a step many turns too long can decrease the merit, and no test of
the search rejects it. The retraction of such a lift leaves the manifold. The bound that exists is
not a property of ``\varphi``: it is the ``2\pi`` of a rotation divided by ``\|\delta\|``, and it
changes at every solver step. The `DEFAULT_LINESEARCH_αmax` of a line search method is a backstop
for every problem, not this bound. The manual section
[A bounded merit is not a bound on the step](@ref) gives the measurements on the SVD problem.

# Why `1`

`1` says *"never step further along the direction than one full turn"*. That is the scale that the
geometry gives, not a value tuned to one problem.

!!! info "This bounds the step, not the direction and not `α`"
    A short direction can take a large ``\alpha`` in an ordinary step, which a ceiling on ``\alpha``
    alone would reject. See [`ensure_descent!`](@ref) and [`linesearch_rejected`](@ref) for the
    safeguards on the direction and on the outcome of the search.

!!! info "Euclidean parameters do not get one"
    [`linesearch_parameters`](@ref) omits `αmax` for an `AbstractVector`, and passes `Inf` for a
    parameter set with no manifold block. There is no geometric scale in either case, and none is
    needed: ``f(x + \alpha{}p)`` grows with ``\alpha``, so the decrease test of the search rejects an
    over-long step. [`_manifold_αmax`](@ref) derives the ceiling over the manifold blocks only.

!!! info "One `α`, one ceiling per block"
    On a parameter set the same ``\alpha`` scales every block, so the ceiling is the *smallest* of the
    per-block ceilings over the manifold blocks — [`_manifold_αmax`](@ref) — and not ``2\pi{}c`` over
    the norm of the whole direction, which makes every block pay for its neighbours.
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
    WT,
    OT} <: AbstractSolver
    algorithm::ALG
    problem::OBJ
    gradient::GT
    hessian::HT
    config::Options{T}
    cache::OCT
    linesearch::LST
    retraction::RT
    # The buffers of every retraction on the step path: a `RetractionWorkspace` for a manifold
    # point, a `NoWorkspace` for any other array, and for a parameter set a tree of these, one per
    # leaf. They are here and not on the cache because the retraction belongs to this type, and
    # both call sites that apply one, `solver_step!` and `trial_iterate!` through the closures of
    # `linesearch_problem`, start from here. See `GeometricOptimizers.retraction_workspace`.
    retraction_workspace::WT
    step_ceiling::T
    observer::OT

    # Every argument is positional, the built `Options` too. See the implementation note of
    # `Optimizer(x, F)`.
    function Optimizer(algorithm::OptimizerMethod, problem::OptimizerProblem{T},
            hessian::Hessian{T}, cache::OptimizerCache, linesearch::LinesearchMethod,
            config::Options{T}, gradient::Gradient{T}, retraction::AbstractRetraction,
            step_ceiling::Real = DEFAULT_STEP_CEILING,
            observer = NoStepObserver()) where {T}
        observed_gradient = _observed_gradient(gradient, observer)
        workspace = retraction_workspace(cache.x, retraction)
        ls_problem = linesearch_problem(
            problem, observed_gradient, cache, retraction, observer, workspace)
        ls = Linesearch(ls_problem, linesearch)
        new{T, typeof(algorithm), typeof(problem), typeof(observed_gradient),
            typeof(hessian), typeof(cache), typeof(ls), typeof(retraction), typeof(workspace),
            typeof(observer)}(
            algorithm, problem, observed_gradient, hessian, config,
            cache, ls, retraction, workspace, T(step_ceiling), observer)
    end
end

function Optimizer(algorithm::OptimizerMethod, problem::OptimizerProblem{T},
        hessian::Hessian{T}, cache::OptimizerCache, linesearch::LinesearchMethod;
        gradient = default_gradient(problem, cache.x), retraction = Cayley(),
        step_ceiling = DEFAULT_STEP_CEILING, observer = NoStepObserver(),
        options_kwargs...) where {T}
    # `_riemannian_gradient` here as in the two methods below, so that every route into the inner
    # constructor projects; see the note there.
    Optimizer(change_precision(T, algorithm), problem, hessian, cache,
        _linesearch_method(T, linesearch), Options(T; options_kwargs...),
        _riemannian_gradient(gradient, cache.x), retraction, step_ceiling, observer)
end

"""
    default_gradient(problem, x)

Return the [`SimpleSolvers.Gradient`](@extref) that [`Optimizer`](@ref) uses if none is
supplied.

A [`NeuralNetworkParameters.NetworkParameters`](@extref) has no default gradient, and this throws
an `ArgumentError` that names `∇F!`: forward mode takes one pass per chunk of parameters, which is
the wrong default for a network of more than a few hundred parameters. Supply the gradient, or opt
in to forward mode or finite differences with the `mode` of `Optimizer(x, F)`.

# Implementation

The [`Manifold`](@ref) method is built from `x`: `GradientAutodiff(problem.F, x)` rebuilds the
point before it calls `problem.F`, where the `AbstractArray` method would hand `problem.F` the flat
vector.

On a device there is no default gradient, and every method throws an `ArgumentError` that names
`∇F!`: `ForwardDiff` and finite differences compute on the host.
"""
function default_gradient(problem::OptimizerProblem{T}, x::AbstractArray) where {T}
    _refuse_default_gradient(x)
    GradientAutodiff{T}(problem.F, length(x))
end
function default_gradient(problem::OptimizerProblem, x::Manifold)
    _refuse_default_gradient(x)
    GradientAutodiff(problem.F, x)
end
function default_gradient(::OptimizerProblem, x::NetworkParameters)
    _refuse_default_gradient(x)
    _no_default_gradient()
end

# The gradient `Optimizer(x, F)` builds without a `∇F!`: `mode` selects automatic differentiation or
# central differences, and anything else is refused rather than read as one of them. A plain vector
# needs `GradientFiniteDifferences`' length constructor, because `SimpleSolvers` has no method for a
# vector; a `Manifold` and a parameter set have one each. With no `mode`, a vector and a `Manifold`
# take `:autodiff`, and a parameter set takes none.
function _default_gradient(F, x, mode)
    mode === nothing || mode === :autodiff || mode === :finitediff ||
        throw(ArgumentError("`mode` is `:autodiff` or `:finitediff`, not `$(repr(mode))`"))
    _refuse_default_gradient(x)
    mode === nothing && x isa NetworkParameters && _no_default_gradient()
    mode === :finitediff ? _finite_differences(F, x) : GradientAutodiff(F, x)
end

# Forward mode takes one pass per chunk of twelve parameters, and finite differences two evaluations
# per parameter, so neither is the default for a parameter set; `mode` opts in to either.
function _no_default_gradient()
    throw(ArgumentError(
        "no default gradient for a parameter set: `mode = :autodiff` takes one forward-mode pass " *
        "per chunk of parameters, and `mode = :finitediff` two evaluations of `F` per parameter. " *
        "Supply the gradient: `∇F!` to `Optimizer(x, F; ∇F! = …)`, or `gradient` to " *
        "`Optimizer(x, problem; gradient = …)`; or opt in with `mode = :autodiff` or " *
        "`mode = :finitediff` in `Optimizer(x, F; mode = …)`."))
end

function _finite_differences(F, x::AbstractVector{T}) where {T}
    GradientFiniteDifferences{T}(F, length(x))
end
_finite_differences(F, x) = GradientFiniteDifferences(F, x)

"""
    _optimizer(x, problem, algorithm, linesearch, gradient, retraction, config)

Convert `algorithm` and `linesearch` to the element type of `x`, build the cache and the Hessian for
`algorithm`, and hand everything to [`Optimizer`](@ref)'s inner constructor.

Every argument is positional; see the implementation note of [`Optimizer(x, F)`](@ref).
"""
function _optimizer(
        x::OptimizerSolution{T}, problem::OptimizerProblem{T}, algorithm::OptimizerMethod,
        linesearch::Union{LinesearchMethod, Real}, gradient::Gradient{T},
        retraction::AbstractRetraction, config::Options{T}, step_ceiling::Real,
        observer) where {T}
    method = change_precision(T, algorithm)
    _check_one_backend(method, x)
    cache = OptimizerCache(method, x)
    hes = Hessian(method, problem, x)
    Optimizer(method, problem, hes, cache, _linesearch_method(T, linesearch),
        config, gradient, retraction, step_ceiling, observer)
end

# For a method other than quasi-Newton, a parameter set with leaves on two backends raises the "mixed
# backends" error before its cache is built. A quasi-Newton cache allocates its matrices on the
# backend of the set and raises it there, so this checks only for the other methods: a second check
# in `_optimizer` stops the keyword body of `Optimizer(x, F)` from inlining, and a BFGS construction
# then allocates more.
_check_one_backend(::QuasiNewtonOptimizerMethod, x) = nothing
_check_one_backend(::OptimizerMethod, x) = (_solution_backend(x); nothing)

# A number is a fixed step size, and a `Static` or a `DecayingStatic` is converted to the element type
# of the parameters, as the method is. A searching line search is taken as it is.
_linesearch_method(::Type{T}, η::Real) where {T} = _linesearch_method(T, Static(η))
# `true` is a `Real` that is `1`, and never meant as a step size
function _linesearch_method(::Type, η::Bool)
    throw(ArgumentError("a step size is a number, not $(η)"))
end
# The check is on the converted step size: `1e-50` is `0` in `Float32`, and `1e40` is `Inf32`.
function _linesearch_method(::Type{T}, ls::Static) where {T}
    s = change_precision(T, ls)
    isfinite(s.α) && s.α > 0 ||
        throw(ArgumentError("a fixed step size is finite and positive in $(T), not $(s.α)"))
    s
end
_linesearch_method(::Type{T}, ls::DecayingStatic) where {T} = change_precision(T, ls)
_linesearch_method(::Type, ls::LinesearchMethod) = ls

# A training step takes what `_linesearch_method` converts, except a searching line search.
function _training_step_size(T::Type, ls::Union{Real, Static, DecayingStatic})
    _linesearch_method(T, ls)
end
function _training_step_size(::Type, ls)
    throw(ArgumentError(
        "a training step takes a step size or a `DecayingStatic`, not a $(typeof(ls)): it has the " *
        "gradient of one minibatch and no objective for a line search to search along."))
end

function Optimizer(x::VT, problem::OptimizerProblem; algorithm::OptimizerMethod = BFGS(),
        linesearch::Union{LinesearchMethod, Real} = default_linesearch(T, algorithm),
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

`∇F!` is the gradient of `F`. Without it, `mode = :autodiff` builds a forward-mode
`ForwardDiff` gradient and `mode = :finitediff` a central-difference one. A vector and a
[`Manifold`](@ref) take `:autodiff` when no `mode` is given. A
[`NeuralNetworkParameters.NetworkParameters`](@extref) takes no default: without `∇F!` and `mode`
this throws an `ArgumentError`. Forward mode costs one pass of `F` per chunk of twelve parameters,
and finite differences two evaluations of `F` per parameter, so either is for a small set.

`linesearch` is a [`SimpleSolvers.LinesearchMethod`](@extref) or a number, which is the fixed step
size `Static(η)`. `algorithm`, and a `Static` or [`DecayingStatic`](@ref) `linesearch`, are converted
to the element type of `x`, so `Optimizer(Float32[1, 2, 3], F; algorithm = Adam())` optimizes in
`Float32`.

# Implementation

!!! warning "Do not add a level of `kwargs...` here"
    This method splats `kwargs...` once. `Options` is built here and passed *positionally* from then
    on, so the return type of the constructor does not depend on which keywords were given; see
    `_optimizer` and the inner constructor. Nested levels of `kwargs...` splatting, with the
    construction and a [`solve!`](@ref) call in one inferred body, make the compile time
    superlinear on some Julia releases, and neither a barrier nor `@nospecialize` prevents it.
    `KNOWN_ISSUES.md` records the measurements.
"""
function Optimizer(x::VT, F::Function; (∇F!) = nothing, mode = nothing,
        algorithm::OptimizerMethod = BFGS(),
        linesearch::Union{LinesearchMethod, Real, Nothing} = nothing,
        retraction::AbstractRetraction = Cayley(), step_ceiling = DEFAULT_STEP_CEILING,
        observer = NoStepObserver(), options_kwargs...) where {
        T, VT <: OptimizerSolution{T}}
    # `T` comes from the `OptimizerSolution{T}` bound and not from `eltype(x)`: for a `NamedTuple` of
    # manifolds the latter is `StiefelManifold{Float64, Matrix{Float64}}` rather than `Float64`.
    _G = if (ismissing(∇F!) | isnothing(∇F!))
        _default_gradient(F, x, mode)
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
retraction_workspace(opt::Optimizer) = opt.retraction_workspace

# What `update!(cache, state, gradient, ·, x)` forms the direction from: the Hessian for `Newton`,
# and the method itself for the first-order and the quasi-Newton methods, which build none.
_direction_rule(opt::Optimizer) = hessian(opt)
function _direction_rule(opt::Optimizer{
        T, <:Union{FirstOrderMethod, QuasiNewtonOptimizerMethod}}) where {T}
    algorithm(opt)
end

"""
    step_observer(opt::Optimizer)

Return the phase observer installed on `opt`. This is a [`NoStepObserver`](@ref) when the caller did
not supply the `observer` keyword to [`Optimizer`](@ref).
"""
step_observer(opt::Optimizer) = opt.observer

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

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: gradient, solver_step!, NewtonState)
julia> f(x) = sum(x .^ 2 + x .^ 3 / 3);

julia> x = [1f0, 2f0]
2-element Vector{Float32}:
 1.0
 2.0

julia> opt = Optimizer(x, f; algorithm = Newton());

julia> state = NewtonState(x);

julia> update!(state, gradient(opt), x);

julia> solver_step!(x, state, opt)
2-element Vector{Float32}:
 0.25
 0.6666666
```

# Extended help

!!! info "A line search that fails does not get its step taken"
    `SimpleSolvers.solve` returns a step length whether or not the search succeeded. A step taken
    from a failed search can be uphill: `Bisection` bisects ``\\varphi'``, so on a non-convex ray it
    can converge on a maximum, and it reports `LINESEARCH_FLOOR`. One uphill move corrupts the secant
    pair, and the next direction can be large enough to leave the manifold.
    `SimpleSolvers.solve_with_status` reports the outcome with the step length, so the two cases are
    told apart; see [`linesearch_rejected`](@ref) and [`restart!`](@ref).

!!! info "A line search that succeeds can still return a step too long for the manifold"
    On a compact manifold ``\\varphi`` is bounded, so a step many orders of magnitude too long can
    decrease the merit, and no test of the search rejects it. This function therefore gives the
    search a `params.αmax` through [`linesearch_parameters`](@ref): a bound from the geometry, because
    ``\\varphi`` gives none. See [`DEFAULT_STEP_CEILING`](@ref).

    The two safeguards are independent. A rejected step is a statement about the direction, and the
    answer is a new direction; the ceiling is a statement about the step, and the answer is a shorter
    step. They meet in one place, so the rejection test takes the ceiling as an argument: a search
    that stops *at* the ceiling with the merit still falling can report `LINESEARCH_FLOOR`, although
    only the steps that this function permits fail to decrease the merit. That case keeps ``Q`` and
    takes the step. See [`linesearch_rejected`](@ref).
"""
function solver_step!(x::OptimizerSolution{T}, state::OptimizerState{T},
        opt::Optimizer{T}) where {T}
    # update cache
    # solve H δx = - ∇f
    # rhs is -g
    update!(cache(opt), state, gradient(opt), _direction_rule(opt), x)
    ensure_descent!(cache(opt), algorithm(opt), config(opt))

    for _ in 1:config(opt).nan_max_iterations
        observe_optimizer_phase(step_observer(opt), :retraction_application) do
            update_section!(section(cache(opt)), section(state), direction(cache(opt)),
                opt.retraction, retraction_workspace(opt))
            _copyto!(solution(cache(opt)), section(cache(opt)))
        end
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
    # `solve_with_status` and not `solve`: the latter returns only the step length, the same for a
    # search that found a decrease and one that gave up. See `linesearch_rejected`. It also keeps the
    # warnings of the line search off the terminal: `solve` is `solve_with_status` plus
    # `linesearch_warnings`, and a solve that cannot progress asks for an impossible decrease at every
    # iteration. This function acts on the outcome.
    #
    # `linesearch_parameters` and not a literal `(x = x, state = state)`: on a manifold it adds the
    # step ceiling `αmax`. It is built below the `NaN` loop, because that loop shrinks the direction
    # and the ceiling is a function of `‖δ‖`.
    #
    # A name, because the ceiling is needed again below: a step that comes back *at* it is not
    # evidence against the direction. See `linesearch_rejected`.
    ls_params = linesearch_parameters(cache(opt), x, state, step_ceiling(opt))
    ls_status = solve_with_status(linesearch(opt), one(T), ls_params)

    # A rejected search means "no step along *this* direction decreases the merit". For a
    # quasi-Newton method that is a statement about `Q`, so throw `Q` away and try the one direction
    # that always descends.
    #
    # `Adam` and `MomentumMethod` are not exempt. A moving average may give a direction that does
    # not descend, but a rejected search returns `α = 1` untouched, so an exemption would take the
    # longest step along that direction. See `linesearch_rejected`.
    #
    # The `αmax` argument exempts a `LINESEARCH_FLOOR` returned *at* the ceiling that this function
    # sets: it says nothing about the direction, only that no permitted step decreases the merit
    # measurably, so that step is taken and the state is not restarted. Euclidean parameters carry no
    # ceiling, so `αmax` is `Inf` there.
    if linesearch_rejected(ls_status, _caller_αmax(T, ls_params))
        config(opt).verbosity ≥ 2 &&
            @warn "the line search returned $(outcome(ls_status)), i.e. no step along the $(algorithm(opt)) direction decreased the merit; restarting the inverse Hessian and searching along the steepest-descent direction instead." maxlog = 1
        restart!(state)
        steepest_descent!(cache(opt))
        observe_optimizer_phase(step_observer(opt), :retraction_application) do
            update_section!(section(cache(opt)), section(state), direction(cache(opt)),
                opt.retraction, retraction_workspace(opt))
            _copyto!(solution(cache(opt)), section(cache(opt)))
        end
        # rebuilt rather than reused: `steepest_descent!` has just replaced the direction, so `‖δ‖`
        # and with it the ceiling are not what they were for the first search.
        ls_params = linesearch_parameters(cache(opt), x, state, step_ceiling(opt))
        ls_status = solve_with_status(linesearch(opt), one(T), ls_params)
    end

    # A second search that reports `LINESEARCH_FLOOR` says that the merit resolves no decrease along
    # steepest descent either: the solve is at the round-off floor of `f`, and the step is zero, so
    # `x_converged` stops the solve. Taking the `α = 1` that such a search returns untouched instead
    # overshoots the minimiser, and the next step comes back: a 2-cycle in which no criterion fires.
    # The other two rejections, `LINESEARCH_EXHAUSTED` and `LINESEARCH_NO_DESCENT`, say nothing
    # about the floor, and their step is taken: a zero step there reports convergence where `‖∇f‖` is
    # of order one, which `test/optimizers/descent_direction_tests.jl` catches.
    α = at_round_off_floor(ls_status, _caller_αmax(T, ls_params)) ? zero(T) :
        steplength(ls_status)
    _rmul!(direction(cache(opt)), α)

    # compute new minimizer
    observe_optimizer_phase(step_observer(opt), :retraction_application) do
        update_section!(section(cache(opt)), section(state), direction(cache(opt)),
            opt.retraction, retraction_workspace(opt))
        _copyto!(solution(cache(opt)), section(cache(opt)))
        _copyto!(x, solution(cache(opt)))
    end

    # `rg` is measured at the iterate this step *ended* at, not at the one it started from; see
    # `refresh_latest_gradient!` and the note on `convergence_measures`. Costs one gradient
    # evaluation per iteration.
    refresh_latest_gradient!(cache(opt), gradient(opt))

    x
end

"""
    solve!(x, state, opt)

Solve the optimization problem described by `opt::`[`Optimizer`](@ref) and store the result in `x`.

# Examples

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: solve!, NewtonState, update!, iteration_number; using Random: seed!; seed!(123))
julia> f(x) = sum(x .^ 2 + x .^ 3 / 3);

julia> x = [1f0, 2f0]
2-element Vector{Float32}:
 1.0
 2.0

julia> opt = Optimizer(x, f; algorithm = Newton());

julia> state = NewtonState(x);

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

`|g(x)|` is ``\\|\\nabla{}f\\|`` at the iterate that the solve returns, so `g_converged` fires when
that residual reaches `f_reltol`. `|g(x) - g(x')|` is the difference of the gradients at the last
two iterates; see [`gradient_difference!`](@ref).

Also see [`solver_step!`](@ref).
"""
function solve!(x::OptimizerSolution{T}, state::OptimizerState, opt::Optimizer{T}) where {T}
    observer = step_observer(opt)

    # `status` below is computed on every iteration regardless, so recording it costs one `Bool` test
    # per iteration when `store_trace` is unset. See `trace`.
    f = observe_optimizer_phase(observer, :objective) do
        value(problem(opt), x)
    end
    initialize_state!(state, opt, x, f)
    tracing = config(opt).store_trace
    _trace = OptimizerTraceEntry{typeof(f), T}[]

    while true
        increase_iteration_number!(state)
        solver_step!(x, state, opt)
        # One objective evaluation per iterate, reused for the state update, the status and the trace
        # entry: all three read the same `x`, and with an observer installed a second call would also
        # emit a second `:objective` pair for a step that only ever evaluated once.
        f = observe_optimizer_phase(observer, :objective) do
            value(problem(opt), x)
        end
        # The state advances *before* the stop check, so that the state `solve!` returns holds the
        # iterate it returns, and the status reads `f` against the objective one step back,
        # `previous_value(state)`.
        update!(state, opt, x, f)
        status = OptimizerStatus(state, cache(opt), f; config = config(opt))
        tracing && push!(_trace,
            OptimizerTraceEntry(iteration_number(state), f, g_residual(status)))
        meets_stopping_criteria(status, opt, state) && break
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

# `solve!` hands the state update the objective value it has already evaluated at `x`. The states
# that record it have their own method; every other state ignores it.
function update!(state::OptimizerState, opt::Optimizer, x::OptimizerSolution, f)
    update!(state, opt, x)
end

# What `solve!` does to `state` before its first step, with the starting point `x` and its objective
# `f`: no iteration is taken, the iterate and the section are at `x`, and the method's own memory is
# reset. So a state used for a second solve repeats the solve of a fresh state, and the first status
# compares `f` against `f(x)`.
function initialize_state!(state::OptimizerState, opt::Optimizer, x::OptimizerSolution, f)
    initialize_state!(state, opt, x)
    state.iterations = 0
    _copyto!(solution(state), x)
    _start_section!(section(state), x)
    state.f = f
    state
end

# The memory a method keeps between steps, as its constructor builds it. The momentum and moment
# states have their method beside their constructor.
function initialize_state!(state::OptimizerState, ::Optimizer, ::OptimizerSolution)
    initialize_state!(state)
end

initialize_state!(state::OptimizerState) = state

# The secant pair and `Q`.
initialize_state!(state::BFGSState) = _poison!(state)

# The gradient at the iterate, which `NewtonState` holds.
function initialize_state!(state::NewtonState, opt::Optimizer, x::AbstractVector)
    gradient(opt)(state.g, x)
    state
end

function warn_iteration_number(state::OptimizerState, config::Options)
    if config.warn_iterations > 0 && iteration_number(state) ≥ config.warn_iterations
        @warn "Optimizer took $(iteration_number(state)) iterations."
    end
end

# `update!(state, x, g, f)` moves the section to `x`, which is the base the next step retracts from.
# `latest_gradient` is `∇f(x)`, which `solver_step!` has just refreshed, so nothing is evaluated.
function update!(state::NewtonState, opt::Optimizer, x::AbstractVector, f)
    update!(state, x, latest_gradient(cache(opt)), f)
    state
end
