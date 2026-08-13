
const SOLUTION_MAX_PRINT_LENGTH = 10

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

The choice of line search does not change where this one ends up:

```jldoctest; setup = :(using GeometricOptimizers; F(x) = sum(sin.(x) .^ 2))
x = ones(3)
algorithm = Newton()
state = OptimizerState(algorithm, x)
optimizer = Optimizer(x, F; algorithm = algorithm, linesearch = Backtracking())

solve!(x, state, optimizer)
x

# output

3-element Vector{Float64}:
 0.0
 0.0
 0.0
```

!!! info "This example needs the descent safeguard"
    ``\\sin^2`` has second derivative ``2\\cos(2x) = -0.83`` at `x = 1`, so the Newton direction there
    points *uphill*. [`ensure_descent!`](@ref) replaces it by the steepest-descent direction; without
    it both solves converge to ``\\pi/2``, where `F` is maximal.

"""
struct Optimizer{T,
    ALG<:OptimizerMethod,
    OBJ<:OptimizerProblem{T},
    GT<:Gradient{T},
    HT<:Hessian{T},
    OCT<:Union{OptimizerCache,NamedTuple},
    LST<:Linesearch,
    RT<:AbstractRetraction} <: AbstractSolver
    algorithm::ALG
    problem::OBJ
    gradient::GT
    hessian::HT
    config::Options{T}
    cache::OCT
    linesearch::LST
    retraction::RT

    # Everything positional, and in particular the `Options` already built. See the note on Julia 1.12
    # below `Optimizer(x, F)`.
    function Optimizer(algorithm::OptimizerMethod, problem::OptimizerProblem{T}, hessian::Hessian{T}, cache::OptimizerCache, linesearch::LinesearchMethod, config::Options{T}, gradient::Gradient{T}, retraction::AbstractRetraction) where {T}
        ls_problem = linesearch_problem(problem, gradient, cache, retraction)
        ls = Linesearch(ls_problem, linesearch)
        new{T,typeof(algorithm),typeof(problem),typeof(gradient),typeof(hessian),typeof(cache),typeof(ls),typeof(retraction)}(algorithm, problem, gradient, hessian, config, cache, ls, retraction)
    end
end

function Optimizer(algorithm::OptimizerMethod, problem::OptimizerProblem{T}, hessian::Hessian{T}, cache::OptimizerCache, linesearch::LinesearchMethod; gradient=default_gradient(problem, cache.x), retraction=Cayley(), options_kwargs...) where {T}
    Optimizer(algorithm, problem, hessian, cache, linesearch, Options(T; options_kwargs...), gradient, retraction)
end

"""
    default_gradient(problem, x)

Return the [`SimpleSolvers.Gradient`](@extref) that [`Optimizer`](@ref) uses if none is
supplied.

# Implementation

The `NamedTuple` method is not just a matter of the *length*: a `Gradient` built for a
`NamedTuple` is called on the flattened parameters, so it has to be constructed from `x`
itself (see `GradientAutodiff(F, ::NamedTuple)`), which composes `problem.F` with the
`unflatten` that belongs to `x`. Sizing it with `length(x)` — the number of entries of the
`NamedTuple` rather than the length of its flattening — used to make the first step fail with
a `DimensionMismatch`.
"""
default_gradient(problem::OptimizerProblem{T}, x::AbstractArray) where {T} = GradientAutodiff{T}(problem.F, length(x))
default_gradient(problem::OptimizerProblem, x::ArrayNamedTuple) = GradientAutodiff(problem.F, x)

"""
    _optimizer(x, problem, algorithm, linesearch, gradient, retraction, config)

Build the cache and the Hessian for `algorithm` and hand everything to [`Optimizer`](@ref)'s inner
constructor.

Takes every argument positionally on purpose; see the note on Julia 1.12 below
[`Optimizer(x, F)`](@ref).
"""
function _optimizer(x::OptimizerSolution{T}, problem::OptimizerProblem{T}, algorithm::OptimizerMethod,
    linesearch::LinesearchMethod, gradient::Gradient{T}, retraction::AbstractRetraction,
    config::Options{T}) where {T}
    # translate to the correct type if we use the momentum method
    algorithm = typeof(algorithm) <: MomentumMethod ? MomentumMethod(T(algorithm.α)) : algorithm
    cache = OptimizerCache(algorithm, x)
    hes = Hessian(algorithm, problem, x)
    Optimizer(algorithm, problem, hes, cache, linesearch, config, gradient, retraction)
end

function Optimizer(x::VT, problem::OptimizerProblem; algorithm::OptimizerMethod=_BFGS(),
    linesearch::LinesearchMethod=default_linesearch(T, algorithm),
    gradient::Union{Gradient,Nothing}=nothing, retraction::AbstractRetraction=Cayley(),
    options_kwargs...) where {T,VT<:OptimizerSolution{T}}
    G = isnothing(gradient) ? default_gradient(problem, x) : gradient
    _optimizer(x, problem, algorithm, linesearch, G, retraction, Options(T; options_kwargs...))
end

@doc raw"""
    Optimizer(x, F; ∇F!, mode, algorithm, linesearch, retraction, options_kwargs...)

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
function Optimizer(x::VT, F::Function; (∇F!)=nothing, mode=:autodiff,
    algorithm::OptimizerMethod=_BFGS(), linesearch::Union{LinesearchMethod,Nothing}=nothing,
    retraction::AbstractRetraction=Cayley(), options_kwargs...) where {T,VT<:OptimizerSolution{T}}
    # `T` comes from the `OptimizerSolution{T}` bound and not from `eltype(x)`: for a `NamedTuple` of
    # manifolds the latter is `StiefelManifold{Float64, Matrix{Float64}}` rather than `Float64`.
    G = if (ismissing(∇F!) | isnothing(∇F!))
        if mode == :autodiff
            GradientAutodiff(F, x)
        else
            GradientFiniteDifferences(F, x)
        end
    else
        GradientFunction(F, ∇F!, x)
    end
    problem = (ismissing(∇F!) | isnothing(∇F!)) ? OptimizerProblem(F, x) : OptimizerProblem(F, ∇F!, x)
    ls = isnothing(linesearch) ? default_linesearch(T, algorithm) : linesearch
    _optimizer(x, problem, algorithm, ls, G, retraction, Options(T; options_kwargs...))
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

check_gradient(opt::Optimizer) = check_gradient(gradient(problem(opt)))
print_gradient(opt::Optimizer) = print_gradient(gradient(problem(opt)))

meets_stopping_criteria(status::OptimizerStatus, opt::Optimizer, state::OptimizerState) = meets_stopping_criteria(status, config(opt), iteration_number(state))

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
    [`SimpleSolvers.solve`](@extref) returns a step length whether or not the search succeeded, so
    taking it unconditionally lets a failed search drive the iteration. On the SVD problem of
    `test/optimizer_convergence/svd_optim.jl`, `_BFGS` + `Bisection` + `Geodesic` used to diverge
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

    See [`linesearch_rejected`](@ref) and [`restart!`](@ref). With the restart the same starting
    point converges in 121 iterations at `check(Y) = 6e-14`.
"""
function solver_step!(x::OptimizerSolution{T}, state::OptimizerState{T}, opt::Optimizer{T,MT}) where {T,MT}
    # update cache
    # solve H δx = - ∇f
    # rhs is -g
    # `Adam`, `AdamWithEuclideanDecay` and `MomentumMethod` need their own parameters to form the direction and
    # have no Hessian, the other methods need the Hessian and have no parameters.
    if MT <: FirstOrderMethodWithState
        update!(cache(opt), state, gradient(opt), algorithm(opt), x)
    else
        update!(cache(opt), state, gradient(opt), hessian(opt), x)
        # A (quasi-)Newton direction only descends where the Hessian is positive definite; see
        # `ensure_descent!`. `Adam` and `MomentumMethod` are excluded on purpose -- their direction is
        # a moving average and is allowed not to descend on an individual step.
        ensure_descent!(cache(opt), algorithm(opt), config(opt))
    end
    typeof(algorithm(opt)) <: Newton && update!(state, gradient(opt), x) # this will have to be removed later

    for _ in 1:config(opt).nan_max_iterations
        update_section!(section(cache(opt)), section(state), direction(cache(opt)), opt.retraction)
        _copyto!(solution(cache(opt)), section(cache(opt)))
        # compute_new_iterate!(solution(cache(opt)), x, one(T), direction(cache(opt)), cache(opt), opt.retraction)
        f = value(problem(opt), solution(cache(opt)))
        if isnan(f) || isinf(f)
            (opt.config.verbosity ≥ 2 && @warn "NaN or Inf detected in optimizer. Reducing length of direction vector.")
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
    # `linesearch_rejected` for what the difference costs.
    ls_status = solve_with_status(linesearch(opt), one(T), (x=x, state=state))

    # A rejected search means "no step along *this* direction decreases the merit". For a
    # quasi-Newton method that is a statement about `Q`, so throw `Q` away and try the one direction
    # that always descends. `Adam` and `MomentumMethod` are excluded for the same reason
    # `ensure_descent!` excludes them: their direction is a moving average with no `Q` behind it, and
    # it is allowed not to descend on an individual step.
    if linesearch_rejected(ls_status) && !(MT <: FirstOrderMethodWithState)
        config(opt).verbosity ≥ 2 && @warn "the line search returned $(outcome(ls_status)), i.e. no step along the $(algorithm(opt)) direction decreased the merit; restarting the inverse Hessian and searching along the steepest-descent direction instead." maxlog = 1
        restart!(state)
        _copyto!(direction(cache(opt)), rhs(cache(opt)))
        update_section!(section(cache(opt)), section(state), direction(cache(opt)), opt.retraction)
        _copyto!(solution(cache(opt)), section(cache(opt)))
        ls_status = solve_with_status(linesearch(opt), one(T), (x=x, state=state))
    end

    # Whatever the second search reports, its step is taken. Substituting `α = 0` instead -- "if even
    # steepest descent cannot decrease the merit, do not move" -- reads as the safer choice and is
    # not: a zero step makes `rxₐ` vanish, `x_converged` fires, and the solve reports convergence at
    # a point where `‖∇f‖` is still of order one. That is what `test/descent_direction_tests.jl`
    # catches, on three of its forty-eight combinations.
    α = steplength(ls_status)
    _rmul!(direction(cache(opt)), α)

    # compute new minimizer
    update_section!(section(cache(opt)), section(state), direction(cache(opt)), opt.retraction)
    _copyto!(solution(cache(opt)), section(cache(opt)))

    _copyto!(x, solution(cache(opt)))
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

    |x - x'|               = 3.05e-05
    |x - x'|/|x'|          = 6.55e+04
    |f(x) - f(x')|         = 9.31e-10
    |f(x) - f(x')|/|f(x')| = 4.30e+09
    |g(x) - g(x')|         = 0.00e+00
    |g(x)|                 = 6.10e-05
, Float32[0.0, 4.656613f-10], 2.1684043f-19, GeometricOptimizers.OptimizerTraceEntry{Float32, Float32}[])

julia> x
2-element Vector{Float32}:
 0.0
 4.656613f-10

julia> iteration_number(state)
5
```

The trailing empty vector is the per-iteration [`trace`](@ref), which is only filled if
`Options(store_trace = true)` asked for it.

Also see [`solver_step!`](@ref).
"""
function solve!(x::OptimizerSolution{T}, state::OptimizerState, opt::Optimizer{T}) where {T}
    initialize_state!(state)

    # `status` below is computed on every iteration regardless, so recording it costs one `Bool` test
    # per iteration when `store_trace` is unset. See `trace`.
    f = value(problem(opt), x)
    tracing = config(opt).store_trace
    _trace = OptimizerTraceEntry{typeof(f),T}[]

    while true
        increase_iteration_number!(state)
        solver_step!(x, state, opt)
        status = OptimizerStatus(state, cache(opt), value(problem(opt), x); config=config(opt))
        tracing && push!(_trace, OptimizerTraceEntry(iteration_number(state), value(problem(opt), x), g_residual(status)))
        meets_stopping_criteria(status, opt, state) && break
        update!(state, opt, x)
    end

    status = OptimizerStatus(state, cache(opt), value(problem(opt), x); config=config(opt))
    warn_iteration_number(state, config(opt))
    OptimizerResult(status, x, value(problem(opt), x), _trace)
end

update!(state::OptimizerState, opt::Optimizer, x::OptimizerSolution) = update!(state, gradient(opt), x)

function initialize_state!(state::OptimizerState)
    state
end

const INITIAL_BFGS_X = 0.12345
const INITIAL_BFGS_G = 0.54321
const INITIAL_BFGS_F = 0.23456

function initialize_state!(state::Union{BFGSState{T},DFPState{T}}) where {T}
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
    update_section!(state.section, gradient_array(cache(opt)), x -> retraction(opt.retraction, x))
    state
end
