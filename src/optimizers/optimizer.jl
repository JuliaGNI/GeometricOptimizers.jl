
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
 1.1102230246251565e-16
 1.1102230246251565e-16
 1.1102230246251565e-16
```
We note that this same problem may have trouble converging with other line searches:

```jldoctest; setup = :(using GeometricOptimizers; F(x) = sum(sin.(x) .^ 2))
x = ones(3)
algorithm = Newton()
state = OptimizerState(algorithm, x)
optimizer = Optimizer(x, F; algorithm = algorithm, linesearch = Backtracking())

solve!(x, state, optimizer)
x

# output

3-element Vector{Float64}:
 1.0
 1.0
 1.0
```

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

    function Optimizer(algorithm::OptimizerMethod, problem::OptimizerProblem{T}, hessian::Hessian{T}, cache::OptimizerCache, linesearch::LinesearchMethod; gradient=GradientAutodiff{T}(problem.F, length(cache.x)), retraction=Cayley(), options_kwargs...) where {T}
        config = Options(T; options_kwargs...)
        ls_problem = linesearch_problem(problem, gradient, cache)
        ls = Linesearch(ls_problem, linesearch)
        new{T,typeof(algorithm),typeof(problem),typeof(gradient),typeof(hessian),typeof(cache),typeof(ls),typeof(retraction)}(algorithm, problem, gradient, hessian, config, cache, ls, retraction)
    end
end

function Optimizer(x::VT, problem::OptimizerProblem; algorithm::OptimizerMethod=_BFGS(), linesearch::LinesearchMethod=Backtracking(), options_kwargs...) where {T,VT<:OptimizerSolution{T}}
    # translate to the correct type if we use the momentum method
    algorithm = typeof(algorithm) <: MomentumMethod ? MomentumMethod(T(algorithm.α)) : algorithm
    cache = OptimizerCache(algorithm, x)
    hes = Hessian(algorithm, problem, x)
    Optimizer(algorithm, problem, hes, cache, linesearch; options_kwargs...)
end

function Optimizer(x::OptimizerSolution, F::Function; (∇F!)=nothing, mode=:autodiff, kwargs...)
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
    Optimizer(x, problem; gradient=G, kwargs...)
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
"""
function solver_step!(x::OptimizerSolution{T}, state::OptimizerState{T}, opt::Optimizer{T,MT}) where {T,MT}
    # update cache
    # solve H δx = - ∇f
    # rhs is -g
    MT <: Adam ? update!(cache(opt), state, gradient(opt), algorithm(opt), x) : update!(cache(opt), state, gradient(opt), hessian(opt), x)
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
    α = solve(linesearch(opt), one(T), (x=x,))
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

    |x - x'|               = 7.82e-03
    |x - x'|/|x'|          = 2.56e+02
    |f(x) - f(x')|         = 6.18e-05
    |f(x) - f(x')|/|f(x')| = 6.63e+04
    |g(x) - g(x')|         = 1.57e-02
    |g(x)|                 = 6.10e-05
, Float32[4.6478817f-8, 3.0517578f-5], 9.313341f-10)

julia> x
2-element Vector{Float32}:
 4.6478817f-8
 3.0517578f-5

julia> iteration_number(state)
4
```

Also see [`solver_step!`](@ref).
"""
function solve!(x::OptimizerSolution, state::OptimizerState, opt::Optimizer)
    initialize_state!(state)

    while true
        increase_iteration_number!(state)
        solver_step!(x, state, opt)
        status = OptimizerStatus(state, cache(opt), value(problem(opt), x); config=config(opt))
        meets_stopping_criteria(status, opt, state) && break
        update!(state, opt, x)
    end

    status = OptimizerStatus(state, cache(opt), value(problem(opt), x); config=config(opt))
    warn_iteration_number(state, config(opt))
    OptimizerResult(status, x, value(problem(opt), x))
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
    _fill!(state.ḡ, T(INITIAL_BFGS_G))
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
