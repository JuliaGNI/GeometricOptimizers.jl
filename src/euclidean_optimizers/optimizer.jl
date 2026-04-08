
const SOLUTION_MAX_PRINT_LENGTH = 10

"""
    EuclideanOptimizer

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
optimizer = EuclideanOptimizer(x, F; algorithm = algorithm, linesearch = Bisection())

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
optimizer = EuclideanOptimizer(x, F; algorithm = algorithm, linesearch = Backtracking())

solve!(x, state, optimizer)
x

# output

3-element Vector{Float64}:
 1.0
 1.0
 1.0
```

"""
struct EuclideanOptimizer{T,
    ALG<:EuclideanOptimizerMethod,
    OBJ<:OptimizerProblem{T},
    GT<:Gradient{T},
    HT<:Hessian{T},
    OCT<:OptimizerCache,
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

    function EuclideanOptimizer(algorithm::EuclideanOptimizerMethod, problem::OptimizerProblem{T}, hessian::Hessian{T}, cache::OptimizerCache, linesearch::LinesearchMethod; gradient=GradientAutodiff{T}(problem.F, length(cache.x)), retraction=Cayley(), options_kwargs...) where {T}
        config = Options(T; options_kwargs...)
        ls_problem = linesearch_problem(problem, gradient, cache)
        ls = Linesearch(ls_problem, linesearch)
        new{T,typeof(algorithm),typeof(problem),typeof(gradient),typeof(hessian),typeof(cache),typeof(ls),typeof(retraction)}(algorithm, problem, gradient, hessian, config, cache, ls, retraction)
    end
end

function EuclideanOptimizer(x::VT, problem::OptimizerProblem; algorithm::EuclideanOptimizerMethod=BFGS(), linesearch::LinesearchMethod=Backtracking(), options_kwargs...) where {T,VT<:AbstractVector{T}}
    cache = OptimizerCache(algorithm, x)
    hes = Hessian(algorithm, problem, x)
    EuclideanOptimizer(algorithm, problem, hes, cache, linesearch; options_kwargs...)
end

function EuclideanOptimizer(x::AbstractVector, F::Function; (∇F!)=nothing, mode=:autodiff, kwargs...)
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
    EuclideanOptimizer(x, problem; gradient=G, kwargs...)
end

config(opt::EuclideanOptimizer) = opt.config
problem(opt::EuclideanOptimizer) = opt.problem
algorithm(opt::EuclideanOptimizer) = opt.algorithm
linesearch(opt::EuclideanOptimizer) = opt.linesearch
hessian(opt::EuclideanOptimizer) = opt.hessian
direction(opt::EuclideanOptimizer) = direction(cache(opt))
rhs(opt::EuclideanOptimizer) = rhs(cache(opt))
cache(opt::EuclideanOptimizer) = opt.cache
gradient(opt::EuclideanOptimizer) = opt.gradient

check_gradient(opt::EuclideanOptimizer) = check_gradient(gradient(problem(opt)))
print_gradient(opt::EuclideanOptimizer) = print_gradient(gradient(problem(opt)))

meets_stopping_criteria(status::OptimizerStatus, opt::EuclideanOptimizer, state::OptimizerState) = meets_stopping_criteria(status, config(opt), iteration_number(state))

function initialize!(opt::EuclideanOptimizer, x::AbstractVector)
    initialize!(cache(opt), x)

    opt
end

"""
    solver_step!(x, state, opt)

Compute a full iterate for an [`EuclideanOptimizer`](@ref).

!!! info
    This also performs a line search.

# Examples

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: solver_step!, NewtonOptimizerState)
julia> f(x) = sum(x .^ 2 + x .^ 3 / 3);

julia> x = [1f0, 2f0]
2-element Vector{Float32}:
 1.0
 2.0

julia> opt = EuclideanOptimizer(x, f; algorithm = Newton());

julia> state = NewtonOptimizerState(x);

julia> update!(state, gradient(opt), x);

julia> solver_step!(x, state, opt)
2-element Vector{Float32}:
 0.25
 0.6666666
```
"""
function solver_step!(x::VT, state::OptimizerState{T}, opt::EuclideanOptimizer{T}) where {T,VT<:Union{AbstractVector{T},Manifold{T}}}
    # update cache
    update!(cache(opt), state, gradient(opt), hessian(opt), x)
    typeof(algorithm(opt)) <: Newton && update!(state, gradient(opt), x) # this will have to be removed later

    # solve H δx = - ∇f
    # rhs is -g
    compute_direction!(opt, state)

    for _ in 1:config(opt).nan_max_iterations
        update_section!(section(cache(opt)), section(state), direction(cache(opt)), opt.retraction)
        solution(cache(opt)) .= section(cache(opt)).Y
        # compute_new_iterate!(solution(cache(opt)), x, one(T), direction(cache(opt)), cache(opt), opt.retraction)
        f = value(problem(opt), solution(cache(opt)))
        if isnan(f) || isinf(f)
            (opt.config.verbosity ≥ 2 && @warn "NaN or Inf detected in optimizer. Reducing length of direction vector.")
            direction(cache(opt)) .*= T(config(opt).nan_factor)
        else
            break
        end
    end

    # apply line search
    α = solve(linesearch(opt), one(T), (x=x,))

    # compute new minimizer
    compute_new_iterate!(x, α, direction(opt))

    x
end

"""
    compute_direction!(opt, state)

Compute the search direction for the optimization problem described by `opt` and store the result in `state`.
"""
compute_direction!(::EuclideanOptimizer, ::OptimizerState)

"""
    solve!(x, state, opt)

Solve the optimization problem described by `opt::`[`EuclideanOptimizer`](@ref) and store the result in `x`.

# Examples

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: solve!, NewtonOptimizerState, update!, iteration_number; using Random: seed!; seed!(123))
julia> f(x) = sum(x .^ 2 + x .^ 3 / 3);

julia> x = [1f0, 2f0]
2-element Vector{Float32}:
 1.0
 2.0

julia> opt = EuclideanOptimizer(x, f; algorithm = Newton());

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
function solve!(x::AbstractVector, state::OptimizerState, opt::EuclideanOptimizer)
    initialize_state!(state)

    while true
        increase_iteration_number!(state)
        solver_step!(x, state, opt)
        status = OptimizerStatus(state, cache(opt), value(problem(opt), x); config=config(opt))
        meets_stopping_criteria(status, opt, state) && break
        update!(state, gradient(opt), x)
    end

    status = OptimizerStatus(state, cache(opt), value(problem(opt), x); config=config(opt))
    warn_iteration_number(state, config(opt))
    OptimizerResult(status, x, value(problem(opt), x))
end

function initialize_state!(state::OptimizerState)
    state
end

const INITIAL_BFGS_X = 0.12345
const INITIAL_BFGS_G = 0.54321
const INITIAL_BFGS_F = 0.23456

function initialize_state!(state::Union{BFGSState{T},DFPState{T}}) where {T}
    state.x̄ .= T(INITIAL_BFGS_X)
    state.ḡ .= T(INITIAL_BFGS_G)
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
function update!(state::NewtonOptimizerState, opt::EuclideanOptimizer, x::AbstractVector)
    update!(state, gradient(opt), x)
    update_section!(state.section, gradient_array(cache(opt)), x -> retraction(opt.retraction, x))
    state
end
