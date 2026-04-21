"""
    AbstractOptimizerProblem

See [`SimpleSolvers.LinesearchProblem`](@extref) and [`OptimizerProblem`](@ref).
"""
abstract type AbstractOptimizerProblem{T<:Number} <: AbstractProblem end

"""
    value(obj::AbstractOptimizerProblem, x)

Evaluates the value at `x` (i.e. computes `obj.F(x)`).
"""
function value(obj::AbstractOptimizerProblem, x::OptimizerSolution)
    obj.F(x)
end

"""
    OptimizerProblem <: AbstractOptimizerProblem

Used in [`Optimizer`](@ref). Also compare this to [`SimpleSolvers.NonlinearProblem`](@extref).

# Examples

```jldoctest; setup = :(using GeometricOptimizers)
julia> x = ones(3); F(x) = sum(sin.(x) .^ 2)
F (generic function with 1 method)

julia> OptimizerProblem(F, x)
OptimizerProblem{Float64, typeof(F), Missing, Missing}(F, missing, missing)
```

!!! info
    If `OptimizerProblem` is called on a single function, the fields for [`SimpleSolvers.Gradient`](@extref) and [`SimpleSolvers.Hessian`](@extref) are `missing`.
"""
mutable struct OptimizerProblem{T,TF<:Callable,TG<:Union{Callable,Missing},TH<:Union{Callable,Missing}} <: AbstractOptimizerProblem{T}
    F::TF
    G::TG
    H::TH
end

function OptimizerProblem(F::Callable, G::TG, H::Callable, ::OptimizerSolution{T}) where {T<:Number,TG<:Union{Callable,Missing}}
    OptimizerProblem{T,typeof(F),TG,typeof(H)}(F, G, H)
end

function OptimizerProblem(F::Callable, G::TG, ::OptimizerSolution{T}) where {T<:Number,TG<:Union{Callable,Missing}}
    OptimizerProblem{T,typeof(F),TG,Missing}(F, G, missing)
end

function OptimizerProblem(F::Callable, x::OptimizerSolution; gradient=missing, hessian=missing)
    ismissing(hessian) ? OptimizerProblem(F, gradient, x) : OptimizerProblem(F, gradient, hessian, x)
end

gradient(prob::OptimizerProblem) = prob.G
hessian(prob::OptimizerProblem) = prob.H

function GradientFunction(prob::OptimizerProblem{T}) where {T}
    GradientFunction{T,typeof(prob.F),typeof(prob.G)}(prob.F, prob.G)
end

function GradientFunction(::OptimizerProblem{T,TF,Missing}) where {T,TF<:Callable}
    error("There is no gradient stored in this `OptimizerProblem`!")
end

function gradient(prob::OptimizerProblem, x::OptimizerSolution)
    grad = GradientFunction(prob)
    grad(x)
end
