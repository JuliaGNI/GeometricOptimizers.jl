
"""
    OptimizerResult

Serves as a diagnostic tool for the [`EuclideanOptimizer`](@ref) and is the return argument of [`solve!`](@ref).

# Keys

- `status::`[`OptimizerStatus`](@ref): current status of the optimization,
- `x`: solution,
- `f`: function value at solution.

"""
mutable struct OptimizerResult{T,YT,VT<:OptimizerSolution{T},OST<:OptimizerStatus{T,YT}}
    status::OST

    x::VT    # current solution
    f::YT    # current function
end

status(result::OptimizerResult) = result.status

solution(result::OptimizerResult) = result.x
Base.minimum(result::OptimizerResult) = result.f
