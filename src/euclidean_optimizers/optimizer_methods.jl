"""
    OptimizerMethod <: SolverMethod

The `OptimizerMethod` is used in [`Optimizer`](@ref) and determines the algorithm that is used.
"""
abstract type OptimizerMethod <: SolverMethod end

"""
    QuasiNewtonOptimizerMethod <: OptimizerMethod

Includes [`_BFGS`](@ref) and [`_DFP`](@ref).
"""
abstract type QuasiNewtonOptimizerMethod <: OptimizerMethod end

struct Newton <: OptimizerMethod end

Hessian(::Newton, ForOBJ::Union{Callable,OptimizerProblem}, x::AbstractVector) = HessianAutodiff(ForOBJ, x)
HessianAutodiff(F::OptimizerProblem, x) = HessianAutodiff(F.F, x)

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct _DFP <: QuasiNewtonOptimizerMethod end

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct _BFGS <: QuasiNewtonOptimizerMethod end

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::_DFP) = print(io, "DFP")
Base.show(io::IO, alg::_BFGS) = print(io, "BFGS")
