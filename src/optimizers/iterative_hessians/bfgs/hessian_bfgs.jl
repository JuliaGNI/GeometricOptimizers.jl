"""
    HessianBFGS <: Hessian

A `struct` derived from [`SimpleSolvers.Hessian`](@extref) to be used for an [`Optimizer`](@ref).
"""
struct HessianBFGS{T,FT<:Callable} <: IterativeHessian{T}
    F::FT

    function HessianBFGS(F::FT, ::OptimizerSolution{T}) where {T,FT<:Callable}
        new{T,FT}(F)
    end
end

HessianBFGS{T}(F::Callable, n::Integer) where {T} = HessianBFGS(F, zeros(T, n))

HessianBFGS(obj::OptimizerProblem, x::OptimizerSolution) = HessianBFGS(obj.F, x)

Hessian(::_BFGS, ForOBJ::Callable, x::OptimizerSolution) = HessianBFGS(ForOBJ, x)

Hessian(::_BFGS, ForOBJ::OptimizerProblem, x::OptimizerSolution) = HessianBFGS(ForOBJ.F, x)

(hes::Hessian)(::AbstractMatrix, ::OptimizerSolution) = error("This has to be called together with a cache.")
