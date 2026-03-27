"""
    HessianBFGS <: Hessian

A `struct` derived from [`SimpleSolvers.Hessian`](@extref) to be used for an [`EuclideanOptimizer`](@ref).
"""
struct HessianBFGS{T,FT<:Callable} <: IterativeHessian{T}
    F::FT

    function HessianBFGS(F::FT, ::AbstractVector{T}) where {T,FT<:Callable}
        new{T,FT}(F)
    end
end

HessianBFGS{T}(F::Callable, n::Integer) where {T} = HessianBFGS(F, zeros(T, n))

HessianBFGS(obj::OptimizerProblem, x::AbstractVector) = HessianBFGS(obj.F, x)

Hessian(::_BFGS, ForOBJ::Callable, x::AbstractVector) = HessianBFGS(ForOBJ, x)

Hessian(::_BFGS, ForOBJ::OptimizerProblem, x::AbstractVector) = HessianBFGS(ForOBJ.F, x)

(hes::Hessian)(::AbstractMatrix, ::AbstractVector) = error("This has to be called together with a cache.")
