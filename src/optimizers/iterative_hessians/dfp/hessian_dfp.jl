"""
    HessianDFP <: Hessian

The [`SimpleSolvers.Hessian`](@extref) corresponding to the [`DFP`](@ref) method.
"""
struct HessianDFP{T,FT<:Callable} <: IterativeHessian{T}
    F::FT

    function HessianDFP(F::FT, ::OptimizerSolution{T}) where {T,FT<:Callable}
        new{T,FT}(F)
    end
end

HessianDFP{T}(F::Callable, n::Integer) where {T} = HessianDFP(F, zeros(T, n))

HessianDFP(obj::OptimizerProblem, x::OptimizerSolution) = HessianDFP(obj.F, x)

Hessian(::DFP, F::Callable, x::OptimizerSolution) = HessianDFP(F, x)

Hessian(::DFP, Obj::OptimizerProblem, x::OptimizerSolution) = HessianDFP(Obj.F, x)

(hes::HessianDFP)(::AbstractMatrix, ::AbstractVector) = error("This has to be called together with a cache.")
