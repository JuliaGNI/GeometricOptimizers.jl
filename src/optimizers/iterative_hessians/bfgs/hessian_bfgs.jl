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

Hessian(::BFGS, ForOBJ::Callable, x::OptimizerSolution) = HessianBFGS(ForOBJ, x)

Hessian(::BFGS, ForOBJ::OptimizerProblem, x::OptimizerSolution) = HessianBFGS(ForOBJ.F, x)

# On this package's own Hessian types and not on `SimpleSolvers.Hessian`, which is where this stood
# until 0.6.1: `OptimizerSolution` is an alias for a union of types this package does not own, so the
# method owned neither side of its own signature and any package loading this one changed what
# calling *any* Hessian on a plain `AbstractVector` meant. The error is about the hessians here --
# they are built from a cache, and there is nothing to say about anyone else's -- so naming them is
# both the fix and the more accurate signature. `IterativeHessian` covers `HessianBFGS` and
# `HessianDFP`; the `NoHessian` arm is in `manifold_optimizers/gradient_optimizer.jl`, where that type
# is defined, because a signature needs its types to exist and that file is included after this one.
# See issue #16.
(hes::IterativeHessian)(::AbstractMatrix, ::OptimizerSolution) =
    error("This has to be called together with a cache.")
