"""
    EuclideanOptimizerMethod <: SolverMethod

The `EuclideanOptimizerMethod` is used in [`EuclideanOptimizer`](@ref) and determines the algorithm that is used.
"""
abstract type EuclideanOptimizerMethod <: SolverMethod end

"""
    QuasiNewtonOptimizerMethod <: EuclideanOptimizerMethod

Includes [`_BFGS`](@ref) and [`_DFP`](@ref).
"""
abstract type QuasiNewtonOptimizerMethod <: EuclideanOptimizerMethod end

struct Newton <: EuclideanOptimizerMethod end

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

"""
The gradient descent algorithm.
"""
struct GradientMethod <: EuclideanOptimizerMethod end

"""
    MomentumMethod

Stores the *descent parameter*.
"""
struct MomentumMethod{T} <: EuclideanOptimizerMethod
    α::T

    MomentumMethod(α::T=DEFAULT_MOMENTUM_α) where {T} = new{T}(α)
end

"""
    Adam
"""
struct Adam{T} <: EuclideanOptimizerMethod
    η::T
    β₁::T
    β₂::T
    δ::T

    Adam(η=1.0f-3, β₁=9.0f-1, β₂=9.9f-1, δ=1.0f-8; T=typeof(η)) = new{T}(T(η), T(β₁), T(β₂), T(δ))
end

function Adam(T::Type)
    Adam(T(1.0f-3))
end

const DEFAULT_MOMENTUM_α = 0.01

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::_DFP) = print(io, "DFP")
Base.show(io::IO, alg::_BFGS) = print(io, "BFGS")
