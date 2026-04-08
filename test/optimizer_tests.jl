using LinearAlgebra
using NaNMath: log
using GeometricOptimizers
using GeometricOptimizers: Newton, _DFP, _BFGS
using GeometricOptimizers: gradient, hessian, linesearch, problem, initialize!, update!, solver_step!
using SimpleSolvers: Static, Backtracking, BierlaireQuadratic, Quadratic, Bisection, GradientAutodiff, GradientFunction
using Test
using Random
Random.seed!(123)

include("optimizers_problems.jl")

struct OptimizerTest{T} <: OptimizerState{T} end

test_optim = OptimizerTest{Float64}()
test_x = zeros(3)
test_obj = OptimizerProblem(F, test_x)

@test_throws MethodError gradient(test_optim)
@test_throws MethodError hessian(test_optim)
@test_throws MethodError linesearch(test_optim)
@test_throws MethodError problem(test_optim)

# test if the correct error is thrown when calling `initialize!` on an `OptimizerState`.
# @test_throws ErrorException initialize!(test_optim, test_x)
@test_throws MethodError update!(test_optim, test_x)
@test_throws MethodError solver_step!(test_x, test_optim)

for T in (Float64, Float32)
    for method in (Newton(), _DFP(), _BFGS())
        for _linesearch in (Static(T(0.1)), Backtracking(T), BierlaireQuadratic(T), Quadratic(T), Bisection(T))
            @testset "$(method) & $(_linesearch) & $(T)" begin
                n = 1
                x = ones(T, n)
                opt = EuclideanOptimizer(x, F; algorithm=method, linesearch=_linesearch)
                state = OptimizerState(method, x)

                @test typeof(gradient(opt)) <: GradientAutodiff

                solve!(x, state, opt)
                @test norm(x) ≈ zero(T) atol = ∛(2000eps(T))
                @test F(x) ≈ F(zero(T)) atol = ∛(2000eps(T))

                x = ones(T, n)
                opt = EuclideanOptimizer(x, F; (∇F!)=∇F!, algorithm=method, linesearch=_linesearch)

                @test typeof(gradient(opt)) <: GradientFunction

                state = OptimizerState(method, x)

                solve!(x, state, opt)
                @test norm(x) ≈ zero(T) atol = ∛(2000eps(T))
                @test F(x) ≈ F(0) atol = ∛(2000eps(T))
            end
        end
    end
end


@testset "Test Nan handling in optimizers" begin

    fnan(x::T) where {T} = log(x) + x^2
    Fnan(x::AbstractVector) = sum(fnan.(x))

    function test_nan_handling_for_optimizers(F, n::Integer, ::Type{T}; kwargs...) where {T}
        x = 0.2 * ones(T, n)
        opt = EuclideanOptimizer(x, F; algorithm=Newton(), linesearch=Static(), verbosity=2, kwargs...)
        state = OptimizerState(Newton(), x)
        solve!(x, state, opt)
    end

    @test_warn "NaN or Inf detected in optimizer. Reducing length of direction vector." test_nan_handling_for_optimizers(Fnan, 1, Float64; max_iterations=5)

end
