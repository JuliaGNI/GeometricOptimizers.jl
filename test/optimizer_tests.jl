using LinearAlgebra
using NaNMath: log
using GeometricOptimizers
using GeometricOptimizers: Newton, _DFP, _BFGS
using GeometricOptimizers: gradient, hessian, linesearch, problem, initialize!, update!, solver_step!
using GeometricOptimizers: DEFAULT_LEARNING_RATE, default_linesearch
using SimpleSolvers: Static, Backtracking, BierlaireQuadratic, Quadratic, Bisection, StrongWolfe, GradientAutodiff, GradientFunction
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
        for _linesearch in (Static(T(0.1)), Backtracking(T), Backtracking(T; expand=true),
            BierlaireQuadratic(T), Quadratic(T), Bisection(T), StrongWolfe(T; c₂=T(0.1)))
            @testset "$(method) & $(_linesearch) & $(T)" begin
                n = 1
                x = ones(T, n)
                opt = Optimizer(x, F; algorithm=method, linesearch=_linesearch)
                state = OptimizerState(method, x)

                @test typeof(gradient(opt)) <: GradientAutodiff

                solve!(x, state, opt)
                @test norm(x) ≈ zero(T) atol = ∛(2000eps(T))
                @test F(x) ≈ F(zero(T)) atol = ∛(2000eps(T))

                x = ones(T, n)
                opt = Optimizer(x, F; (∇F!)=∇F!, algorithm=method, linesearch=_linesearch)

                @test typeof(gradient(opt)) <: GradientFunction

                state = OptimizerState(method, x)

                solve!(x, state, opt)
                @test norm(x) ≈ zero(T) atol = ∛(2000eps(T))
                @test F(x) ≈ F(0) atol = ∛(2000eps(T))
            end
        end
    end
end


# The first-order methods produce a *direction* of their own and nothing more, so the step
# length is entirely the `α` of the line search: for them a fixed learning rate is
# `Static(η)`, which is why it is the default. `Adam` used to carry an `η` field instead, which
# meant the step was scaled twice — once by `η` and once by the line search, whose default was
# `Backtracking` and therefore not `1`.
#
# The methods that come with a Hessian (or an approximation of it) produce a direction whose
# *length* is meaningful, so they keep `Backtracking`, which starts its trial step at `α = 1`
# and shortens it if it has to — and, with `expand = true`, lengthens it if the first trial is
# accepted and longer ones keep improving the merit.
@testset "the default line search matches the method" begin
    for T in (Float64, Float32)
        # The `AdamFamily` methods are the ones that keep a fixed step: their direction is a moving
        # average and is not required to descend on an individual step, so a sufficient-decrease
        # search has nothing to work with. For `AdamWithEuclideanDecay` a fixed step is also what
        # makes `λ` mean what it is documented to mean — the merit does not contain the penalty, so
        # a searching `α` would be picked partly in order to undo the decay, and `αλ` would be
        # whatever the search settled on. See `default_linesearch`.
        for method in (Adam(T), AdamWithEuclideanDecay(T))
            ls = default_linesearch(T, method)
            @test ls isa Static{T}
            @test ls.α == T(DEFAULT_LEARNING_RATE)

            x = ones(T, 3)
            @test linesearch(Optimizer(x, F; algorithm=method)).method isa Static{T}
        end

        # Everything else searches. `GradientMethod` and `MomentumMethod` joined this group in 0.2.0,
        # once a line search could take its trial step through a retraction; before that `Static` was
        # the only thing that worked on manifold parameters, so they had no choice. `_DFP` rejoined it
        # in SimpleSolvers 0.11: it needs a search that can *lengthen* a step, and until `expand` that
        # ruled `Backtracking` out for it entirely (49_679 iterations on the SVD problem, against 830
        # with the expansion phase). See `default_linesearch`.
        for method in (GradientMethod(), MomentumMethod(T(0.1)), Newton(), _BFGS(), _DFP())
            ls = default_linesearch(T, method)
            @test ls isa Backtracking{T}
            # the property the default turns on, and the one a future SimpleSolvers bump could drop
            # silently: without it `_DFP` does not converge on the SVD problem at all
            @test ls.expand

            x = ones(T, 3)
            opt_ls = linesearch(Optimizer(x, F; algorithm=method)).method
            @test opt_ls isa Backtracking{T}
            @test opt_ls.expand
        end
    end

    # `DEFAULT_LEARNING_RATE` is written as a `Float64` literal so that the `Float32` default is
    # `1f-3` and not `Float32(1.0e-3)` rounded through `Float64` — i.e. so that `Static`'s `α`
    # prints as `0.001` for both element types.
    @test default_linesearch(Float32, Adam(Float32)).α === 1.0f-3
    @test default_linesearch(Float32, AdamWithEuclideanDecay(Float32)).α === 1.0f-3

    # `Adam` no longer takes a learning rate, and `β₁`, `β₂` and `δ` are keyword arguments, so
    # an old positional call fails instead of quietly setting `β₁ = 0.01`.
    @test_throws MethodError Adam(0.01)
end

@testset "Test Nan handling in optimizers" begin

    fnan(x::T) where {T} = log(x) + x^2
    Fnan(x::AbstractVector) = sum(fnan.(x))

    function test_nan_handling_for_optimizers(F, n::Integer, ::Type{T}; kwargs...) where {T}
        x = 0.2 * ones(T, n)
        opt = Optimizer(x, F; algorithm=Newton(), linesearch=Static(), verbosity=2, kwargs...)
        state = OptimizerState(Newton(), x)
        solve!(x, state, opt)
    end

    @test_warn "NaN or Inf detected in optimizer. Reducing length of direction vector." test_nan_handling_for_optimizers(Fnan, 1, Float64; max_iterations=5)

end
