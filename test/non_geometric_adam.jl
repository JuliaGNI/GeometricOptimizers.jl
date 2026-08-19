using GeometricOptimizers
using GeometricOptimizers: cache, direction, first_moment, second_moment,
    increase_iteration_number!, iteration_number, solver_step!
using Test

function linear_stiefel_objective(C)
    Y -> sum(C .* Y.A)
end

@testset "NonGeometricAdam constructor and scope" begin
    method = NonGeometricAdam(Float32; β₁=0.8, β₂=0.95, δ=1f-6)
    @test method.β₁ isa Float32
    @test method.β₂ isa Float32
    @test method.δ isa Float32
    @test_throws ArgumentError NonGeometricAdam(; β₁=1.0)
    @test_throws ArgumentError NonGeometricAdam(; β₂=-0.1)
    @test_throws ArgumentError NonGeometricAdam(; δ=-1.0)
    @test_throws ArgumentError OptimizerState(method, rand(3))
    @test_throws ArgumentError OptimizerState(method, (Y=rand(StiefelManifold, 4, 2), z=rand(3)))
end

@testset "NonGeometricAdam ambient moments" begin
    for T in (Float64, Float32)
        Y = rand(StiefelManifold{T}, 5, 2)
        C = T[1 2; -3 4; 2 -1; 1 0; -2 3]
        β₁, β₂, δ, η = T(0.5), T(0.25), T(0.1), T(0.01)
        method = NonGeometricAdam(T; β₁, β₂, δ)
        objective = linear_stiefel_objective(C)
        opt = Optimizer(Y, objective; algorithm=method, linesearch=Static(η), retraction=Cayley())
        state = OptimizerState(method, Y)

        Y_before = copy(Y.A)
        increase_iteration_number!(state)
        solver_step!(Y, state, opt)
        G = rgrad(StiefelManifold(Y_before), C)
        expected_m₁ = (one(T) - β₁) .* G
        expected_m₂ = (one(T) - β₂) .* (G .* G)
        @test first_moment(cache(opt)) ≈ expected_m₁ rtol=T === Float32 ? 5f-5 : 1e-12
        @test second_moment(cache(opt)) ≈ expected_m₂ rtol=T === Float32 ? 5f-5 : 1e-12
        @test iteration_number(state) == 1
        @test check(Y) < (T === Float32 ? 5f-5 : 1e-12)
        @test direction(cache(opt)) isa StiefelLieAlgHorMatrix
        @test size(Matrix(direction(cache(opt)))) == (5, 5)
        @test size(first_moment(cache(opt))) == (5, 2)

        increase_iteration_number!(state)
        solver_step!(Y, state, opt)
        @test iteration_number(state) == 2
        @test check(Y) < (T === Float32 ? 5f-5 : 1e-12)
    end
end

@testset "NonGeometricAdam differs from Adam" begin
    Y₁ = rand(StiefelManifold, 6, 3)
    Y₂ = copy(Y₁)
    C = randn(6, 3)
    objective = linear_stiefel_objective(C)
    opt₁ = Optimizer(Y₁, objective; algorithm=Adam(), linesearch=Static(0.01), retraction=Cayley())
    opt₂ = Optimizer(Y₂, objective; algorithm=NonGeometricAdam(), linesearch=Static(0.01), retraction=Cayley())
    state₁ = OptimizerState(opt₁.algorithm, Y₁)
    state₂ = OptimizerState(opt₂.algorithm, Y₂)
    increase_iteration_number!(state₁)
    increase_iteration_number!(state₂)
    solver_step!(Y₁, state₁, opt₁)
    solver_step!(Y₂, state₂, opt₂)
    @test !isapprox(Y₁.A, Y₂.A; atol=1e-12, rtol=1e-12)
end
