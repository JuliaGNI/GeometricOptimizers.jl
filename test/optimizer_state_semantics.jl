using GeometricOptimizers
using GeometricOptimizers: value, previous_value, solution, previous_solution, trace,
                           status,
                           gradient, cache, solver_step!, update!,
                           increase_iteration_number!,
                           latest_gradient_is_current,
                           AbstractLieAlgHorMatrix, StiefelLieAlgHorMatrix,
                           GrassmannLieAlgHorMatrix, _fill!, _zero
using Test

# qualified and not imported, so that this file loads on a tree that lacks either
get_backend(x) = GeometricOptimizers.KernelAbstractions.get_backend(x)
zero_tangent(x) = GeometricOptimizers.zero_tangent(x)
import Random

Random.seed!(1234)

# What a state holds once `solve!` returns, and what the status it returns compares. Every figure
# here is an equality between two stored numbers, so every comparison is `==` and none is `≈`: a
# match is the same pair of numbers or it is not. See issue #108 and `KNOWN_ISSUES.md` A24.

objective(x) = sum(x .^ 4) + sum(x .^ 2)

const ITERATIONS = 6

vector_methods() = (GradientMethod(), MomentumMethod(), Adam(), BFGS(), DFP(), Newton())
function stiefel_methods()
    (GradientMethod(), MomentumMethod(), Adam(), BFGS(), DFP(),
        ScalarMomentAdam())
end

function solve_for(method, x)
    state = OptimizerState(method, x)
    opt = Optimizer(x, objective; algorithm = method, store_trace = true,
        max_iterations = ITERATIONS, warn_iterations = 0)
    result = solve!(x, state, opt)
    state, result, opt
end

function fixtures(T)
    (
        (:vector, vector_methods(), () -> T[1, 2, 3]),
        (:stiefel, stiefel_methods(), () -> rand(StiefelManifold{T}, 5, 3)))
end

@testset "the state holds the iterate `solve!` returns, and its objective" begin
    for T in (Float32, Float64), (name, methods, x₀) in fixtures(T), method in methods
        x = x₀()
        state, result, opt = solve_for(method, x)
        @testset "$(nameof(typeof(method))) on $(name), $(T)" begin
            @test minimum(result) isa T
            @test value(state) == minimum(result)
            @test solution(state) == x
            @test solution(state) == solution(result)
            # the gradient the state holds is the one at `x`, the gradient `solver_step!` refreshed
            # at the accepted iterate (issue A10); `BFGSState` holds only the previous one
            method isa GeometricOptimizers.QuasiNewtonOptimizerMethod ||
                @test gradient(state) ==
                      GeometricOptimizers.latest_gradient(GeometricOptimizers.cache(opt))
        end
    end
end

@testset "`Δf` spans one step" begin
    for T in (Float32, Float64), (name, methods, x₀) in fixtures(T), method in methods
        x = x₀()
        state, result = solve_for(method, x)
        entries = trace(result)
        @testset "$(nameof(typeof(method))) on $(name), $(T)" begin
            @test length(entries) ≥ 2
            @test status(result).Δf == entries[end].f - entries[end - 1].f
            @test previous_value(state) == entries[end - 1].f
        end
    end
end

# `Newton`'s state moves its section to the iterate, as every other state does, so the cache reuses
# the gradient `solver_step!` refreshed instead of evaluating it again (issue A13). Rosenbrock, so
# that neither the gradient nor the step is zero inside the window.
@testset "`Newton` reuses the refreshed gradient" begin
    rosenbrock(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    x = [-1.2, 1.0]
    state = OptimizerState(Newton(), x)
    opt = Optimizer(x, rosenbrock; algorithm = Newton())
    for k in 1:6
        increase_iteration_number!(state)
        @test latest_gradient_is_current(cache(opt), state, x) == (k > 1)
        solver_step!(x, state, opt)
        update!(state, opt, x, rosenbrock(x))
    end
    @test rosenbrock(x) > 1e-8
end

# `zero` of a point is not a tangent vector: issue #21. `zero_tangent` is the internal name for the
# zero horizontal lift, on the backend of the point.
@testset "`zero` of a manifold point is not its horizontal lift" begin
    for T in (Float32, Float64)
        Y = rand(StiefelManifold{T}, 6, 3)
        @test !(zero(Y) isa AbstractLieAlgHorMatrix)
        @test zero_tangent(Y) isa StiefelLieAlgHorMatrix{T}
        @test iszero(zero_tangent(Y))
        @test get_backend(zero_tangent(Y).B) == get_backend(Y.A)
        @test _zero(Y) isa StiefelLieAlgHorMatrix{T}

        Z = rand(GrassmannManifold{T}, 6, 3)
        @test !(zero(Z) isa AbstractLieAlgHorMatrix)
        @test zero_tangent(Z) isa GrassmannLieAlgHorMatrix{T}
        @test iszero(zero_tangent(Z))
        @test get_backend(zero_tangent(Z).B) == get_backend(Z.A)
    end
end

# An uninitialised gradient on a manifold reads as `NaN`, never as `0`: issue #22. The point is not
# poisoned; filling one is an error rather than a silent no-op.
@testset "an uninitialised manifold gradient cannot read as zero" begin
    for T in (Float32, Float64)
        Y = rand(StiefelManifold{T}, 6, 3)
        for method in (BFGS(), DFP())
            state = OptimizerState(method, Y)
            for tangent in (state.ḡ, state.s)
                @test all(isnan, tangent.A.S)
                @test all(isnan, tangent.B)
            end
            @test isnan(previous_value(state))
            @test previous_value(state) isa T
        end
        @test_throws ErrorException _fill!(Y, T(NaN))
    end
end

@testset "an iteration-count warning is a log record" begin
    x = [1.0, 2.0, 3.0]
    state = OptimizerState(GradientMethod(), x)
    opt = Optimizer(x, objective; algorithm = GradientMethod(), max_iterations = 3,
        warn_iterations = 1)
    @test_logs (:warn, r"iterations") match_mode=:any solve!(x, state, opt)
end
