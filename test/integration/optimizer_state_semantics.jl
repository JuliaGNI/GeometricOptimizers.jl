using GeometricOptimizers
using GeometricOptimizers: value, previous_value, solution, trace, status,
                           gradient, cache, solver_step!, update!,
                           increase_iteration_number!,
                           latest_gradient_is_current,
                           AbstractLieAlgHorMatrix, StiefelLieAlgHorMatrix,
                           GrassmannLieAlgHorMatrix, _fill!, _zero, initialize_state!,
                           iteration_number, section
using JLArrays: JLArray
using Test

include("../helpers/eltypes.jl")

# qualified and not imported, so that this file loads on a tree that lacks either
get_backend(x) = GeometricOptimizers.KernelAbstractions.get_backend(x)
zero_tangent(x) = GeometricOptimizers.zero_tangent(x)
import Random

Random.seed!(1234)

# What a state holds once `solve!` returns, and what the status it returns compares. Every figure
# here is an equality between two stored numbers, so every comparison is `==` and none is `≈`: a
# match is the same pair of numbers or it is not. See issue #108 and issue A24 in `CHANGELOG.md`.

objective(x) = sum(x .^ 4) + sum(x .^ 2)

const ITERATIONS = 6

# the starting point of the vector problems; not dyadic, so that it rounds in `Float32`
start(::Type{T}) where {T} = T[1.1, 2.3, 3.7]

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

function fixtures(T, rng)
    (
        (:vector, vector_methods(), () -> start(T)),
        (:stiefel, stiefel_methods(), () -> rand(rng, StiefelManifold{T}, 5, 3)))
end

@testset "the state holds the iterate `solve!` returns, and its objective, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1)
    for (name, methods, x₀) in fixtures(T, rng), method in methods

        x = x₀()
        state, result, opt = solve_for(method, x)
        @testset "$(nameof(typeof(method))) on $(name), $(T)" begin
            @test eltype(minimum(result)) == T
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

@testset "`Δf` spans one step, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(2)
    for (name, methods, x₀) in fixtures(T, rng), method in methods

        x = x₀()
        state, result = solve_for(method, x)
        entries = trace(result)
        @testset "$(nameof(typeof(method))) on $(name), $(T)" begin
            @test eltype(status(result).Δf) == T
            @test length(entries) ≥ 2
            @test status(result).Δf == entries[end].f - entries[end - 1].f
            @test previous_value(state) == entries[end - 1].f
        end
    end
end

# The first status compares against the objective at the start, which `solve!` records in the state.
@testset "the first `Δf` is `f(x₁) - f(x₀)`, $T" for T in REAL_ELTYPES
    for method in vector_methods()
        x = start(T)
        f₀ = objective(x)
        state = OptimizerState(method, x)
        opt = Optimizer(x, objective; algorithm = method, max_iterations = 1,
            warn_iterations = 0)
        result = solve!(x, state, opt)
        @testset "$(nameof(typeof(method))), $(T)" begin
            @test eltype(status(result).Δf) == T
            @test status(result).Δf == objective(x) - f₀
            @test previous_value(state) == f₀
        end
    end
end

# `Newton`'s state moves its section to the iterate, as every other state does, so the cache reuses
# the gradient `solver_step!` refreshed instead of evaluating it again (issue A13). Rosenbrock, so
# that neither the gradient nor the step is zero inside the window.
@testset "`Newton` reuses the refreshed gradient, $T" for T in REAL_ELTYPES
    rosenbrock(x) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    x = T[-1.2, 1.0]
    state = OptimizerState(Newton(), x)
    opt = Optimizer(x, rosenbrock; algorithm = Newton())
    for k in 1:6
        increase_iteration_number!(state)
        @test latest_gradient_is_current(cache(opt), state, x) == (k > 1)
        solver_step!(x, state, opt)
        update!(state, opt, x, rosenbrock(x))
    end
    @test eltype(rosenbrock(x)) == T
    @test rosenbrock(x) isa T
    # the window ends far from the minimum, at 1.54 in both precisions, so the objective there is
    # not round-off
    @test rosenbrock(x) > √eps(T)
end

# `zero` of a point is not a tangent vector: issue #21. `zero_tangent` is the internal name for the
# zero horizontal lift, on the backend of the point.
@testset "`zero` of a manifold point is not its horizontal lift" begin
    for T in REAL_ELTYPES
        Y = rand(StiefelManifold{T}, 6, 3)
        @test !(zero(Y) isa AbstractLieAlgHorMatrix)
        @test zero_tangent(Y) isa StiefelLieAlgHorMatrix{T}
        @test iszero(zero_tangent(Y))
        @test get_backend(zero_tangent(Y).B) == get_backend(Y.A)
        @test _zero(Y) isa StiefelLieAlgHorMatrix{T}
        # `alloc_h` of a point is sized by the lift, 12 for `St(6, 3)`, not by the dense storage
        @test size(GeometricOptimizers.alloc_h(Y)) == (12, 12)

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
    for T in REAL_ELTYPES
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
    for T in REAL_ELTYPES,
        x₀ in (() -> T[1, 2, 3], () -> rand(StiefelManifold{T}, 5, 3))

        x = x₀()
        state = OptimizerState(GradientMethod(), x)
        opt = Optimizer(x, objective; algorithm = GradientMethod(), max_iterations = 3,
            warn_iterations = 1)
        result = @test_logs (:warn, r"iterations") match_mode=:any solve!(x, state, opt)
        @test minimum(result) isa T
    end
end

# `initialize_state!` resets a quasi-Newton state that a solve has written, so a reused state carries
# no secant pair and no `Q` into the next solve, and starts a `NewtonState` at the starting point.
@testset "`initialize_state!` resets a quasi-Newton state and starts a Newton state at x₀, $T" for T in REAL_ELTYPES
    for method in (BFGS(), DFP())
        x = start(T)
        state, _, opt = solve_for(method, x)
        @test eltype(state.Q) == T
        @test !isnan(previous_value(state))
        @test state.Q != one(state.Q)
        initialize_state!(state, opt, x, objective(x))
        @test value(state) == objective(x)
        @test isnan(previous_value(state))
        @test all(isnan, state.s)
        @test all(isnan, state.ḡ)
        @test state.Q == one(state.Q)
    end

    x = start(T)
    state = OptimizerState(Newton(), x)
    opt = Optimizer(x, objective; algorithm = Newton())
    initialize_state!(state, opt, x, objective(x))
    @test eltype(value(state)) == T
    @test solution(state) == x
    @test value(state) == objective(x)
    @test gradient(state) == gradient(opt)(x)
    @test GeometricOptimizers.section(state).Y == x
end

# A state used for a second `solve!` repeats the solve of a fresh state from the same start: the
# iteration count, the iterate, the section, the objective and the method's own memory start again.
@testset "a state used for a second solve repeats a fresh state's solve, $T" for T in REAL_ELTYPES
    for method in vector_methods()
        state, _, opt = solve_for(method, start(T))
        x = T[-0.6, 0.3, -1.4]
        result = solve!(x, state, opt)

        x_fresh = T[-0.6, 0.3, -1.4]
        state_fresh, result_fresh = solve_for(method, x_fresh)
        @testset "$(nameof(typeof(method))), $(T)" begin
            @test eltype(minimum(result)) == T
            @test iteration_number(state) == iteration_number(state_fresh)
            @test x == x_fresh
            @test minimum(result) == minimum(result_fresh)
            @test value(state) == value(state_fresh)
            @test trace(result) == trace(result_fresh)
        end
    end
end

# A first solve that overflows leaves Adam's second moment at `Inf`, which the zero weight of the
# next solve's first step does not clear (`0 ⋅ Inf` is `NaN`), so the reset has to. The start is
# scaled to the range of `T`: the square of the gradient `4x³` overflows where `16x⁶ > floatmax(T)`,
# while the objective `x⁴` stays finite.
@testset "a state that an overflowing solve left non-finite repeats a fresh state's solve, $T" for T in REAL_ELTYPES
    state, _, opt = solve_for(Adam(), T[1, 2, 3] .* (4 * floatmax(T)^(1 / T(6))))
    @test any(!isfinite, GeometricOptimizers.second_moment(state))
    x = start(T)
    result = solve!(x, state, opt)
    x_fresh = start(T)
    _, result_fresh = solve_for(Adam(), x_fresh)
    @test eltype(x) == T
    @test x == x_fresh
    @test trace(result) == trace(result_fresh)
end

# On a manifold the section of a state that starts elsewhere gets the frame of its new anchor. A
# state that starts where it was built keeps its frame, so a fresh state's solve draws no random
# number.
@testset "`initialize_state!` gives a manifold section the frame of its start, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(3)
    for method in stiefel_methods()
        Y = rand(rng, StiefelManifold{T}, 5, 3)
        state = OptimizerState(method, Y)
        opt = Optimizer(Y, objective; algorithm = method)
        λ = copy(section(state).λ)
        initialize_state!(state, opt, Y, objective(Y))
        @test section(state).λ == λ

        Z = rand(rng, StiefelManifold{T}, 5, 3)
        initialize_state!(state, opt, Z, objective(Z))
        @test section(state).Y == Z
        frame = Matrix(section(state))
        @test eltype(frame) == T
        # the round-off of the orthonormal completion of a `5 × 3` point, a few `eps(T)` per entry
        @test frame' * frame ≈ one(frame' * frame) atol=10eps(T)
    end
end

# The first iteration compares against `f(x₀)` and not against a sentinel, so a solve that starts
# at the minimiser reports no increase and converges in `x`. The start is the minimiser `0`, which
# is exact in every precision: the assertion needs that exactness.
@testset "a quasi-Newton solve from the minimiser does not read as an increase, $T" for T in REAL_ELTYPES
    shifted(x) = 1 + sum(abs2, x)
    for method in (BFGS(), DFP())
        x = zeros(T, 3)
        state = OptimizerState(method, x)
        opt = Optimizer(
            x, shifted; algorithm = method, max_iterations = 1, warn_iterations = 0)
        result = solve!(x, state, opt)
        @test eltype(x) == T
        @test !status(result).f_increased
        @test status(result).x_converged
    end
end

# `_alloc_q` keeps the array type of a plain vector, so a device vector gets a device `Q`.
@testset "a device vector gets a device `Q`" begin
    for T in REAL_ELTYPES
        @test OptimizerState(BFGS(), JLArray(rand(T, 5))).Q isa JLArray{T, 2}
    end
end
