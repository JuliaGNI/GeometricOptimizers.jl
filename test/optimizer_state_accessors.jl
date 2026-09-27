using GeometricOptimizers
using GeometricOptimizers: value, previous_value, solution, previous_solution,
                           gradient, previous_gradient
using Test
import Random

Random.seed!(1234)

# `value` and `previous_value` are how a caller reads the objective a state holds, and this file is
# the only one under `test/` that calls either. The four first-order manifold states carry both
# fields and answer both; `NewtonState` carries both and answers both; `BFGSState` carries
# only the previous one and answers only `previous_value`.
#
# The fields are written directly here rather than through a solve, so that a failure names the
# accessor and not the optimizer that fed it. Nothing here runs a solve, so nothing here constrains
# what the solve loop leaves in a state when it returns; that seam is issue #108.

@testset "the first-order manifold states report both objective values" begin
    Y = rand(StiefelManifold{Float64}, 5, 3)

    for state in (GradientState(Y), MomentumState(Y), AdamState(Y),
        ScalarMomentAdamState(Y))
        state.f = 3.5
        state.f̄ = 7.25

        @test value(state) === 3.5
        @test previous_value(state) === 7.25
    end
end

@testset "`NewtonState` reports both objective values" begin
    state = NewtonState([1.0, 2.0])
    state.f = 3.5
    state.f̄ = 7.25

    @test value(state) === 3.5
    @test previous_value(state) === 7.25
end

# `update!` shifts the barred fields and then writes the unbarred ones, so the unbarred fields are
# the current iterate's. `solution` and `gradient` read the unbarred ones, which is what `value`
# reads on the same object and what the same names read on every other state. See issue #106.
@testset "`NewtonState` reads the current iterate, not the previous one" begin
    state = NewtonState([0.0, 0.0])
    state.x .= [1.0, 2.0]
    state.x̄ .= [3.0, 4.0]
    state.g .= [5.0, 6.0]
    state.ḡ .= [7.0, 8.0]
    state.f = 3.5
    state.f̄ = 7.25

    # The first and third are the whole point: they read one iterate, and `value(state)` above reads
    # the same one. A wrong vector here is a wrong answer and not a missing method, so only an
    # equality assertion catches it — which is why these two are written out rather than assumed.
    @test solution(state) == [1.0, 2.0]
    @test previous_solution(state) == [3.0, 4.0]
    @test gradient(state) == [5.0, 6.0]
    @test previous_gradient(state) == [7.0, 8.0]
end

# `BFGSState` holds a pair, as every other state does. `DFPState` is an alias for it, so both
# methods are the same method.
@testset "`BFGSState` reports both objective values and both iterates" begin
    for T in (Float32, Float64), state in (BFGSState(zeros(T, 2)), DFPState(zeros(T, 2)))

        state.x .= T[1, 2]
        state.x̄ .= T[3, 4]
        state.f = T(3.5)
        state.f̄ = T(7.25)

        @test value(state) === T(3.5)
        @test previous_value(state) === T(7.25)
        @test solution(state) == T[1, 2]
        @test previous_solution(state) == T[3, 4]
        @test eltype(solution(state)) == T
    end
end
