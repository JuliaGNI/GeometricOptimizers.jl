using GeometricOptimizers
using GeometricOptimizers: value, previous_value, solution, previous_solution,
                           gradient, previous_gradient
using GeometricOptimizers: AdamState, GradientState, MomentumState, ScalarMomentAdamState
using Test
import Random

Random.seed!(1234)

# `value` and `previous_value` are how a caller reads the objective a state holds, and nothing in
# `test/` called either before this file. The four first-order manifold states carry both fields and
# answer both; `NewtonOptimizerState` carries both and answered neither; `BFGSState` carries only the
# previous one.
#
# The fields are written directly here rather than through a solve, so that a failure names the
# accessor and not the optimizer that fed it.

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

@testset "`NewtonOptimizerState` reports both objective values" begin
    state = NewtonOptimizerState([1.0, 2.0])
    state.f = 3.5
    state.f̄ = 7.25

    @test value(state) === 3.5
    @test previous_value(state) === 7.25
end

# `update!` shifts the barred fields and then writes the unbarred ones, so the unbarred fields are
# the current iterate's. `solution` and `gradient` read the barred ones, which made them disagree
# with `value` on the same object and with the same names on every other state. See issue #106.
@testset "`NewtonOptimizerState` reads the current iterate, not the previous one" begin
    state = NewtonOptimizerState([0.0, 0.0])
    state.x .= [1.0, 2.0]
    state.x̄ .= [3.0, 4.0]
    state.g .= [5.0, 6.0]
    state.ḡ .= [7.0, 8.0]
    state.f = 3.5
    state.f̄ = 7.25

    # The first and third are the whole point: they read one iterate, and `value(state)` above reads
    # the same one. On the pre-change source they returned the *previous* iterate's vectors — a
    # wrong answer rather than a missing method, which is why nothing caught them.
    @test solution(state) == [1.0, 2.0]
    @test previous_solution(state) == [3.0, 4.0]
    @test gradient(state) == [5.0, 6.0]
    @test previous_gradient(state) == [7.0, 8.0]
end

# `BFGSState` holds one iterate and one objective, not a pair: `update!` writes `x̄` and `f̄` at the
# end of the iteration, and the next iteration reads them as the previous ones. `DFPState` is an
# alias for it, so both methods are the same method.
@testset "`BFGSState` reports the one objective it holds" begin
    state = BFGSState([1.0, 2.0])
    state.f̄ = 7.25

    @test previous_value(state) === 7.25

    dfp = DFPState([1.0, 2.0])
    dfp.f̄ = 1.5
    @test previous_value(dfp) === 1.5

    # And there is no current objective to report, because the solve loop owns it. The method table
    # and not `applicable`: a narrower `value(::BFGSState{Float32})` leaves `!applicable` passing on
    # this `Float64` state, so it would not catch the method coming back.
    @test isempty(methods(value, Tuple{BFGSState}))
end
