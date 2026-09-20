using GeometricOptimizers
using GeometricOptimizers: value, previous_value
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
        OptimizerState(ScalarMomentAdam(), Y))
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

# `BFGSState` holds one iterate and one objective, not a pair: `update!` writes `x̄` and `f̄` at the
# end of the iteration, and the next iteration reads them as the previous ones. `DFPState` is an
# alias for it, so both methods are the same method.
@testset "`BFGSState` reports the one objective it holds" begin
    state = BFGSState([1.0, 2.0])
    state.f̄ = 7.25

    @test previous_value(state) === 7.25
    @test previous_value(DFPState([1.0, 2.0])) isa Float64

    # and there is no current objective to report, because the solve loop owns it
    @test !applicable(value, state)
end
