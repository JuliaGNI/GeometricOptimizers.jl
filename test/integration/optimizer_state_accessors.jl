using GeometricOptimizers
using GeometricOptimizers: value, previous_value, solution, previous_solution,
                           gradient, previous_gradient
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

# `value` and `previous_value` are how a caller reads the objective a state holds, and this file is
# the only one under `test/` that calls either. The four first-order manifold states carry both
# fields and answer both; `NewtonState` carries both and answers both; `BFGSState` carries
# only the previous one and answers only `previous_value`.
#
# The fields are written directly here rather than through a solve, so that a failure names the
# accessor and not the optimizer that fed it. Nothing here runs a solve, so nothing here constrains
# what the solve loop leaves in a state when it returns; that seam is issue #108.
#
# The stored values are distinct numbers and nothing here does arithmetic on them, so their being
# exact in `Float32` hides nothing: an accessor that reads the wrong field returns another of them,
# and one that converts returns another type, which `===` catches.
@testset "the first-order manifold states report both objective values, $T" for T in REAL_ELTYPES
    Y = rand(Random.Xoshiro(1), StiefelManifold{T}, 5, 3)

    for state in (GradientState(Y), MomentumState(Y), AdamState(Y),
        ScalarMomentAdamState(Y))
        state.f = T(3.5)
        state.f̄ = T(7.25)

        @test eltype(value(state)) == T
        @test value(state) === T(3.5)
        @test previous_value(state) === T(7.25)
    end
end

@testset "`NewtonState` reports both objective values, $T" for T in REAL_ELTYPES
    state = NewtonState(T[1, 2])
    state.f = T(3.5)
    state.f̄ = T(7.25)

    @test eltype(value(state)) == T
    @test value(state) === T(3.5)
    @test previous_value(state) === T(7.25)
end

# `update!` shifts the barred fields and then writes the unbarred ones, so the unbarred fields are
# the current iterate's. `solution` and `gradient` read the unbarred ones, which is what `value`
# reads on the same object and what the same names read on every other state. See issue #106.
@testset "`NewtonState` reads the current iterate, not the previous one, $T" for T in REAL_ELTYPES
    state = NewtonState(zeros(T, 2))
    state.x .= T[1, 2]
    state.x̄ .= T[3, 4]
    state.g .= T[5, 6]
    state.ḡ .= T[7, 8]
    state.f = T(3.5)
    state.f̄ = T(7.25)

    # The first and third are the whole point: they read one iterate, and `value(state)` above reads
    # the same one. A wrong vector here is a wrong answer and not a missing method, so only an
    # equality assertion catches it — which is why these two are written out rather than assumed.
    @test eltype(solution(state)) == T
    @test solution(state) == T[1, 2]
    @test previous_solution(state) == T[3, 4]
    @test gradient(state) == T[5, 6]
    @test previous_gradient(state) == T[7, 8]
end

# `BFGSState` holds a pair, as every other state does. `DFPState` is an alias for it, so both
# methods are the same method.
@testset "`BFGSState` reports both objective values and both iterates, $T" for T in REAL_ELTYPES
    for state in (BFGSState(zeros(T, 2)), DFPState(zeros(T, 2)))
        state.x .= T[1, 2]
        state.x̄ .= T[3, 4]
        state.f = T(3.5)
        state.f̄ = T(7.25)

        @test value(state) === T(3.5)
        @test previous_value(state) === T(7.25)
        @test solution(state) == T[1, 2]
        @test previous_solution(state) == T[3, 4]
        @test eltype(solution(state)) == T
        # no current gradient to report: the quasi-Newton cache holds it
        @test_throws MethodError gradient(state)
    end
end
