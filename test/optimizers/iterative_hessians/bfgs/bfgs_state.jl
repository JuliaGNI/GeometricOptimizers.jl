# The state update of `BFGS` transports its section in the optimizer's `RetractionWorkspace`, as every
# other `update_section!` on the step path does. Without one the retraction is taken in fresh
# `N × N` arrays once per iteration of every quasi-Newton solve on a manifold, so the bytes of one
# `update!` grow with `N`; with one they do not. And the section is the one the transport without a
# workspace writes: to the bit at `N = 40`, and to the rounding of one addition at `N = 400`, where the
# two sum their products differently (see the second testset).

using GeometricOptimizers
using GeometricOptimizers: update!, solver_step!, increase_iteration_number!,
                           initialize_state!,
                           problem, value, section, update_section!, GlobalSection
using Test
import Random

manifold(::Val{:Stiefel}) = StiefelManifold
manifold(::Val{:Grassmann}) = GrassmannManifold

# an optimizer and a state one iteration into a solve, at the point `x` that iteration produced
function stepped(lift, ::Type{T}, N, retraction) where {T}
    Random.seed!(1234)
    x = rand(Random.Xoshiro(N), manifold(Val(lift)){T}, N, 3)
    F(Z) = sum(abs2, Z .- T(0.3)) + sum(sin.(Z))
    opt = Optimizer(x, F; algorithm = BFGS(), retraction = retraction)
    state = OptimizerState(BFGS(), x)
    initialize_state!(state)
    increase_iteration_number!(state)
    solver_step!(x, state, opt)
    (opt = opt, state = state, x = x, f = value(problem(opt), x))
end

function _measured_update!(state, opt, x, f)
    (update!(state, opt, x, f); @allocated update!(state, opt, x, f))
end

@testset "one BFGS state update allocates the same at N = 40 and N = 400, $lift, $T" for lift in (:Stiefel,
        :Grassmann),
    T in (Float32, Float64)

    for retraction in (Cayley(), Geodesic())
        small, large = stepped(lift, T, 40, retraction), stepped(lift, T, 400, retraction)
        @test _measured_update!(small.state, small.opt, small.x, small.f) ==
              _measured_update!(large.state, large.opt, large.x, large.f)
    end
end

@testset "the BFGS state update writes the section the transport without a workspace writes, $lift, $T" for lift in (:Stiefel,
        :Grassmann),
    T in (Float32, Float64)

    for retraction in (Cayley(), Geodesic()), N in (40, 400)

        s = stepped(lift, T, N, retraction)
        before = deepcopy(section(s.state))
        update!(s.state, s.opt, s.x, s.f)

        expected = deepcopy(before)
        update_section!(expected, s.state.s, retraction, nothing)
        @test eltype(section(s.state).λ) == T
        if N == 40
            @test section(s.state).λ == expected.λ
        else
            # The workspace adds the second product into the first with a five-argument `mul!`;
            # without one the two products are summed after. In `Float64` OpenBLAS splits the inner
            # dimension `N - n = 397` into blocks, and the two forms then round differently, by one
            # addition per block and entry: `100eps(T)` relative. Below about `N = 300` they agree to
            # the bit, as at `N = 40` above.
            @test isapprox(section(s.state).λ, expected.λ; rtol = 100eps(T))
        end
    end
end
