# `trial_slope` under `Cayley` takes the retraction's differential in the optimizer's workspace
# (issue #73), so what it allocates away from `α = 0` is what it allocates at `α = 0`: the gradient
# functor and `global_rep`, which are the same at every `α`, and nothing for the differential. The
# slope itself is the one the allocating differential gives, to the bit.

using GeometricOptimizers
using GeometricOptimizers: trial_slope, trial_iterate!, retraction_workspace, initialize!,
                           cache,
                           gradient, update!, direction, section, solution, global_rep,
                           _dot,
                           _similar, _direction_rule
using Test
import Random

include("../helpers/reference_retractions.jl")

manifold(::Val{:Stiefel}) = StiefelManifold
manifold(::Val{:Grassmann}) = GrassmannManifold

# a BFGS optimizer on the point, its cache warmed through one `update!`
function slope_fixture(lift, ::Type{T}) where {T}
    Random.seed!(1234)
    Y = rand(manifold(Val(lift)){T}, 6, 3)
    F(Z) = sum(abs2, Z .- T(0.3)) + sum(sin.(Z))
    opt = Optimizer(Y, F; algorithm = BFGS(), retraction = Cayley())
    state = OptimizerState(BFGS(), Y)
    c = cache(opt)
    initialize!(c, Y)
    update!(c, state, gradient(opt), _direction_rule(opt), Y)
    (opt = opt, cache = c, params = (x = Y, state = state),
        differential = _similar(direction(c)), workspace = retraction_workspace(opt))
end

function slope_at(f, α)
    trial_iterate!(f.cache, f.params, α, Cayley(), f.workspace)
    trial_slope(gradient(f.opt), f.cache, Cayley(), α, f.differential, f.workspace)
end

function _measured_slope(gradient_instance, cache, retraction, α, differential, workspace)
    trial_slope(gradient_instance, cache, retraction, α, differential, workspace)
    @allocated trial_slope(gradient_instance, cache, retraction, α, differential, workspace)
end

function measured_slope(f, α)
    _measured_slope(gradient(f.opt), f.cache, Cayley(), α, f.differential,
        f.workspace)
end

@testset "trial_slope allocates as much at α = 0.5 as at α = 0, $lift, $T" for lift in (:Stiefel, :Grassmann),
    T in (Float32, Float64)

    f = slope_fixture(lift, T)
    trial_iterate!(f.cache, f.params, T(0.5), Cayley(), f.workspace)
    at_half = measured_slope(f, T(0.5))
    trial_iterate!(f.cache, f.params, zero(T), Cayley(), f.workspace)
    at_zero = measured_slope(f, zero(T))
    @test at_half == at_zero
    @test (@inferred trial_slope(
        gradient(f.opt), f.cache, Cayley(), T(0.5), f.differential,
        f.workspace)) isa T
end

@testset "trial_slope is the slope of the allocating differential, $lift, $T" for lift in (:Stiefel,
        :Grassmann),
    T in (Float32, Float64)

    f = slope_fixture(lift, T)
    for α in T.((0, 0.5, 2))
        slope = slope_at(f, α)
        g = global_rep(section(f.cache), gradient(f.opt)(solution(f.cache)))
        @test slope isa T
        @test slope == _dot(g, reference_cayley_differential(direction(f.cache), α))
    end
end
