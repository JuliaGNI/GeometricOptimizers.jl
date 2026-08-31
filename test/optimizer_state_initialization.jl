using GeometricOptimizers
using GeometricOptimizers: AdamState, MomentumState, GradientState
using GeometricOptimizers: first_moment, second_moment, _second_moment, momentum
using GeometricOptimizers: cache, gradient_array, increase_iteration_number!, solver_step!,
                           update!, _square, _mul
using SimpleSolvers: Static
using LinearAlgebra: norm
using Test
import Random

Random.seed!(1234)

const A = randn(5, 3)

manifold_error(x::StiefelManifold) = norm(A - x * x' * A)
named_tuple_error(ps::NetworkParameters) = norm(A - ps.w * ps.w' * A) + norm(ps.b)

# both a bare `Manifold` and a whole set of parameters are tested
function problems()
    ((rand(StiefelManifold, 5, 3), manifold_error),
        (NetworkParameters((w = rand(StiefelManifold, 5, 3), b = randn(3))),
            named_tuple_error))
end

_all_zero(a::AbstractArray) = all(iszero, a)
_all_zero(a::NetworkParameters) = all(_all_zero, values(a))

_isapprox(a::AbstractArray, b::AbstractArray) = isapprox(a, b)
function _isapprox(a::NetworkParameters, b::NetworkParameters)
    all(_isapprox(a[k], b[k]) for k in keys(a))
end

# The moments of an `AdamState` and the momentum of a `MomentumState` are read in the first
# call to `update!(::OptimizerCache, ...)`, i.e. before they are written to for the first
# time. They therefore have to be initialized with zeros; initializing them with `_similar`
# makes the first optimizer step depend on uninitialized memory.
@testset "the optimizer states are initialized with zeros" begin
    for (x, _) in problems()
        state = AdamState(x)
        @test _all_zero(first_moment(state))
        @test _all_zero(second_moment(state))
        @test _all_zero(_second_moment(state))

        @test _all_zero(momentum(MomentumState(x)))
    end
end

# The moments are stored in bias-corrected form, i.e.
#   m₁ ← ((β₁ - β₁ᵗ)/(1 - β₁ᵗ))⋅m₁ + ((1 - β₁)/(1 - β₁ᵗ))⋅∇L,
#   m₂ ← ((β₂ - β₂ᵗ)/(1 - β₂ᵗ))⋅m₂ + ((1 - β₂)/(1 - β₂ᵗ))⋅∇L⊙∇L,
# so for `t = 1` the first moment is the gradient and the second moment is its square. Note
# that this also checks that the square root that goes into the direction
# `-m₁/(√m₂ + δ)` is not applied to `m₂` itself.
#
# `increase_iteration_number!` has to be called before the step, exactly as `solve!` does it —
# this testset used to be the only loop in the suite that left it out, which is how the
# off-by-one in the bias correction (`_t = t + 1`, so `t = 2` in the first step) survived: it
# is the one call sequence in which `t + 1` gives the right answer. `test/optimizer_step_formulas.jl`
# pins the resulting step size.
@testset "the first Adam step" begin
    x = NetworkParameters((w = rand(StiefelManifold, 5, 3), b = randn(3)))
    algorithm = Adam()
    optimizer = Optimizer(x, named_tuple_error; algorithm = algorithm, linesearch = Static(0.01))
    state = AdamState(x)

    ps = deepcopy(x)
    increase_iteration_number!(state)
    solver_step!(ps, state, optimizer)
    update!(state, optimizer, ps)

    g = gradient_array(cache(optimizer))
    @test _isapprox(first_moment(state), g)
    @test _isapprox(second_moment(state), _square(g))
end

# A regression test for the uninitialized moments: those made the optimizers depend on
# whatever happened to be in memory. Note that the seed has to be fixed for every run: the
# [`GlobalSection`](@ref) is drawn at random and `Adam` is not equivariant with respect to a
# change of section (its moments are updated element-wise).
@testset "the same seed gives the same result" begin
    for algorithm in (GradientMethod(), MomentumMethod(0.5), Adam(), AdamWithEuclideanDecay())
        x = NetworkParameters((w = rand(StiefelManifold, 5, 3), b = randn(3)))
        results = map(1:2) do _
            Random.seed!(1234)
            ps = deepcopy(x)
            optimizer = Optimizer(ps, named_tuple_error; algorithm = algorithm, linesearch = Static(0.01))
            state = OptimizerState(algorithm, ps)
            for _ in 1:5
                increase_iteration_number!(state)
                solver_step!(ps, state, optimizer)
                update!(state, optimizer, ps)
            end
            named_tuple_error(ps)
        end
        @test results[1] == results[2]
    end
end
