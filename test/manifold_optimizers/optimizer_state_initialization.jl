using GeometricOptimizers
using GeometricOptimizers: AdamState, MomentumState, GradientState
using GeometricOptimizers: first_moment, second_moment, _second_moment, momentum
using GeometricOptimizers: cache, gradient_array, increase_iteration_number!, solver_step!,
                           update!, _square, _mul
using SimpleSolvers: Static
using LinearAlgebra: norm
using Test
import Random

include("../helpers/eltypes.jl")

# The `GlobalSection` of a state is drawn from the global generator, so it is seeded too.
Random.seed!(1234)

manifold_error(A) = x -> norm(A - x * x' * A)
named_tuple_error(A) = ps -> norm(A - ps.w * ps.w' * A) + norm(ps.b)

# both a bare `Manifold` and a whole set of parameters are tested
function problems(rng, ::Type{T}) where {T}
    A = randn(rng, T, 5, 3)
    ((rand(rng, StiefelManifold{T}, 5, 3), manifold_error(A)),
        (
            NetworkParameters((
                w = rand(rng, StiefelManifold{T}, 5, 3), b = randn(rng, T, 3))),
            named_tuple_error(A)))
end

_all_zero(a::AbstractArray) = all(iszero, a)
_all_zero(a::NetworkParameters) = all(_all_zero, values(a))

_isapprox(a::AbstractArray, b::AbstractArray) = isapprox(a, b)
function _isapprox(a::NetworkParameters, b::NetworkParameters)
    all(_isapprox(a[k], b[k]) for k in keys(a))
end

# the element type of every entry, which is one type where the state was allocated in that of `x`
_eltypes(a::AbstractArray) = (eltype(a),)
_eltypes(a::NetworkParameters) = Tuple(unique(eltype(v) for v in values(a)))

# The moments of an `AdamState` and the momentum of a `MomentumState` are read in the first
# call to `update!(::OptimizerCache, ...)`, i.e. before they are written to for the first
# time. They therefore have to be initialized with zeros; initializing them with `_similar`
# makes the first optimizer step depend on uninitialized memory.
@testset "the optimizer states are initialized with zeros, $T" for T in REAL_ELTYPES
    for (x, _) in problems(Random.Xoshiro(1), T)
        state = AdamState(x)
        @test _eltypes(first_moment(state)) == (T,)
        @test _all_zero(first_moment(state))
        @test _all_zero(second_moment(state))
        @test _all_zero(_second_moment(state))

        @test _eltypes(momentum(MomentumState(x))) == (T,)
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
# `increase_iteration_number!` has to be called before the step, exactly as `solve!` does it. A
# loop that leaves it out is the one call sequence in which an off-by-one in the bias correction
# (`_t = t + 1`, so `t = 2` in the first step) gives the right answer.
# `test/manifold_optimizers/optimizer_step_formulas.jl` pins the resulting step size.
@testset "the first Adam step, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(2)
    A = randn(rng, T, 5, 3)
    x = NetworkParameters((w = rand(rng, StiefelManifold{T}, 5, 3), b = randn(rng, T, 3)))
    algorithm = Adam()
    optimizer = Optimizer(x, named_tuple_error(A); algorithm = algorithm,
        linesearch = Static(T(0.01)))
    state = AdamState(x)

    ps = deepcopy(x)
    increase_iteration_number!(state)
    solver_step!(ps, state, optimizer)
    update!(state, optimizer, ps)

    g = gradient_array(cache(optimizer))
    @test _eltypes(first_moment(state)) == (T,)
    @test _eltypes(second_moment(state)) == (T,)
    @test _isapprox(first_moment(state), g)
    @test _isapprox(second_moment(state), _square(g))
end

# A regression test for the uninitialized moments: those made the optimizers depend on
# whatever happened to be in memory. Note that the seed has to be fixed for every run: the
# [`GlobalSection`](@ref) is drawn at random and `Adam` is not equivariant with respect to a
# change of section (its moments are updated element-wise).
@testset "the same seed gives the same result, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(3)
    A = randn(rng, T, 5, 3)
    f = named_tuple_error(A)
    for algorithm in (GradientMethod(), MomentumMethod(; α = 0.5), Adam(), AdamWithEuclideanDecay())
        x = NetworkParameters((
            w = rand(rng, StiefelManifold{T}, 5, 3), b = randn(rng, T, 3)))
        results = map(1:2) do _
            Random.seed!(1234)
            ps = deepcopy(x)
            optimizer = Optimizer(ps, f; algorithm = algorithm, linesearch = Static(T(0.01)))
            state = OptimizerState(algorithm, ps)
            for _ in 1:5
                increase_iteration_number!(state)
                solver_step!(ps, state, optimizer)
                update!(state, optimizer, ps)
            end
            f(ps)
        end
        @test eltype(results[1]) == T
        @test results[1] == results[2]
    end
end
