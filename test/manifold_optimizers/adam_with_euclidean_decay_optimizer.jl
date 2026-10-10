using GeometricOptimizers
using GeometricOptimizers: AdamState, Manifold, first_moment, second_moment, check,
                           increase_iteration_number!, solver_step!, update!, _square,
                           _weight_decay!, _is_decayable
using SimpleSolvers: Static
using LinearAlgebra: I, norm, tr, Symmetric
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/manifold_tolerance.jl")

# `AdamWithEuclideanDecay` is [`Adam`](@ref) plus *decoupled* weight decay: the direction is
# `-m₁/(√m₂ + δ) - λx` instead of `-m₁/(√m₂ + δ)`, and `λx` never enters the moments. This
# file pins the three things that distinguishes it from `Adam` and from `Adam` on an
# `L²`-penalized objective:
#
#   1. the decay is applied to the direction and scaled by the learning rate,
#   2. the moments are the ones of `Adam` — the decay does not touch them,
#   3. it does nothing to a weight that lives on a manifold.
#
# The objective is *linear*, so its gradient is the constant `C` at every iterate. For a
# constant gradient the bias-corrected moments are exactly `m₁ = C` and `m₂ = C ⊙ C` at every
# iteration (the two recursion factors are a convex combination), so the `Adam` part of the
# direction is exactly `-C/(|C| + δ)`, i.e. `-sign(C)` up to `δ = 1e-8`, and the whole recursion
# has a closed form. The entries of `C` and of the start are not dyadic, so that they round in
# `Float32`.
coefficients(::Type{T}) where {T} = T[1.3, -2.7, 0.45]
start(::Type{T}) where {T} = T[1.1, -1.7, 0.6]
linear_objective(C) = x -> sum(C .* x)

# the learning rate, i.e. the `α` of the `Static` line search — see `default_linesearch`
const η = 0.01

# deliberately much larger than `DEFAULT_WEIGHT_DECAY`, so that a step that forgot the decay is
# not within tolerance of one that applied it
const λ = 0.5

# the mixed problem: a `NamedTuple` that holds a manifold weight and an ordinary one, i.e. the
# case that tells `AdamWithEuclideanDecay` apart from `Adam`. `‖b‖` is in the objective so that
# `b` has a gradient of its own to be decayed against.
Random.seed!(1234)
named_tuple_error(A) = ps -> norm(A - ps.w * ps.w' * A) + norm(ps.b)
stiefel_error(A) = Y -> norm(A - Y * Y' * A)

# a bare `Manifold`, a whole set of parameters and an ordinary `Vector` — the three kinds of
# parameters the unified interface accepts
function problems(rng, ::Type{T}) where {T}
    A = randn(rng, T, 5, 3)
    ((rand(rng, StiefelManifold{T}, 5, 3), stiefel_error(A)),
        (
            NetworkParameters((
                w = rand(rng, StiefelManifold{T}, 5, 3), b = randn(rng, T, 3))),
            named_tuple_error(A)),
        (start(T), linear_objective(coefficients(T))))
end

_isequal(a::AbstractArray, b::AbstractArray) = a == b
function _isequal(a::NetworkParameters, b::NetworkParameters)
    all(_isequal(a[k], b[k]) for k in keys(a))
end

"""
    run!(ps, algorithm, f, steps)

Take `steps` optimizer steps on `f`, exactly as `solve!` does it but without its stopping
criteria, and return the parameters.
"""
function run!(ps, algorithm, f, steps; α = η)
    Random.seed!(1234)
    optimizer = Optimizer(ps, f; algorithm = algorithm, linesearch = Static(α))
    state = OptimizerState(algorithm, ps)
    for _ in 1:steps
        increase_iteration_number!(state)
        solver_step!(ps, state, optimizer)
        update!(state, optimizer, ps)
    end
    ps
end

# `x ← x - η⋅s - ηλ⋅x`, with `s = C/(|C| + δ)`, is an affine recursion, hence
#
#     xₖ = (1 - ηλ)ᵏx₀ - (s/λ)(1 - (1 - ηλ)ᵏ),
#
# which is the whole method in one line: the first term is the decay, the second is `Adam`, and
# the fixed point `-s/λ ≈ -sign(C)/λ` is where the two balance. Getting the decay coupled into the
# gradient instead (`m₁ ← ... + λx`, i.e. `L²` regularization) fails this: the second moment
# then rescales the penalty and the trajectory is a different one.
@testset "the decayed Adam recursion, in closed form, $T" for T in REAL_ELTYPES
    steps = 20
    C, x₀ = coefficients(T), start(T)
    x = run!(copy(x₀), AdamWithEuclideanDecay(; λ = T(λ)), linear_objective(C), steps;
        α = T(η))

    δ = Adam().δ
    s = C ./ (abs.(C) .+ T(δ))
    decayed = (1 - T(η) * T(λ))^steps
    @test eltype(x) == T
    # round-off of `steps` steps of a few operations each: measured at most 0.25 `eps(T)` per step,
    # relative, in both precisions over 20 draws of `C` and `x₀`
    @test x ≈ decayed * x₀ - s / T(λ) * (1 - decayed) rtol = steps * eps(T)

    # the fixed point is approached, not overshot, and it is `Adam`'s step size divided by `λ`
    @test all(abs.(x) .< 1 / T(λ))
end

# With no gradient at all nothing is left of the step but the decay, so the weights shrink by
# `1 - ηλ` per iteration. This is the property that gives weight decay its name and the one
# that a coupled implementation would get right as well — it is here to pin the *factor*.
@testset "a vanishing gradient leaves the decay by itself, $T" for T in REAL_ELTYPES
    steps = 15
    x₀ = start(T)
    x = run!(copy(x₀), AdamWithEuclideanDecay(; λ = T(λ)), x -> 0 * sum(x), steps;
        α = T(η))

    @test eltype(x) == T
    # round-off of `steps` products: measured at most 0.16 `eps(T)` per step, relative, in both
    # precisions over 20 draws of `x₀`
    @test x ≈ (1 - T(η) * T(λ))^steps * x₀ rtol = steps * eps(T)
end

# The decoupling itself: after a step the moments have to be those of the *unpenalized*
# gradient. If `λx` had been added to the gradient — which is what `Adam` on
# `objective(x) + λ/2‖x‖²` does — then `m₁` would be `C + λx₀` here.
@testset "the weight decay stays out of the moments, $T" for T in REAL_ELTYPES
    C = coefficients(T)
    x = start(T)
    algorithm = AdamWithEuclideanDecay(; λ = T(λ))
    optimizer = Optimizer(x, linear_objective(C); algorithm = algorithm,
        linesearch = Static(T(η)))
    state = AdamState(x)

    increase_iteration_number!(state)
    solver_step!(x, state, optimizer)
    update!(state, optimizer, x)

    @test eltype(first_moment(state)) == T
    @test first_moment(state) ≈ C
    @test second_moment(state) ≈ _square(C)
end

# `λ = 0` is the one setting in which `AdamWithEuclideanDecay` and `Adam` are the same method for
# *every* kind of parameter, so it is the cheapest check that nothing else was changed along the
# way.
#
# The comparison is exact rather than approximate, and structurally so rather than by luck: on an
# ordinary array the decay is `δ .-= 0.0 .* x`, and `d - 0.0` is `d` bit for bit in IEEE-754 for
# every finite `d` (`-0.0` is the one value it can flip, and `==` does not distinguish the two
# zeros); on a manifold entry `_weight_decay!` is the no-op. So the two runs execute the same
# arithmetic in the same order and there is no rounding to absorb. Asserting `≈` here would let a
# decay that leaked in at the size of the tolerance pass.
@testset "λ = 0 is Adam, $T" for T in REAL_ELTYPES
    for (ps, f) in problems(Random.Xoshiro(1), T)
        adam = run!(deepcopy(ps), Adam(), f, 10; α = T(η))
        adamw = run!(deepcopy(ps), AdamWithEuclideanDecay(; λ = zero(T)), f, 10; α = T(η))

        @test eltype(f(adamw)) == T
        @test _isequal(adam, adamw)
        @test f(adam) == f(adamw)
    end
end

# The justification for the no-op below, checked rather than taken on trust: weight decay is
# the gradient of `λ/2‖x‖²`, and that function is constant on both manifolds of this package
# (`‖Y‖_F² = tr(YᵀY) = n`), so its Riemannian gradient vanishes identically.
@testset "the Riemannian gradient of the weight-decay penalty vanishes, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(2)
    for Y in (rand(rng, StiefelManifold{T}, 6, 3), rand(rng, GrassmannManifold{T}, 6, 3))
        G = rgrad(Y, T(λ) * Y.A)
        @test eltype(G) == T
        # `λY` less its tangent projection is `λ` times the round-off of `YᵀY = I`: measured at
        # most 3.2 `λ⋅eps(T)` in both precisions over 20 draws
        @test norm(G) < 8 * T(λ) * eps(T)
    end
end

# Hence the method *is* `Adam` on a bare manifold — for any `λ`, and not just for a small one.
# `_weight_decay!` is a no-op on the pair (`AbstractLieAlgHorMatrix` direction, `Manifold`
# weight) that a manifold produces; the alternative, decaying the horizontal representation of
# the update, would shrink the optimizer's own step rather than the weight and is not what
# decoupled weight decay means. See issue #28 and `docs/src/weight_decay.md`. Because the run is
# then `Adam` under another name, it is also the case that has to be *said* — the warning is
# asserted here rather than merely tolerated.
@testset "weight decay does nothing to a manifold weight, $T" for T in REAL_ELTYPES
    target = T[0.0, 0.0, 1.2]
    f(Y::StiefelManifold) = norm(vec(Y) - target)
    x₀ = StiefelManifold(T[0.0; sqrt(T(0.5)); sqrt(T(0.5));;])

    adam = run!(deepcopy(x₀), Adam(), f, 25; α = T(0.1))
    adamw = @test_logs (:warn, r"none of the parameters") match_mode = :any run!(
        deepcopy(x₀), AdamWithEuclideanDecay(; λ = T(λ)), f, 25; α = T(0.1))

    @test eltype(adamw) == T
    @test adamw isa StiefelManifold{T}
    @test adamw.A == adam.A                     # bit for bit, not just to a tolerance
    # and it is still on the manifold, to the round-off of 25 retractions of a `3 × 1` point
    @test check(adamw) < 100 * eps(T)
    @test f(adamw) < f(x₀)                      # and it optimized
end

# The same run on a `GrassmannManifold`. `test/integration/grassmann_optimizer_tests.jl` covers the
# solve itself. What this testset adds is that the *weight-decay* claim above holds on the second
# manifold and not only on the first, which is what the `rgrad` identity two testsets up predicts.
#
# The objective is the Rayleigh quotient rather than a distance to a target point: on the Grassmann
# manifold `Y` and `YO` are the same point, and a distance is not a function of it.
@testset "weight decay does nothing to a Grassmann weight either, $T" for T in REAL_ELTYPES
    M = Symmetric(T[3.0 0.5 0.0; 0.5 2.0 0.1; 0.0 0.1 1.0])
    f(Y::GrassmannManifold) = -tr(Y' * M * Y)
    x₀ = rand(Random.Xoshiro(3), GrassmannManifold{T}, 3, 1)

    adam = run!(deepcopy(x₀), Adam(), f, 25; α = T(0.1))
    adamw = @test_logs (:warn, r"none of the parameters") match_mode = :any run!(
        deepcopy(x₀), AdamWithEuclideanDecay(; λ = T(λ)), f, 25; α = T(0.1))

    @test eltype(adamw) == T
    @test adamw isa GrassmannManifold{T}
    @test adamw.A == adam.A                     # bit for bit, as on the Stiefel manifold
    @test check(adamw) < 100 * eps(T)           # the round-off of 25 retractions, as above
    @test f(adamw) < f(x₀)
end

# `_weight_decay!` directly, rather than only through a 25-step run. The no-op is not an omission:
# the direction on a manifold is a horizontal lift, of a different shape from the point, so `λx`
# could not be subtracted from it even if it were nonzero.
@testset "the no-op of `_weight_decay!` is a no-op, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(4)
    for (Y, B) in (
        (rand(rng, StiefelManifold{T}, 6, 3), rand(rng, StiefelLieAlgHorMatrix{T}, 6, 3)),
        (rand(rng, GrassmannManifold{T}, 6, 3),
        rand(rng, GrassmannLieAlgHorMatrix{T}, 6, 3)))
        B₀ = copy(B)
        @test _weight_decay!(B, Y, T(λ)) === B      # returned in place, as the array method is
        @test B == B₀                               # and untouched
        @test size(B) != size(Y)                    # the shapes it would have to reconcile
    end

    # the ordinary method for comparison, on the same call shape: `δ - λx`, against the same
    # difference in `BigFloat`
    δ, x = T[1.1, 0.9, 1.3], T[2.2, 4.1, 6.3]
    δ_decayed = _weight_decay!(copy(δ), x, T(λ))
    @test eltype(δ_decayed) == T
    @test δ_decayed ≈ T.(big.(δ) .- big(T(λ)) .* big.(x))
end

# The trait carries the geometry, and it is declared on the two concrete manifolds rather than on
# `Manifold`: what makes the decay vanish is that both are *compact*, which a manifold added later
# need not be. It therefore has to answer for itself instead of inheriting a no-op that may be
# wrong for it — the failure mode this guards is a silent one.
struct NoncompactTestManifold{T} <: Manifold{T}
    A::Matrix{T}
end

@testset "a manifold that has not decided is an error, not a no-op" begin
    Y = NoncompactTestManifold(randn(3, 2))

    @test_throws ErrorException _is_decayable(Y)
    message = try
        _is_decayable(Y)
    catch e
        sprint(showerror, e)
    end
    @test occursin("NoncompactTestManifold", message)
    @test occursin("compact", message)

    # and the two that have decided are unaffected by the absence of a fallback
    @test !_is_decayable(rand(StiefelManifold, 5, 3))
    @test !_is_decayable(rand(GrassmannManifold, 5, 3))
end

# The public entry point, rather than the hand-rolled loop the rest of this file uses: `solve!`
# adds the stopping criteria and `OptimizerStatus`, and nothing about the decay should disturb
# either. `warn_iterations = 0` because `Adam` on a fixed step is expected to use its budget.
@testset "solve! runs the decayed method end to end, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(5)
    f = named_tuple_error(randn(rng, T, 5, 3))
    ps = NetworkParameters((
        w = rand(rng, StiefelManifold{T}, 5, 3), b = 10 * randn(rng, T, 3)))
    algorithm = AdamWithEuclideanDecay(; λ = T(λ))

    optimizer = Optimizer(
        ps, f; algorithm = algorithm, linesearch = Static(T(η)),
        max_iterations = 200, warn_iterations = 0)
    state = OptimizerState(algorithm, ps)
    f₀, b₀ = f(ps), norm(ps.b)
    solve!(ps, state, optimizer)

    @test state isa AdamState
    @test eltype(f(ps)) == T
    @test f(ps) < f₀
    @test norm(ps.b) < b₀                           # the decay pulled `b` in
    @test check(ps.w) < manifold_tolerance(T)       # and `w` is still on the manifold
end

# The case the method exists for: a network whose parameters mix a manifold with ordinary
# weights. The decay has to reach the ordinary ones and leave the manifold ones alone, in the same
# step. Only the *first* step can be compared against `Adam` component-wise — after it the two
# runs sit at different `b`, and the gradient with respect to `w` sees that.
@testset "a NamedTuple is decayed entry by entry, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(6)
    f = named_tuple_error(randn(rng, T, 5, 3))
    ps₀ = NetworkParameters((w = rand(rng, StiefelManifold{T}, 5, 3), b = randn(rng, T, 3)))

    adam = run!(deepcopy(ps₀), Adam(), f, 1; α = T(η))
    adamw = run!(deepcopy(ps₀), AdamWithEuclideanDecay(; λ = T(λ)), f, 1; α = T(η))

    @test eltype(adamw.b) == T
    @test adamw.w.A == adam.w.A                     # the manifold entry is untouched ...
    @test adamw.b ≈ adam.b - T(η) * T(λ) * ps₀.b    # ... and the ordinary one is decayed
    @test adamw.b ≉ adam.b                          # by an amount that is actually visible
end

# Over many steps the decay has to keep pulling the ordinary entry towards zero: `b` appears in
# the objective as `‖b‖`, whose `Adam` direction has magnitude ≈ 1 per component whatever `b`
# is, so without the decay it oscillates around zero at the scale of the learning rate instead
# of settling.
@testset "the decay shrinks the ordinary entries of a NamedTuple, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(7)
    f = named_tuple_error(randn(rng, T, 5, 3))
    ps₀ = NetworkParameters((
        w = rand(rng, StiefelManifold{T}, 5, 3), b = 10 * randn(rng, T, 3)))

    adam = run!(deepcopy(ps₀), Adam(), f, 200; α = T(η))
    adamw = run!(deepcopy(ps₀), AdamWithEuclideanDecay(; λ = T(λ)), f, 200; α = T(η))

    @test eltype(adamw.b) == T
    @test norm(adamw.b) < norm(adam.b)
    @test norm(adamw.b) < norm(ps₀.b)
    @test check(adamw.w) < manifold_tolerance(T)    # the manifold entry survives 200 steps
end

# The method carries no element type of the parameters: `Optimizer` converts it to theirs.
@testset "the optimizer converts the method to the element type of the parameters, $T" for T in REAL_ELTYPES
    @test AdamWithEuclideanDecay() isa AdamWithEuclideanDecay{Float64}
    x = T[1, 2, 3]
    opt = Optimizer(x, x -> sum(abs2, x); algorithm = AdamWithEuclideanDecay(; λ = 0.5))
    @test opt.algorithm isa AdamWithEuclideanDecay{T}
    @test eltype(opt.algorithm.λ) == T
    @test opt.algorithm.λ === T(0.5)
    # the method only produces a direction; the learning rate is the line search's `α`
    @test !hasproperty(AdamWithEuclideanDecay(), :η)
end

# A `λ` that cannot reach a single weight is silent otherwise — the run converges and nothing
# says the decay was dropped — so it is warned about once, when the optimizer is built. The
# `NamedTuple` above must *not* warn: one decayable entry is enough for the setting to mean
# something.
@testset "a λ that cannot reach anything is warned about" begin
    A = randn(5, 3)
    Y = rand(StiefelManifold, 5, 3)
    f(Y::StiefelManifold) = norm(A - Y * Y' * A)

    @test !_is_decayable(Y)
    @test _is_decayable(randn(3))
    @test _is_decayable(NetworkParameters((w = Y, b = randn(3))))

    @test_logs (:warn, r"none of the parameters") match_mode = :any Optimizer(
        deepcopy(Y), f; algorithm = AdamWithEuclideanDecay(; λ = λ), linesearch = Static(η))
    # `λ = 0` asks for no decay in the first place, and a `NamedTuple` with an ordinary entry
    # gets one, so neither has anything to warn about
    @test_logs Optimizer(deepcopy(Y), f; algorithm = AdamWithEuclideanDecay(; λ = 0.0), linesearch = Static(η))
    @test_logs Optimizer(
        NetworkParameters((w = deepcopy(Y), b = randn(3))), named_tuple_error(A);
        algorithm = AdamWithEuclideanDecay(; λ = λ), linesearch = Static(η))
end
