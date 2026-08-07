using GeometricOptimizers
using GeometricOptimizers: AdamState, MomentumState, momentum, cache, direction,
    increase_iteration_number!, solver_step!, update!
using Test

# The update formulas of the two stateful first-order methods, pinned on ordinary (Euclidean)
# parameters: a `Vector` carries no global section and no retraction, so what is left of a step
# is exactly the recursion.
#
# The objective is *linear*, so its gradient is the constant `C` at every iterate. That is what
# makes the momentum recursion checkable in closed form, and it is also what tells an
# accumulator apart from momentum: on a constant gradient the former diverges where the latter
# saturates.
const C = [1.0, -2.0, 0.5]
objective(x::AbstractVector) = sum(C .* x)

# `η` is the learning rate, i.e. the `α` of the `Static` line search — the methods themselves
# only produce a direction, see `default_linesearch`.
const η = 0.01

# With `t = 1` the bias-corrected moments are `m₁ = ∇L` and `m₂ = ∇L⊙∇L`, so the first Adam
# direction is `-∇L/(|∇L| + δ)`, i.e. `-sign(∇L)` up to `δ = 1e-8`, and the first step is `η`
# in absolute value — component-wise and independently of how large the gradient is. This is
# the property that the off-by-one in the bias correction broke: with `t = 2` in the first step
# the factors come out as `0.7425` instead of `1`, and nothing in the suite noticed, because
# the one test that looked at the first step was also the one that did not increase the
# iteration number.
@testset "the first Adam step is the learning rate times sign(∇L)" begin
    x = [1.0, -2.0, 0.5]
    algorithm = Adam()
    optimizer = Optimizer(x, objective; algorithm=algorithm, linesearch=Static(η))
    state = AdamState(x)

    x₀ = copy(x)
    increase_iteration_number!(state)
    solver_step!(x, state, optimizer)

    @test x - x₀ ≈ -η * sign.(C) rtol = 1e-6
end

# The momentum recursion is `p ← αp + ∇L` with the direction `-p`. It used to be `p ← p + α∇L`
# with the direction `-(∇L + p)`, which is an undamped accumulator: on a constant gradient it
# grows linearly and without bound instead of saturating at `∇L/(1 - α)`. See issue #18.
@testset "the momentum is damped rather than accumulated" begin
    α = 0.9
    steps = 10

    x = zeros(3)
    algorithm = MomentumMethod(α)
    optimizer = Optimizer(x, objective; algorithm=algorithm, linesearch=Static(η))
    state = MomentumState(x)

    p = zeros(3)
    for _ in 1:steps
        p = α * p + C                          # the recursion, spelled out
        x_before = copy(x)

        increase_iteration_number!(state)
        solver_step!(x, state, optimizer)

        # the cache forms the direction from the *previous* momentum and the current gradient,
        # so it has to arrive at the same `p` as the state does afterwards
        @test direction(cache(optimizer)) ≈ -η * p
        @test x - x_before ≈ -η * p

        update!(state, optimizer, x)
        @test momentum(state) ≈ p
    end

    # the closed form of the recursion, and the bound that separates it from the accumulator:
    # after these ten steps the accumulator would sit at `steps * α * C = 9C`, and it would keep
    # going, whereas `p` never leaves `C/(1 - α) = 10C`
    @test momentum(state) ≈ C * (1 - α^steps) / (1 - α)
    @test all(abs.(momentum(state)) .< abs.(C) ./ (1 - α))
end
