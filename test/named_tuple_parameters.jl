using GeometricOptimizers
using GeometricOptimizers: ArrayNamedTuple, OptimizerCache, OptimizerSolution,
    ParameterHandling, Cayley, Geodesic, check, increase_iteration_number!, solver_step!
using SimpleSolvers: Static, l2norm
using Test
import Random

# `scripts/mnist.jl` stores the parameters of a transformer in a flat `NamedTuple` that mixes
# `StiefelManifold`s with ordinary matrices and vectors, that is made up of `Float32`s and
# whose gradient is computed by hand. These three cases are covered below, together with the
# property that matters for all of them: the `StiefelManifold` entries have to *stay* on the
# manifold over the course of the optimization (this is what `check` measures).

const N, n, m = 6, 3, 4

Random.seed!(1234)

const B₀ = randn(N, m)

# the ranges that the three parameters occupy in `ParameterHandling.flatten(ps)[1]`
const ranges = (1:(N*n), (N*n+1):(N*n+n*m), (N*n+n*m+1):(N*n+n*m+N))

"""
    test_problem(T)

Return `F` and `∇F!` for ``F(Y, W, b) = \\frac{1}{2}\\|YW + b - B\\|^2`` with
``Y\\in{}St(N, n)``, ``W\\in\\mathbb{R}^{n\\times{}m}`` and ``b\\in\\mathbb{R}^N``.

The gradients are ``\\partial_YF = RW^T``, ``\\partial_WF = Y^TR`` and
``\\partial_bF = \\sum_jR_{\\bullet{}j}`` with ``R = YW + b - B``. As required by
[`GradientFunction`](@ref), `∇F!` operates on the *flattened* parameters.
"""
function test_problem(::Type{T}) where {T}
    B = T.(B₀)

    F(ps::NamedTuple) = sum(abs2, ps.Y * ps.W .+ ps.b .- B) / 2

    function ∇F!(g::AbstractVector, v::AbstractVector)
        Y = reshape(view(v, ranges[1]), N, n)
        W = reshape(view(v, ranges[2]), n, m)
        b = view(v, ranges[3])
        R = Y * W .+ b .- B
        copyto!(view(g, ranges[1]), vec(R * W'))
        copyto!(view(g, ranges[2]), vec(Y' * R))
        copyto!(view(g, ranges[3]), vec(sum(R; dims=2)))
        g
    end

    F, ∇F!
end

initial_parameters(::Type{T}) where {T} =
    (Y=rand(Random.Xoshiro(1234), StiefelManifold{T}, N, n), W=randn(Random.Xoshiro(5678), T, n, m), b=zeros(T, N))

"""
    optimize(T, algorithm; kwargs...)

Take `steps` optimizer steps and return the parameters, the value of `check(ps.Y)` after
*every* step and the values of the objective before and after every step.

Note that the seed has to be fixed for every run: the [`GlobalSection`](@ref) is drawn at
random and the iterates depend on it.
"""
function optimize(::Type{T}, algorithm; steps=20, η=0.1, retraction=Cayley(), hand_written_gradient=false) where {T}
    Random.seed!(1234)
    F, ∇F! = test_problem(T)
    ps = initial_parameters(T)
    optimizer = if hand_written_gradient
        Optimizer(ps, F; (∇F!)=∇F!, algorithm=algorithm, linesearch=Static(T(η)), retraction=retraction)
    else
        Optimizer(ps, F; algorithm=algorithm, linesearch=Static(T(η)), retraction=retraction)
    end
    state = OptimizerState(algorithm, ps)

    checks = T[]
    losses = T[F(ps)]
    for _ in 1:steps
        increase_iteration_number!(state)
        solver_step!(ps, state, optimizer)
        update!(state, optimizer, ps)
        push!(checks, check(ps.Y))
        push!(losses, F(ps))
    end

    ps, checks, losses
end

# Note that `Adam` has to be constructed with the element type of the parameters: unlike
# `MomentumMethod`, which the `Optimizer` converts, an `Adam{Float64}` does not dispatch to
# `OptimizerCache(::Adam{T}, ::OptimizerSolution{T})` for `Float32` parameters. That mismatch
# now errors with a message that says so, see the testset at the bottom of this file.
algorithms(::Type{T}) where {T} = (GradientMethod(), MomentumMethod(T(0.1)), Adam(T))
retractions() = (Geodesic(), Cayley())

# `ArrayTuple` used to be written as `Tuple{Vararg{AT}} where {AT<:AbstractArray{T}}`, which
# Julia's diagonal rule makes *homogeneous*: a `NamedTuple` that stores a `StiefelManifold`
# and an ordinary `Matrix` at the same time was then not an `ArrayNamedTuple` and could not
# be passed to the `Optimizer`.
@testset "a heterogeneous NamedTuple is a valid set of parameters" begin
    for T in (Float64, Float32)
        ps = initial_parameters(T)
        @test length(unique(typeof.(values(ps)))) == 3      # a manifold, a matrix and a vector
        @test ps isa ArrayNamedTuple{T}
        @test ps isa OptimizerSolution{T}
    end
end

# `ParameterHandling.flatten` defaults to `Float64`, so `Float32` parameters were silently
# promoted and the flattened vector no longer matched the parameters.
@testset "the parameters are flattened to their own element type" begin
    for T in (Float64, Float32)
        ps = initial_parameters(T)
        v, unflatten = ParameterHandling.flatten(ps)
        @test v isa Vector{T}
        @test length(v) == N * n + n * m + N
        ps′ = unflatten(v)
        @test ps′.Y ≈ ps.Y
        @test ps′.W == ps.W
        @test ps′.b == ps.b
        @test typeof(ps′.Y) == typeof(ps.Y)
    end
end

# The property that the whole exercise is about: a retraction maps back onto the manifold, so
# `Y` has to satisfy `YᵀY = I` after every single step — for every algorithm, every
# retraction and both element types.
#
# The tolerance is a round-off tolerance and nothing else. The deviation grows very slowly
# with the number of iterations — measured over all the combinations below it is `5`, `10`,
# `20` and `45` times `eps(T)` for `5`, `20`, `80` and `320` steps, i.e. it behaves like a
# random walk that the retraction keeps pulling back rather than like an accumulating error.
# The worst value observed at the `20` steps taken here is `10 * eps(T)`, so the tolerance
# leaves a factor of `10`. A retraction that actually left the manifold would be off by the
# order of the step size, i.e. by `1e-1`.
const MANIFOLD_TOLERANCE_IN_EPS = 100

@testset "the manifold property is preserved during the optimization" begin
    for T in (Float64, Float32), algorithm in algorithms(T), retraction in retractions()
        tol = MANIFOLD_TOLERANCE_IN_EPS * eps(T)
        _, checks, losses = optimize(T, algorithm; retraction=retraction)
        @test check(initial_parameters(T).Y) < tol   # the starting point is on the manifold ...
        @test maximum(checks) < tol                  # ... and so is every iterate
        @test last(losses) < first(losses) / 2       # the optimization actually did something
    end
end

# Only the projections of the attention layers of `scripts/mnist.jl` are on a manifold, so
# the ordinary entries must be updated as ordinary (Euclidean) parameters.
@testset "the parameters that are not on a manifold are optimized as well" begin
    for T in (Float64, Float32)
        ps₀ = initial_parameters(T)
        ps, _, _ = optimize(T, GradientMethod())
        @test ps.W ≉ ps₀.W
        @test ps.b ≉ ps₀.b
        @test ps.Y ≉ ps₀.Y
    end
end

# `GradientFunction(F, ∇F!, nt::NamedTuple)` is what lets `scripts/mnist.jl` use `Zygote`
# instead of the default `ForwardDiff`; `∇F!` is called on the flattened parameters.
@testset "a hand written gradient gives the same result as the default one" begin
    for T in (Float64, Float32), algorithm in algorithms(T)
        ps₁, _, losses₁ = optimize(T, algorithm; hand_written_gradient=false)
        ps₂, checks₂, losses₂ = optimize(T, algorithm; hand_written_gradient=true)
        @test ps₁.Y ≈ ps₂.Y
        @test ps₁.W ≈ ps₂.W
        @test ps₁.b ≈ ps₂.b
        @test losses₁ ≈ losses₂
        @test maximum(checks₂) < MANIFOLD_TOLERANCE_IN_EPS * eps(T)
    end
end
