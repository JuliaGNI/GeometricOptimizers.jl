# The training step: `TrainingOptimizer` and `optimization_step!`, the methods that carry no element
# type, and the step-size keywords.

using GeometricOptimizers
using GeometricOptimizers: section, solution, iteration_number, step_size,
                           default_step_size,
                           PrecomputedGradient, DecayingStatic, AdamOptimizerWithDecay
using NeuralNetworkParameters: flatten, mapparameters
using LinearAlgebra: norm
using Random: Random
using Test

const PRECISIONS = (Float32, Float64)

# The iterates of GeometricMachineLearning 0.8's `optimization_step!`; see
# `scripts/gml_reference/generate.jl`.
include(joinpath(@__DIR__, "data", "gml_reference.jl"))

_flat(x::NetworkParameters) = flatten(x)[1]
_flat(x::AbstractArray) = vec(Matrix(x))
_flat(x::AbstractVector) = x

# One parameter set of each shape, with a Euclidean gradient for it.
function _parameters(::Type{T}, shape) where {T}
    if shape === :vector
        T[1, -2, 3]
    elseif shape === :stiefel
        rand(Random.Xoshiro(1), StiefelManifold{T}, 5, 2)
    else
        NetworkParameters((
            L1 = (weight = rand(Random.Xoshiro(2), StiefelManifold{T}, 5, 2),),
            L2 = (W = T[1 2; 3 4], b = T[1, -1]),
            L3 = (S = SymmetricMatrix(T[1, 2, 3], 2),)))
    end
end
_gradient(x::AbstractVector{T}) where {T} = T[0.5, -1, 2]
_gradient(x::StiefelManifold{T}) where {T} = T.(reshape(1:10, 5, 2)) ./ 10
_gradient(x::NetworkParameters) = mapparameters(_leaf_gradient, x)
_leaf_gradient(x::StiefelManifold{T}) where {T} = T.(reshape(1:10, 5, 2)) ./ 10
_leaf_gradient(x::SymmetricMatrix{T}) where {T} = SymmetricMatrix(T[1, -1, 2] ./ 4, 2)
_leaf_gradient(x::AbstractArray{T}) where {T} = fill(T(1 // 4), size(x))

const SHAPES = (:vector, :stiefel, :network)

@testset "a method without an element type solves Float32 parameters" begin
    F(x) = sum(abs2, x .- 1)
    for T in PRECISIONS, algorithm in (Adam(), AdamWithEuclideanDecay(), MomentumMethod())

        x = T[1, 2, 3]
        opt = Optimizer(
            x, F; algorithm = algorithm, max_iterations = 20, warn_iterations = 0)
        result = solve!(x, OptimizerState(algorithm, x), opt)
        @test eltype(x) === T
        @test minimum(result) isa T
        @test all(getfield(opt.algorithm, f) isa T
        for f in fieldnames(typeof(opt.algorithm)))
    end
    for T in PRECISIONS
        Y = rand(Random.Xoshiro(3), StiefelManifold{T}, 5, 2)
        G(Y) = sum(abs2, Y.A .- 1)
        opt = Optimizer(Y, G; algorithm = ScalarMomentAdam(), max_iterations = 5,
            warn_iterations = 0)
        result = solve!(Y, OptimizerState(opt.algorithm, Y), opt)
        @test minimum(result) isa T
        @test opt.algorithm isa ScalarMomentAdam{T}
    end
end

@testset "the methods and schedules take no element type" begin
    @test_throws MethodError Adam(Float32)
    @test_throws MethodError MomentumMethod(0.5)
    @test_throws MethodError ScalarMomentAdam(Float32)
    @test_throws MethodError AdamWithEuclideanDecay(Float32)
    @test_throws MethodError DecayingStatic(Float32)
    @test_throws MethodError AdamOptimizerWithDecay(100, Float32)

    @test Adam(; β₁ = 0.8).β₁ == 0.8
    @test MomentumMethod(; α = 0.5).α == 0.5
    @test DecayingStatic(; η₁ = 1e-1, η₂ = 1e-3, n = 10).n == 10
    @test AdamOptimizerWithDecay(100; η₁ = 1e-1).linesearch.η₁ == 1e-1

    for T in PRECISIONS
        opt = TrainingOptimizer(_parameters(T, :vector); algorithm = MomentumMethod())
        @test opt.method isa MomentumMethod{T}
        @test opt.method.α === T(0.01)
        # `Optimizer` converts once, as `TrainingOptimizer` does
        x = _parameters(T, :vector)
        @test Optimizer(x, x -> sum(abs2, x); algorithm = MomentumMethod()).algorithm isa
              MomentumMethod{T}
        @test Optimizer(x, x -> sum(abs2, x); algorithm = Adam()).algorithm isa Adam{T}
    end
end

@testset "step sizes: a number, a Static, a DecayingStatic, and nothing else" begin
    @test default_step_size(GradientMethod()) === 1e-2
    @test default_step_size(MomentumMethod()) === 1e-2
    @test default_step_size(Adam()) === 1e-3

    for T in PRECISIONS, shape in SHAPES

        x = _parameters(T, shape)
        opt = TrainingOptimizer(x; algorithm = GradientMethod(), linesearch = 1e-2)
        @test opt.linesearch isa Static
        @test step_size(opt.linesearch, 1) === T(1e-2)
        opt = TrainingOptimizer(x; algorithm = Adam())
        @test step_size(opt.linesearch, 1) === T(1e-3)
        opt = TrainingOptimizer(x; algorithm = Adam(), linesearch = Static(0.5))
        @test step_size(opt.linesearch, 7) === T(0.5)

        opt = TrainingOptimizer(x; algorithm = Adam(),
            linesearch = DecayingStatic(; η₁ = 1e-2, η₂ = 1e-4, n = 10))
        ls = opt.linesearch
        @test ls isa DecayingStatic{T}
        for t in (1, 5, 10)
            @test step_size(ls, t) == ls.γ^t * ls.η₁
            @test step_size(ls, t) isa T
        end

        @test_throws ArgumentError TrainingOptimizer(
            x; algorithm = Adam(), linesearch = Backtracking())
        @test_throws TypeError TrainingOptimizer(
            x; algorithm = Adam(), retraction = GeometricOptimizers.cayley)
        @test TrainingOptimizer(x; algorithm = Adam(), retraction = Cayley()).retraction ===
              Cayley()
        @test TrainingOptimizer(x; algorithm = Adam(), retraction = Geodesic()).retraction ===
              Geodesic()
    end

    # `Optimizer` reads a number the same way
    for T in PRECISIONS
        x = _parameters(T, :vector)
        opt = Optimizer(x, x -> sum(abs2, x); algorithm = GradientMethod(), linesearch = 1e-2)
        @test GeometricOptimizers.method(opt.linesearch) === Static(T(1e-2))
        @test_throws TypeError Optimizer(
            x, x -> sum(abs2, x); retraction = GeometricOptimizers.cayley)
    end
end

@testset "after k steps the state, the cache and x agree" begin
    methods = (
        GradientMethod(), MomentumMethod(; α = 0.5), Adam(), AdamWithEuclideanDecay())
    for T in PRECISIONS, shape in SHAPES, m in methods
        # a weight decay on a bare manifold warns that it does nothing
        method = m isa AdamWithEuclideanDecay && shape === :stiefel ?
                 AdamWithEuclideanDecay(; λ = 0.0) : m
        x = _parameters(T, shape)
        opt = TrainingOptimizer(x; algorithm = method)
        for k in 1:3
            @test optimization_step!(x, opt, _gradient(x)) === x
            @test iteration_number(opt.state) == k
            @test section(opt.state) == section(opt.cache)
            @test x == solution(opt.cache)
            @test eltype(_flat(x)) === T
        end
    end
    for T in PRECISIONS
        Y = _parameters(T, :stiefel)
        opt = TrainingOptimizer(Y; algorithm = ScalarMomentAdam())
        for k in 1:3
            optimization_step!(Y, opt, _gradient(Y))
            @test iteration_number(opt.state) == k
            @test section(opt.state) == section(opt.cache)
            @test Y == solution(opt.cache)
        end
    end
end

@testset "a step moves the parameters the way the method says" begin
    # `GradientMethod` on a vector is `x ← x - η∇L`, and `MomentumMethod` accumulates `p ← αp + ∇L`
    for T in PRECISIONS
        x = _parameters(T, :vector)
        g = _gradient(x)
        opt = TrainingOptimizer(x; algorithm = GradientMethod(), linesearch = 1 // 4)
        optimization_step!(x, opt, g)
        @test x == T[1, -2, 3] .- g ./ 4

        x = _parameters(T, :vector)
        opt = TrainingOptimizer(x; algorithm = MomentumMethod(; α = 1 // 2), linesearch = 1)
        optimization_step!(x, opt, g)
        optimization_step!(x, opt, g)
        @test x == T[1, -2, 3] .- g .- (g ./ 2 .+ g)
    end
end

@testset "PrecomputedGradient projects onto the tangent space" begin
    for T in PRECISIONS
        Y = _parameters(T, :stiefel)
        dp = _gradient(Y)
        @test PrecomputedGradient(Y, dp)(Y) == rgrad(Y, dp)
        x = _parameters(T, :vector)
        @test PrecomputedGradient(x, _gradient(x))(x) == _gradient(x)
        ps = _parameters(T, :network)
        dps = _gradient(ps)
        @test PrecomputedGradient(ps, dps)(ps) == rgrad(ps, dps)
    end
end

# The bound for `GradientMethod` and `MomentumMethod` on a model with manifold leaves is the measured
# difference of 1 to 2 `eps(T)` relative between GML 0.8's per-layer step and a step over the whole
# parameter set. A model without a manifold leaf matches exactly. `Adam` matches exactly once the
# random completion of each `GlobalSection` in the state is the one the reference drew; see
# `scripts/gml_reference/generate.jl` for why seeding the RNG equally does not give that.
_has_manifold(::Manifold) = true
_has_manifold(::AbstractArray) = false
_has_manifold(x::NamedTuple) = any(_has_manifold, values(x))
_exact(entry) = entry.method === :Adam || !_has_manifold(entry.x₀)

_copy_completions!(λY::GlobalSection, ::Nothing) = λY
_copy_completions!(λY::GlobalSection, λ::AbstractMatrix) = (copyto!(λY.λ, λ); λY)
function _copy_completions!(sections::NamedTuple, completions::NamedTuple)
    foreach(k -> _copy_completions!(sections[k], completions[k]), keys(sections))
end

function _method(entry)
    entry.method === :GradientMethod && return GradientMethod()
    entry.method === :MomentumMethod && return MomentumMethod(; α = 0.5)
    Adam()
end

@testset "the training step reproduces GeometricMachineLearning 0.8" begin
    for entry in GML_REFERENCE
        T = entry.T
        # a copy, because the step writes into the arrays of `x`
        x = NetworkParameters(deepcopy(entry.x₀))
        opt = TrainingOptimizer(x; algorithm = _method(entry), linesearch = entry.η)
        _copy_completions!(section(opt.state), entry.completions)
        for (dp, reference) in zip(entry.gradients, entry.iterates)
            optimization_step!(x, opt, NetworkParameters(dp))
            x₀₈ = _flat(NetworkParameters(reference))
            @test eltype(_flat(x)) === T
            if _exact(entry)
                @test _flat(x) == x₀₈
            else
                @test norm(_flat(x) - x₀₈) ≤ 4eps(T) * norm(x₀₈)
            end
        end
    end
end

@testset "the training step is public" begin
    @test Base.ispublic(GeometricOptimizers, :TrainingOptimizer)
    @test Base.ispublic(GeometricOptimizers, :optimization_step!)
    @test Base.ispublic(GeometricOptimizers, :PrecomputedGradient)
    @test Base.ispublic(GeometricOptimizers, :default_step_size)
    @test Base.ispublic(GeometricOptimizers, :step_size)
end
