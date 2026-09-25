# The training step: `TrainingOptimizer` and `optimization_step!`, the methods that carry no element
# type, and the step-size keywords.

using GeometricOptimizers
using GeometricOptimizers: section, solution, iteration_number, step_size,
                           default_step_size,
                           PrecomputedGradient, DecayingStatic, AdamOptimizerWithDecay
using NeuralNetworkParameters: flatten, mapparameters, params
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
            L3 = (S = SymmetricMatrix(T[1, 2, 3], 2), K = SkewSymMatrix(T[1, -2, 3], 3),
                Lo = StrictlyLowerTriangular(T[2, 1, -1], 3),
                Up = StrictlyUpperTriangular(T[-3, 1, 2], 3))))
    end
end
const VectorStorage = Union{
    SymmetricMatrix, SkewSymMatrix, StrictlyLowerTriangular, StrictlyUpperTriangular}
function _storage_gradient(x::VectorStorage)
    typeof(x).name.wrapper(
        eltype(x).(collect(1:length(x.S))) ./ 8, x.n)
end
_gradient(x::AbstractVector{T}) where {T} = T[0.5, -1, 2]
_gradient(x::StiefelManifold{T}) where {T} = T.(reshape(1:10, 5, 2)) ./ 10
_gradient(x::NetworkParameters) = mapparameters(_leaf_gradient, x)
_leaf_gradient(x::StiefelManifold{T}) where {T} = T.(reshape(1:10, 5, 2)) ./ 10
_leaf_gradient(x::SymmetricMatrix{T}) where {T} = SymmetricMatrix(T[1, -1, 2] ./ 4, 2)
_leaf_gradient(x::VectorStorage) = _storage_gradient(x)
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

# The bound for `GradientMethod` and `MomentumMethod` on a model with manifold leaves is round-off
# headroom. With the completions copied the difference measured 0 on aarch64 macOS, and without
# them at most 0.57 `eps(T)` relative, from the second orthonormalisation of the completion; BLAS
# and QR rounding differs by platform, which 4 `eps(T)` covers. A model without a manifold leaf
# matches exactly. `Adam` matches exactly once the
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

@testset "a gradient of another element type is refused at the entry" begin
    for (T, S) in ((Float32, Float64), (Float64, Float32)), shape in SHAPES

        x = _parameters(T, shape)
        opt = TrainingOptimizer(x; algorithm = Adam())
        dp = _gradient(_parameters(S, shape))
        @test_throws ArgumentError optimization_step!(x, opt, dp)
        @test iteration_number(opt.state) == 0
        # parameters and gradient agree, and the optimizer was built for the other type
        @test_throws ArgumentError optimization_step!(_parameters(S, shape), opt, dp)
        @test iteration_number(opt.state) == 0
    end
end

@testset "AdamWithEuclideanDecay shrinks every structured leaf by 1 - ηλ" begin
    # The decay is decoupled: the step is Adam's plus `-ηλx`, and the structured matrices are
    # linear in their storage, so on the storage it is Adam's step minus `ηλ` times the storage.
    for T in PRECISIONS
        leaves() = NetworkParameters((L = (
            S = SymmetricMatrix(T[1, 2, 3, 4, 5, 6], 3), K = SkewSymMatrix(T[1, -2, 3], 3),
            Lo = StrictlyLowerTriangular(T[2, 1, -1], 3),
            Up = StrictlyUpperTriangular(T[-3, 1, 2], 3)),))
        η, λ = T(1 // 10), T(1 // 4)
        x₀, x_adam, x_awd = leaves(), leaves(), leaves()
        dp = mapparameters(_storage_gradient, x₀)
        optimization_step!(
            x_adam, TrainingOptimizer(x_adam; algorithm = Adam(),
                linesearch = η), dp)
        optimization_step!(x_awd,
            TrainingOptimizer(x_awd; algorithm = AdamWithEuclideanDecay(; λ), linesearch = η),
            dp)
        for k in (:S, :K, :Lo, :Up)
            a, w, s = x_adam.L[k].S, x_awd.L[k].S, x₀.L[k].S
            @test eltype(w) === T
            @test norm(w - (a - η * λ * s)) ≤ 4eps(T) * norm(s)
        end
    end
end

@testset "the positional Optimizer converts the method" begin
    x = Float32[1, 2, 3]
    problem = GeometricOptimizers.OptimizerProblem(x -> sum(abs2, x), x)
    method = Adam()
    opt = Optimizer(method, problem, GeometricOptimizers.Hessian(method, problem, x),
        GeometricOptimizers.OptimizerCache(method, x), Static(0.01f0); max_iterations = 3,
        warn_iterations = 0)
    @test opt.algorithm isa Adam{Float32}
    @test minimum(solve!(x, OptimizerState(method, x), opt)) isa Float32
end

@testset "a gradient of the wrong shape is refused before the step counts" begin
    for T in PRECISIONS
        x = _parameters(T, :vector)
        opt = TrainingOptimizer(x; algorithm = Adam())
        @test_throws DimensionMismatch optimization_step!(x, opt, T[1, 2])
        @test iteration_number(opt.state) == 0
        ps = _parameters(T, :network)
        opt = TrainingOptimizer(ps; algorithm = Adam())
        dp = NetworkParameters((
            L1 = (weight = ones(T, 4, 2),), L2 = (W = ones(T, 2, 2),
                b = ones(T, 2)),
            L3 = params(_gradient(ps)).L3))
        @test_throws DimensionMismatch optimization_step!(ps, opt, dp)
        @test iteration_number(opt.state) == 0
    end
end

@testset "a step size is finite and positive" begin
    x = _parameters(Float64, :vector)
    for η in (NaN, Inf, -1, 0, true, Static(-1.0), Static(NaN))
        @test_throws ArgumentError TrainingOptimizer(x; algorithm = Adam(), linesearch = η)
        @test_throws ArgumentError Optimizer(x, x -> sum(abs2, x); linesearch = η)
    end
    # the check is on the step size in the element type of the parameters
    y = _parameters(Float32, :vector)
    for η in (1e-50, 1e40)
        @test_throws ArgumentError TrainingOptimizer(y; algorithm = Adam(), linesearch = η)
        @test_throws ArgumentError Optimizer(y, y -> sum(abs2, y); linesearch = η)
    end
end

@testset "the training step is public" begin
    @test Base.ispublic(GeometricOptimizers, :TrainingOptimizer)
    @test Base.ispublic(GeometricOptimizers, :optimization_step!)
    @test Base.ispublic(GeometricOptimizers, :PrecomputedGradient)
    @test Base.ispublic(GeometricOptimizers, :default_step_size)
    @test Base.ispublic(GeometricOptimizers, :step_size)
end
