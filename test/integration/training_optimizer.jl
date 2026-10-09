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

include("../helpers/eltypes.jl")

# The iterates of GeometricMachineLearning 0.8's `optimization_step!`; see
# `scripts/gml_reference/generate.jl`.
include(joinpath(@__DIR__, "..", "helpers", "gml_reference.jl"))

_flat(x::NetworkParameters) = flatten(x)[1]
_flat(x::AbstractArray) = vec(Matrix(x))
_flat(x::AbstractVector) = x

# One parameter set of each shape, with a Euclidean gradient for it. The integer patterns are scaled
# by `1.1`, which is not a power of 2, so that they round in `Float32`.
_scaled(x::AbstractArray{T}) where {T} = T(1.1) .* x
function _parameters(::Type{T}, shape) where {T}
    if shape === :vector
        _scaled(T[1, -2, 3])
    elseif shape === :stiefel
        rand(Random.Xoshiro(1), StiefelManifold{T}, 5, 2)
    else
        NetworkParameters((
            L1 = (weight = rand(Random.Xoshiro(2), StiefelManifold{T}, 5, 2),),
            L2 = (W = _scaled(T[1 2; 3 4]), b = _scaled(T[1, -1])),
            L3 = (S = SymmetricMatrix(_scaled(T[1, 2, 3]), 2),
                K = SkewSymMatrix(_scaled(T[1, -2, 3]), 3),
                Lo = StrictlyLowerTriangular(_scaled(T[2, 1, -1]), 3),
                Up = StrictlyUpperTriangular(_scaled(T[-3, 1, 2]), 3))))
    end
end
const VectorStorage = Union{
    SymmetricMatrix, SkewSymMatrix, StrictlyLowerTriangular, StrictlyUpperTriangular}
function _storage_gradient(x::VectorStorage)
    typeof(x).name.wrapper(
        eltype(x).(collect(1:length(x.S))) ./ 10, x.n)
end
_gradient(x::AbstractVector{T}) where {T} = T[0.7, -1.3, 2.1]
_gradient(x::StiefelManifold{T}) where {T} = T.(reshape(1:10, 5, 2)) ./ 10
_gradient(x::NetworkParameters) = mapparameters(_leaf_gradient, x)
_leaf_gradient(x::StiefelManifold{T}) where {T} = T.(reshape(1:10, 5, 2)) ./ 10
_leaf_gradient(x::SymmetricMatrix{T}) where {T} = SymmetricMatrix(T[1, -1, 2] ./ 10, 2)
_leaf_gradient(x::VectorStorage) = _storage_gradient(x)
_leaf_gradient(x::AbstractArray{T}) where {T} = fill(T(3 // 10), size(x))

const SHAPES = (:vector, :stiefel, :network)

@testset "a method without an element type solves parameters of either precision, $T" for T in REAL_ELTYPES
    F(x) = sum(abs2, x .- 1)
    for algorithm in (Adam(), AdamWithEuclideanDecay(), MomentumMethod())
        x = _parameters(T, :vector)
        opt = Optimizer(
            x, F; algorithm = algorithm, max_iterations = 20, warn_iterations = 0)
        result = solve!(x, OptimizerState(algorithm, x), opt)
        @test eltype(x) === T
        @test minimum(result) isa T
        @test all(getfield(opt.algorithm, f) isa T
        for f in fieldnames(typeof(opt.algorithm)))
    end
    Y = rand(Random.Xoshiro(3), StiefelManifold{T}, 5, 2)
    G(Y) = sum(abs2, Y.A .- 1)
    opt = Optimizer(Y, G; algorithm = ScalarMomentAdam(), max_iterations = 5,
        warn_iterations = 0)
    result = solve!(Y, OptimizerState(opt.algorithm, Y), opt)
    @test eltype(Y) == T
    @test minimum(result) isa T
    @test opt.algorithm isa ScalarMomentAdam{T}
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

    for T in REAL_ELTYPES
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

@testset "step sizes: a number, a Static, a DecayingStatic, and nothing else, $T" for T in REAL_ELTYPES
    @test default_step_size(GradientMethod()) === 1e-2
    @test default_step_size(MomentumMethod()) === 1e-2
    @test default_step_size(Adam()) === 1e-3

    for shape in SHAPES
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
            @test eltype(step_size(ls, t)) == T
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
    x = _parameters(T, :vector)
    opt = Optimizer(x, x -> sum(abs2, x); algorithm = GradientMethod(), linesearch = 1e-2)
    @test GeometricOptimizers.method(opt.linesearch) === Static(T(1e-2))
    @test_throws TypeError Optimizer(
        x, x -> sum(abs2, x); retraction = GeometricOptimizers.cayley)
end

@testset "after k steps the state, the cache and x agree, $T" for T in REAL_ELTYPES
    methods = (
        GradientMethod(), MomentumMethod(; α = 0.5), Adam(), AdamWithEuclideanDecay())
    for shape in SHAPES, m in methods
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
    Y = _parameters(T, :stiefel)
    opt = TrainingOptimizer(Y; algorithm = ScalarMomentAdam())
    for k in 1:3
        optimization_step!(Y, opt, _gradient(Y))
        @test iteration_number(opt.state) == k
        @test section(opt.state) == section(opt.cache)
        @test Y == solution(opt.cache)
        @test eltype(Y) == T
    end
end

# `GradientMethod` on a vector is `x ← x - η∇L`, and `MomentumMethod` accumulates `p ← αp + ∇L`.
# Both are compared exactly: `η = 1/4` and `α = 1/2` are powers of 2, so the expected value is
# rounded in the same operations as the step.
@testset "a step moves the parameters the way the method says, $T" for T in REAL_ELTYPES
    x₀ = _parameters(T, :vector)
    x = copy(x₀)
    g = _gradient(x)
    opt = TrainingOptimizer(x; algorithm = GradientMethod(), linesearch = 1 // 4)
    optimization_step!(x, opt, g)
    @test eltype(x) == T
    @test x == x₀ .- g ./ 4

    x = copy(x₀)
    opt = TrainingOptimizer(x; algorithm = MomentumMethod(; α = 1 // 2), linesearch = 1)
    optimization_step!(x, opt, g)
    optimization_step!(x, opt, g)
    @test eltype(x) == T
    @test x == x₀ .- g .- (g ./ 2 .+ g)
end

@testset "PrecomputedGradient projects onto the tangent space, $T" for T in REAL_ELTYPES
    Y = _parameters(T, :stiefel)
    dp = _gradient(Y)
    @test eltype(PrecomputedGradient(Y, dp)(Y)) == T
    @test PrecomputedGradient(Y, dp)(Y) == rgrad(Y, dp)
    x = _parameters(T, :vector)
    @test PrecomputedGradient(x, _gradient(x))(x) == _gradient(x)
    ps = _parameters(T, :network)
    dps = _gradient(ps)
    @test PrecomputedGradient(ps, dps)(ps) == rgrad(ps, dps)
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

@testset "the training step reproduces GeometricMachineLearning 0.8, $T" for T in REAL_ELTYPES
    for entry in filter(entry -> entry.T === T, GML_REFERENCE)
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

@testset "a gradient of another element type is refused at the entry, $T" for T in REAL_ELTYPES
    for S in filter(!=(T), REAL_ELTYPES), shape in SHAPES

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

@testset "AdamWithEuclideanDecay shrinks every structured leaf by 1 - ηλ, $T" for T in REAL_ELTYPES
    # The decay is decoupled: the step is Adam's plus `-ηλx`, and the structured matrices are
    # linear in their storage, so on the storage it is Adam's step minus `ηλ` times the storage.
    leaves() = NetworkParameters((L = (
        S = SymmetricMatrix(_scaled(T[1, 2, 3, 4, 5, 6]), 3),
        K = SkewSymMatrix(_scaled(T[1, -2, 3]), 3),
        Lo = StrictlyLowerTriangular(_scaled(T[2, 1, -1]), 3),
        Up = StrictlyUpperTriangular(_scaled(T[-3, 1, 2]), 3)),))
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
        # the round-off of the two steps and of the expected value, a few operations each:
        # measured at most 0.93 `eps(T)⋅‖s‖` in both precisions over 20 draws of the storage
        @test norm(w - (a - η * λ * s)) ≤ 4eps(T) * norm(s)
    end
end

@testset "the positional Optimizer converts the method, $T" for T in REAL_ELTYPES
    x = _parameters(T, :vector)
    problem = GeometricOptimizers.OptimizerProblem(x -> sum(abs2, x), x)
    method = Adam()
    opt = Optimizer(method, problem, GeometricOptimizers.Hessian(method, problem, x),
        GeometricOptimizers.OptimizerCache(method, x), Static(T(0.01)); max_iterations = 3,
        warn_iterations = 0)
    @test opt.algorithm isa Adam{T}
    result = solve!(x, OptimizerState(method, x), opt)
    @test eltype(minimum(result)) == T
    @test minimum(result) isa T
end

@testset "a gradient of the wrong shape is refused before the step counts" begin
    for T in REAL_ELTYPES
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
