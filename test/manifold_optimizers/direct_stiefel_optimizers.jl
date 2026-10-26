# The three Stiefel optimizers that step in the tangent space `T_Y St(k, M)` itself: `DirectMomentum`,
# `DirectScalarMomentAdam` and `FactoredStiefelAdam`.
#
# Every expected iterate is a dense expression: the closed forms of `rgrad`, the `M × M` Cayley
# transform `(𝕀 - W/2)⁻¹(𝕀 + W/2)Y`, the dense tangent projection and the dense differential of the
# Cayley transform, and the recursions of each method transcribed on dense matrices. The column
# norms of the horizontal part are sums over the rows here, where the package forms them otherwise.

using GeometricOptimizers
using GeometricOptimizers: AdamFamily, FirstOrderMethodWithState, change_precision,
                           default_step_size, direction, gradient, solution,
                           DirectStiefelCache, FactoredStiefelAdamCache,
                           DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM_α, NoWorkspace,
                           OptimizerCache
using NeuralNetworkParameters: NetworkParameters, params
using LinearAlgebra: LinearAlgebra, I, norm, qr
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/allocations.jl")

const METHODS = (DirectMomentum, DirectScalarMomentAdam, FactoredStiefelAdam)

# The tangent projection at `Y`, `Ξ - Y(YᵀΞ + ΞᵀY)/2`, in dense form.
proj(Y, Ξ) = Ξ - Y * (Y' * Ξ + Ξ' * Y) / 2

# The generator of the Cayley transform at `Y` along `Δ`, as an `M × M` matrix.
generator(Y, Δ) = Δ * Y' - Y * Δ' + Y * Δ' * Y * Y'

dense_cayley(Y, Δ) = (I - generator(Y, Δ) / 2) \ ((I + generator(Y, Δ) / 2) * Y)

dense_transport(Y_new, Y, Δ, Ξ, ::ProjectionTransport) = proj(Y_new, Ξ)
function dense_transport(Y_new, Y, Δ, Ξ, ::DifferentiatedCayleyTransport)
    L = I - generator(Y, Δ) / 2
    L \ (generator(Y, proj(Y, Ξ)) * (L \ Y))
end

dense_rgrad(Y, G, ::CanonicalMetric) = G - Y * G' * Y
dense_rgrad(Y, G, ::EuclideanMetric) = proj(Y, G)

skew(X) = (X - X') / 2

# The bias-corrected weights of step `t`, as `Adam` stores its moments.
weights(β, t) = ((β - β^t) / (1 - β^t), (1 - β) / (1 - β^t))

# The memory of each method before its first step, and one dense step: the new point, the step `Δ`
# and the new memory, transported to the new point.
dense_memory(::DirectMomentum, M, k) = (p = zeros(M, k),)
dense_memory(::DirectScalarMomentAdam, M, k) = (B = zeros(M, k), v = 0.0)
dense_memory(::FactoredStiefelAdam, M, k) = (B = zeros(M, k), V = zeros(k, k), v = zeros(k))

function dense_step(m::DirectMomentum, Y, G, mem, t, η)
    p = m.α * mem.p + dense_rgrad(Y, G, m.metric)
    Δ = -η * p
    Y_new = dense_cayley(Y, Δ)
    Y_new, Δ, (p = dense_transport(Y_new, Y, Δ, p, m.transport),)
end

function dense_step(m::DirectScalarMomentAdam, Y, G, mem, t, η)
    GR = dense_rgrad(Y, G, m.metric)
    c₁₁, c₁₂ = weights(m.β₁, t)
    c₂₁, c₂₂ = weights(m.β₂, t)
    B = c₁₁ * mem.B + c₁₂ * GR
    v = c₂₁ * mem.v + c₂₂ * sum(abs2, GR)
    D = B / (sqrt(v) + m.δ)
    Δ = -min(η, m.α_max / (norm(D) + m.δ)) * D
    Y_new = dense_cayley(Y, Δ)
    Y_new, Δ, (B = dense_transport(Y_new, Y, Δ, B, m.transport), v = v)
end

function dense_step(m::FactoredStiefelAdam, Y, G, mem, t, η)
    GR = dense_rgrad(Y, G, m.metric)
    c₁₁, c₁₂ = weights(m.β₁, t)
    c₂₁, c₂₂ = weights(m.β₂, t)
    B = c₁₁ * mem.B + c₁₂ * GR
    V = c₂₁ * mem.V + c₂₂ * skew(Y' * GR) .^ 2
    K = GR - Y * (Y' * GR)
    v = c₂₁ * mem.v + c₂₂ * vec(sum(abs2, K; dims = 1))
    Ã = skew(Y' * B) ./ (sqrt.(V) .+ m.δ)
    K̃ = (B - Y * (Y' * B)) ./ (sqrt.(v') .+ m.δ)
    D = Y * Ã + K̃
    Δ = -min(η, m.α_max / (norm(D) + m.δ)) * D
    Y_new = dense_cayley(Y, Δ)
    Y_new, Δ, (B = dense_transport(Y_new, Y, Δ, B, m.transport), V = V, v = v)
end

# A sequence of synthetic Euclidean gradients, the same for every method.
gradients(rng, ::Type{T}, M, k, n) where {T} = [randn(rng, T, M, k) for _ in 1:n]

tangent_defect(Y, Δ) = norm(Y' * Δ + Δ' * Y)

# The whole messages of the refusing guards. A test of a guard matches its whole message, and the
# patterns are checked against Base's own messages below.
const HYPERPARAMETER_GUARD = r"^(α|β₁|β₂) must satisfy 0 ≤ (α|β₁|β₂) < 1$|^(δ|α_max) must be positive$"
const RETRACTION_GUARD = r"^(DirectMomentum|DirectScalarMomentAdam|FactoredStiefelAdam) retracts with the direct Cayley transform and takes `retraction = Cayley\(\)` only$"

function message(f)
    try
        f()
        nothing
    catch e
        e isa Union{ArgumentError, DimensionMismatch} ? (typeof(e), e.msg) : (typeof(e), "")
    end
end

function refuses(f, pattern)
    (m = message(f); m !== nothing && m[1] === ArgumentError &&
                         occursin(pattern, m[2]))
end

@testset "the guard patterns match no message of Base" begin
    for f in (() -> ones(2, 2) * ones(3, 3), () -> ones(2) .+ ones(3),
        () -> sqrt(-1.0), () -> Int[][1])
        m = message(f)
        @test m !== nothing
        @test !occursin(HYPERPARAMETER_GUARD, something(m[2], ""))
        @test !occursin(RETRACTION_GUARD, something(m[2], ""))
    end
end

@testset "the constructors and their defaults" begin
    m = DirectMomentum()
    @test m isa OptimizerMethod
    @test m.α === DEFAULT_MOMENTUM_α
    @test m.metric === CanonicalMetric()
    @test m.transport === ProjectionTransport()
    for method in (DirectScalarMomentAdam, FactoredStiefelAdam)
        a = method()
        @test a isa OptimizerMethod
        @test (a.β₁, a.β₂, a.δ, a.α_max) === (9.0e-1, 9.9e-1, 1.0e-8, 1.0)
        @test a.metric === CanonicalMetric()
        @test a.transport === ProjectionTransport()
        @test a isa AdamFamily
    end
    @test !(DirectMomentum() isa AdamFamily)
    for method in METHODS
        @test method() isa FirstOrderMethodWithState
        @test nameof(method) in names(GeometricOptimizers)
        # no step size and no element type of the parameters: the keywords are the coefficients
        @test !(:η in fieldnames(typeof(method())))
        @test_throws MethodError method(1.0e-3)
        a = method(; metric = EuclideanMetric(), transport = DifferentiatedCayleyTransport())
        @test a.metric === EuclideanMetric()
        @test a.transport === DifferentiatedCayleyTransport()
    end
    @test_throws MethodError FactoredStiefelAdam(1e-3)
    @test DirectMomentum(; α = 1 // 2).α === 0.5
    @test FactoredStiefelAdam(; α_max = 2).α_max === 2.0
    for name in (:DirectMomentumState, :DirectScalarMomentAdamState, :FactoredStiefelAdamState)
        @test name in names(GeometricOptimizers)
    end
    for name in (:DirectStiefelCache, :FactoredStiefelAdamCache)
        @test isdefined(GeometricOptimizers, name)
        @test !(name in names(GeometricOptimizers))
    end
end

@testset "change_precision and default_step_size" begin
    @test default_step_size(DirectMomentum()) === 1.0e-2
    @test default_step_size(DirectScalarMomentAdam()) === DEFAULT_LEARNING_RATE
    @test default_step_size(FactoredStiefelAdam()) === DEFAULT_LEARNING_RATE
    for T in REAL_ELTYPES
        m = change_precision(T, DirectMomentum(; α = 0.3, metric = EuclideanMetric()))
        @test m.α === T(0.3)
        @test m.metric === EuclideanMetric()
        for method in (DirectScalarMomentAdam, FactoredStiefelAdam)
            a = change_precision(T,
                method(; β₁ = 0.8, β₂ = 0.95, δ = 1.0e-6, α_max = 0.5,
                    transport = DifferentiatedCayleyTransport()))
            @test (a.β₁, a.β₂, a.δ, a.α_max) === (T(0.8), T(0.95), T(1.0e-6), T(0.5))
            @test a.transport === DifferentiatedCayleyTransport()
        end
    end
end

@testset "a hyperparameter out of its range is refused" begin
    for bad in (-0.1, 1.0, 1.5, NaN)
        @test refuses(() -> DirectMomentum(; α = bad), HYPERPARAMETER_GUARD)
        @test message(() -> DirectMomentum(; α = bad)) ==
              (ArgumentError, "α must satisfy 0 ≤ α < 1")
        for method in (DirectScalarMomentAdam, FactoredStiefelAdam)
            @test message(() -> method(; β₁ = bad)) ==
                  (ArgumentError, "β₁ must satisfy 0 ≤ β₁ < 1")
            @test message(() -> method(; β₂ = bad)) ==
                  (ArgumentError, "β₂ must satisfy 0 ≤ β₂ < 1")
        end
    end
    for bad in (0.0, -0.0, -1.0e-8, NaN),
        method in (DirectScalarMomentAdam, FactoredStiefelAdam)

        @test message(() -> method(; δ = bad)) == (ArgumentError, "δ must be positive")
        @test message(() -> method(; α_max = bad)) ==
              (ArgumentError, "α_max must be positive")
        @test refuses(() -> method(; α_max = bad), HYPERPARAMETER_GUARD)
    end
    # the edges of the ranges are accepted
    @test DirectMomentum(; α = 0).α === 0.0
    @test FactoredStiefelAdam(; β₁ = 0, β₂ = 0).β₂ === 0.0
    @test FactoredStiefelAdam(; β₁ = prevfloat(1.0)).β₁ === prevfloat(1.0)
    @test FactoredStiefelAdam(; δ = nextfloat(0.0)).δ === nextfloat(0.0)
end

@testset "only a StiefelManifold is a leaf of these methods, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1)
    for method in METHODS
        for x in (rand(rng, T, 6), rand(rng, T, 6, 2), rand(rng, GrassmannManifold{T}, 6, 2))
            @test_throws MethodError OptimizerCache(method(), x)
            @test_throws MethodError OptimizerState(method(), x)
            @test_throws MethodError TrainingOptimizer(x; algorithm = method())
        end
        # `solve!`'s line search retracts through a section, so `Optimizer` takes none of them
        Y = rand(rng, StiefelManifold{T}, 6, 2)
        @test_throws MethodError Optimizer(Y, y -> sum(abs2, y.A); algorithm = method())
    end
end

@testset "no GlobalSection and no RetractionWorkspace, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(2)
    M, k = 500, 4
    Y = rand(rng, StiefelManifold{T}, M, k)
    for method in METHODS
        opt = TrainingOptimizer(Y; algorithm = method())
        @test opt.workspace isa DirectCayleyWorkspace{T}
        @test opt.gradient_buffer isa NoWorkspace
        @test opt.method == change_precision(T, method())
        @test !hasfield(typeof(opt.cache), :section)
        @test !hasfield(typeof(opt.state), :section)
        # A `GlobalSection` of this point holds an `M × (M - k)` matrix and its `qr` an `M × M` one,
        # 2 MB in `Float64`; the caches, the state and the workspace hold `O(Mk + k²)`, about
        # 16 `M × k` matrices, 128 kB in `Float64`. A quarter of one `M × M` matrix lies between.
        build(m, Y) = TrainingOptimizer(Y; algorithm = m())
        build(method, Y)
        @test (@allocated build(method, Y)) < M^2 * sizeof(T) ÷ 4
        @test (@allocated OptimizerCache(method(), Y)) < M^2 * sizeof(T) ÷ 4
        @test (@allocated OptimizerState(method(), Y)) < M^2 * sizeof(T) ÷ 4
    end
    # the control: the bound is below what one `GlobalSection` allocates
    @test (@allocated GlobalSection(Y)) > M^2 * sizeof(T) ÷ 4
end

@testset "another retraction than Cayley is refused, $T" for T in REAL_ELTYPES
    Y = rand(Random.Xoshiro(3), StiefelManifold{T}, 8, 2)
    for method in METHODS
        @test refuses(
            () -> TrainingOptimizer(Y; algorithm = method(), retraction = Geodesic()),
            RETRACTION_GUARD)
        @test message(() -> TrainingOptimizer(Y; algorithm = method(), retraction = Geodesic()))[2] ==
              "$(nameof(method)) retracts with the direct Cayley transform and takes " *
              "`retraction = Cayley()` only"
        @test TrainingOptimizer(Y; algorithm = method(), retraction = Cayley()) isa
              TrainingOptimizer
    end
end

const VARIANTS = ((CanonicalMetric(), ProjectionTransport()),
    (EuclideanMetric(), ProjectionTransport()),
    (CanonicalMetric(), DifferentiatedCayleyTransport()),
    (EuclideanMetric(), DifferentiatedCayleyTransport()))

@testset "five steps of $(nameof(method)) are the dense recursion, $T" for T in REAL_ELTYPES,
    method in METHODS

    # `α_max = 1e-2` caps every step of the two Adam methods, whose `‖D‖_F` is about 1 or more
    caps = method === DirectMomentum ? (nothing,) : (1.0, 1.0e-2)
    # at the scale `1e-4` the second moments are of the order of `δ = 1e-8`, so that a `δ` inside
    # the square root differs from a `δ` outside it by far more than the tolerance
    for (metric, transport) in VARIANTS, α_max in caps, scale in (1, 1.0e-4)
        rng = Random.Xoshiro(4)
        M, k, η = 16, 4, T(5.0e-2)
        Y = rand(rng, StiefelManifold{T}, M, k)
        Gs = [T(scale) * G for G in gradients(rng, T, M, k, 5)]
        m = α_max === nothing ? method(; metric, transport) :
            method(; metric, transport, α_max)
        opt = TrainingOptimizer(Y; algorithm = m, linesearch = η)
        Yd, mem = Float64.(Y.A), dense_memory(m, M, k)
        for (t, G) in enumerate(Gs)
            Y_before = copy(Y.A)
            @test optimization_step!(Y, opt, G) === Y
            @test Y.A != Y_before
            @test eltype(Y.A) === T
            Yd, Δd, mem = dense_step(m, Yd, Float64.(G), mem, t, Float64(η))
            Δ = direction(opt.cache)
            @test eltype(Δ) === T
            @test norm(Δ - Δd) ≤ sqrt(eps(T)) * norm(Δd)
            α_max === 1.0e-2 && @test norm(Δ) ≤ T(α_max) * (1 + 8 * eps(T))
            @test norm(Y.A - Yd) ≤ sqrt(eps(T)) * norm(Yd)
            # the step is tangent at the point it is taken from, with no re-projection
            @test tangent_defect(solution(opt.cache).A, Δ) ≤ 20 * eps(T) * norm(Δ)
            # the stored first moment is a tangent vector at the new point
            B = method === DirectMomentum ? opt.state.p : opt.state.B₁
            @test tangent_defect(Y.A, B) ≤ 20 * eps(T) * norm(B)
            @test norm(B - (method === DirectMomentum ? mem.p : mem.B)) ≤
                  sqrt(eps(T)) * norm(B)
        end
    end
end

@testset "FactoredStiefelAdam keeps V_W symmetric and Ã skew-symmetric exactly, $T" for T in REAL_ELTYPES
    for metric in (CanonicalMetric(), EuclideanMetric())
        rng = Random.Xoshiro(5)
        Y = rand(rng, StiefelManifold{T}, 16, 4)
        opt = TrainingOptimizer(Y; algorithm = FactoredStiefelAdam(; metric), linesearch = T(5e-2))
        for G in gradients(rng, T, 16, 4, 5)
            optimization_step!(Y, opt, G)
            @test opt.state.V_W == opt.state.V_W'
            @test opt.cache.Ã == -opt.cache.Ã'
            @test all(≥(0), opt.state.v_K)
            @test length(opt.state.v_K) == 4
        end
    end
end

@testset "a zero gradient is a zero step, $T" for T in REAL_ELTYPES, method in METHODS

    rng = Random.Xoshiro(6)
    Y = rand(rng, StiefelManifold{T}, 12, 3)
    Y₀ = copy(Y.A)
    opt = TrainingOptimizer(Y; algorithm = method(), linesearch = T(1.0e-1))
    optimization_step!(Y, opt, zeros(T, 12, 3))
    @test gradient(opt.cache) == zeros(T, 12, 3)
    @test direction(opt.cache) == zeros(T, 12, 3)
    @test Y.A == Y₀
    # a later step from a nonzero gradient is finite
    optimization_step!(Y, opt, randn(rng, T, 12, 3))
    @test all(isfinite, Y.A)
    @test Y.A != Y₀
end

@testset "a gradient of another shape or element type is refused before the count moves, $T" for T in REAL_ELTYPES,
    method in METHODS

    Y = rand(Random.Xoshiro(14), StiefelManifold{T}, 8, 2)
    Y₀ = copy(Y.A)
    opt = TrainingOptimizer(Y; algorithm = method())
    @test message(() -> optimization_step!(Y, opt, zeros(T, 8, 3))) == (DimensionMismatch,
        "the gradient does not have the shape of the parameters it is a gradient of")
    U = T === Float32 ? Float64 : Float32
    @test message(() -> optimization_step!(Y, opt, zeros(U, 8, 2)))[1] === ArgumentError
    @test opt.state.iterations == 0
    @test Y.A == Y₀
end

random_orthogonal(rng, ::Type{T}, M) where {T} = T.(Matrix(qr(randn(rng, M, M)).Q))

@testset "$(nameof(method)) is left O(M)-equivariant, $T" for T in REAL_ELTYPES,
    method in METHODS

    for (metric, transport) in VARIANTS
        rng = Random.Xoshiro(7)
        M, k = 16, 4
        Q = random_orthogonal(rng, T, M)
        Y = rand(rng, StiefelManifold{T}, M, k)
        Yq = StiefelManifold(Q * Y.A)
        Gs = gradients(rng, T, M, k, 5)
        m = method(; metric, transport)
        opt = TrainingOptimizer(Y; algorithm = m, linesearch = T(5.0e-2))
        optq = TrainingOptimizer(Yq; algorithm = m, linesearch = T(5.0e-2))
        for G in Gs
            optimization_step!(Y, opt, G)
            optimization_step!(Yq, optq, Q * G)
        end
        @test norm(Yq.A - Q * Y.A) ≤ 100 * eps(T)
    end
end

# `min ½‖Y - A‖²` over `St(4, 24)` for an `A` on the manifold, whose minimum `0` is at `Y = A`.
@testset "$(nameof(method)) solves the Stiefel Procrustes problem, $T" for T in REAL_ELTYPES,
    method in METHODS

    rng = Random.Xoshiro(8)
    A = rand(rng, StiefelManifold{T}, 24, 4).A
    Y = rand(rng, StiefelManifold{T}, 24, 4)
    f(Y) = sum(abs2, Y.A - A) / 2
    f₀ = f(Y)
    opt = TrainingOptimizer(Y; algorithm = method(), linesearch = T(5.0e-2))
    defects = T[]
    for _ in 1:150
        optimization_step!(Y, opt, Y.A - A)
        push!(defects, norm(Y.A' * Y.A - I))
    end
    @test f(Y) ≤ f₀ / 1000
    @test maximum(defects) ≤ 50 * eps(T)
    @test eltype(Y.A) === T
end

function mixed_set(::Type{T}, M; k = 4) where {T}
    rng = Random.Xoshiro(9)
    NetworkParameters((L1 = (weight = rand(rng, StiefelManifold{T}, M, k),),
        L2 = (W = randn(rng, T, 5, 3), b = randn(rng, T, 5))))
end

function mixed_gradient(ps, seed)
    (rng = Random.Xoshiro(seed);
        NetworkParameters((
            L1 = (weight = randn(rng, eltype(ps.L1.weight), size(ps.L1.weight)...),),
            L2 = (
                W = randn(rng, eltype(ps.L2.W), 5, 3), b = randn(rng, eltype(ps.L2.b), 5)))))
end

function step_allocations(x::X, opt::O, dp::D) where {X, O, D}
    optimization_step!(x, opt, dp)
    @allocated optimization_step!(x, opt, dp)
end

@testset "a CompositeMethod steps a Stiefel leaf with FactoredStiefelAdam, $T" for T in REAL_ELTYPES
    ps = mixed_set(T, 20)
    method = CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam())
    opt = TrainingOptimizer(ps; algorithm = method, linesearch = T(1e-3))
    @test opt.workspace.L1.weight isa DirectCayleyWorkspace{T}
    @test opt.workspace.L2.b isa NoWorkspace
    @test opt.gradient_buffer.L1.weight isa NoWorkspace
    @test opt.cache.L1.weight isa FactoredStiefelAdamCache{T}
    @test !hasfield(typeof(opt.cache.L1.weight), :section)
    # the leaf moves as under a `TrainingOptimizer` of its own
    Y = StiefelManifold(copy(ps.L1.weight.A))
    alone = TrainingOptimizer(Y; algorithm = FactoredStiefelAdam(), linesearch = T(1e-3))
    for s in 1:3
        dp = mixed_gradient(ps, s)
        W₀ = copy(ps.L2.W)
        optimization_step!(ps, opt, dp)
        optimization_step!(Y, alone, dp.L1.weight)
        @test ps.L1.weight.A == Y.A
        @test ps.L2.W != W₀
        @test norm(ps.L1.weight.A' * ps.L1.weight.A - I) ≤ 50 * eps(T)
    end
    @test eltype(ps.L1.weight.A) === T
end

# One `M × M` temporary of the Stiefel leaf is 1.9 MB in `Float32` between `M = 200` and `M = 800`,
# and one `M × k` temporary 9.6 kB; the tolerance of `n_independent` is 1 024 bytes.
@testset "the step of a Stiefel leaf allocates the same at M = 200 and M = 800, $T" for T in REAL_ELTYPES
    composite = CompositeMethod(; manifold = FactoredStiefelAdam(), array = Adam())
    bytes = map((200, 800)) do M
        ps = mixed_set(T, M)
        opt = TrainingOptimizer(ps; algorithm = composite, linesearch = T(1e-3))
        step_allocations(ps, opt, mixed_gradient(ps, 1))
    end
    @test n_independent(bytes...)
    for method in METHODS
        bytes = map((200, 800)) do M
            Y = rand(Random.Xoshiro(10), StiefelManifold{T}, M, 4)
            opt = TrainingOptimizer(Y; algorithm = method(), linesearch = T(1e-3))
            step_allocations(Y, opt, randn(Random.Xoshiro(11), T, M, 4))
        end
        @test n_independent(bytes...)
    end
    # the control: the barrier sees the allocations of `Adam`'s step on the same leaf, which
    # `KNOWN_ISSUES.md` records
    Y = rand(Random.Xoshiro(12), StiefelManifold{T}, 200, 4)
    opt = TrainingOptimizer(Y; algorithm = Adam(), linesearch = T(1e-3))
    @test step_allocations(Y, opt, randn(Random.Xoshiro(13), T, 200, 4)) > 0
end
