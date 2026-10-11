# The training step of the `Adam` family on a parameter set: what it allocates, and what it computes.
#
# `Adam`, `AdamWithEuclideanDecay` and `AdamOptimizerWithDecay` update both moments of every leaf in
# place, so a step after warm-up allocates nothing on a set of dense and vector-storage leaves. On a
# set with a `StiefelManifold` leaf the step allocates what the `GradientMethod` step allocates there:
# the Riemannian gradient, its horizontal lift and the Cayley retraction of that leaf, which
# `KNOWN_ISSUES.md` records. `Adam` adds nothing to it.
#
# `scripts/parameter_set_step_cost.jl` measures the same steps and names the frames.

using GeometricOptimizers
using GeometricOptimizers: AdamOptimizerWithDecay
using NeuralNetworkParameters: NetworkParameters, mapparameters, params, flatten
using LinearAlgebra: norm
using Random: Xoshiro
using Test

include("../helpers/eltypes.jl")
include("../helpers/allocations.jl")

# "Model A": eight `Dense(64, 64)` layers and one `Dense(64, 1)`, 33 345 parameters.
function model_A(::Type{T}) where {T}
    rng = Xoshiro(1)
    layer(m, n) = (W = randn(rng, T, m, n) ./ T(8), b = randn(rng, T, m) ./ T(8))
    NetworkParameters(NamedTuple{ntuple(i -> Symbol(:L, i), 9)}(
        (ntuple(_ -> layer(64, 64), 8)..., layer(1, 64))))
end

# Model A with a `SymmetricMatrix` and a `SkewSymMatrix` weight: dense and vector-storage leaves.
function structured_set(::Type{T}) where {T}
    rng = Xoshiro(3)
    p = params(model_A(T))
    NetworkParameters(merge(p,
        (L4 = (W = rand(rng, SymmetricMatrix{T}, 64), b = p.L4.b),
            L5 = (W = rand(rng, SkewSymMatrix{T}, 64), b = p.L5.b))))
end

# The same with a `64 × 8` `StiefelManifold` weight in `L6`.
function stiefel_set(::Type{T}) where {T}
    rng = Xoshiro(4)
    p = params(structured_set(T))
    NetworkParameters(merge(p,
        (L6 = (W = rand(rng, StiefelManifold{T}, 64, 8), b = p.L6.b),)))
end

# A Euclidean gradient of the shape of `x`: a dense matrix where `x` is a manifold point.
_gradient_leaf(rng, x::Manifold{T}) where {T} = randn(rng, T, size(x)...) ./ 100
function _gradient_leaf(rng, x::Union{SymmetricMatrix{T}, SkewSymMatrix{T}}) where {T}
    typeof(x).name.wrapper(randn(rng, T, length(x.S)) ./ 100, x.n)
end
_gradient_leaf(rng, x::AbstractArray{T}) where {T} = randn(rng, T, size(x)...) ./ 100
function random_gradient(x, seed)
    (rng = Xoshiro(seed); mapparameters(y -> _gradient_leaf(rng, y), x))
end

# The barrier: concrete argument types, one warm-up call, then the measured call.
function allocations(x::X, opt::O, dp::D) where {X, O, D}
    optimization_step!(x, opt, dp)
    @allocated optimization_step!(x, opt, dp)
end
flatten_allocations(x::X) where {X} = (flatten(x); @allocated flatten(x))

const ADAM_FAMILY = (
    "Adam" => () -> (algorithm = Adam(),),
    "AdamWithEuclideanDecay" => () -> (algorithm = AdamWithEuclideanDecay(λ = 0.1),),
    "AdamOptimizerWithDecay" => () -> AdamOptimizerWithDecay(100))

@testset "the $name step allocates nothing on $setname, $T" for T in REAL_ELTYPES,
    (name, kwargs) in ADAM_FAMILY,
    (setname, build) in (("model A", model_A), (
        "dense and vector-storage leaves", structured_set))
    x = build(T)
    dp = random_gradient(x, 6)
    opt = TrainingOptimizer(x; kwargs()...)
    before = copy(flatten(x)[1])
    @test (@inferred optimization_step!(x, opt, dp)) === x
    @test flatten(x)[1] != before
    @test eltype(flatten(x)[1]) == T
    @test allocations(x, opt, dp) == 0
    # the control: a barrier of this form sees an allocation where there is one
    @test flatten_allocations(dp) > 0
end

# A `64 × 8` horizontal lift holds `28 + 448 = 476` numbers: 1 904 bytes in `Float32` and 3 808 in
# `Float64`. One more moment temporary of the Stiefel leaf alone is therefore above the tolerance of
# `n_independent`, 1 024 bytes, and one of a dense `64 × 64` leaf is 16 kB.
@testset "the $name step on a set with a Stiefel leaf allocates what a gradient step does, $T" for T in REAL_ELTYPES,
    (name, kwargs) in ADAM_FAMILY

    x = stiefel_set(T)
    dp = random_gradient(x, 6)
    adam = allocations(x, TrainingOptimizer(x; kwargs()...), dp)
    y = stiefel_set(T)
    gradient_step = allocations(y, TrainingOptimizer(y; algorithm = GradientMethod()), dp)
    @test gradient_step > 0
    @test n_independent(adam, gradient_step)
end

# The `Adam` step written out for leaves whose step is `x + αδ`: a dense array and the storage vector
# of a vector-storage matrix. Both moments are stored bias-corrected (see the docstring of `Adam`).
function reference_adam(x₀::NetworkParameters{T}, gradients, α::T; β₁ = T(0.9),
        β₂ = T(0.99), δ = T(1.0e-8)) where {T}
    x = [copy(vec(leaf)) for leaf in _storages(x₀)]
    m₁ = [zero(xᵢ) for xᵢ in x]
    m₂ = [zero(xᵢ) for xᵢ in x]
    for (t, dp) in enumerate(gradients)
        fac₁₁ = (β₁ - β₁^t) / (1 - β₁^t)
        fac₁₂ = (1 - β₁) / (1 - β₁^t)
        fac₂₁ = (β₂ - β₂^t) / (1 - β₂^t)
        fac₂₂ = (1 - β₂) / (1 - β₂^t)
        for (i, leaf) in enumerate(_storages(dp))
            g = vec(leaf)
            m₁[i] = fac₁₁ .* m₁[i] .+ fac₁₂ .* g
            m₂[i] = fac₂₁ .* m₂[i] .+ fac₂₂ .* (g .^ 2)
            x[i] = x[i] .+ α .* -(m₁[i] ./ (sqrt.(m₂[i]) .+ δ))
        end
    end
    x
end
_storages(x::NetworkParameters) = _storages(params(x))
_storages(x::NamedTuple) = reduce(vcat, map(_storages, values(x)); init = Any[])
_storages(x::Union{SymmetricMatrix, SkewSymMatrix}) = Any[x.S]
_storages(x::AbstractArray) = Any[x]

@testset "20 Adam steps are the Adam recursion, leaf by leaf, $T" for T in REAL_ELTYPES
    α = T(1) / 100
    x = structured_set(T)
    expected = reference_adam(x, [random_gradient(x, 10 + t) for t in 1:20], α)
    opt = TrainingOptimizer(x; algorithm = Adam(), linesearch = α)
    for t in 1:20
        optimization_step!(x, opt, random_gradient(x, 10 + t))
    end
    # A fused update rounds as the two passes it replaces, or within a few ULPs of them.
    for (leaf, reference) in zip(_storages(x), expected)
        @test eltype(leaf) == T
        @test norm(vec(leaf) - reference) ≤ 4eps(T) * norm(reference)
    end
end
