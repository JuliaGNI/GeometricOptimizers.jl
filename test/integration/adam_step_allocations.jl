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
using GeometricOptimizers: AdamOptimizerWithDecay, direction, first_moment, gradient_array,
                           second_moment
using NeuralNetworkParameters: NetworkParameters, params, flatten
using LinearAlgebra: norm
using Random: Random, Xoshiro
using Test

include("../helpers/eltypes.jl")
include("../helpers/allocations.jl")
include("../helpers/parameter_set_step.jl")

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
    @test step_allocations(x, opt, dp) == 0
    # the control: a barrier of this form sees an allocation where there is one
    @test flatten_allocations(dp) > 0
end

# Whether the walk over the leaves specialises on the function it is handed can depend on what the
# process compiled before, so this file's own order cannot show every case. A fresh process takes the
# step on small dense sets first and then on sets with a vector-storage weight, in both precisions.
@testset "the Adam step allocates nothing on a vector-storage weight after a dense one, in a fresh process" begin
    script = joinpath(@__DIR__, "..", "helpers", "adam_step_after_dense.jl")
    project = dirname(Base.active_project())
    lines = readlines(`$(Base.julia_cmd()) --startup-file=no --project=$project $script`)
    @test length(lines) == 20
    for line in lines
        name, bytes = split(line, ": ")
        @test (name, parse(Int, bytes)) == (name, 0)
    end
end

# A `64 × 8` horizontal lift holds `28 + 448 = 476` numbers: 1 904 bytes in `Float32` and 3 808 in
# `Float64`. One more moment temporary of the Stiefel leaf alone is therefore above the tolerance of
# `n_independent`, 1 024 bytes, and one of a dense `64 × 64` leaf is 16 kB.
@testset "the $name step on a set with a Stiefel leaf allocates what a gradient step does, $T" for T in REAL_ELTYPES,
    (name, kwargs) in ADAM_FAMILY

    x = stiefel_set(T)
    dp = random_gradient(x, 6)
    adam = step_allocations(x, TrainingOptimizer(x; kwargs()...), dp)
    y = stiefel_set(T)
    gradient_step = step_allocations(y, TrainingOptimizer(y; algorithm = GradientMethod()), dp)
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

# A set with a `StiefelManifold` and a `GrassmannManifold` weight.
function manifold_set(::Type{T}) where {T}
    rng = Xoshiro(7)
    NetworkParameters((
        L1 = (W = rand(rng, StiefelManifold{T}, 16, 4), b = randn(rng, T, 16)),
        L2 = (W = rand(rng, GrassmannManifold{T}, 16, 4),)))
end

flat(x) = flatten(x)[1]

# On a manifold leaf the step is a retraction and not `x + αδ`, so `reference_adam` does not apply.
# The moments and the direction of such a leaf are horizontal lifts, and the recursion acts
# elementwise on their free parameters, which `flatten` writes out. So after every step the flattened
# moments and the flattened direction are the recursion applied to the flattened lift of the gradient
# that the cache holds, on every block of a Stiefel and a Grassmann lift. The section and the
# retraction do not enter.
@testset "20 Adam steps are the Adam recursion on Stiefel and Grassmann leaves, $T" for T in REAL_ELTYPES
    # the random completion of each `GlobalSection`
    Random.seed!(1234)
    α, β₁, β₂, δ = T(1) / 100, T(0.9), T(0.99), T(1.0e-8)
    x = manifold_set(T)
    before = copy(flat(x))
    opt = TrainingOptimizer(x; algorithm = Adam(), linesearch = α)
    m₁ = zero(flat(first_moment(opt.state)))
    m₂ = zero(m₁)
    for t in 1:20
        optimization_step!(x, opt, random_gradient(x, 30 + t))
        fac₁₁ = (β₁ - β₁^t) / (1 - β₁^t)
        fac₁₂ = (1 - β₁) / (1 - β₁^t)
        fac₂₁ = (β₂ - β₂^t) / (1 - β₂^t)
        fac₂₂ = (1 - β₂) / (1 - β₂^t)
        g = flat(gradient_array(opt.cache))
        m₁ = fac₁₁ .* m₁ .+ fac₁₂ .* g
        m₂ = fac₂₁ .* m₂ .+ fac₂₂ .* g .^ 2
        d = α .* .-(m₁ ./ (sqrt.(m₂) .+ δ))
        @test norm(flat(first_moment(opt.state)) - m₁) ≤ 4eps(T) * norm(m₁)
        @test norm(flat(second_moment(opt.state)) - m₂) ≤ 4eps(T) * norm(m₂)
        @test norm(flat(direction(opt.cache)) - d) ≤ 4eps(T) * norm(d)
    end
    @test eltype(flat(x)) == T
    @test flat(x) != before
end
