# The cost of one `optimization_step!` of a `TrainingOptimizer`, on the parameter set of
# `StiefelLayer(64, 8) + Dense(64, 64) + Dense(64, 64)`, in both precisions.
#
#     julia --startup-file=no --project=. scripts/training_step_cost.jl
#
# `scripts/gml_reference/step_cost.jl` measures GeometricMachineLearning 0.8's `optimization_step!`
# on the same parameter set, in its own environment. BLAS runs on one thread in both, and each figure
# is the minimum over repeated calls after two warm-up steps: 5 calls for the allocation, 200 for the
# time.
using GeometricOptimizers
using LinearAlgebra: BLAS
using Random: Xoshiro
using Test: @inferred

BLAS.set_num_threads(1)

function parameters(::Type{T}) where {T}
    rng = Xoshiro(1)
    NetworkParameters((L1 = (weight = rand(rng, StiefelManifold{T}, 64, 8),),
        L2 = (W = randn(rng, T, 64, 64), b = zeros(T, 64)),
        L3 = (W = randn(rng, T, 64, 64), b = zeros(T, 64))))
end

function gradient(x::NetworkParameters{T}) where {T}
    rng = Xoshiro(2)
    NetworkParameters((L1 = (weight = rand(rng, T, 64, 8) ./ 100,),
        L2 = (W = rand(rng, T, 64, 64) ./ 100, b = rand(rng, T, 64) ./ 100),
        L3 = (W = rand(rng, T, 64, 64) ./ 100, b = rand(rng, T, 64) ./ 100)))
end

for T in (Float64, Float32), algorithm in (GradientMethod(), MomentumMethod(), Adam())

    x = parameters(T)
    dp = gradient(x)
    opt = TrainingOptimizer(x; algorithm = algorithm)
    @inferred optimization_step!(x, opt, dp)
    optimization_step!(x, opt, dp)
    bytes = minimum(@allocated(optimization_step!(x, opt, dp)) for _ in 1:5)
    time = minimum(@elapsed(optimization_step!(x, opt, dp)) for _ in 1:200)
    println(rpad(nameof(typeof(algorithm)), 16), rpad(T, 9), "alloc/step = ",
        round(bytes / 1024; digits = 1), " KiB   t/step = ", round(time * 1e6; digits = 1),
        " μs")
end
