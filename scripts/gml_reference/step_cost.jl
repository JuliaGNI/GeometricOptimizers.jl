# The cost of one GeometricMachineLearning 0.8 `optimization_step!`, on the parameter set
# `scripts/training_step_cost.jl` measures a `TrainingOptimizer` on, measured the same way.
#
#     julia --startup-file=no --project=scripts/gml_reference scripts/gml_reference/step_cost.jl
using GeometricMachineLearning
import GeometricOptimizers as GO
using LinearAlgebra: BLAS
using Random: Xoshiro
using NeuralNetworkParameters: params

BLAS.set_num_threads(1)

function parameters(::Type{T}) where {T}
    rng = Xoshiro(1)
    NetworkParameters((L1 = (weight = rand(rng, GO.StiefelManifold{T}, 64, 8),),
        L2 = (W = randn(rng, T, 64, 64), b = zeros(T, 64)),
        L3 = (W = randn(rng, T, 64, 64), b = zeros(T, 64))))
end

function gradient(::Type{T}) where {T}
    rng = Xoshiro(2)
    (L1 = (weight = rand(rng, T, 64, 8) ./ 100,),
        L2 = (W = rand(rng, T, 64, 64) ./ 100, b = rand(rng, T, 64) ./ 100),
        L3 = (W = rand(rng, T, 64, 64) ./ 100, b = rand(rng, T, 64) ./ 100))
end

for T in (Float64, Float32),
    method in (GO.GradientMethod(), GO.MomentumMethod(), GO.Adam(T))

    ps = parameters(T)
    dp = gradient(T)
    opt = Optimizer(method, ps)
    λY = GlobalSection(ps)
    optimization_step!(opt, λY, ps, dp)
    optimization_step!(opt, λY, ps, dp)
    bytes = minimum(@allocated(optimization_step!(opt, λY, ps, dp)) for _ in 1:5)
    time = minimum(@elapsed(optimization_step!(opt, λY, ps, dp)) for _ in 1:200)
    println(rpad(nameof(typeof(method)), 16), rpad(T, 9), "alloc/step = ",
        round(bytes / 1024; digits = 1), " KiB   t/step = ", round(time * 1e6; digits = 1),
        " μs")
end
