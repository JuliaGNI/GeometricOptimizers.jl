# Run by `test/integration/adam_step_allocations.jl` in a process of its own: the `Adam` step on small
# sets of one weight and a bias, a dense weight first, then each vector-storage weight. Whether the walk
# over the leaves specialises on a function it is handed can depend on what the process compiled
# before, so the order is fixed here and the process is fresh. Prints one line per set: its name and
# the bytes of one step after warm-up, behind a function barrier.
using GeometricOptimizers
using NeuralNetworkParameters: NetworkParameters
using Random: Xoshiro

include(joinpath(@__DIR__, "parameter_set_step.jl"))

function weights(::Type{T}) where {T}
    (
        "Matrix" => () -> randn(Xoshiro(1), T, 16, 16),
        "SymmetricMatrix" => () -> SymmetricMatrix(randn(Xoshiro(1), T, 136), 16),
        "SkewSymMatrix" => () -> SkewSymMatrix(randn(Xoshiro(1), T, 120), 16),
        "StrictlyLowerTriangular" =>
            () -> StrictlyLowerTriangular(randn(Xoshiro(1), T, 120), 16),
        "StrictlyUpperTriangular" =>
            () -> StrictlyUpperTriangular(randn(Xoshiro(1), T, 120), 16))
end

for T in (Float32, Float64), (name, W) in weights(T)

    for (shape, x) in (
        "flat" => NetworkParameters((W = W(), b = randn(Xoshiro(3), T, 16))),
        "nested" => NetworkParameters((L1 = (W = W(), b = randn(Xoshiro(3), T, 16)),)))
        dp = random_gradient(x)
        opt = TrainingOptimizer(x; algorithm = Adam())
        println("$T $name $shape: ", minimum(step_allocations(x, opt, dp) for _ in 1:3))
    end
end
