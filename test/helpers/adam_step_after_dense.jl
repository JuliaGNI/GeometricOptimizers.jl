# Run by `test/integration/adam_step_allocations.jl` in a process of its own: the `Adam` step on small
# sets of one weight and a bias, a dense weight first, then each vector-storage weight. Whether the walk
# over the leaves specialises on a function it is handed can depend on what the process compiled
# before, so the order is fixed here and the process is fresh. Prints one line per set: its name and
# the bytes of one step after warm-up, behind a function barrier.
using GeometricOptimizers
using NeuralNetworkParameters: NetworkParameters, mapparameters
using Random: Xoshiro

function gradient_leaf(rng, x::Union{SymmetricMatrix, SkewSymMatrix, AbstractTriangular})
    typeof(x).name.wrapper(randn(rng, eltype(x), length(x.S)) ./ 100, x.n)
end
gradient_leaf(rng, x::AbstractArray{T}) where {T} = randn(rng, T, size(x)...) ./ 100
random_gradient(x) = (rng = Xoshiro(6); mapparameters(y -> gradient_leaf(rng, y), x))

function allocations(x::X, opt::O, dp::D) where {X, O, D}
    optimization_step!(x, opt, dp)
    @allocated optimization_step!(x, opt, dp)
end

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
        println("$T $name $shape: ", minimum(allocations(x, opt, dp) for _ in 1:3))
    end
end
