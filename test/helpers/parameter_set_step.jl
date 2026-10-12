# The fixtures of a training step on a parameter set, shared by `test/integration/adam_step_allocations.jl`,
# `test/helpers/adam_step_after_dense.jl`, `test/manifold_optimizers/scalar_moment_adam_optimizer.jl`
# and `scripts/parameter_set_step_cost.jl`: model A, a random gradient of the shape of a set, and the
# barrier that reads the bytes of one step.
using GeometricOptimizers
using NeuralNetworkParameters: NetworkParameters, mapparameters
using Random: Xoshiro

# "Model A": eight `Dense(64, 64)` layers and one `Dense(64, 1)`, 33 345 parameters.
function model_A(::Type{T}) where {T}
    rng = Xoshiro(1)
    layer(m, n) = (W = randn(rng, T, m, n) ./ T(8), b = randn(rng, T, m) ./ T(8))
    NetworkParameters(NamedTuple{ntuple(i -> Symbol(:L, i), 9)}(
        (ntuple(_ -> layer(64, 64), 8)..., layer(1, 64))))
end

# A Euclidean gradient of the shape of a leaf: a matrix of the leaf's own type with random storage
# where the leaf is a vector-storage matrix, and a dense array where it is a dense array or a manifold
# point.
function gradient_leaf(rng, x::Union{SymmetricMatrix, SkewSymMatrix, AbstractTriangular})
    typeof(x).name.wrapper(randn(rng, eltype(x), length(x.S)) ./ 100, x.n)
end
gradient_leaf(rng, x::AbstractArray{T}) where {T} = randn(rng, T, size(x)...) ./ 100

function random_gradient(x, seed = 6)
    rng = Xoshiro(seed)
    mapparameters(y -> gradient_leaf(rng, y), x)
end

# The barrier: concrete argument types, one warm-up step, then the measured step.
function step_allocations(x::X, opt::O, dp::D) where {X, O, D}
    optimization_step!(x, opt, dp)
    @allocated optimization_step!(x, opt, dp)
end
