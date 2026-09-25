# The iterates `solve!` produces for the three first-order methods on the SVD problem of
# `test/optimizer_convergence/svd_optim.jl`, as bit patterns.
#
# Run it on two trees and compare the output: equal lines mean `solve!` took the same steps to the
# last bit. The training step (`optimization_step!`) shares its state updates with `solve!`, so this is
# the check that a change to those updates did not move `solve!`.
#
#     julia --startup-file=no --project=. scripts/svd_first_order_iterates.jl
using GeometricOptimizers
using GeometricOptimizers: StiefelManifold
using LinearAlgebra: norm
import Random

const A = include(joinpath(
    @__DIR__, "..", "test", "optimizer_convergence", "svd_matrix.jl"))
objective(ps::NetworkParameters) = norm(A - ps.w₁ * ps.w₂' * A)

function starting_point(n)
    Random.seed!(1234)
    NetworkParameters((w₁ = rand(StiefelManifold, size(A, 1), n),
        w₂ = rand(StiefelManifold, size(A, 1), n)))
end

for retraction in (GeometricOptimizers.Geodesic(), GeometricOptimizers.Cayley())
    for algorithm in (GradientMethod(), MomentumMethod(), Adam())
        ps = starting_point(3)
        state = OptimizerState(algorithm, ps)
        optimizer = Optimizer(
            ps, objective; retraction = retraction, algorithm = algorithm,
            linesearch = Static(0.01), max_iterations = 1000, warn_iterations = 0)
        solve!(ps, state, optimizer)
        bits = hash((reinterpret(UInt64, vec(Matrix(ps.w₁))),
            reinterpret(UInt64, vec(Matrix(ps.w₂)))))
        println(rpad(nameof(typeof(retraction)), 10), rpad(nameof(typeof(algorithm)), 16),
            "iterations = ", GeometricOptimizers.iteration_number(state),
            "  f = ", objective(ps), "  bits = ", string(bits; base = 16))
    end
end
