# The iterates `solve!` produces for every method of `test/verification/svd_optim.jl`, and for a
# vector problem and a set of structured parameters besides, in both precisions, as bit patterns.
#
# Run it on two trees and compare the output: equal lines mean `solve!` took the same steps to the
# last bit. Each line names a solve and gives its iteration count and a hash of the bits of the final
# iterate, of the objective at every iteration (`store_trace = true`) and of the parameters at every
# evaluation of the objective. A hash is only comparable with one taken on the same Julia and the
# same machine.
#
#     julia --startup-file=no --project=. scripts/svd_optim_iterates.jl
using GeometricOptimizers
using GeometricOptimizers: Geodesic, Cayley, iteration_number, trace
using NeuralNetworkParameters: flatten
using SimpleSolvers: Static, Backtracking, Bisection, Quadratic, BierlaireQuadratic,
                     StrongWolfe
using LinearAlgebra: norm
import Random

const A = include(joinpath(@__DIR__, "..", "test", "helpers", "svd_matrix.jl"))

bits(v::AbstractVector{Float64}) = reinterpret(UInt64, v)
bits(v::AbstractVector{Float32}) = reinterpret(UInt32, v)
flat(x::AbstractVector) = x
flat(x) = flatten(x)[1]

# The trace holds no parameters, so the objective is wrapped to hash them at every call: `solve!`
# evaluates it at each trial point of the line search and at each iterate. A call at another element
# type, such as one of automatic differentiation, is skipped.
struct Recorded{F} <: Function
    f::F
    hashes::Vector{UInt}
end
function (r::Recorded)(x)
    v = flat(x)
    eltype(v) <: Union{Float32, Float64} && push!(r.hashes, hash(bits(v)))
    r.f(x)
end

function record(label, x, state, result, recorded)
    fs = [entry.f for entry in trace(result)]
    h = hash((bits(flat(x)), bits(fs), recorded.hashes))
    println(rpad(label, 64), "iterations = ", lpad(iteration_number(state), 5),
        "  bits = ", string(h; base = 16))
end

function run!(label, x, objective, algorithm, linesearch, max_iterations;
        retraction = Cayley())
    state = OptimizerState(algorithm, x)
    recorded = Recorded(objective, UInt[])
    optimizer = Optimizer(x, recorded; retraction = retraction, algorithm = algorithm,
        linesearch = linesearch, max_iterations = max_iterations, warn_iterations = 0,
        store_trace = true)
    record(label, x, state, solve!(x, state, optimizer), recorded)
end

# The SVD problem, as `test/verification/svd_optim.jl` poses it, in `T`.
function svd_point(::Type{T}, seed) where {T}
    Random.seed!(seed)
    NetworkParameters((w₁ = rand(StiefelManifold{T}, size(A, 1), 3),
        w₂ = rand(StiefelManifold{T}, size(A, 1), 3)))
end
svd_objective(Aₜ) = ps -> norm(Aₜ - ps.w₁ * ps.w₂' * Aₜ)

# the fixed-budget first-order solves, in both precisions
for T in (Float64, Float32), retraction in (Geodesic(), Cayley())

    for algorithm in (GradientMethod(), MomentumMethod(), Adam())
        run!("svd $T $(nameof(typeof(retraction))) $(nameof(typeof(algorithm)))",
            svd_point(T, 1234), svd_objective(T.(A)), algorithm, Static(T(0.01)), 1000;
            retraction)
    end
end

# the converging quasi-Newton solves, and the A1b seeds
for retraction in (Geodesic(), Cayley())
    for (algorithm, name, linesearch) in ((BFGS(), "Backtracking", Backtracking(Float64)),
        (BFGS(), "Backtracking(expand)", Backtracking(Float64; expand = true)),
        (BFGS(), "Bisection", Bisection(Float64)), (
        BFGS(), "Quadratic", Quadratic(Float64)),
        (BFGS(), "BierlaireQuadratic", BierlaireQuadratic(Float64)),
        (DFP(), "Bisection", Bisection(Float64)),
        (DFP(), "StrongWolfe(c₂ = 0.1)", StrongWolfe(Float64; c₂ = 0.1)))
        run!(
            "svd Float64 $(nameof(typeof(retraction))) $(nameof(typeof(algorithm))) $name",
            svd_point(Float64, 1234), svd_objective(A), algorithm, linesearch, 5000; retraction)
    end
end
for linesearch in (Quadratic(Float64), BierlaireQuadratic(Float64)), seed in (2, 8)

    run!("svd A1b seed $seed $(nameof(typeof(linesearch)))", svd_point(Float64, seed),
        svd_objective(A), BFGS(), linesearch, 5000)
end

# a vector problem, every method
function rosenbrock(x)
    sum((1 .- x[1:(end - 1)]) .^ 2 .+ 10 .* (x[2:end] .- x[1:(end - 1)] .^ 2) .^ 2)
end
for T in (Float64, Float32)
    for (algorithm, linesearch) in ((GradientMethod(), Static(T(0.01))),
        (MomentumMethod(), Static(T(0.01))), (Adam(), Static(T(0.01))),
        (AdamWithEuclideanDecay(), Static(T(0.01))), (BFGS(), Backtracking(T)),
        (DFP(), Backtracking(T)), (Newton(), Backtracking(T)))
        run!("vector $T $(nameof(typeof(algorithm)))", T[-1.2, 1.0, 0.5, -0.3],
            rosenbrock, algorithm, linesearch, 200)
    end
end

# every structured leaf a parameter set may hold, and a bare Stiefel point
function structured_point(::Type{T}) where {T}
    Random.seed!(4321)
    NetworkParameters((S = rand(SymmetricMatrix{T}, 3), K = rand(SkewSymMatrix{T}, 3),
        L = rand(StrictlyLowerTriangular{T}, 3), U = rand(StrictlyUpperTriangular{T}, 3),
        Y = rand(StiefelManifold{T}, 5, 2), Z = rand(GrassmannManifold{T}, 5, 2),
        W = rand(T, 2, 3)))
end
function structured_objective(ps)
    sum(abs2, parent(ps.S) .- 1) + sum(abs2, parent(ps.K) .- 1) +
    sum(abs2, parent(ps.L) .- 1) +
    sum(abs2, parent(ps.U) .- 1) + sum(abs2, ps.Y .- 1) + sum(abs2, ps.Z .- 1) +
    sum(abs2, ps.W .- 1)
end
for T in (Float64, Float32)
    for (algorithm, linesearch) in ((GradientMethod(), Static(T(0.01))),
        (MomentumMethod(), Static(T(0.01))), (Adam(), Static(T(0.01))),
        (AdamWithEuclideanDecay(), Static(T(0.01))), (BFGS(), Backtracking(T)),
        (DFP(), Backtracking(T)))
        run!("structured $T $(nameof(typeof(algorithm)))", structured_point(T),
            structured_objective, algorithm, linesearch, 100)
    end
    Random.seed!(99)
    run!("stiefel $T ScalarMomentAdam", rand(StiefelManifold{T}, 5, 2),
        Y -> sum(abs2, Y .- 1), ScalarMomentAdam(), Static(T(0.01)), 100)
end
