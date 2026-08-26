# What a `solve!` allocates, per shape of solution and per quasi-Newton method.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/optimizer_allocations.jl
#
# This is the harness behind the table in the 0.6.0 CHANGELOG entry on the flat buffers, and it exists
# because of the rule the *Open Issues* preamble states: treat a number as reproducible only where the
# harness that produced it is named. It is also the answer to the obvious cheaper measurement, which is
# wrong — `@allocated update!(cache, state, x, g)` needs one call to compile and one to measure, and
# calling `update!` twice at one iterate leaves `Δg` identically zero, because the cache advances
# `state.ḡ` as soon as it has used it. The `γᵀQγ` and both `outer!`s sit inside the
# `curvature_is_usable` branch, so such a measurement reports the cost of *not* running them.
#
# A whole `solve!` of a fixed length has no such trap: the branch fires on the iterations that earn it,
# and every other cost of a step is in the figure too. Most of what remains is not the flat buffers and
# is untouched by that release — the gradient evaluation, and the parameter trees that `global_rep`,
# `retraction_differential` and the elementwise walks build per line-search evaluation.
#
# `test/flat_buffer_allocations.jl` is the complement: it pins the individual sites at exactly zero,
# which is the claim being made, where this script says what that is worth end to end.

using GeometricOptimizers
using NeuralNetworkParameters: NetworkParameters
using SimpleSolvers: Static
import Random

const N, n, m = 6, 3, 4
const ITERATIONS = 20

Random.seed!(1234)
const B = randn(N, m)

# One problem per member of `OptimizerSolution`. The manifold blocks are what make the flat/ambient
# distinction bite: `Q` is sized by the flattening of the horizontal lift and the parameters are not.
vector_problem() = (randn(Random.Xoshiro(3), 12), v -> sum(abs2, v))

manifold_problem() = (rand(Random.Xoshiro(4), StiefelManifold{Float64}, N, n),
                      Y -> sum(abs2, Y * ones(n, m) .- B) / 2)

namedtuple_problem() = ((Y = rand(Random.Xoshiro(1), StiefelManifold{Float64}, N, n),
                         W = randn(Random.Xoshiro(2), n, m), b = zeros(N)),
                        ps -> sum(abs2, ps.Y * ps.W .+ ps.b .- B) / 2)

container_problem() = let (ps, _) = namedtuple_problem()
    (NetworkParameters((L1 = (Y = ps.Y,), L2 = (W = ps.W, b = ps.b))),
     ps -> sum(abs2, ps.L1.Y * ps.L2.W .+ ps.L2.b .- B) / 2)
end

const PROBLEMS = (("Vector", vector_problem), ("Manifold", manifold_problem),
                  ("NamedTuple", namedtuple_problem), ("container", container_problem))

function solve_once(x, F, algorithm)
    Random.seed!(1234)
    optimizer = Optimizer(x, F; algorithm = algorithm, linesearch = Static(0.1),
                          max_iterations = ITERATIONS)
    solve!(x, OptimizerState(algorithm, x), optimizer)
end

function main()
    println("solve!, ", ITERATIONS, " iterations, bytes allocated")
    for algorithm in (BFGS(), DFP()), (name, make) in PROBLEMS
        label = rpad(string(nameof(typeof(algorithm)), " ", name), 24)
        # a fresh problem for each of the two calls: `solve!` writes into `x`, so measuring the second
        # call on the first call's answer would be measuring a solve that starts at the minimum
        try
            x, F = make()
            solve_once(x, F, algorithm)                      # compile
            x, F = make()
            println("  ", label, @allocated(solve_once(x, F, algorithm)))
        catch e
            # `main` before 0.6.0 has no method for a container; say so rather than dying half way
            println("  ", label, "unsupported (", nameof(typeof(e)), ")")
        end
    end
end

main()
