# The iteration count at which each first-order solve of the two-sphere problem of
# `test/manifold_linesearch_tests.jl` stops, under the default tolerances, and the stop measures of
# its last iterations. Run it on two trees to compare where a change to the stopping logic moves the
# stop.
#
#   julia --startup-file=no --project=<test environment> scripts/two_sphere_stop.jl [trace]
#
# With `trace`, it also prints the measures of the last iterations of
# `GradientMethod` + `BierlaireQuadratic` + `Cayley`.

using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, StiefelManifold, iteration_number, status
using NeuralNetworkParameters: NetworkParameters
using SimpleSolvers: Static, Backtracking, Bisection, Quadratic, BierlaireQuadratic,
                     StrongWolfe,
                     l2norm
import Random

const TARGET = [0.0, 0.0, 1.2]
const TARGET₂ = [0.0, 1.5, 0.0]
two_spheres(ps) = l2norm(vec(ps.w₁), TARGET) + l2norm(vec(ps.w₂), TARGET₂)

function ps₀()
    Random.seed!(1234)
    NetworkParameters((w₁ = StiefelManifold([0.0; sqrt(0.5); sqrt(0.5);;]),
        w₂ = StiefelManifold([sqrt(0.5); 0.0; sqrt(0.5);;])))
end

const LINESEARCHES = (
    Static(0.1), Backtracking(Float64), Backtracking(Float64; expand = true),
    Bisection(Float64), Quadratic(Float64), BierlaireQuadratic(Float64),
    StrongWolfe(Float64; c₂ = 0.1))

function solve_case(method, linesearch, retraction; store_trace = false)
    ps = ps₀()
    state = OptimizerState(method, ps)
    opt = Optimizer(ps, two_spheres; algorithm = method, linesearch = linesearch,
        retraction = retraction, max_iterations = 1000, store_trace = store_trace)
    result = solve!(ps, state, opt)
    (iteration_number(state), status(result), result)
end

println("Julia ", VERSION)
for method in (GradientMethod(), MomentumMethod(; α = 0.1)), ls in LINESEARCHES,
    retraction in (Geodesic(), Cayley())

    n, s, _ = solve_case(method, ls, retraction)
    println(rpad(nameof(typeof(method)), 16), rpad(nameof(typeof(ls)), 20),
        rpad(nameof(typeof(retraction)), 10), lpad(n, 5),
        "  x_conv=", s.x_converged, " f_conv=", s.f_converged, " g_conv=", s.g_converged)
end

if "trace" in ARGS
    n, s, result = solve_case(GradientMethod(), BierlaireQuadratic(Float64), Cayley();
        store_trace = true)
    println("\nGradientMethod + BierlaireQuadratic + Cayley: ", n, " iterations")
    for e in result.trace[max(1, end - 5):end]
        println(e)
    end
    println(s)
end
