using GeometricOptimizers
using GeometricOptimizers: solver_step!
using GeometricOptimizers: cache, update_section!, section, direction, hessian, solution, cayley, problem
using SimpleSolvers
using Test
using SimpleSolvers: l2norm
using Random: seed!

seed!(1234)

f(x::StiefelManifold) = l2norm(vec(x), [0.0, 0.0, 1.2])

const sol = StiefelManifold([0.; 0.; 1.;;])
const α = 0.1

x = StiefelManifold([0.; sqrt(.5); sqrt(.5);;])

opt = EuclideanOptimizer(x, f; algorithm=GradientMethod(), linesearch=Static(α))

state = GradientState(x)

solve!(x, GradientState(x), opt)
@test isapprox(x, sol; rtol=1e-6)
