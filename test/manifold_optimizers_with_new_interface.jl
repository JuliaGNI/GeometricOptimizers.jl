using GeometricOptimizers
using SimpleSolvers
using Test
using SimpleSolvers: l2norm
using Random: seed!

seed!(123)

f(x::StiefelManifold) = l2norm(vec(x), [0.0, 0.0, 1.2])

const solution = [0., 0., 1.]

x = rand(StiefelManifold, 3, 1)

o = EuclideanOptimizer(x, f; algorithm=GradientMethod(), linesearch=Static(0.1))

solve!(x, GradientState(x), o)
@test x ≈ solution
