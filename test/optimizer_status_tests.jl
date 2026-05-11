
using SimpleSolvers
using Test

using Random: seed!
seed!(123)

include("optimizers_problems.jl")


n = 1
x = ones(n)
nl = QuasiNewtonOptimizer(x, F)

@test config(nl) == nl.config

solve!(x, nl)
