using GeometricOptimizers
using Test

# Julia only errors on a dangling `export` when the name is actually *resolved*, so an exported name
# that nothing defines is silent: the package loads, the docs build, the suite passes, and
# `using GeometricOptimizers; BFGSOptimizer()` is an `UndefVarError` for the user. That is how
# `NewtonOptimizer`, `BFGSOptimizer` and `DFPOptimizer` — transcribed from SimpleSolvers' export
# list in `0eab6b1`, and already dead there — survived from March 2026 until they were removed by
# hand. The check below is one line and it closes the class rather than those three instances.
@testset "every exported name is defined" begin
    @test isempty(filter(n -> !isdefined(GeometricOptimizers, n), names(GeometricOptimizers)))
end

# A spot check on the names a user reaches for first, so that a rename which drops one from the
# export list fails here and not in a downstream package. `Optimizer`'s default `algorithm` is
# `BFGS()`, which is the reason these in particular are exported rather than internal.
@testset "the optimizer methods and their states are exported" begin
    for name in (:Newton, :BFGS, :DFP,
        :GradientMethod, :MomentumMethod, :Adam,
        :NewtonOptimizerState, :BFGSState, :DFPState,
        :GradientState, :MomentumState, :AdamState)
        @test name in names(GeometricOptimizers)
    end

    # The caches are the half that stays internal, for every method alike.
    for name in (:BFGSCache, :DFPCache, :NewtonOptimizerCache,
        :GradientCache, :MomentumCache, :AdamCache)
        @test isdefined(GeometricOptimizers, name)
        @test !(name in names(GeometricOptimizers))
    end
end
