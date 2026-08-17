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

    # The caches are the half that stays internal, for every method alike: they are `solver_step!`
    # scratch, and nothing outside a step should be reading one. `GeometricMachineLearning`
    # re-exported the three first-order ones until 0.5, which is the only reason exporting them was
    # ever considered; it reaches them qualified now.
    for name in (:BFGSCache, :DFPCache, :NewtonOptimizerCache,
        :GradientCache, :MomentumCache, :AdamCache, :OptimizerCache)
        @test isdefined(GeometricOptimizers, name)
        @test !(name in names(GeometricOptimizers))
    end
end

# The interface a package that walks its own parameter tree needs -- `GeometricMachineLearning` does,
# for a neural network, and imports every one of these. They were internal until 0.4, so GML named
# them `GeometricOptimizers.`-qualified and re-exported them; that is what this list replaces. See
# the *Exports* entry in the changelog.
@testset "the manifold, section and retraction interface is exported" begin
    for name in (
        # the geometry
        :Manifold, :StiefelManifold, :GrassmannManifold,
        :rgrad, :metric, :check, :Ω,
        # the structured matrices and the lifts
        :SkewSymMatrix, :SymmetricMatrix, :LowerTriangular, :UpperTriangular,
        :AbstractTriangular, :StiefelProjection,
        :AbstractLieAlgHorMatrix, :StiefelLieAlgHorMatrix, :GrassmannLieAlgHorMatrix,
        # global sections
        :GlobalSection, :global_section, :global_rep,
        :apply_section, :apply_section!, :update_section!,
        # retractions
        :AbstractRetraction, :Geodesic, :Cayley, :geodesic, :cayley, :retraction,
        # the optimizer types a caller dispatches on
        :OptimizerMethod, :OptimizerState, :OptimizerSolution)
        @test name in names(GeometricOptimizers)
    end
end
