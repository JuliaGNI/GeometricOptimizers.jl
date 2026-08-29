# Type piracy, as a property rather than a list.
#
# Goal 2 of the ecosystem plan is "no type piracy in the GML ecosystem", and this package was the
# last holder: eight methods as of 0.6.0, on `SimpleSolvers`' `Gradient`, `GradientAutodiff`,
# `GradientFunction`, `Hessian` and `alloc_h` and on `GeometricBase`' `l2norm`, none of them
# dispatching on a type this package owns. Five went upstream -- `L2norm` over a parameter set into
# `NeuralNetworkParameters` 0.3.0 proper, the rest to
# `SimpleSolvers` 0.13.2's `ext/SimpleSolversNeuralNetworkParametersExt.jl` -- one moved to
# `SimpleSolvers` proper, and two were de-pirated in place -- `RiemannianGradient` for the projecting
# functor, and this package's own `Hessian` types for the call-with-a-cache error.
#
# A list of sites goes stale, and this package's own audit did, twice and in both directions: the
# 0.6.0 changelog counted one open site where there were five, and recorded two as open after they had
# been fixed. A test does not go stale. Anything added here that pirates a method fails this, whether
# or not anybody remembers to update a table.
#
# Only the piracy check. `Aqua.test_all` would also run `test_ambiguities`, which reports 139 -- almost
# all of them between this package's structured matrices and `LinearAlgebra`/`ArrayLayouts` methods on
# `AbstractMatrix`, and none of them touched by the piracy work. That is its own piece of work and not
# this one; turning it on before then would only mean a test that is expected to fail.

using Aqua
using GeometricOptimizers
using Test

# `test_piracies`, plural: `Aqua.test_piracy` was renamed in 0.8 and no longer exists.
Aqua.test_piracies(GeometricOptimizers)

# Where each of the eight went, asserted at the seam. `Aqua.test_piracies` above says that this package
# defines no pirated method; these say that the methods still *exist*, and in a package that owns one
# side of the signature. Without them a move upstream is indistinguishable from a deletion, which is
# how a capability gets lost in a de-piracy pass.
#
# The two seams land in different packages, and for the same reason in both cases — whichever package
# can *test* the method. `GradientAutodiff`, `GradientFunction` and `alloc_h` are `SimpleSolvers`',
# which owns the generics. `L2norm` over a parameter set is `NeuralNetworkParameters`'
# (`src/norms.jl`), which owns the type and the walk the method is written in;
# `NNP/test/geometric_base_tests.jl` pins its behaviour. What is asserted here is only that this
# package reaches both, since a solve that quietly fell back to a different `l2norm` would still run.

using GeometricBase
using GeometricBase.Utils: L2norm, l2norm
using NeuralNetworkParameters
using NeuralNetworkParameters: NetworkParameters, flatlength
using SimpleSolvers
using SimpleSolvers: Gradient, GradientAutodiff, GradientFunction, alloc_h

@testset "the five methods that went upstream are upstream" begin
    ps = NetworkParameters((L1 = (W = [3.0 0.0; 0.0 4.0], b = [0.0, 0.0]), L2 = (W = [0.0 0.0], b = [12.0])))

    ss_ext = Base.get_extension(SimpleSolvers, :SimpleSolversNeuralNetworkParametersExt)
    @test ss_ext isa Module

    # `NeuralNetworkParameters` itself and not an extension of it: `GeometricBase` is a hard
    # dependency there, which this ecosystem takes anyway.
    @test which(L2norm, Tuple{typeof(ps)}).module === NeuralNetworkParameters
    for m in (which(GradientAutodiff, Tuple{typeof(l2norm),typeof(ps)}),
              which(GradientFunction, Tuple{typeof(l2norm),Function,typeof(ps)}),
              which(alloc_h, Tuple{typeof(ps)}))
        @test m.module === ss_ext
    end

    # And the sixth, which went to `SimpleSolvers` proper rather than to its extension.
    @test which(GradientAutodiff, Tuple{typeof(l2norm),Matrix{Float64}}).module === SimpleSolvers

    # The quadrature sum survived the move, and so did this package's leaf methods: `l2norm` of a
    # lift is over the free parameters, which is what the fold upstream calls at every leaf.
    @test l2norm(ps) ≈ 13.0
    @test size(alloc_h(ps)) == (flatlength(ps), flatlength(ps))
end

@testset "the two that stayed are owned" begin
    Y = StiefelManifold([1.0 0.0; 0.0 1.0; 0.0 0.0])
    ps = NetworkParameters((L1 = (weight = Y,),))

    # The projecting functor: wrapped, not pirated, and `Optimizer` is what wraps.
    F(p) = l2norm(p)
    opt = Optimizer(ps, F; algorithm = GradientMethod())
    @test gradient(opt) isa RiemannianGradient
    @test gradient(opt).gradient isa GradientAutodiff
    # Idempotent, so handing `Optimizer` a gradient that is already wrapped is not a second wrapper.
    @test RiemannianGradient(gradient(opt)) === gradient(opt)

    # A `Manifold` needs no wrapper: `Manifold` is this package's type, so the method on the abstract
    # `Gradient` is already owned.
    optY = Optimizer(Y, F; algorithm = GradientMethod())
    @test gradient(optY) isa GradientAutodiff
    @test which(gradient(optY), Tuple{typeof(Y)}).module === GeometricOptimizers

    # A plain vector, likewise.
    optv = Optimizer(ones(3), F; algorithm = GradientMethod())
    @test optv.gradient isa GradientAutodiff

    # And the Hessian functor's error, now on this package's own Hessian types.
    @test_throws ErrorException HessianBFGS(F, ones(3))(zeros(3, 3), ones(3))
end

# `Optimizer` never builds this one, because a plain `Matrix` is not an `OptimizerSolution` -- the
# union is `AbstractVector`, `Manifold`, `NetworkParameters` -- and `_riemannian_gradient` has no
# `Matrix` method either. It is here for a caller who wraps by hand, which is why the type is
# exported, and it is the owned replacement for the `(::Gradient{T})(::Matrix{T})` this release
# deletes. Asserted so that it is covered rather than merely present: a method nothing reaches is
# indistinguishable from one that no longer works.
@testset "the matrix functor, which `Optimizer` does not build" begin
    x = [1.0 2.0; 3.0 4.0]
    G(A) = sum(abs2, A)
    grad = RiemannianGradient(GradientAutodiff(G, x))

    @test grad(x) ≈ rgrad(x, 2x)
    # a `Matrix` is not an `OptimizerSolution`, which is why nothing here builds the wrapper for one
    @test !(x isa GeometricOptimizers.OptimizerSolution)
end
