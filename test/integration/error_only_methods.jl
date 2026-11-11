# The guards that raise an error by design, and the calls that have no method.
#
# Two guards are on a `Manifold`, `similar` and `fill!`, because without them `Base`'s generic methods
# build an array of the point's shape that is not a point. `Newton`'s scope check is the third: without
# it `Newton` on a `Manifold` or a parameter set fails with an error that does not state the scope.
# The fourth is the weight-decay guard of `AdamWithEuclideanDecay`.
# The other refusals below are calls with no method, which raise `Base`'s `MethodError`.
using GeometricOptimizers
using GeometricOptimizers: NoHessian, OptimizerCache, hessian, inverse_hessian,
                           OptimizerMethod
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

struct MethodWithoutState <: OptimizerMethod end

@testset "the manifold guards still refuse, $T" for T in REAL_ELTYPES
    for Y in (rand(StiefelManifold{T}, 5, 2), rand(GrassmannManifold{T}, 5, 2))
        @test_throws ErrorException similar(Y)
        @test_throws ErrorException fill!(Y, zero(T))
    end
end

@testset "Newton refuses a point and a parameter set, $T" for T in REAL_ELTYPES
    for x in (rand(StiefelManifold{T}, 6, 3), rand(GrassmannManifold{T}, 6, 3),
        NetworkParameters((W = rand(T, 3, 3),)))
        @test_throws "Newton optimizes an AbstractVector only" OptimizerState(Newton(), x)
        @test_throws "Newton optimizes an AbstractVector only" OptimizerCache(Newton(), x)
        @test_throws "Newton optimizes an AbstractVector only" OptimizerState(
            Newton(), x, GeometricOptimizers._zero(x))
    end
    @test OptimizerState(Newton(), T[1, 2, 3]) isa NewtonState{T}
end

@testset "the optimizers that do not step a SymplecticStiefelManifold refuse it, $T" for T in REAL_ELTYPES
    # The base point of `Sp(2, 4)`, and not a draw: `rand` of this type can raise in `Float32`.
    U = SymplecticStiefelManifold(T[1 0; 0 0; 0 1; 0 0])

    # Every first-order and quasi-Newton method is built on the point.
    for method in (Adam(), GradientMethod(), MomentumMethod(), BFGS(), DFP())
        @test OptimizerCache(method, U) isa OptimizerCache{T}
        @test OptimizerState(method, U) isa OptimizerState{T}
    end

    # This method refuses the point — bare, at `λ = 0`, or after an ordinary leaf in a container —
    # with an `ArgumentError` of its own: the manifold is not compact, so `rgrad(U, λU)` does not
    # vanish away from orthosymplectic points, and a horizontal lift cannot apply `-λU` entrywise.
    @test_throws ArgumentError TrainingOptimizer(U; algorithm = AdamWithEuclideanDecay())
    @test_throws ArgumentError OptimizerCache(AdamWithEuclideanDecay(; λ = zero(T)), U)
    @test_throws ArgumentError TrainingOptimizer(
        NetworkParameters((b = ones(T, 3), U = U)); algorithm = AdamWithEuclideanDecay())

    # These take a `StiefelManifold` by their signature, so the call has no method.
    for method in (ScalarMomentAdam(), DirectMomentum(), DirectScalarMomentAdam(),
        FactoredStiefelAdam())
        @test_throws MethodError TrainingOptimizer(U; algorithm = method, linesearch = T(1e-2))
    end
end

@testset "a deleted error-only method gives a MethodError, $T" for T in REAL_ELTYPES
    # `ScalarMomentAdam` takes a single `StiefelManifold`, by its signature
    for x in (T[1, 2, 3], rand(GrassmannManifold{T}, 4, 2), NetworkParameters((W = rand(T, 3),)))
        @test_throws MethodError OptimizerState(ScalarMomentAdam(), x)
        @test_throws MethodError OptimizerState(ScalarMomentAdam(), x, x)
        @test_throws MethodError OptimizerCache(ScalarMomentAdam(), x)
    end
    Y = rand(StiefelManifold{T}, 5, 2)
    @test_throws MethodError ScalarMomentAdamState(Y, rand(GrassmannLieAlgHorMatrix{T}, 5, 2))
    @test OptimizerState(ScalarMomentAdam(), Y) isa ScalarMomentAdamState{T}

    # a method with no state of its own
    @test_throws MethodError OptimizerState(MethodWithoutState(), T[1, 2, 3])

    # the Hessian placeholder of the methods that build none has no functor
    @test_throws MethodError NoHessian{T}()(zeros(T, 3, 3), ones(T, 3))

    # a quasi-Newton cache holds neither the Hessian nor its inverse
    cache = OptimizerCache(BFGS(), T[1, 2, 3])
    @test_throws MethodError hessian(cache)
    @test_throws MethodError inverse_hessian(cache)
end
