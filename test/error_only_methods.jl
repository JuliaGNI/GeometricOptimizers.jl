# What is left of the methods that existed only to raise an error.
#
# Two guards stay: `similar` and `fill!` on a `Manifold`, because without them `Base`'s generic
# methods build an array of the point's shape that is not a point. So does `Newton`'s scope check,
# because without it `Newton` runs on a `Manifold` in the ambient space with no error. The others are
# gone, and a call that reached one now gets `Base`'s `MethodError`.
using GeometricOptimizers
using GeometricOptimizers: NoHessian, OptimizerCache, hessian, inverse_hessian,
                           OptimizerMethod
using Test
import Random

Random.seed!(1234)

struct MethodWithoutState <: OptimizerMethod end

@testset "the manifold guards still refuse, $T" for T in (Float32, Float64)
    for Y in (rand(StiefelManifold{T}, 5, 2), rand(GrassmannManifold{T}, 5, 2))
        @test_throws ErrorException similar(Y)
        @test_throws ErrorException fill!(Y, zero(T))
    end
end

@testset "Newton refuses a point and a parameter set, $T" for T in (Float32, Float64)
    for x in (rand(StiefelManifold{T}, 6, 3), rand(GrassmannManifold{T}, 6, 3),
        NetworkParameters((W = rand(T, 3, 3),)))
        @test_throws "Newton optimizes an AbstractVector only" OptimizerState(Newton(), x)
        @test_throws "Newton optimizes an AbstractVector only" OptimizerCache(Newton(), x)
        @test_throws "Newton optimizes an AbstractVector only" OptimizerState(
            Newton(), x, GeometricOptimizers._zero(x))
    end
    @test OptimizerState(Newton(), T[1, 2, 3]) isa NewtonState{T}
end

@testset "a deleted error-only method gives a MethodError, $T" for T in (Float32, Float64)
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
