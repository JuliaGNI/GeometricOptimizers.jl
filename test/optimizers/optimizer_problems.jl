using GeometricOptimizers
using GeometricOptimizers: gradient, value
import Random
import SimpleSolvers
using Test

include("../helpers/eltypes.jl")

function F(x)
    1 + sum(x .^ 2)
end

function G!(g, x)
    g .= 0
    for i in eachindex(x, g)
        g[i] = 2x[i]
    end
end

const n = 2

# test if the correct value is returned and if the counter goes up
function return_correct_value(obj1::OptimizerProblem, obj2::OptimizerProblem, x::AbstractVector, y::Number)
    @test eltype(value(obj2, x)) == eltype(x)
    @test value(obj1, x) == value(obj2, x) == y
end

function return_correct_gradients(obj1::OptimizerProblem, obj2::OptimizerProblem,
        x::AbstractVector, z::AbstractVector)
    @test eltype(gradient(obj2, x)) == eltype(x)
    @test gradient(obj2, x) == z
    @test_throws "There is no gradient stored in this `OptimizerProblem`!" gradient(obj1, x)
end

@testset "an OptimizerProblem returns the value and the gradient it was given, $T" for T in REAL_ELTYPES
    x = rand(Random.Xoshiro(123), T, n)
    f = F(x)
    g = SimpleSolvers.alloc_g(x)

    G!(g, x)

    obj1 = OptimizerProblem(F, zero(x))
    obj2 = OptimizerProblem(F, G!, zero(x))

    # test value-related functionality (clear Objective object after every run)
    for (x_temp, y_temp) in zip((x, 2x), (f, F(2x)))
        return_correct_value(obj1, obj2, x_temp, y_temp)
    end

    # test gradient-related functionality (clear Objective object after every run); `4x` is the
    # gradient `2(2x)` exactly, because a factor of 2 does not round
    for (x_temp, z_temp) in zip((x, 2x), (g, 4x))
        return_correct_gradients(obj1, obj2, x_temp, z_temp)
    end
end
