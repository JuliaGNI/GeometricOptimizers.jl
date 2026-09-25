using Test
using LinearAlgebra
using GeometricOptimizers
using GeometricOptimizers: Ω, metric, geodesic
import Random

Random.seed!(123)

function correct_format(n::Integer, N::Integer, T::DataType)
    A_skew = rand(SkewSymMatrix{T}, N)
    A_hor = StiefelLieAlgHorMatrix(A_skew, n)

    for i in 1:n
        for j in 1:N
            @test abs(A_hor[i, j] - A_skew[i, j]) < eps(T)
        end
    end

    for i in (n + 1):N
        for j in 1:n
            @test abs(A_hor[i, j] - A_skew[i, j]) < eps(T)
        end
        for j in (n + 1):N
            @test abs(A_hor[i, j]) < eps(T)
        end
    end
end

function metric_test(n::Integer, N::Integer, T::DataType)
    Y = rand(StiefelManifold{T}, N, n)
    Δ₁ = rgrad(Y, rand(T, N, n))
    Δ₂ = rgrad(Y, rand(T, N, n))
    @test T(0.5) * tr(Ω(Y, Δ₁)' * Ω(Y, Δ₂)) ≈ metric(Y, Δ₁, Δ₂)
    # A bare `Float64` literal anywhere in `metric` hands a `StiefelManifold{Float32}` a `Float64`
    # back, and returns nothing at all on a backend that has no `Float64`, e.g. Metal.
    @test metric(Y, Δ₁, Δ₂) isa T
end

# This multiplies the adjoint of a `StiefelManifold` by another `StiefelManifold` and checks the
# result equals `Y.A' * Z.A`, i.e. that both operands are unwrapped and their underlying storage is
# multiplied. It runs over the same `(N, n, T)` sweep as `correct_format` and `metric_test`.
function adjoint_mul_test(n::Integer, N::Integer, T::DataType)
    Y = rand(StiefelManifold{T}, N, n)
    Z = rand(StiefelManifold{T}, N, n)
    @test Y' * Z ≈ Y.A' * Z.A
end

for N in (20, 10)
    for n in (5, 3)
        for T in (Float64, Float32)
            correct_format(n, N, T)
            metric_test(n, N, T)
            adjoint_mul_test(n, N, T)
        end
    end
end

# The `rgrad` doctest above constructs a `StiefelManifold` of integer element type, so `metric` has
# to accept one. `(T(1) / 2)` does; `T(1//2)` would throw `InexactError` for `T = Int`.
let Y = StiefelManifold([1 0; 0 1; 0 0; 0 0]), Δ₁ = [1 2; 3 4; 5 6; 7 8],
    Δ₂ = [8 7; 6 5; 4 3; 2 1]

    @test metric(Y, Δ₁, Δ₂) isa Float64
end

# A row vector on the left is the one shape `*(::AbstractMatrix, ::OwnedMatrix)` does not settle
# on its own: `LinearAlgebra` has its own method for that left operand, narrower there and wider on
# the right, so neither wins. The two row-vector methods in `src/ambiguities.jl` settle it. `Y` is
# rectangular, so a method that swapped or dropped an operand would not conform.
@testset "a row vector times a StiefelManifold" begin
    for T in (Float32, Float64), N in (20, 10), n in (5, 3)
        Y = rand(StiefelManifold{T}, N, n)
        v = rand(T, N)

        @test v' * Y ≈ v' * Matrix(Y)
        @test transpose(v) * Y ≈ transpose(v) * Matrix(Y)
        @test size(v' * Y) == (1, n)
    end
end
