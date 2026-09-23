# Ambiguities between two of this package's own methods, as a property rather than a list, and the
# sweeps that check every product and sum `src/ambiguities.jl` routes against its dense answer.
#
# The assertion below is that the set is empty rather than that it has some size. A count goes stale
# the moment a method is added, and it says nothing about which pairs are in it.

using GeometricOptimizers
using GeometricOptimizers: StiefelProjection, sr!
using LinearAlgebra: qr!
using Test
import Random

Random.seed!(1234)

@testset "no ambiguity between two of this package's own methods" begin
    isown(m) = parentmodule(m) === GeometricOptimizers
    own = filter(p -> isown(first(p)) && isown(last(p)),
        Test.detect_ambiguities(GeometricOptimizers; recursive = false))

    @test isempty(own)

    # Printed rather than merely counted, so a failure names the pair instead of a number.
    for (m₁, m₂) in own
        @info "own-vs-own ambiguity" m₁ m₂
    end
end

# One instance of every type that takes part, at three shapes: `6 × 6`, `6 × 2` with its adjoint,
# and `2 × 2`. A product is checked for every pair that conforms, so a rectangular operand meets
# every type on the side where it fits. A square `StiefelProjection` is the identity, so a method
# that dropped an operand or took the product the other way round would still agree with the dense
# product there; the rectangular one cannot hide that.
const N, n = 6, 2

qpoint(m, k) = StiefelManifold(Matrix(qr!(randn(m, m)).Q)[:, 1:k])

const OPERANDS = let
    Q = Matrix(qr!(randn(N, N)).Q)
    Y = StiefelManifold(Q)
    U = rand(SymplecticStiefelManifold, N, N)
    Uᵣ = rand(SymplecticStiefelManifold, N, n)
    Yᵣ = qpoint(N, n)
    S = sr!(randn(N, N ÷ 2 + 1)).S
    # `sr!` asserts an even number of columns, so `n ÷ 2 + 1` has to be even as well
    Sₙ = sr!(randn(n, n ÷ 2 + 1)).S
    ["Y" => Y, "Y'" => Y', "Yview" => StiefelManifold(view(Q, :, :)),
        "U" => U, "U'" => U', "S" => S, "inv(S)" => inv(S),
        "E" => StiefelProjection(N, N),
        "LowerTriangular" => rand(LowerTriangular{Float64}, N),
        "UpperTriangular" => rand(UpperTriangular{Float64}, N),
        "SkewSym" => rand(SkewSymMatrix, N), "Sym" => rand(SymmetricMatrix, N),
        "StiefelHor" => rand(StiefelLieAlgHorMatrix, N, N ÷ 2),
        "GrassmannHor" => rand(GrassmannLieAlgHorMatrix, N, N ÷ 2),
        # `6 × 2` and `2 × 6`
        "Yᵣ" => Yᵣ, "Yᵣ'" => Yᵣ', "Uᵣ" => Uᵣ, "Uᵣ'" => Uᵣ', "Eᵣ" => StiefelProjection(N, n),
        # `2 × 2` and `2 × 1`
        "Yₙ" => qpoint(n, 1), "Uₙ" => rand(SymplecticStiefelManifold, n, n), "Sₙ" => Sₙ,
        "inv(Sₙ)" => inv(Sₙ), "Eₙ" => StiefelProjection(n, 1),
        "LowerTriangularₙ" => rand(LowerTriangular{Float64}, n),
        "UpperTriangularₙ" => rand(UpperTriangular{Float64}, n),
        "SkewSymₙ" => rand(SkewSymMatrix, n), "Symₙ" => rand(SymmetricMatrix, n),
        "StiefelHorₙ" => rand(StiefelLieAlgHorMatrix, n, 1),
        "GrassmannHorₙ" => rand(GrassmannLieAlgHorMatrix, n, 1)]
end

@testset "a product of two owned matrices agrees with the dense product" begin
    for (lname, L) in OPERANDS, (rname, R) in OPERANDS

        size(L, 2) == size(R, 1) || continue
        @testset "$lname * $rname" begin
            @test L * R ≈ Matrix(L) * Matrix(R)
        end
    end
end

@testset "a vector and a row vector against an owned matrix" begin
    for (name, A) in OPERANDS
        @testset "$name" begin
            v, w = randn(size(A, 1)), randn(size(A, 2))
            @test v' * A ≈ v' * Matrix(A)
            @test transpose(v) * A ≈ transpose(v) * Matrix(A)
            @test A * w ≈ Matrix(A) * w
            M, P = randn(size(A)...), randn(size(A, 2), size(A, 1))
            @test A * P ≈ Matrix(A) * P
            @test P * A ≈ P * Matrix(A)
            @test A + M ≈ Matrix(A) + M
            @test M + A ≈ M + Matrix(A)
            @test A - M ≈ Matrix(A) - M
            @test M - A ≈ M - Matrix(A)
        end
    end
end

@testset "a sum or difference of two owned matrices agrees with the dense one" begin
    for (lname, L) in OPERANDS, (rname, R) in OPERANDS

        size(L) == size(R) || continue
        @testset "$lname ± $rname" begin
            @test L + R ≈ Matrix(L) + Matrix(R)
            @test L - R ≈ Matrix(L) - Matrix(R)
        end
    end
end

# An owned operand whose element type differs from the other's reaches the untyped fallback of its
# kernel, and so `LinearAlgebra`'s or `Base`'s own answer. That answer goes through `mul!`, and the
# `mul!` kernels of `SkewSymMatrix` and `SymmetricMatrix` take one element type only, so a product
# that reaches one of them with two element types raises a `MethodError`. The `broken` cases are
# those.
@testset "mixed element types" begin
    for A in (rand(SkewSymMatrix{Float32}, N), rand(SymmetricMatrix{Float32}, N),
        rand(LowerTriangular{Float32}, N), rand(StiefelLieAlgHorMatrix{Float32}, N, n),
        StiefelManifold(Float32.(Matrix(qr!(randn(N, N)).Q))))
        kernel_on_left = A isa Union{SkewSymMatrix, SymmetricMatrix}
        M, v = randn(N, N), randn(N)
        B = rand(SkewSymMatrix, N)
        @test A * M≈Matrix(A) * M broken=kernel_on_left
        @test M * A ≈ M * Matrix(A)
        @test A * v≈Matrix(A) * v broken=kernel_on_left
        @test v' * A≈v' * Matrix(A) broken=A isa SymmetricMatrix
        @test A + M ≈ Matrix(A) + M
        @test M + A ≈ M + Matrix(A)
        @test A * B≈Matrix(A) * Matrix(B) broken=kernel_on_left
        @test B * A≈Matrix(B) * Matrix(A) broken=true
    end
end

@testset "vcat and hcat of two StiefelProjections" begin
    E = StiefelProjection(5, 2)
    F = StiefelProjection(4, 2)

    @test vcat(E, F) ≈ vcat(Matrix(E), Matrix(F))
    @test hcat(E, E) ≈ hcat(Matrix(E), Matrix(E))
end

# The same-type and same-pair methods that keep a structure are narrower than the `(Owned, Owned)`
# ones, and so still answer.
@testset "the structure-preserving methods still win" begin
    A = rand(SkewSymMatrix, N)
    C = rand(StiefelLieAlgHorMatrix, N, N ÷ 2)

    @test C + A isa SkewSymMatrix
    @test A + C isa SkewSymMatrix
    @test C + A ≈ Matrix(C) + Matrix(A)
    @test A + C ≈ Matrix(A) + Matrix(C)
    @test A + A isa SkewSymMatrix
    @test A - A isa SkewSymMatrix
    @test C + C isa StiefelLieAlgHorMatrix
    @test rand(LowerTriangular{Float64}, N) + rand(LowerTriangular{Float64}, N) isa
          LowerTriangular

    # Built in the packed representation and not through `SkewSymMatrix(::AbstractMatrix)`, which
    # would halve a difference and so widen an integer element type to a float one.
    Cᵢ = StiefelLieAlgHorMatrix(SkewSymMatrix(collect(1:3), 3), collect(reshape(4:12, 3, 3)), N, 3)
    Aᵢ = SkewSymMatrix(collect(1:(N * (N - 1) ÷ 2)), N)

    @test eltype(Cᵢ + Aᵢ) === Int
    @test Cᵢ + Aᵢ ≈ Matrix(Cᵢ) + Matrix(Aᵢ)
end
