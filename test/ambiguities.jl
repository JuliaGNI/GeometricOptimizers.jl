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

include("helpers/eltypes.jl")

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

qpoint(rng, T, m, k) = StiefelManifold(Matrix(qr!(randn(rng, T, m, m)).Q)[:, 1:k])

function operands(rng, ::Type{T}) where {T}
    Q = Matrix(qr!(randn(rng, T, N, N)).Q)
    Y = StiefelManifold(Q)
    U = rand(rng, SymplecticStiefelManifold{T}, N, N)
    Uᵣ = rand(rng, SymplecticStiefelManifold{T}, N, n)
    Yᵣ = qpoint(rng, T, N, n)
    S = sr!(randn(rng, T, N, N ÷ 2 + 1)).S
    # `sr!` asserts an even number of columns, so `n ÷ 2 + 1` has to be even as well
    Sₙ = sr!(randn(rng, T, n, n ÷ 2 + 1)).S
    skew = rand(rng, SkewSymMatrix{T}, N)
    lift = rand(rng, StiefelLieAlgHorMatrix{T}, N, N ÷ 2)
    grass = rand(rng, GrassmannLieAlgHorMatrix{T}, N, N ÷ 2)
    ["Y" => Y, "Y'" => Y', "Yview" => StiefelManifold(view(Q, :, :)),
        "G" => GrassmannManifold(Q), "G'" => GrassmannManifold(Q)',
        "U" => U, "U'" => U', "S" => S, "inv(S)" => inv(S),
        "E" => StiefelProjection(T, N, N),
        "StrictlyLowerTriangular" => rand(rng, StrictlyLowerTriangular{T}, N),
        "StrictlyUpperTriangular" => rand(rng, StrictlyUpperTriangular{T}, N),
        "SkewSym" => skew, "SkewSym'" => skew', "Sym" => rand(rng, SymmetricMatrix{T}, N),
        "StiefelHor" => lift, "StiefelHor'" => lift',
        "GrassmannHor" => grass, "GrassmannHor'" => grass',
        # `6 × 2` and `2 × 6`
        "Yᵣ" => Yᵣ, "Yᵣ'" => Yᵣ', "Uᵣ" => Uᵣ, "Uᵣ'" => Uᵣ', "Eᵣ" =>
            StiefelProjection(T, N, n),
        "Eᵣ'" => StiefelProjection(T, N, n)',
        # `2 × 2` and `2 × 1`
        "Yₙ" => qpoint(rng, T, n, 1), "Uₙ" => rand(rng, SymplecticStiefelManifold{T}, n, n),
        "Sₙ" => Sₙ, "inv(Sₙ)" => inv(Sₙ), "Eₙ" => StiefelProjection(T, n, 1),
        "StrictlyLowerTriangularₙ" => rand(rng, StrictlyLowerTriangular{T}, n),
        "StrictlyUpperTriangularₙ" => rand(rng, StrictlyUpperTriangular{T}, n),
        "SkewSymₙ" => rand(rng, SkewSymMatrix{T}, n), "Symₙ" =>
            rand(rng, SymmetricMatrix{T}, n),
        "StiefelHorₙ" => rand(rng, StiefelLieAlgHorMatrix{T}, n, 1),
        "GrassmannHorₙ" => rand(rng, GrassmannLieAlgHorMatrix{T}, n, 1)]
end

const OPERANDS = Dict(T => operands(Random.Xoshiro(1), T) for T in REAL_ELTYPES)

@testset "a product of two owned matrices agrees with the dense product, $T" for T in REAL_ELTYPES
    for (lname, L) in OPERANDS[T], (rname, R) in OPERANDS[T]

        size(L, 2) == size(R, 1) || continue
        @testset "$lname * $rname" begin
            @test eltype(L * R) == T
            @test L * R ≈ Matrix(L) * Matrix(R)
        end
    end
end

@testset "a vector and a row vector against an owned matrix, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(2)
    for (name, A) in OPERANDS[T]
        @testset "$name" begin
            v, w = randn(rng, T, size(A, 1)), randn(rng, T, size(A, 2))
            @test eltype(v' * A) == T
            @test v' * A ≈ v' * Matrix(A)
            @test transpose(v) * A ≈ transpose(v) * Matrix(A)
            @test A * w ≈ Matrix(A) * w
            M, P = randn(rng, T, size(A)...), randn(rng, T, size(A, 2), size(A, 1))
            @test A * P ≈ Matrix(A) * P
            @test P * A ≈ P * Matrix(A)
            @test A + M ≈ Matrix(A) + M
            @test M + A ≈ M + Matrix(A)
            @test A - M ≈ Matrix(A) - M
            @test M - A ≈ M - Matrix(A)
        end
    end
end

@testset "a sum or difference of two owned matrices agrees with the dense one, $T" for T in REAL_ELTYPES
    for (lname, L) in OPERANDS[T], (rname, R) in OPERANDS[T]

        size(L) == size(R) || continue
        @testset "$lname ± $rname" begin
            @test eltype(L + R) == T
            @test L + R ≈ Matrix(L) + Matrix(R)
            @test L - R ≈ Matrix(L) - Matrix(R)
        end
    end
end

# An owned operand whose element type differs from the other's reaches the untyped fallback of its
# kernel, and so `LinearAlgebra`'s or `Base`'s own answer. That answer goes through `mul!`, where the
# kernels take one element type only, so a pair with two reaches the five-argument generic `mul!`.
@testset "mixed element types" begin
    for A in (rand(SkewSymMatrix{Float32}, N), rand(SymmetricMatrix{Float32}, N),
        rand(StrictlyLowerTriangular{Float32}, N), rand(StiefelLieAlgHorMatrix{Float32}, N, n),
        StiefelManifold(Float32.(Matrix(qr!(randn(N, N)).Q))))
        M, v = randn(N, N), randn(N)
        B = rand(SkewSymMatrix, N)
        @test A * M ≈ Matrix(A) * M
        @test M * A ≈ M * Matrix(A)
        @test A * v ≈ Matrix(A) * v
        @test v' * A ≈ v' * Matrix(A)
        @test A + M ≈ Matrix(A) + M
        @test M + A ≈ M + Matrix(A)
        @test A * B ≈ Matrix(A) * Matrix(B)
        @test B * A ≈ Matrix(B) * Matrix(A)
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
@testset "the structure-preserving methods still win, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(3)
    A = rand(rng, SkewSymMatrix{T}, N)
    C = rand(rng, StiefelLieAlgHorMatrix{T}, N, N ÷ 2)

    @test C + A isa SkewSymMatrix
    @test A + C isa SkewSymMatrix
    @test eltype(C + A) == T
    @test C + A ≈ Matrix(C) + Matrix(A)
    @test A + C ≈ Matrix(A) + Matrix(C)
    @test A + A isa SkewSymMatrix
    @test A - A isa SkewSymMatrix
    @test C + C isa StiefelLieAlgHorMatrix
    @test rand(rng, StrictlyLowerTriangular{T}, N) +
          rand(rng, StrictlyLowerTriangular{T}, N) isa
          StrictlyLowerTriangular
end

@testset "the structure-preserving methods keep an integer element type" begin
    # Built in the packed representation and not through `SkewSymMatrix(::AbstractMatrix)`, which
    # would halve a difference and so widen an integer element type to a float one.
    Cᵢ = StiefelLieAlgHorMatrix(SkewSymMatrix(collect(1:3), 3), collect(reshape(4:12, 3, 3)), N, 3)
    Aᵢ = SkewSymMatrix(collect(1:(N * (N - 1) ÷ 2)), N)

    @test eltype(Cᵢ + Aᵢ) === Int
    @test Cᵢ + Aᵢ ≈ Matrix(Cᵢ) + Matrix(Aᵢ)
end
