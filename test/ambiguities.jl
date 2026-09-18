# Ambiguities between two of this package's own methods, as a property rather than a list.
#
# Each such pair is a standoff between an `Owned ∘ AbstractMatrix` method and an
# `AbstractMatrix ∘ Owned` one. For two owned operands neither is more specific, so an ordinary
# product or sum of two of them raised a `MethodError`.
# `src/ambiguities.jl` carries the tie-breakers and says what each one returns.
#
# The assertion below is that the set is empty rather than that it has some size. A count goes stale
# the moment a method is added -- the one in this suite's own Aqua header read 139 where
# `detect_ambiguities` reported 224 -- and it says nothing about which pairs are in it. An empty set
# also catches the next pair somebody adds without anyone having to remember this file exists.

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

    # `global_rep` is excluded by name, and it is the only exclusion. Its two pairs intersect at a
    # `GlobalSection` that is anchored on a manifold and carries no lift, which the constructors
    # cannot build; the comment on `global_rep` in `src/optimizers/named_tuple_wrapper.jl` carries
    # the `typeintersect`-and-witness triage that establishes it. A tie-breaker for a shape that
    # cannot exist would be there for the checker and for nothing else.
    unresolved = filter(p -> first(p).name !== :global_rep, own)

    @test isempty(unresolved)

    # Printed rather than merely counted, so a failure names the pair instead of a number.
    for (m₁, m₂) in unresolved
        @info "unresolved own-vs-own ambiguity" m₁ m₂
    end
end

# One instance of every type that takes part, all `6 × 6` so that every combination conforms. The
# product of two of them is checked against the product of their dense forms, which is the answer
# every tie-breaker in `src/ambiguities.jl` is written to give.
const N = 6

const LEFT = let
    Y = StiefelManifold(Matrix(qr!(randn(N, N)).Q))
    S = sr!(randn(N, N ÷ 2 + 1)).S
    ["Y'" => Y',
        "Y" => Y,
        "U" => rand(SymplecticStiefelManifold, N, N),
        "S" => S,
        "inv(S)" => inv(S),
        "SkewSym" => rand(SkewSymMatrix, N),
        "Sym" => rand(SymmetricMatrix, N)]
end

const RIGHT = let
    S = sr!(randn(N, N ÷ 2 + 1)).S
    ["Y" => StiefelManifold(Matrix(qr!(randn(N, N)).Q)),
        "U" => rand(SymplecticStiefelManifold, N, N),
        "S" => S,
        "inv(S)" => inv(S),
        "LowerTriangular" => rand(LowerTriangular{Float64}, N),
        "UpperTriangular" => rand(UpperTriangular{Float64}, N),
        "SkewSym" => rand(SkewSymMatrix, N),
        "Sym" => rand(SymmetricMatrix, N)]
end

@testset "a product of two owned matrices agrees with the dense product" begin
    for (lname, L) in LEFT, (rname, R) in RIGHT

        @testset "$lname * $rname" begin
            @test L * R ≈ Matrix(L) * Matrix(R)
        end
    end
end

# The sum's three types. `StiefelProjection` is square here for the same reason as above: the other
# two are `N × N` and `+` conforms only if all three are.
const SUMMANDS = ["StiefelLieAlgHor" => rand(StiefelLieAlgHorMatrix, N, N ÷ 2),
    "SkewSym" => rand(SkewSymMatrix, N),
    "StiefelProjection" => StiefelProjection(N, N)]

@testset "a sum of two owned matrices agrees with the dense sum" begin
    for (lname, L) in SUMMANDS, (rname, R) in SUMMANDS

        @testset "$lname + $rname" begin
            @test L + R ≈ Matrix(L) + Matrix(R)
        end
    end
end

# `*(::Adjoint{T, StiefelManifold{T, AT}}, ::StiefelManifold)` in `src/manifolds/stiefel_manifold.jl`
# is what separates that pair, and it bound `AT` to both operands. A pair whose storage types differ
# therefore missed it and was left to the two generic methods, which both apply and neither of which
# is more specific. Both points below hold the same entries; only the type of the array holding them
# differs.
@testset "an adjoint point times a point whose storage type differs" begin
    Q = Matrix(qr!(randn(N, N)).Q)
    Y = StiefelManifold(Q)
    Yᵥ = StiefelManifold(view(Q, :, :))

    @test Y' * Yᵥ ≈ Matrix(Y)' * Matrix(Yᵥ)
    @test Yᵥ' * Y ≈ Matrix(Yᵥ)' * Matrix(Y)
end

@testset "vcat and hcat of two StiefelProjections" begin
    E = StiefelProjection(5, 2)
    F = StiefelProjection(4, 2)

    @test vcat(E, F) ≈ vcat(Matrix(E), Matrix(F))
    @test hcat(E, E) ≈ hcat(Matrix(E), Matrix(E))
end

# The one pair whose sum keeps a structure, rather than landing on a dense matrix the way every
# other tie-breaker does: both operands are skew-symmetric, so the sum is.
@testset "a horizontal lift plus a skew-symmetric matrix stays skew-symmetric" begin
    C = rand(StiefelLieAlgHorMatrix, N, N ÷ 2)
    A = rand(SkewSymMatrix, N)

    @test C + A isa SkewSymMatrix
    @test A + C isa SkewSymMatrix
    @test C + A ≈ Matrix(C) + Matrix(A)
    @test A + C ≈ Matrix(A) + Matrix(C)

    # Built in the packed representation and not through `SkewSymMatrix(::AbstractMatrix)`, which
    # would halve a difference and so widen an integer element type to a float one.
    Cᵢ = StiefelLieAlgHorMatrix(SkewSymMatrix(collect(1:3), 3), collect(reshape(4:12, 3, 3)), N, 3)
    Aᵢ = SkewSymMatrix(collect(1:(N * (N - 1) ÷ 2)), N)

    @test eltype(Cᵢ + Aᵢ) === Int
    @test Cᵢ + Aᵢ ≈ Matrix(Cᵢ) + Matrix(Aᵢ)
end
