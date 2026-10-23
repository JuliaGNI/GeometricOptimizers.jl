# The method families both horizontal lifts share (`src/lie_algebras/lift_families.jl`), and the
# bounds check of their `getindex`.
using GeometricOptimizers
using GeometricOptimizers: add!
using LinearAlgebra: mul!
using Test
import Random

include("../helpers/eltypes.jl")

# The families bind both operands to one concrete lift, so a Stiefel lift beside a Grassmann lift
# reaches the generic `AbstractArray` fallback, which has no `setindex!` to write with. A family
# written over the abstract type would pair the blocks of the two by position and return normally.
@testset "a mixed pair of lifts is refused, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1)
    S = rand(rng, StiefelLieAlgHorMatrix{T}, 6, 2)
    G = rand(rng, GrassmannLieAlgHorMatrix{T}, 6, 2)
    @test_throws CanonicalIndexError copyto!(G, S)
    @test_throws CanonicalIndexError mul!(G, S, T(2))
    @test_throws CanonicalIndexError add!(G, S, G)
end

@testset "similar gives a zero lift of the same type, $L, $T" for L in (
        StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix),
    T in REAL_ELTYPES

    B = rand(Random.Xoshiro(2), L{T}, 6, 2)
    @test typeof(similar(B)) == typeof(B)
    @test similar(B) == zero(B)
    @test (size(similar(B, 8, 3)), similar(B, 8, 3).n) == ((8, 8), 3)
end

# The last line of each `getindex` answers `zero(T)` for the zero block, so an index outside the
# matrix has to raise before it, not reach it.
@testset "an index outside a lift raises, $L" for L in (
    StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix)
    B = rand(Random.Xoshiro(3), L{Float64}, 6, 2)
    for (i, j) in ((0, 1), (-1, 2), (7, 1), (1, 7), (7, 7))
        @test_throws BoundsError B[i, j]
    end
end
