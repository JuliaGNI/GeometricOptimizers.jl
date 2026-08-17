using GeometricOptimizers
using GeometricOptimizers: VectorStorageMatrix, _add!, _rac!, _square!, _div!, _rmul!
using Test
import Random

Random.seed!(1234)

# The four types that store their free parameters in one vector. Everything below is written on
# `parent`, which is that vector, because that is the whole point of `VectorStorageMatrix`: the
# generic `AbstractArray` methods broadcast, and none of these four has a `setindex!` to broadcast
# through. Before these methods existed, a `SymmetricMatrix` or a triangular matrix could not be an
# optimizer parameter at all -- `GeometricMachineLearning` carried its own copies of them to work
# around it (GeometricMachineLearning#234).
const TYPES = (SkewSymMatrix, SymmetricMatrix, LowerTriangular, UpperTriangular)

const N = 5

@testset "$(nameof(MT))" for MT in TYPES
    @testset "similar preserves the type" begin
        # Not a detail: the optimizer caches allocate their scratch with `similar` and then require
        # every one of them to have the same type as the parameter. The `AbstractArray` fallback
        # returns a dense `Matrix`, which makes the cache constructors inapplicable.
        for T in (Float32, Float64)
            A = rand(MT{T}, N)
            S = similar(A)
            @test typeof(S) == typeof(A)
            @test size(S) == size(A)
            @test length(parent(S)) == length(parent(A))
        end
    end

    @testset "fill! writes the storage" begin
        A = fill!(similar(rand(MT{Float64}, N)), 3.0)
        @test all(parent(A) .== 3.0)
        # `NaN` is what the caches actually poison scratch arrays with, and it has to survive the
        # round trip rather than being swallowed by a comparison somewhere.
        @test all(isnan, parent(fill!(similar(A), NaN)))
    end

    @testset "the elementwise primitives act on the free parameters" begin
        A = rand(MT{Float64}, N)
        B = rand(MT{Float64}, N)

        a = copy(A)
        @test _add!(a, B) === a
        @test parent(a) ≈ parent(A) .+ parent(B)

        a = copy(A)
        @test _add!(a, 2.0) === a
        @test parent(a) ≈ parent(A) .+ 2.0

        # squares of a known matrix, so that `_rac!` inverts `_square!` exactly rather than to a
        # tolerance that a sign error could hide in
        sq = similar(A)
        @test _square!(sq, A) === sq
        @test parent(sq) ≈ parent(A) .^ 2

        rt = similar(A)
        @test _rac!(rt, sq) === rt
        @test parent(rt) ≈ abs.(parent(A))

        d = similar(A)
        @test _div!(d, A, B) === d
        @test parent(d) ≈ parent(A) ./ parent(B)

        m = copy(A)
        @test _rmul!(m, 2.0) === m
        @test parent(m) ≈ 2 .* parent(A)
    end

    @testset "the primitives leave the represented matrix consistent" begin
        # `_rmul!` writes the storage, but what it has to *mean* is scaling the matrix. For a
        # skew-symmetric matrix the two differ by a sign in the upper triangle, so this checks the
        # full dense matrix and not only `parent`.
        A = rand(MT{Float64}, N)
        dense = Matrix(A)
        @test Matrix(_rmul!(copy(A), 2.0)) ≈ 2 .* dense

        B = rand(MT{Float64}, N)
        @test Matrix(_add!(copy(A), B)) ≈ dense .+ Matrix(B)
    end

    @testset "update_section! is addition on the free parameters" begin
        # These are ordinary vector-space parameters, so the extended retraction is addition and the
        # section carries no `λ`. The retraction argument is ignored, which is what the two calls
        # below assert by giving different ones and expecting the same answer.
        A = rand(MT{Float64}, N)
        B = rand(MT{Float64}, N)

        Λᵗ = GlobalSection(copy(A))
        Λ⁽ᵗ⁻¹⁾ = GlobalSection(copy(A))
        @test update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B, Cayley()) === Λᵗ
        @test parent(Λᵗ.Y) ≈ parent(A) .+ parent(B)
        @test Λᵗ.λ === nothing

        Λ₂ = GlobalSection(copy(A))
        update_section!(Λ₂, GlobalSection(copy(A)), B, Geodesic())
        @test parent(Λ₂.Y) ≈ parent(Λᵗ.Y)

        # the in-place three-argument form, which is what `update_section!(Λ, B, retraction)` is
        Λ₃ = GlobalSection(copy(A))
        update_section!(Λ₃, B, Cayley())
        @test parent(Λ₃.Y) ≈ parent(A) .+ parent(B)
    end

    @testset "every one of them is a VectorStorageMatrix" begin
        @test rand(MT{Float64}, N) isa VectorStorageMatrix
        @test rand(MT{Float32}, N) isa VectorStorageMatrix{Float32}
    end
end

# Addition against a dense matrix has to commute. `SkewSymMatrix` had
# `+(B::AbstractMatrix, A::SkewSymMatrix) = B + A` in `GeometricMachineLearning`, which calls itself
# and dies with a `StackOverflowError`; the method here is `A + B` and always was. The loop closes
# the class rather than that one instance.
@testset "addition with a dense matrix commutes -- $(nameof(MT))" for MT in TYPES
    A = rand(MT{Float64}, N)
    B = rand(N, N)
    @test A + B ≈ B + A
    @test A + B ≈ Matrix(A) + B
end
