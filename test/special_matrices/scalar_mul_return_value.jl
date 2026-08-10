using GeometricOptimizers
using GeometricOptimizers: _mul
using LinearAlgebra
using Test
import Random

# Regression test for issue #17. The scalar `mul!` methods of the custom matrix types used to
# end on their last inner `mul!(C.S, A.S, α)` (or `C.A` / `C.B`), and so returned that inner
# *field* instead of the destination `C`. The mutation was always correct — only the return
# value was wrong — which is why it stayed hidden until `_mul(α, a) = _rmul!(_copy(a), α)` in
# `src/optimizers/named_tuple_wrapper.jl` started using the return value. At that point a
# `StiefelLieAlgHorMatrix` silently degraded into a plain `Matrix` of the wrong shape, and
# `MomentumMethod` / `Adam` on a bare `Manifold` died with `DimensionMismatch` /
# `CanonicalIndexError`.
#
# `LinearAlgebra`'s contract is that `mul!` and `rmul!` return their destination argument, so
# what is asserted here is identity (`===`), not just equality.

Random.seed!(1234)

const N, n = 5, 2

# One representative of every type that defines a scalar `mul!`. `triangular.jl` defines a
# single method for `AbstractTriangular`, which is why both of its subtypes appear.
instances() = (
    rand(SkewSymMatrix, n),
    rand(SymmetricMatrix, n),
    rand(GeometricOptimizers.LowerTriangular, n),
    rand(GeometricOptimizers.UpperTriangular, n),
    rand(StiefelLieAlgHorMatrix, N, n),
    rand(GrassmannLieAlgHorMatrix, N, n),
)

name(A) = string(typeof(A).name.name)

@testset "scalar mul! and rmul! return their destination" begin
    for A in instances()
        @testset "$(name(A))" begin
            α = 2.0
            expected = α .* Matrix(A)

            # mul!(C, A, α) — the destination is returned, and the result is right.
            C = zero(A)
            @test mul!(C, A, α) === C
            @test Matrix(C) ≈ expected

            # mul!(C, α, A) — the commuted method forwards to the one above.
            C = zero(A)
            @test mul!(C, α, A) === C
            @test Matrix(C) ≈ expected

            # rmul!(C, α) — defined as mul!(C, C, α), so it inherited the same bug.
            C = copy(A)
            @test rmul!(C, α) === C
            @test Matrix(C) ≈ expected
        end
    end
end

# The reason the above matters: `_mul` builds its result out of the return value, so a wrong
# return type propagates straight into the optimizer caches.
@testset "_mul preserves the type and value of its argument" begin
    for A in instances()
        @testset "$(name(A))" begin
            original = Matrix(A)
            B = _mul(2.0, A)

            @test B isa typeof(A)                # the structured type survives ...
            @test Matrix(B) ≈ 2.0 .* original    # ... with the right entries ...
            @test Matrix(A) ≈ original           # ... and the input is left alone
        end
    end
end

# `_mul` needs `copy`, which for these types needs `similar`; both were missing or wrong for
# `GrassmannLieAlgHorMatrix`, whose `similar` passed the two-parameter concrete type to
# `zeros`, which has no such method.
@testset "copy and zero are structure-preserving" begin
    for A in instances()
        @testset "$(name(A))" begin
            @test zero(A) isa typeof(A)
            @test iszero(Matrix(zero(A)))

            B = copy(A)
            @test B isa typeof(A)
            @test Matrix(B) ≈ Matrix(A)

            # the copy is independent of the original
            rmul!(B, 2.0)
            @test Matrix(B) ≈ 2.0 .* Matrix(A)
        end
    end
end

# The horizontal Lie algebra types are the ones whose `copy` goes through the generic
# `AbstractArray` fallback, so for them `similar` has to return the structured type rather
# than a dense matrix. (The special matrix types above deliberately return a dense `Matrix`
# from `similar`, so they are not included here.)
@testset "similar is structure-preserving for the horizontal Lie algebra types" begin
    for A in (rand(StiefelLieAlgHorMatrix, N, n), rand(GrassmannLieAlgHorMatrix, N, n))
        @testset "$(name(A))" begin
            @test similar(A) isa typeof(A)
            @test size(similar(A)) == size(A)
        end
    end
end
