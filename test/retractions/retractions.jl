using Test
using LinearAlgebra: I, norm
using GeometricOptimizers
using GeometricOptimizers: AbstractRetraction, geodesic, cayley, retraction, check
import Random

Random.seed!(123)

include("../grassmann_test_help.jl")

# Every one of these takes a step of `Δ / 1000`, so they say nothing about a retraction's behaviour
# at a step of any size — which is how bugs.md A1 survived. The `check` assertion is the one that
# holds the retraction on the manifold; `test/retractions/exponential_accuracy.jl` is what exercises
# it at a lift norm large enough to matter.
const MANIFOLD_TOLERANCE = 1e-5     # `Float32`; `check` cannot go much below `1e-6` in that format

function geodesic_retraction_for_stiefel_manifold(N::Integer, n::Integer, T::Type = Float32)
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = geodesic(Y, Δ / 1000)
    @test check(Y₁) < MANIFOLD_TOLERANCE
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function cayley_retraction_for_stiefel_manifold(N::Integer, n::Integer, T::Type = Float32)
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = cayley(Y, Δ / 1000)
    @test check(Y₁) < MANIFOLD_TOLERANCE
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function geodesic_retraction_for_grassmann_manifold(N::Integer, n::Integer, T::Type = Float32)
    Y = rand(GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = geodesic(Y, Δ / 1000)
    @test check(Y₁) < MANIFOLD_TOLERANCE
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function cayley_retraction_for_grassmann_manifold(N::Integer, n::Integer, T::Type = Float32)
    Y = rand(GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = cayley(Y, Δ / 1000)
    @test check(Y₁) < MANIFOLD_TOLERANCE
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

# A retraction that is passed to the `Optimizer` but has no `retraction` method has to say so.
# The fallback used to have an empty body, so it returned `nothing`, and the step then failed
# further downstream (in `_copyto!`, with a `MethodError` about `Nothing`) — which pointed at
# the wrong place entirely.
struct UnimplementedRetraction <: AbstractRetraction end

@testset "an unimplemented retraction reports itself" begin
    R = UnimplementedRetraction()
    x = rand(3, 3)

    @test_throws "UnimplementedRetraction" retraction(R, x)
    @test_throws ErrorException R(x)                # through the callable form as well
    @test retraction(GeometricOptimizers.Cayley(), x) == cayley(x)
    @test retraction(GeometricOptimizers.Geodesic(), x) == geodesic(x)
end

# `lift_factors` writes the lift as `B̂ * B̄'`, and its upper-right block is the one `getindex` builds
# as `-B.B[j, i]` — entrywise, without conjugating. Spelt `-B.B'` the block conjugates, so the
# factorisation reproduced a different matrix from the lift it came from. The two are one expression
# on a real element type, which is why this testset is here: it is what the real path cannot see.
# `+(::StiefelLieAlgHorMatrix, ::AbstractMatrix)` rebuilds the same block and is pinned the same way
# in `test/lie_algebras/stiefel_lie_algebra_horizontal.jl`.
@testset "the lift factorisation reproduces the lift on a complex element type" begin
    for C in (StiefelLieAlgHorMatrix(
        SkewSymMatrix(randn(ComplexF64, 2, 2)), randn(ComplexF64, 2, 2), 4, 2),
        GrassmannLieAlgHorMatrix(randn(ComplexF64, 2, 2), 4, 2))
        B̂, B̄ = GeometricOptimizers.lift_factors(C)
        @test B̂ * B̄' ≈ Matrix(C)
    end

    # and the real path, where the two spellings are one expression
    for C in (StiefelLieAlgHorMatrix(SkewSymMatrix(randn(2, 2)), randn(2, 2), 4, 2),
        GrassmannLieAlgHorMatrix(randn(2, 2), 4, 2))
        B̂, B̄ = GeometricOptimizers.lift_factors(C)
        @test B̂ * B̄' ≈ Matrix(C)
    end
end

# `cayley(::AbstractLieAlgHorMatrix)` evaluates a regrouping of the Cayley transform in the
# `N × 2n` factors of `lift_factors`, so nothing in it is read off the definition and every step of
# the regrouping is a place to lose a factor or a transpose. The retraction assertions below catch
# only part of that. A sign slip in the `2n × 2n` inverse reaches them, but barely: it lands at a
# `check` residual of 2e-5 against a `MANIFOLD_TOLERANCE` of 1e-5, where the correct grouping sits
# at 4e-7. `B̄'` spelt `transpose(B̄)` does not reach them at all, because the two are one expression
# on the real `Float32` points they run on. This pins the value against the definition, on the
# dense lift where there is nothing to group, and on a complex element type as well.
@testset "the Cayley retraction of a lift is the Cayley transform of the lift" begin
    for C in (StiefelLieAlgHorMatrix(SkewSymMatrix(randn(3, 3)), randn(5, 3), 8, 3),
        GrassmannLieAlgHorMatrix(randn(5, 3), 8, 3),
        StiefelLieAlgHorMatrix(
        SkewSymMatrix(randn(ComplexF64, 2, 2)), randn(ComplexF64, 2, 2), 4, 2),
        GrassmannLieAlgHorMatrix(randn(ComplexF64, 2, 2), 4, 2))
        B̄ = Matrix(C)
        @test cayley(C) ≈ (I - B̄ / 2) \ (I + B̄ / 2)
    end
end

T = Float32

for N in 3:5
    for n in 1:N
        @test geodesic_retraction_for_stiefel_manifold(N, n, T)
        @test cayley_retraction_for_stiefel_manifold(N, n, T)
        grassmann_test_help(geodesic_retraction_for_grassmann_manifold(N, n, T), N, n)
        grassmann_test_help(cayley_retraction_for_grassmann_manifold(N, n, T), N, n)
    end
end
