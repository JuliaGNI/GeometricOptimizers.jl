using Test
using LinearAlgebra: I, norm
using GeometricOptimizers
using GeometricOptimizers: AbstractRetraction, geodesic, cayley, retraction, check
import Random

include("../helpers/eltypes.jl")
include("../helpers/manifold_tolerance.jl")

Random.seed!(123)

include("../helpers/grassmann_test_help.jl")

# Every one of these takes a step of `h = √eps(T)`, so they say nothing about a retraction's
# behaviour at a step of any size — which is how bugs.md A1 survived. The `check` assertion is the
# one that holds the retraction on the manifold; `test/retractions/exponential_accuracy.jl` is what
# exercises it at a lift norm large enough to matter.
#
# `(R(Y, hΔ) - Y) / h` is a forward difference, so it differs from `Δ` by a truncation error of
# `O(h‖Δ‖²)` and a round-off error of `O(eps(T) / h)`; `h = √eps(T)` balances the two, and both are
# then `O(√eps(T))`. Measured worst relative error over 20 seeds and every shape below: `4.0√eps(T)`
# in `Float32` and `4.1√eps(T)` in `Float64`. The bound leaves a factor four; a retraction whose
# first derivative is not `Δ` misses it by `O(1)`.
finite_difference_step(::Type{T}) where {T} = sqrt(eps(T))
finite_difference_tolerance(::Type{T}) where {T} = 16 * sqrt(eps(T))

function geodesic_retraction_for_stiefel_manifold(N::Integer, n::Integer, T::Type, rng)
    Y = rand(rng, StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(rng, T, N, n))
    h = finite_difference_step(T)
    Y₁ = geodesic(Y, h * Δ)
    @test eltype(Y₁) == T
    @test check(Y₁) < manifold_tolerance(T)
    norm((Y₁ - Y) / h - Δ) / norm(Δ) < finite_difference_tolerance(T)
end

function cayley_retraction_for_stiefel_manifold(N::Integer, n::Integer, T::Type, rng)
    Y = rand(rng, StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(rng, T, N, n))
    h = finite_difference_step(T)
    Y₁ = cayley(Y, h * Δ)
    @test eltype(Y₁) == T
    @test check(Y₁) < manifold_tolerance(T)
    norm((Y₁ - Y) / h - Δ) / norm(Δ) < finite_difference_tolerance(T)
end

function geodesic_retraction_for_grassmann_manifold(N::Integer, n::Integer, T::Type, rng)
    Y = rand(rng, GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(rng, T, N, n))
    h = finite_difference_step(T)
    Y₁ = geodesic(Y, h * Δ)
    @test eltype(Y₁) == T
    @test check(Y₁) < manifold_tolerance(T)
    norm((Y₁ - Y) / h - Δ) / norm(Δ) < finite_difference_tolerance(T)
end

function cayley_retraction_for_grassmann_manifold(N::Integer, n::Integer, T::Type, rng)
    Y = rand(rng, GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(rng, T, N, n))
    h = finite_difference_step(T)
    Y₁ = cayley(Y, h * Δ)
    @test eltype(Y₁) == T
    @test check(Y₁) < manifold_tolerance(T)
    norm((Y₁ - Y) / h - Δ) / norm(Δ) < finite_difference_tolerance(T)
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
@testset "the lift factorisation reproduces the lift on a complex element type, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(71)
    for C in (StiefelLieAlgHorMatrix(
        SkewSymMatrix(randn(rng, Complex{T}, 2, 2)), randn(rng, Complex{T}, 2, 2), 4, 2),
        GrassmannLieAlgHorMatrix(randn(rng, Complex{T}, 2, 2), 4, 2))
        B̂, B̄ = GeometricOptimizers.lift_factors(C)
        result = B̂ * B̄'
        @test eltype(result) == Complex{T}
        @test result ≈ Matrix(C)
    end

    # and the real path, where the two spellings are one expression
    for C in (StiefelLieAlgHorMatrix(SkewSymMatrix(randn(rng, T, 2, 2)), randn(rng, T, 2, 2), 4, 2),
        GrassmannLieAlgHorMatrix(randn(rng, T, 2, 2), 4, 2))
        B̂, B̄ = GeometricOptimizers.lift_factors(C)
        result = B̂ * B̄'
        @test eltype(result) == T
        @test result ≈ Matrix(C)
    end
end

# `cayley(::AbstractLieAlgHorMatrix)` evaluates a regrouping of the Cayley transform in the
# `N × 2n` factors of `lift_factors`, so nothing in it is read off the definition and every step of
# the regrouping is a place to lose a factor or a transpose. The retraction assertions below catch
# only part of that. A sign slip in the `2n × 2n` inverse was measured at a `Float32` `check`
# residual of 2e-5 (at the step `Δ / 1000` these assertions once took), where the correct grouping
# sits at 4e-7: both are below
# `manifold_tolerance(Float32)`, 4.9e-4, so `check` does not separate them in that format. `B̄'`
# spelt `transpose(B̄)` does not reach them at all, because the two are one expression on the real
# points they run on. This pins the value against the definition, on the dense lift where there is
# nothing to group, and on a complex element type as well. The element type of the result is what
# fails if a `Float64` literal enters the formula and promotes a `Float32` lift.
@testset "the Cayley retraction of a lift is the Cayley transform of the lift, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(72)
    for C in (StiefelLieAlgHorMatrix(SkewSymMatrix(randn(rng, T, 3, 3)), randn(rng, T, 5, 3), 8, 3),
        GrassmannLieAlgHorMatrix(randn(rng, T, 5, 3), 8, 3))
        B̄ = Matrix(C)
        result = cayley(C)
        @test eltype(result) == T
        @test result ≈ (I - B̄ / 2) \ (I + B̄ / 2)
    end

    for C in (StiefelLieAlgHorMatrix(
        SkewSymMatrix(randn(rng, Complex{T}, 2, 2)), randn(rng, Complex{T}, 2, 2), 4, 2),
        GrassmannLieAlgHorMatrix(randn(rng, Complex{T}, 2, 2), 4, 2))
        B̄ = Matrix(C)
        result = cayley(C)
        @test eltype(result) == Complex{T}
        @test result ≈ (I - B̄ / 2) \ (I + B̄ / 2)
    end
end

@testset "a step of √eps(T) along each retraction is the tangent vector, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for N in 3:5
        for n in 1:N
            @test geodesic_retraction_for_stiefel_manifold(N, n, T, rng)
            @test cayley_retraction_for_stiefel_manifold(N, n, T, rng)
            grassmann_test_help(geodesic_retraction_for_grassmann_manifold(N, n, T, rng), N, n)
            grassmann_test_help(cayley_retraction_for_grassmann_manifold(N, n, T, rng), N, n)
        end
    end
end
