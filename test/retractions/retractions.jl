using Test
using LinearAlgebra: norm
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

function geodesic_retraction_for_stiefel_manifold(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = geodesic(Y, Δ / 1000)
    @test check(Y₁) < MANIFOLD_TOLERANCE
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function cayley_retraction_for_stiefel_manifold(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = cayley(Y, Δ / 1000)
    @test check(Y₁) < MANIFOLD_TOLERANCE
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function geodesic_retraction_for_grassmann_manifold(N::Integer, n::Integer, T::Type=Float32)
    Y = rand(GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))
    Y₁ = geodesic(Y, Δ / 1000)
    @test check(Y₁) < MANIFOLD_TOLERANCE
    norm(1000 * (Y₁ - Y) - Δ) / norm(Δ) < 1e-2
end

function cayley_retraction_for_grassmann_manifold(N::Integer, n::Integer, T::Type=Float32)
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

T = Float32

for N ∈ 3:5
    for n ∈ 1:N
        @test geodesic_retraction_for_stiefel_manifold(N, n, T)
        @test cayley_retraction_for_stiefel_manifold(N, n, T)
        grassmann_test_help(geodesic_retraction_for_grassmann_manifold(N, n, T), N, n)
        grassmann_test_help(cayley_retraction_for_grassmann_manifold(N, n, T), N, n)
    end
end