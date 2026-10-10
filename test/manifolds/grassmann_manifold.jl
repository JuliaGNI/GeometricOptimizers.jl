# Every bound below is a small multiple of `eps(T)`, and holds in `Float32` and `Float64` alike:
# over 200 seeds of this sweep the residuals of `check`, of the gradient and of the tangent space
# stay under `6eps(T)` in both precisions. The coordinate chart inverts the leading `n × n` block of
# the point, so its residual is that block's condition number times `eps(T)` (measured at most
# `0.25eps(T)` times it in both precisions), and its bound is written in that condition number.

using Test
using LinearAlgebra
using GeometricOptimizers
using GeometricOptimizers: Ω, metric, check
using GeometricOptimizers: global_section, global_rep
import Random

include("../helpers/eltypes.jl")

# `GlobalSection` draws its completion from the global generator
Random.seed!(1234)

function check_gradient(rng, T, N::Integer, n::Integer)
    Y = rand(rng, GrassmannManifold{T}, N, n)

    #element of the tangent space
    Δ = rgrad(Y, randn(rng, T, N, n))
    A = randn(rng, T, N, n)
    V = rgrad(Y, A)
    norm(tr(Δ'*A) - metric(Y, Δ, V))/N/n
end

function tangent_space_rep(rng, T, N::Integer, n::Integer)
    Y = rand(rng, GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, randn(rng, T, N, n))
    Y.A' * Δ
end

function gloabl_tangent_space_representation(rng, T, N::Integer, n::Integer)
    Y = rand(rng, GrassmannManifold{T}, N, n)
    Δ = rgrad(Y, randn(rng, T, N, n))
    λY = GlobalSection(Y)
    global_rep(λY, Δ)
end

# The chart and the condition number of the block it inverts.
function coordinate_chart_rep(rng, T, N::Integer, n::Integer)
    Y = rand(rng, GrassmannManifold{T}, N, n)
    Y₁ = Y.A[1:n, 1:n]
    Y.A = Y.A*inv(Y₁)
    Y, cond(Y₁)
end

function metric_test(rng, T, N, n)
    Y = rand(rng, GrassmannManifold{T}, N, n)
    Δ₁ = rgrad(Y, rand(rng, T, N, n))
    Δ₂ = rgrad(Y, rand(rng, T, N, n))
    @test T(0.5) * tr(Ω(Y, Δ₁)' * Ω(Y, Δ₂)) ≈ metric(Y, Δ₁, Δ₂)
    @test eltype(metric(Y, Δ₁, Δ₂)) == T
end

function run_tests(rng, T, N, n)
    # round-off of a CholeskyQR2 and of `O(Nn)` products on `N ≤ 10`: measured under `6eps(T)`
    tol = 16eps(T)
    # `check` used to be defined for `StiefelManifold` only, so the one function that measures
    # distance from the manifold was a `MethodError` here — see bugs.md A3. The representative of a
    # `GrassmannManifold` point satisfies `YᵗY = I` just as the Stiefel one does.
    Y = rand(rng, GrassmannManifold{T}, N, n)
    @test eltype(Y) == T
    @test check(Y) < tol
    @test check_gradient(rng, T, N, n) < tol
    @test norm(tangent_space_rep(rng, T, N, n)[1:n, 1:n])/N/n < tol
    B = gloabl_tangent_space_representation(rng, T, N, n)
    @test typeof(B) <: GrassmannLieAlgHorMatrix
    @test eltype(B) == T
    # the inversion amplifies round-off by the condition number of the inverted block, measured at
    # most `0.25eps(T)` times it
    C, κ = coordinate_chart_rep(rng, T, N, n)
    @test eltype(C) == T
    @test norm(C[1:n, 1:n]-I(n)) / N / n < 2eps(T) * κ
    metric_test(rng, T, N, n)
end

@testset "Grassmann manifold, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1234)
    for N in 1:10
        for n in 1:(N - 1)
            run_tests(rng, T, N, n)
        end
    end
end
