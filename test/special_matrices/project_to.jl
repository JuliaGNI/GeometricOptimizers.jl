using Test
using LinearAlgebra
using GeometricOptimizers
using GeometricOptimizers: ProjectTo
using JLArrays
import Random

Random.seed!(1234)

# `ProjectTo` turns a dense cotangent `dA = ∂L/∂A` into the natural cotangent of a structured matrix:
# the Frobenius projection onto its tangent space. That is the matrix `G` of the structure whose
# pairing `sum(G .* Ȧ)` with every tangent `Ȧ` is the derivative of `L` along `Ȧ`. The reference is a
# central difference of `L` along each storage direction, taken in `Float64`, so it does not share a
# formula with the projection it checks.
#
# The natural cotangent is not the gradient in the storage, `∂L/∂S`: an off-diagonal entry of a
# `SymmetricMatrix` appears twice in the matrix, so `∂L/∂S` is twice the off-diagonal of `G`. The
# natural cotangent is what AD can add and project again, because the projection is linear and
# idempotent; `∂L/∂S` held in a structured matrix is not, and a second projection of a sum of two
# of them doubles the off-diagonal entries.

# A nonlinear loss with a nonsymmetric weight, so that `dA` is neither symmetric nor skew.
loss(A, W) = sum(sin.(W .* Matrix(A)))
dense_cotangent(A, W) = W .* cos.(W .* Matrix(A))

function directional_derivatives_by_fd(X, S, n, W; h = 1e-5)
    map(eachindex(S)) do k
        e = zero(S)
        e[k] = h
        (loss(X(S + e, n), W) - loss(X(S - e, n), W)) / (2h)
    end
end

function unit_storage_direction(X, n, k)
    e = zeros(storage_length(X, n))
    e[k] = 1
    Matrix(X(e, n))
end

storage_length(::Type{SymmetricMatrix}, n) = n * (n + 1) ÷ 2
storage_length(::Type{SkewSymMatrix}, n) = n * (n - 1) ÷ 2
storage_length(::Type{GeometricOptimizers.StrictlyLowerTriangular}, n) = n * (n - 1) ÷ 2
storage_length(::Type{GeometricOptimizers.StrictlyUpperTriangular}, n) = n * (n - 1) ÷ 2

const STRUCTURED = (
    SymmetricMatrix, SkewSymMatrix, GeometricOptimizers.StrictlyLowerTriangular,
    GeometricOptimizers.StrictlyUpperTriangular)

@testset "ProjectTo gives the natural cotangent: $X, $T" for X in STRUCTURED,
    T in (Float32, Float64)

    n = 4
    S = randn(storage_length(X, n))
    W = randn(n, n)
    reference = directional_derivatives_by_fd(X, S, n, W)

    project = ProjectTo(X(T.(S), n))
    G = project(T.(dense_cotangent(X(S, n), W)))
    @test G isa X
    @test eltype(G) == T
    pairings = [sum(Matrix(G) .* unit_storage_direction(X, n, k)) for k in eachindex(S)]
    @test pairings ≈ reference rtol = √eps(T)

    # A structured cotangent passes unchanged, and a dense one that already has the structure
    # projects to itself.
    @test project(G).S == G.S
    @test project(Matrix(G)).S ≈ G.S

    # A weight used twice in a loss gets two cotangents, and AD adds them as dense matrices before
    # it projects the sum. The result must be the projection of the summed dense cotangent.
    dA₁, dA₂ = randn(T, n, n), randn(T, n, n)
    @test project(Matrix(project(dA₁)) + Matrix(project(dA₂))).S ≈ project(dA₁ + dA₂).S
end

@testset "ProjectTo on a device: $X, $T" for X in (SymmetricMatrix, SkewSymMatrix),
    T in (Float32, Float64)

    JLArrays.allowscalar(false)
    n = 4
    S = randn(T, storage_length(X, n))
    dA = randn(T, n, n)
    G = ProjectTo(X(jl(S), n))(jl(dA))
    @test G.S isa JLArray{T}
    @test Array(G.S) == ProjectTo(X(S, n))(dA).S
end
