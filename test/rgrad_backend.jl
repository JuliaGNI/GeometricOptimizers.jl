# A gradient lives on the backend of its parameter.
#
# `rgrad` at a point on a device refuses an ambient gradient on another backend, with the
# "mixed backends" `ArgumentError`: copying it over would hide the caller that allocated it on the
# host and pay a transfer per leaf per step. At a host point it asks the gradient nothing, so a host
# gradient that `KernelAbstractions` cannot place keeps the host path. A parameter set whose leaves
# are on two backends is refused when the `Optimizer` is built, and the gradient an `Optimizer`
# computes for a device set has every leaf on the backend of its parameter leaf.
#
# `JLArray` stands in for the device: its backend is a `KernelAbstractions.GPU`.

using GeometricOptimizers
using GeometricOptimizers: StrictlyLowerTriangular, StrictlyUpperTriangular, gradient, rgrad
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions, get_backend
using LinearAlgebra: qr
using NeuralNetworkParameters: NetworkParameters, flatten, foldstorage, mapstorage
using Random
using Test

allowscalar(false)

# A gradient `KernelAbstractions` cannot place: `get_backend` raises for it. It stands in for any
# matrix type outside `KernelAbstractions`' reach that a host caller may hand to `rgrad`.
struct OpaqueMatrix{T} <: AbstractMatrix{T}
    A::Matrix{T}
end

Base.size(A::OpaqueMatrix) = size(A.A)
Base.getindex(A::OpaqueMatrix, i::Int, j::Int) = A.A[i, j]

const N, n = 6, 3
const device = get_backend(JLArray(zeros(1)))

@test !isdefined(GeometricOptimizers, :_match_backend)

@testset "rgrad and the backend of the gradient: $MT, $T" for MT in (StiefelManifold, GrassmannManifold),
    T in (Float32, Float64)

    rng = Random.Xoshiro(11)
    point = Matrix{T}(qr(randn(rng, T, N, n)).Q)[:, 1:n]
    host_gradient = randn(rng, T, N, n)
    host_Y, device_Y = MT(copy(point)), MT(JLArray(point))

    # a host gradient at a device point is refused at the entry
    @test_throws ArgumentError rgrad(device_Y, host_gradient)
    @test_throws "mixed backends" rgrad(device_Y, host_gradient)

    # a device gradient at a device point stays there and is the host answer
    Δ = rgrad(device_Y, JLArray(host_gradient))
    @test get_backend(Δ) == device
    @test Array(Δ) ≈ rgrad(host_Y, host_gradient) rtol = √eps(T)

    # a host point never asks the gradient for its backend
    @test_throws ArgumentError get_backend(OpaqueMatrix(host_gradient))
    @test rgrad(host_Y, OpaqueMatrix(host_gradient)) ≈ rgrad(host_Y, host_gradient)
    @test rgrad(MT(view(point, :, 1:n)), OpaqueMatrix(host_gradient)) ≈
          rgrad(host_Y, host_gradient)

    # a device gradient at a host point is not refused: on `JLArrays` the products of the host
    # point with the device gradient run, and the projection comes back on the device
    Δ_mixed = rgrad(host_Y, JLArray(host_gradient))
    @test Δ_mixed isa JLArray{T, 2}
    @test Array(Δ_mixed) ≈ rgrad(host_Y, host_gradient) rtol = √eps(T)
end

function device_set(rng, ::Type{T}) where {T}
    point() = Matrix{T}(qr(randn(rng, T, N, n)).Q)[:, 1:n]
    host = NetworkParameters((
        Y = StiefelManifold(point()), G = GrassmannManifold(point()),
        S = rand(rng, SymmetricMatrix{T}, n), K = rand(rng, SkewSymMatrix{T}, n),
        Lo = rand(rng, StrictlyLowerTriangular{T}, n), Up = rand(rng, StrictlyUpperTriangular{T}, n),
        W = randn(rng, T, n, 2)))
    host, mapstorage(JLArray, host)
end

storages(ps) = foldstorage((acc, s) -> (acc..., s), (), ps)

@testset "every gradient leaf is on the backend of its parameter leaf, $T" for T in (Float32, Float64)
    host, ps = device_set(Random.Xoshiro(12), T)
    F(ps) = foldstorage((acc, s) -> acc + sum(abs2, s), zero(T), ps)
    ∇F!(g, v) = (g .= 2 .* v; g)

    optimizer = Optimizer(ps, F; (∇F!) = ∇F!, algorithm = GradientMethod())
    g = gradient(optimizer)(ps)
    host_g = gradient(Optimizer(host, F; (∇F!) = ∇F!, algorithm = GradientMethod()))(host)

    for k in keys(ps)
        @testset "$k" begin
            for s in storages(getproperty(g, k))
                @test get_backend(s) == device
                @test s isa JLArray{T}
            end
            @test first(flatten(mapstorage(Array, getproperty(g, k)))) ≈
                  first(flatten(getproperty(host_g, k))) rtol = √eps(T)
        end
    end
end

@testset "a parameter set on two backends is refused at construction, $T" for T in (Float32, Float64)
    host, ps = device_set(Random.Xoshiro(13), T)
    mixed = NetworkParameters((Y = ps.Y, W = host.W))
    F(ps) = sum(abs2, parent(ps.Y)) + sum(abs2, ps.W)
    ∇F!(g, v) = (g .= 2 .* v; g)

    @test_throws "mixed backends" Optimizer(mixed, F; (∇F!) = ∇F!, algorithm = GradientMethod())
    @test Optimizer(ps, F; (∇F!) = ∇F!, algorithm = GradientMethod()) isa Optimizer
end
