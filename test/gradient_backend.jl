# `rgrad` tolerates an ambient gradient that is not on the point's backend.
#
# TEMPORARY, together with `GeometricOptimizers._match_backend`, which is what this file pins. The
# defect it works around is not in this package. Delete this file with the shim; the two issues that
# close it are named beside `_match_backend`. `similar_backend.jl` next door pins the *allocation*
# side; this is the *input* side.
#
# `JLArray` stands in for the device, as it does in `similar_backend.jl`: its backend is a
# `KernelAbstractions.GPU`, so `rgrad` takes the mismatched-backend branch without a GPU present.

using GeometricOptimizers
using GeometricOptimizers: GrassmannManifold, StiefelManifold, _match_backend, rgrad
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: qr!
using Random
using Test

# A gradient `KernelAbstractions` cannot place: `parent` returns it, so `get_backend` throws. It
# stands in for any matrix type outside `KernelAbstractions`' reach that a host caller may hand to
# `rgrad`, and it is why the shim must decide from the point alone.
struct OpaqueMatrix{T} <: AbstractMatrix{T}
    A::Matrix{T}
end

Base.size(A::OpaqueMatrix) = size(A.A)
Base.getindex(A::OpaqueMatrix, i::Int, j::Int) = A.A[i, j]

Random.seed!(1234)

const T = Float32
const N, n = 6, 3

const device = KernelAbstractions.get_backend(JLArray(zeros(T, 1)))
const host = KernelAbstractions.get_backend(zeros(T, 1))

# Drawn on the host and moved over rather than through `rand(device, …)`, for the reason
# `similar_backend.jl` gives: one representative for every type below, so a failure names the type.
# The device draw itself works — `device_orthonormalization.jl` is where that is asserted.
const host_point = Matrix(qr!(randn(T, N, n)).Q)[:, 1:n]
const host_gradient = randn(T, N, n)

@testset "$MT" for MT in (StiefelManifold, GrassmannManifold)
    host_Y = MT(copy(host_point))
    device_Y = MT(JLArray(host_point))

    # The point is on a device and the gradient is not. This is the pendulum stage's case, and the
    # whole reason the shim exists.
    @testset "host gradient, device point" begin
        Δ = rgrad(device_Y, host_gradient)

        @test KernelAbstractions.get_backend(Δ) == device
        # The Riemannian gradient itself is unchanged by where it was computed.
        @test Array(Δ) ≈ rgrad(host_Y, host_gradient)
    end

    # Both already on the device: the shim must not insert a copy, so `rgrad` sees the very array it
    # was handed.
    @testset "device gradient, device point" begin
        device_gradient = JLArray(host_gradient)
        Δ = rgrad(device_Y, device_gradient)

        @test _match_backend(device_Y, device_gradient) === device_gradient
        @test KernelAbstractions.get_backend(Δ) == device
        @test Array(Δ) ≈ rgrad(host_Y, host_gradient)
    end

    # The host path is the one every existing caller is on, and it reaches the arithmetic untouched:
    # `_match_backend` returns its second argument identically, without asking that argument for a
    # backend. That is what keeps element types `KernelAbstractions` cannot allocate, such as
    # `ForwardDiff.Dual`, working on a host point.
    @testset "host gradient, host point" begin
        @test _match_backend(host_Y, host_gradient) === host_gradient
        @test KernelAbstractions.get_backend(rgrad(host_Y, host_gradient)) == host
    end

    # The same, for a host point whose storage is not a plain `Array`. The decision has to come from
    # the point's backend rather than from the type of its storage, or a wrapped point sends an
    # unplaceable gradient to `get_backend` and a working host call becomes an `ArgumentError`.
    @testset "host gradient, host point with wrapped storage" begin
        wrapped_Y = MT(view(host_point, :, 1:n))
        opaque_gradient = OpaqueMatrix(host_gradient)

        @test _match_backend(wrapped_Y, host_gradient) === host_gradient
        @test _match_backend(wrapped_Y, opaque_gradient) === opaque_gradient
        @test rgrad(wrapped_Y, host_gradient) ≈ rgrad(host_Y, host_gradient)
    end
end
