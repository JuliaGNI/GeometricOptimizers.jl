# `rgrad` tolerates an ambient gradient that is not on the point's backend.
#
# TEMPORARY, together with `GeometricOptimizers._match_backend`, which is what this file pins. The
# defect it works around is not in this package: see the two issues linked from
# GeometricOptimizers#79. Delete this file when they are closed and the shim comes out.
#
# `similar_backend.jl` next door pins the *allocation* side, which is what made
# `Optimizer(Adam(), network)` a `MethodError` on a device-resident network. This is the next thing
# the same pendulum stage hit once that was fixed, and it is on the *input* side:
# `GeometricMachineLearning`'s `_gml_rgrad(x::Manifold, dp) = rgrad(x, dp)` passes the pullback's
# leaf through unchanged, that leaf came back on the host for a device parameter, and `∇L' * Y.A`
# then pairs a host matrix with a device one. On CUDA that is
# `ArgumentError: Illegal conversion of a CUDA.DeviceMemory to a Ptr{Float32}` from a CPU `gemm!`
# (`GMLDatasets#12`, run `20260903T191704Z_smoke`, RTX 4090).
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

Random.seed!(1234)

const T = Float32
const N, n = 6, 3

const device = KernelAbstractions.get_backend(JLArray(zeros(T, 1)))
const host = KernelAbstractions.get_backend(zeros(T, 1))

# Drawn on the host and moved over rather than through `rand(device, …)`, for the reason
# `similar_backend.jl` gives: the device `rand` materializes its QR factor as `typeof(A)(qr!(A).Q)`,
# which `JLArray` does not implement.
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
    # `_match_backend` returns its second argument identically, without asking either argument for a
    # backend. That is what keeps element types `KernelAbstractions` cannot allocate, such as
    # `ForwardDiff.Dual`, working on a host point.
    @testset "host gradient, host point" begin
        @test _match_backend(host_Y, host_gradient) === host_gradient
        @test KernelAbstractions.get_backend(rgrad(host_Y, host_gradient)) == host
    end
end
