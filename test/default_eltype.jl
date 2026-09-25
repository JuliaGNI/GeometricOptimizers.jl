# The element type a `rand` uses when the caller names a backend.
#
# The element type nobody named is `default_eltype(backend)`: `Float64` on the host, because that is
# what `zeros(n)` gives, and `Float32` on a device, because that is the width an accelerator is
# built for. It used to be those same two values as bare literals in two `rand` methods, with
# nothing saying why. The values do not change; the rule is now stated and is what this file
# asserts against, so that adding a backend does not mean editing a literal here.
#
# **The rule does not consult `supports_float64`.** A backend being able to hold a `Float64` is not
# a reason to hand it one. An element type the caller *did* name and the backend cannot hold is
# refused by the backend's own allocation, not narrowed.

using GeometricOptimizers
using GeometricOptimizers: default_eltype
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions, CPU, GPU, supports_float64
using Random
using Test

Random.seed!(2026)

const jl_backend = KernelAbstractions.get_backend(JLArray(zeros(Float32, 1)))

# Two stand-in devices, so that both answers a real backend can give are checked where no device is
# available. `Metal` declares `supports_float64` false itself; a backend that declares nothing takes
# KernelAbstractions' `true`, which is what `CUDA` and the `JLArray` backend do.
struct _NoFloat64GPU <: GPU end
KernelAbstractions.supports_float64(::_NoFloat64GPU) = false

struct _Float64GPU <: GPU end

# `_Float64GPU` allocates host arrays, which makes it a device the whole device `rand` will actually
# run on: the real method, the orthonormalization and the type application included. It is here for
# the element type and not for the draw -- `JLArrays` is a real device backend and runs the draw, in
# `device_orthonormalization.jl`, but it declares itself `Float64`-capable and so cannot stand on
# both sides of the distinction the rule turns on.
function KernelAbstractions.allocate(::_Float64GPU, ::Type{T}, dims::Tuple; kwargs...) where {T}
    Array{T}(undef, dims)
end

@testset "the default is the host's width on the host and single precision on a device" begin
    @test default_eltype(CPU()) === Float64
    @test default_eltype(jl_backend) === Float32
    @test default_eltype(_Float64GPU()) === Float32
    @test default_eltype(_NoFloat64GPU()) === Float32
end

@testset "supporting Float64 is not a reason to be handed one" begin
    # the distinction this rule turns on: both stand-ins are devices, they disagree about `Float64`,
    # and the default is the same for both
    @test supports_float64(_Float64GPU())
    @test !supports_float64(_NoFloat64GPU())
    @test default_eltype(_Float64GPU()) === default_eltype(_NoFloat64GPU())

    # and the same holds through `rand` itself, which is where a caller meets it
    for MT in (StiefelManifold, GrassmannManifold)
        @test eltype(rand(Random.default_rng(), _Float64GPU(), MT, 5, 3)) === Float32
    end
end

@testset "the device draw honours a manifold type that names its storage array" begin
    # the same case as on the host path, and the `GPU` arm used to write `MT{typeof(A)}`
    # unconditionally, which is a `TypeError` for an already concrete `MT`
    N, n = 5, 3
    for MT in (StiefelManifold{Float32, Matrix{Float32}},
        GrassmannManifold{Float32, Matrix{Float32}})
        Y = rand(Random.default_rng(), _Float64GPU(), MT, N, n)
        @test typeof(Y) === MT
        @test GeometricOptimizers.check(Y) < 10 * eps(eltype(Y))
    end
end

@testset "a device that carries Float64 draws one when asked" begin
    for T in (Float32, Float64)
        @test eltype(rand(
            Random.default_rng(), _Float64GPU(), StiefelManifold{T}, 5, 3)) === T
    end
end

@testset "the backend-taking `rand` uses the rule on the host" begin
    N, n = 5, 3
    for MT in (StiefelManifold, GrassmannManifold)
        Y = rand(CPU(), MT, N, n)
        # the rule and not the value, so that this line survives a backend being added
        @test eltype(Y) === default_eltype(CPU())
        @test Y isa MT
        @test GeometricOptimizers.check(Y) < 10 * eps(eltype(Y))
    end
end

@testset "naming the element type still fixes it" begin
    N, n = 5, 3
    for T in (Float32, Float64)
        @test eltype(rand(CPU(), StiefelManifold{T}, N, n)) === T
        @test eltype(rand(StiefelManifold{T}, N, n)) === T
    end
end

@testset "a manifold type that names its storage array too" begin
    # `StiefelManifold{T}` and `StiefelManifold{T, AT}` reach the same method. The second used to
    # have one of its own, written for `StiefelManifold` alone, so the same call on
    # `GrassmannManifold{T, AT}` was a `TypeError`.
    N, n = 5, 3
    for MT in (StiefelManifold{Float64, Matrix{Float64}},
        GrassmannManifold{Float32, Matrix{Float32}})
        Y = rand(Random.default_rng(), CPU(), MT, N, n)
        @test typeof(Y) === MT
        @test GeometricOptimizers.check(Y) < 10 * eps(eltype(Y))
    end
end
