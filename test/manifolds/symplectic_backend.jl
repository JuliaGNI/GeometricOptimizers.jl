# What a `SymplecticStiefelManifold` does on a device backend, and what it refuses.
#
# Three of its four operations run wherever the point is, and one does not. The split is not
# arbitrary and is the whole content of this file:
#
#   * `check`, `rgrad` and `metric` are products against the Poisson tensor. The tensor is built for
#     the point's backend, so they run there. `metric` also forms `inv(U'U)`, so it needs an `lu`
#     from the backend on top of that — `Metal` supplies one and `JLArrays` does not, which is what
#     the fourth testset below pins.
#   * `global_section` orthogonalizes its completion with the symplectic SR decomposition, which is
#     a host factorization — `sr!`'s factor is materialized with `Matrix`. A device point is refused
#     with a message rather than left to fail inside `sr!`, which is the same answer
#     `rand(::GPU, ::Type{<:SymplecticStiefelManifold}, …)` gives and for the same reason.
#
# `JLArrays` stands in for the device, as it does in `similar_backend.jl` and
# `device_orthonormalization.jl`, and `allowscalar(false)` is what makes this a test rather than a
# description: without it a scalar index merely warns.

using GeometricOptimizers
using GeometricOptimizers: check, global_section, metric, _poisson_tensor
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: CPU
using Random
using Test

Random.seed!(1618)

const T = Float32
const N2, n2 = 6, 4

allowscalar(false)

const host_point = rand(SymplecticStiefelManifold{T}, N2, n2)
device_point() = SymplecticStiefelManifold(JLArray(host_point.A))

@testset "the Poisson tensor is built where its point is" begin
    U = device_point()

    J = _poisson_tensor(U, N2)
    @test J isa JLArray{T, 2}
    @test size(J) == (N2, N2)
    @test Array(J) == _poisson_tensor(CPU(), T, N2)

    # the size is a separate argument because `check` needs both `2N` and `2n` of one point
    @test size(_poisson_tensor(U, n2)) == (n2, n2)

    # and a host point still gets the host spelling, which is not a `KernelAbstractions` allocation
    @test _poisson_tensor(host_point, N2) isa Matrix{T}
end

@testset "check runs on the device and agrees with the host" begin
    U = device_point()

    # Agreement with the host is the assertion, and there is deliberately no absolute bound on the
    # residual. A symplectic factor is not orthogonal, its condition number is unbounded and `sr!`
    # has no re-orthogonalization step, so `check` of a drawn point is not a machine-precision
    # quantity — the entry for `sr` in `CHANGELOG.md` measures it growing with the problem size.
    # At `Float32` and `6 × 4` it is around `2e-5`, which is the manifold's property and not this
    # backend's.
    @test check(U) ≈ check(host_point)
end

@testset "rgrad runs on the device and agrees with the host" begin
    U = device_point()
    ∇L = rand(T, N2, n2)

    Δ = rgrad(U, JLArray(∇L))
    @test Δ isa JLArray{T, 2}
    @test size(Δ) == (N2, n2)
    @test Array(Δ) ≈ rgrad(host_point, ∇L)
end

@testset "the adjoint of a point multiplies without scalar indexing" begin
    # `U'` is where all three of the working operations go — `U'U`, `U'J`, `U'JU` — so without a
    # method of its own the adjoint carries its wrapper into the generic product.
    U = device_point()
    B = JLArray(rand(T, N2, n2))

    @test Array(U' * B) ≈ Array(U.A)' * Array(B)
    @test Array(U' * U) ≈ Array(U.A)' * Array(U.A)
end

@testset "metric stops in lu, which is JLArrays' gap and not this package's" begin
    # `metric` forms `inv(U'U)`. `JLArrays` supplies no `lu`, so the generic fallback scalar-indexes
    # — the same place `cayley` stops on this backend, and not a wrapper.
    #
    # **`Metal` does supply it and `metric` runs there**: measured on an M4 Max under
    # `Metal.allowscalar(false)`, see the pull request. Pinned here so that the day `JLArrays` gains
    # an `lu` this assertion fails and says so.
    #
    # The match is on the message and not on `ErrorException`, for the reason the two `@test_throws`
    # in `device_multiply.jl` give: any `error()` on either path satisfies the type, so the type
    # alone would pin nothing about where this stops.
    U = device_point()
    Δ = JLArray(rgrad(host_point, rand(T, N2, n2)))

    @test_throws "Scalar indexing is disallowed" metric(U, Δ, Δ)
end

@testset "global_section refuses a device point, and says why" begin
    U = device_point()

    @test_throws ArgumentError global_section(U)

    err = try
        global_section(U)
        nothing
    catch e
        e
    end
    # the message has to name the reason and the way out, since raising it here rather than letting
    # `sr!` fail is the whole point
    @test occursin("host", err.msg)
    @test occursin("SymplecticStiefelManifold", err.msg)

    # and the host point is untouched by the refusal
    λ = global_section(host_point)
    @test λ isa Matrix{T}
    @test size(λ) == (N2, N2 - n2)
end
