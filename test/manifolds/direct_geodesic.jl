# The direct geodesic retraction and the exponential transport.
#
# Every expected value is a dense expression: the `M × M` exponential `exp(W)Y` of the generator, and
# `exp(W)P_YΞ` for the transport. The retraction is also checked against `geodesic`, which reaches
# the same point through a global section.

using GeometricOptimizers
using GeometricOptimizers: ScaledSquaring, NativePade, AugmentedPade, TaylorSeries, ProjectedSkew
using LinearAlgebra: I, norm
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/allocations.jl")

const ALGORITHMS = (ScaledSquaring(), NativePade(), AugmentedPade(), TaylorSeries())

proj(Y, Ξ) = Ξ - Y * (Y' * Ξ + Ξ' * Y) / 2
generator(Y, Δ) = Δ * Y' - Y * Δ' + Y * Δ' * Y * Y'
dense_geodesic(Y, Δ) = exp(generator(Y, Δ)) * Y
dense_transport(Y, Δ, Ξ) = exp(generator(Y, Δ)) * proj(Y, Ξ)

function point_and_tangent(rng, ::Type{T}, M, k; scale = 1 / 2) where {T}
    Y = rand(rng, StiefelManifold{T}, M, k)
    Δ = proj(Y.A, randn(rng, T, M, k))
    Y, T(scale) * Δ / norm(Δ)
end

tangent_defect(Y, Δ) = norm(Y' * Δ + Δ' * Y)

const POINT_GUARD = r"^a matrix of size \(\d+, \d+\) at a point of size \(\d+, \d+\)$"
const WORKSPACE_GUARD = r"^a workspace for points of size \(\d+, \d+\) at a point of size \(\d+, \d+\)$"
const PROJECTED_SKEW_GUARD = "ProjectedSkew exponentiates a skew-symmetric matrix, and the " *
                             "2k × 2k matrix VU of a direct geodesic is not skew-symmetric; use " *
                             "another exponential algorithm"

function refuses(f, ::Type{E}, pattern) where {E}
    err = try
        f()
        nothing
    catch e
        e
    end
    err isa E && occursin(pattern, err.msg)
end

@testset "direct geodesic retraction, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(2)
    # `k = 1` has `A = 0`, and `k = M` has `K = 0`; the scale 5 halves the argument of `𝔄`
    for (M, k) in ((20, 4), (20, 1), (5, 5)), scale in (1 / 2, 5), trial in 1:5

        Y, Δ = point_and_tangent(rng, T, M, k; scale)
        reference = dense_geodesic(Float64.(Y.A), Float64.(Δ))
        for algorithm in ALGORITHMS
            Y_new = geodesic_direct(Y, Δ, algorithm)
            @test Y_new isa StiefelManifold{T}
            @test eltype(Y_new.A) === T
            @test norm(Y_new.A - reference) ≤ 200eps(T)
            @test norm(Y_new.A' * Y_new.A - I) ≤ 200eps(T)
        end
        # the default algorithm is `ScaledSquaring`, and the point is the one `geodesic` reaches
        # through a global section
        Y_new = geodesic_direct(Y, Δ)
        @test Y_new.A == geodesic_direct(Y, Δ, ScaledSquaring()).A
        @test norm(Y_new.A - geodesic(Y, Δ).A) ≤ 200eps(T)

        ws = DirectCayleyWorkspace(Y)
        Z = StiefelManifold(zero(Y.A))
        @test geodesic_direct!(Z, Y, Δ, ws) === Z
        @test Z.A == Y_new.A

        Y_alias = StiefelManifold(copy(Y.A))
        geodesic_direct!(Y_alias, Y_alias, Δ, ws)
        @test Y_alias.A == Y_new.A

        geodesic_direct!(Z, Y, zero(Y.A), ws)
        @test Z.A == Y.A
    end

    # a step off the tangent space is the retraction of its tangent projection
    Y, Δ = point_and_tangent(rng, T, 20, 4)
    S = randn(rng, T, 4, 4)
    @test norm(geodesic_direct(Y, Δ + Y.A * (S + S')).A - geodesic_direct(Y, Δ).A) ≤ 50eps(T)

    ws = DirectCayleyWorkspace(Y)
    Z = StiefelManifold(zero(Y.A))
    @test refuses(() -> geodesic_direct(Y, randn(rng, T, 20, 3)), DimensionMismatch, POINT_GUARD)
    @test refuses(() -> geodesic_direct!(Z, Y, randn(rng, T, 20, 3), ws), DimensionMismatch,
        POINT_GUARD)
    @test refuses(() -> geodesic_direct!(StiefelManifold(zeros(T, 20, 3)), Y, Δ, ws),
        DimensionMismatch, POINT_GUARD)
    @test refuses(
        () -> geodesic_direct!(
            Z, Y, Δ, DirectCayleyWorkspace(rand(rng, StiefelManifold{T}, 21, 4))),
        DimensionMismatch, WORKSPACE_GUARD)
    @test refuses(() -> geodesic_direct!(Z, Y, Δ, ws, ProjectedSkew()), ArgumentError,
        PROJECTED_SKEW_GUARD)
    @test Z.A == zero(Y.A)
end

@testset "direct geodesic retraction: first-order slope" begin
    rng = Random.Xoshiro(3)
    τ = 1.0e-4
    for (M, k) in ((20, 4), (20, 1), (5, 5)), trial in 1:10

        Y, Δ = point_and_tangent(rng, Float64, M, k)
        slope = (geodesic_direct(Y, τ * Δ).A - Y.A) / τ
        @test norm(slope - Δ) ≤ 2τ * norm(Δ)^2
    end
end

@testset "exponential transport, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(4)
    @test ExponentialTransport() isa AbstractVectorTransport
    @test ExponentialTransport() === ExponentialTransport(ScaledSquaring())
    for (M, k) in ((20, 4), (20, 1), (5, 5)), trial in 1:10

        Y, Δ = point_and_tangent(rng, T, M, k)
        Y_new = geodesic_direct(Y, Δ)
        ws = DirectCayleyWorkspace(Y)
        Ξ_t = proj(Y.A, randn(rng, T, M, k))
        S = randn(rng, T, k, k)
        Ξ = Ξ_t + Y.A * (S + S')

        reference = dense_transport(Float64.(Y.A), Float64.(Δ), Float64.(Ξ))
        for algorithm in ALGORITHMS
            transported = transport(Y_new, Y, Δ, Ξ, ExponentialTransport(algorithm))
            @test eltype(transported) === T
            @test norm(transported - reference) ≤ 200eps(T) * norm(Ξ)
            @test tangent_defect(Y_new.A, transported) ≤ 200eps(T) * norm(Ξ)
        end
        transported = transport(Y_new, Y, Δ, Ξ, ExponentialTransport())
        # the part of `Ξ` off the tangent space at `Y` is dropped, and `exp(W)` is orthogonal
        @test norm(transported - transport(Y_new, Y, Δ, Ξ_t, ExponentialTransport())) ≤
              50eps(T) * norm(Ξ)
        @test abs(norm(transported) - norm(Ξ_t)) ≤ 50eps(T) * norm(Ξ)
        # `exp(W)` carries `Y` to the new point
        @test norm(transport(Y_new, Y, Δ, Y.A * (S - S'), ExponentialTransport()) -
                   Y_new.A * (S - S')) ≤ 50eps(T) * norm(S)
        # it reads only the size of `Y_new`
        @test transport(Y, Y, Δ, Ξ, ExponentialTransport()) == transported

        out = zero(Ξ)
        @test transport!(out, Y_new, Y, Δ, Ξ, ExponentialTransport(), ws) === out
        @test out == transported
        Ξ_alias = copy(Ξ)
        transport!(Ξ_alias, Y_new, Y, Δ, Ξ_alias, ExponentialTransport(), ws)
        @test Ξ_alias == transported

        @test norm(transport(Y, Y, zero(Y.A), Ξ, ExponentialTransport()) - Ξ_t) ≤
              10eps(T) * norm(Ξ)

        wrong = randn(rng, T, M, k + 1)
        @test refuses(() -> transport(Y_new, Y, Δ, wrong, ExponentialTransport()),
            DimensionMismatch, POINT_GUARD)
        @test refuses(() -> transport(Y_new, Y, wrong, Ξ, ExponentialTransport()),
            DimensionMismatch, POINT_GUARD)
        @test refuses(
            () -> transport!(zero(Ξ), Y_new, Y, Δ, Ξ, ExponentialTransport(),
                DirectCayleyWorkspace(rand(rng, StiefelManifold{T}, M + 1, k))),
            DimensionMismatch, WORKSPACE_GUARD)
        @test refuses(() -> transport(Y_new, Y, Δ, Ξ, ExponentialTransport(ProjectedSkew())),
            ArgumentError, PROJECTED_SKEW_GUARD)
    end
end

function retraction_allocations(Y_new, Y, Δ, ws, algorithm)
    geodesic_direct!(Y_new, Y, Δ, ws, algorithm)
    @allocated geodesic_direct!(Y_new, Y, Δ, ws, algorithm)
end

function transport_allocations(out, Y_new, Y, Δ, Ξ, tr, ws)
    transport!(out, Y_new, Y, Δ, Ξ, tr, ws)
    @allocated transport!(out, Y_new, Y, Δ, Ξ, tr, ws)
end

@testset "allocations do not grow with M, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(6)
    k = 4
    measured = map((200, 800)) do M
        Y, Δ = point_and_tangent(rng, T, M, k)
        ws = DirectCayleyWorkspace(Y)
        @test all(a -> length(a) ≤ 16k^2 && eltype(a) === T, ws.𝔄_scratch)
        Y_new = StiefelManifold(zero(Y.A))
        Ξ = randn(rng, T, M, k)
        (retraction_allocations(Y_new, Y, Δ, ws, ScaledSquaring()),
            retraction_allocations(Y_new, Y, Δ, ws, NativePade()),
            transport_allocations(zero(Ξ), Y_new, Y, Δ, Ξ, ExponentialTransport(), ws))
    end
    for i in 1:3
        @test n_independent(measured[1][i], measured[2][i])
    end
    # the native algorithms evaluate `𝔄` in the workspace
    @test measured[1][1] == 0
    @test measured[1][2] == 0
    @test measured[1][3] == 0
end

@testset "the new names are exported" begin
    for name in (:geodesic_direct, :geodesic_direct!, :ExponentialTransport)
        @test Base.isexported(GeometricOptimizers, name)
    end
end
