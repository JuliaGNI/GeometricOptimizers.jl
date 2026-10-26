# The Stiefel metric traits, the direct Cayley retraction and the two direct vector transports.
#
# Every expected value is a dense expression: the closed form of `rgrad`, the `M × M` Cayley
# transform `(𝕀 - W/2)⁻¹(𝕀 + W/2)Y`, its differential, or a central finite difference of
# `cayley_direct`. Only the finite difference goes through the rank-`2k` factorization, and it
# checks the transport against the retraction, whose own values the dense transform checks.

using GeometricOptimizers
using LinearAlgebra: LinearAlgebra, I, norm, tr
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/allocations.jl")

# The tangent projection at `Y`, `Ξ - Y(YᵀΞ + ΞᵀY)/2`, in dense form.
proj(Y, Ξ) = Ξ - Y * (Y' * Ξ + Ξ' * Y) / 2

# The generator of the Cayley transform at `Y` along `Δ`, as an `M × M` matrix.
generator(Y, Δ) = Δ * Y' - Y * Δ' + Y * Δ' * Y * Y'

dense_cayley(Y, Δ) = (I - generator(Y, Δ) / 2) \ ((I + generator(Y, Δ) / 2) * Y)

function dense_transport(Y, Δ, Ξ)
    L = I - generator(Y, Δ) / 2
    L \ (generator(Y, proj(Y, Ξ)) * (L \ Y))
end

# A point of `St(k, M)` and a tangent vector at it of Frobenius norm `scale`.
function point_and_tangent(rng, ::Type{T}, M, k; scale = 1 / 2) where {T}
    Y = rand(rng, StiefelManifold{T}, M, k)
    Δ = proj(Y.A, randn(rng, T, M, k))
    Y, T(scale) * Δ / norm(Δ)
end

tangent_defect(Y, Δ) = norm(Y' * Δ + Δ' * Y)

# The whole messages of the two size guards. A size error raised later, by `mul!` or a
# broadcast, is also a `DimensionMismatch`, and Base's matrix-product message holds the words
# "a matrix of size" too, so a test of a guard matches its whole message.
const POINT_GUARD = r"^a matrix of size \(\d+, \d+\) at a point of size \(\d+, \d+\)$"
const WORKSPACE_GUARD = r"^a workspace for points of size \(\d+, \d+\) at a point of size \(\d+, \d+\)$"

# `f()` raises a `DimensionMismatch` whose whole message matches `pattern`.
function refuses(f, pattern)
    err = try
        f()
        nothing
    catch e
        e
    end
    err isa DimensionMismatch && occursin(pattern, err.msg)
end

@testset "the guard patterns match only their own guard, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(7)
    Y = rand(rng, StiefelManifold{T}, 20, 4)
    Δ = zero(Y.A)
    Z = StiefelManifold(zero(Y.A))
    ws = DirectCayleyWorkspace(rand(rng, StiefelManifold{T}, 19, 4))
    point = () -> cayley_direct(Y, randn(rng, T, 20, 3))
    workspace = () -> cayley_direct!(Z, Y, Δ, ws)
    @test refuses(point, POINT_GUARD)
    @test !refuses(point, WORKSPACE_GUARD)
    @test refuses(workspace, WORKSPACE_GUARD)
    @test !refuses(workspace, POINT_GUARD)
    # the size errors of a matrix product and of a broadcast
    for f in (() -> randn(rng, T, 20, 4) * randn(rng, T, 5, 4),
        () -> randn(rng, T, 20, 4) .+ randn(rng, T, 20, 3))
        @test_throws DimensionMismatch f()
        @test !refuses(f, POINT_GUARD)
        @test !refuses(f, WORKSPACE_GUARD)
    end
end

@testset "Stiefel metric traits" begin
    @test CanonicalMetric() isa StiefelMetric
    @test EuclideanMetric() isa StiefelMetric

    @testset "$T" for T in REAL_ELTYPES
        rng = Random.Xoshiro(1)
        M, k = 16, 4
        Y = rand(rng, StiefelManifold{T}, M, k)
        for trial in 1:20
            G = randn(rng, T, M, k)
            canonical = rgrad(Y, G, CanonicalMetric())
            euclidean = rgrad(Y, G, EuclideanMetric())
            @test eltype(canonical) === T
            @test eltype(euclidean) === T
            # the two-argument form is the canonical metric, bit for bit
            @test canonical == rgrad(Y, G)
            @test norm(canonical - (G - Y.A * G' * Y.A)) ≤ 10eps(T) * norm(G)
            @test norm(euclidean - (G - Y.A * (Y.A' * G + G' * Y.A) / 2)) ≤
                  10eps(T) * norm(G)
            @test tangent_defect(Y.A, canonical) ≤ 10eps(T) * norm(G)
            @test tangent_defect(Y.A, euclidean) ≤ 10eps(T) * norm(G)

            # the Riesz identity: the gradient of each metric represents `Δ ↦ tr(GᵀΔ)`
            Δ = proj(Y.A, randn(rng, T, M, k))
            for m in (CanonicalMetric(), EuclideanMetric())
                value = metric(Y, rgrad(Y, G, m), Δ, m)
                @test value isa T
                @test abs(value - tr(G' * Δ)) ≤ 20eps(T) * norm(G) * norm(Δ)
            end
            Δ₂ = proj(Y.A, randn(rng, T, M, k))
            @test metric(Y, Δ, Δ₂, CanonicalMetric()) == metric(Y, Δ, Δ₂)
            @test abs(metric(Y, Δ, Δ₂, EuclideanMetric()) - tr(Δ' * Δ₂)) ≤
                  10eps(T) * norm(Δ) * norm(Δ₂)
        end

        # the two metrics differ on a tangent vector with a component in the span of `Y`
        Δ = proj(Y.A, randn(rng, T, M, k))
        @test !isapprox(metric(Y, Δ, Δ, CanonicalMetric()), metric(Y, Δ, Δ, EuclideanMetric()))

        wrong = randn(rng, T, M, k + 1)
        for m in (CanonicalMetric(), EuclideanMetric())
            @test refuses(() -> rgrad(Y, wrong, m), POINT_GUARD)
            @test refuses(() -> rgrad(Y, randn(rng, T, M + 1, k), m), POINT_GUARD)
            @test refuses(() -> metric(Y, wrong, Δ, m), POINT_GUARD)
            @test refuses(() -> metric(Y, Δ, wrong, m), POINT_GUARD)
            # two arguments of one wrong size have a well-defined dot product
            @test refuses(() -> metric(Y, wrong, wrong, m), POINT_GUARD)
        end
    end
end

@testset "direct Cayley retraction, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(2)
    # `k = 1` has `A = 0`, and `k = M` has `K = 0`
    for (M, k) in ((20, 4), (20, 1), (5, 5)), trial in 1:10

        Y, Δ = point_and_tangent(rng, T, M, k)
        Y_new = cayley_direct(Y, Δ)
        @test Y_new isa StiefelManifold{T}
        @test eltype(Y_new.A) === T
        reference = dense_cayley(Float64.(Y.A), Float64.(Δ))
        @test norm(Y_new.A - reference) ≤ 50eps(T)
        @test norm(Y_new.A' * Y_new.A - I) ≤ 20eps(T)

        # the in-place form returns its first argument and agrees with the out-of-place one
        ws = DirectCayleyWorkspace(Y)
        Z = StiefelManifold(zero(Y.A))
        @test cayley_direct!(Z, Y, Δ, ws) === Z
        @test Z.A == Y_new.A

        # writing into the point itself gives the same bits as writing into another one
        Y_alias = StiefelManifold(copy(Y.A))
        cayley_direct!(Y_alias, Y_alias, Δ, ws)
        @test Y_alias.A == Y_new.A

        # a zero step returns the point
        cayley_direct!(Z, Y, zero(Y.A), ws)
        @test norm(Z.A - Y.A) ≤ 10eps(T)
        @test Z.A == Y.A
    end

    Y, Δ = point_and_tangent(rng, T, 20, 4)
    ws = DirectCayleyWorkspace(Y)
    Z = StiefelManifold(zero(Y.A))
    @test refuses(() -> cayley_direct(Y, randn(rng, T, 20, 3)), POINT_GUARD)
    @test refuses(() -> cayley_direct(Y, randn(rng, T, 19, 4)), POINT_GUARD)
    @test refuses(() -> cayley_direct!(Z, Y, randn(rng, T, 20, 3), ws), POINT_GUARD)
    @test refuses(() -> cayley_direct!(StiefelManifold(zeros(T, 20, 3)), Y, Δ, ws),
        POINT_GUARD)
    for M_ws in (21, 19)
        @test refuses(
            () -> cayley_direct!(
                Z, Y, Δ, DirectCayleyWorkspace(rand(rng, StiefelManifold{T}, M_ws, 4))),
            WORKSPACE_GUARD)
    end
    # a refused call leaves the output alone
    @test Z.A == zero(Y.A)
end

@testset "direct Cayley retraction: first-order slope" begin
    rng = Random.Xoshiro(3)
    τ = 1.0e-4
    for (M, k) in ((20, 4), (20, 1), (5, 5)), trial in 1:10

        Y, Δ = point_and_tangent(rng, Float64, M, k)
        slope = (cayley_direct(Y, τ * Δ).A - Y.A) / τ
        @test norm(slope - Δ) ≤ 2τ * norm(Δ)^2
    end
end

@testset "vector transports, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(4)
    @test ProjectionTransport() isa AbstractVectorTransport
    @test DifferentiatedCayleyTransport() isa AbstractVectorTransport
    for (M, k) in ((20, 4), (20, 1), (5, 5)), trial in 1:10

        Y, Δ = point_and_tangent(rng, T, M, k)
        Y_new = cayley_direct(Y, Δ)
        ws = DirectCayleyWorkspace(Y)
        # a tangent vector at `Y`, and a vector off the tangent space with the same projection
        Ξ_t = proj(Y.A, randn(rng, T, M, k))
        S = randn(rng, T, k, k)
        Ξ = Ξ_t + Y.A * (S + S')

        projected = transport(Y_new, Y, Δ, Ξ, ProjectionTransport())
        @test eltype(projected) === T
        @test norm(projected - proj(Y_new.A, Ξ)) ≤ 10eps(T) * norm(Ξ)
        @test tangent_defect(Y_new.A, projected) ≤ 50eps(T) * norm(Ξ)

        differentiated = transport(Y_new, Y, Δ, Ξ, DifferentiatedCayleyTransport())
        @test eltype(differentiated) === T
        reference = dense_transport(Float64.(Y.A), Float64.(Δ), Float64.(Ξ))
        @test norm(differentiated - reference) ≤ 50eps(T) * norm(Ξ)
        @test tangent_defect(Y_new.A, differentiated) ≤ 50eps(T) * norm(Ξ)
        # the part of `Ξ` off the tangent space at `Y` is dropped
        @test norm(differentiated -
                   transport(Y_new, Y, Δ, Ξ_t, DifferentiatedCayleyTransport())) ≤
              50eps(T) * norm(Ξ)
        # it reads only the size of `Y_new`: another point gives the same result
        @test transport(Y, Y, Δ, Ξ, DifferentiatedCayleyTransport()) == differentiated

        # the in-place forms agree with the out-of-place ones, also when `Ξ_out === Ξ`
        for (tr, expected) in ((ProjectionTransport(), projected),
            (DifferentiatedCayleyTransport(), differentiated))
            out = zero(Ξ)
            @test transport!(out, Y_new, Y, Δ, Ξ, tr, ws) === out
            @test out == expected
            Ξ_alias = copy(Ξ)
            transport!(Ξ_alias, Y_new, Y, Δ, Ξ_alias, tr, ws)
            @test Ξ_alias == expected
        end

        # at a zero step both transports are the tangent projection at `Y`
        Δ₀ = zero(Y.A)
        for tr in (ProjectionTransport(), DifferentiatedCayleyTransport())
            @test norm(transport(Y, Y, Δ₀, Ξ, tr) - Ξ_t) ≤ 10eps(T) * norm(Ξ)
        end

        wrong = randn(rng, T, M, k + 1)
        for tr in (ProjectionTransport(), DifferentiatedCayleyTransport())
            @test refuses(() -> transport(Y_new, Y, Δ, wrong, tr), POINT_GUARD)
            @test refuses(() -> transport(Y_new, Y, wrong, Ξ, tr), POINT_GUARD)
            @test refuses(
                () -> transport(StiefelManifold(zeros(T, M + 1, k)), Y, Δ, Ξ, tr),
                POINT_GUARD)
            @test refuses(() -> transport!(copy(wrong), Y_new, Y, Δ, Ξ, tr, ws),
                POINT_GUARD)
            @test refuses(
                () -> transport!(zero(Ξ), Y_new, Y, Δ, Ξ, tr,
                    DirectCayleyWorkspace(rand(rng, StiefelManifold{T}, M + 1, k))),
                WORKSPACE_GUARD)
            # a refused call leaves the output alone
            out = copy(wrong)
            @test refuses(() -> transport!(out, Y_new, Y, Δ, Ξ, tr, ws), POINT_GUARD)
            @test out == wrong
        end
        # the guard refuses a `Y_new` with one column, which the differentiated transport would
        # otherwise accept, because it reads no entry of `Y_new`
        if k > 1
            @test refuses(
                () -> transport(StiefelManifold(randn(rng, T, M, 1)), Y, Δ, Ξ,
                    DifferentiatedCayleyTransport()),
                POINT_GUARD)
        end
    end
end

@testset "differentiated Cayley transport is the derivative of the retraction" begin
    rng = Random.Xoshiro(5)
    s = 1.0e-6
    for (M, k) in ((20, 4), (20, 1), (5, 5)), trial in 1:10

        Y, Δ = point_and_tangent(rng, Float64, M, k)
        Y_new = cayley_direct(Y, Δ)
        Ξ_t = proj(Y.A, randn(rng, M, k))
        S = randn(rng, k, k)
        for Ξ in (Ξ_t, Ξ_t + Y.A * (S + S'))
            difference = (cayley_direct(Y, Δ + s * Ξ).A - cayley_direct(Y, Δ - s * Ξ).A) /
                         2s
            @test norm(transport(Y_new, Y, Δ, Ξ, DifferentiatedCayleyTransport()) -
                       difference) ≤
                  1.0e-8
        end
    end
end

# A workspace whose arrays are `O(Mk + k²)`, and a retraction whose allocation does not grow with
# `M`. One `M × M` temporary at `M = 200` is 160 kB in `Float32`, far above the 1024 B tolerance.
function retraction_allocations(Y_new, Y, Δ, ws)
    cayley_direct!(Y_new, Y, Δ, ws)
    @allocated cayley_direct!(Y_new, Y, Δ, ws)
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
        arrays = [getfield(ws, f)
                  for f in fieldnames(typeof(ws))
                  if getfield(ws, f) isa AbstractArray]
        @test !isempty(arrays)
        @test all(a -> length(a) ≤ 2k * M, arrays)
        @test all(a -> eltype(a) === T, arrays)
        Y_new = StiefelManifold(zero(Y.A))
        Ξ = randn(rng, T, M, k)
        out = zero(Ξ)
        (retraction_allocations(Y_new, Y, Δ, ws),
            transport_allocations(out, Y_new, Y, Δ, Ξ, ProjectionTransport(), ws),
            transport_allocations(out, Y_new, Y, Δ, Ξ, DifferentiatedCayleyTransport(), ws))
    end
    for i in 1:3
        @test n_independent(measured[1][i], measured[2][i])
    end
end

@testset "the new names are exported" begin
    for name in (:StiefelMetric, :CanonicalMetric, :EuclideanMetric,
        :cayley_direct, :cayley_direct!, :DirectCayleyWorkspace,
        :AbstractVectorTransport, :ProjectionTransport, :DifferentiatedCayleyTransport,
        :transport, :transport!)
        @test Base.isexported(GeometricOptimizers, name)
    end
end
