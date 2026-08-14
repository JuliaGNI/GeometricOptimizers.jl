using Test
using LinearAlgebra: norm, I
using GeometricOptimizers
using GeometricOptimizers: geodesic, check
using GeometricOptimizers: ScaledSquaring, AugmentedPade, ProjectedSkew, TaylorSeries
import Random

Random.seed!(1234)

# The regression net for bugs.md A1: `geodesic` silently left the manifold for a lift of norm ≳ 50,
# and nothing in the suite would have noticed, because every retraction test took a step of
# `Δ / 1000` and `check` existed for one of the two manifolds.

# `‖B̄‖` throughout is the Frobenius norm of the full `N × N` lift, which is what `bugs.md` and the
# CHANGELOG quote. With `N, n = 20, 3` and this seed the scales below give
# ‖B̄‖ ≈ 0.66, 2.9, 5.9, 18, 39, 64, 180, 384.
const NORM_SCALES = (0.1, 0.5, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0)

const ALGORITHMS = (ScaledSquaring(), AugmentedPade(), ProjectedSkew())

stiefel_lift(T, N, n, s) = T(s) * rand(StiefelLieAlgHorMatrix{T}, N, n)
grassmann_lift(T, N, n, s) = T(s) * rand(GrassmannLieAlgHorMatrix{T}, N, n)

const LIFTS = (("Stiefel", stiefel_lift), ("Grassmann", grassmann_lift))

@testset "every algorithm stays on the manifold at every lift norm" begin
    N, n = 20, 3
    for (name, lift) in LIFTS, s in NORM_SCALES
        B = lift(Float64, N, n, s)
        reference = exp(Matrix(B))

        for algorithm in ALGORITHMS
            Y = geodesic(B, algorithm)
            @test check(Y) < 1e-12
            # ... and it is still the exponential map, not merely something orthogonal. A retraction
            # that re-orthonormalised its result would pass the line above and fail this one.
            @test norm(Matrix(Y) - reference) / norm(reference) < 1e-10
        end
    end
end

@testset "the algorithms agree with each other" begin
    N, n = 20, 3
    for (name, lift) in LIFTS, s in NORM_SCALES
        B = lift(Float64, N, n, s)
        Y = Matrix(geodesic(B, first(ALGORITHMS)))

        for algorithm in ALGORITHMS[2:end]
            @test norm(Matrix(geodesic(B, algorithm)) - Y) / norm(Y) < 1e-10
        end
    end
end

# This is the behaviour every version up to 0.2.0 had, and the reason it is kept: without it the
# table in `TaylorSeries`' docstring is unreproducible. Measured `check` on the Stiefel lifts above:
#
#   ‖B̄‖  |  0.66      5.9       18       39        64       180        384
#   ----- | ------- -------- -------- -------- -------- --------- ----------
#   check | 4.5e-16  2.5e-15  4.0e-12  4.7e-06  2.1e+02   1.3e+67   1.3e+186
#
# Making the termination test relative to the partial sum rather than absolute — the obvious first
# fix — was measured not to change any of these. The loss is cancellation inside the sum, not the
# point at which the summation stops.
@testset "the unscaled series is accurate only for a small lift" begin
    N, n = 20, 3
    for (name, lift) in LIFTS
        @test check(geodesic(lift(Float64, N, n, 1.0), TaylorSeries())) < 1e-12   # ‖B̄‖ ≈ 6
        @test check(geodesic(lift(Float64, N, n, 12.0), TaylorSeries())) > 1e-6   # ‖B̄‖ ≈ 64
        @test check(geodesic(lift(Float64, N, n, 60.0), TaylorSeries())) > 1e-6   # ‖B̄‖ ≈ 384
    end
end

@testset "degenerate shapes" begin
    for (name, lift) in LIFTS, (N, n) in ((5, 5), (5, 1), (4, 2)), algorithm in ALGORITHMS
        # `n == N` leaves `B.B` empty, and for Grassmann it makes the lift identically zero.
        @test check(geodesic(lift(Float64, N, n, 3.0), algorithm)) < 1e-12
    end

    for (name, lift) in LIFTS, algorithm in ALGORITHMS
        # A zero lift has to give back the identity exactly, not to round-off — a scaling loop that
        # divided by a norm rather than testing it would produce a `NaN` here.
        @test Matrix(geodesic(lift(Float64, 6, 2, 0.0), algorithm)) == I
    end
end

# `Float32` has its own floor: `check` cannot go below about `1e-6` there whatever the algorithm, so
# the `1e-12` above is a `Float64` statement and this is the corresponding `Float32` one. Measured
# worst case over the sweep is `5.7e-5` for `ScaledSquaring` and `1.6e-6` for `ProjectedSkew`, the
# latter being flat in `‖B̄‖` because its orthogonality is structural.
@testset "Float32 stays on the manifold to Float32 precision" begin
    N, n = 20, 3
    for (name, lift) in LIFTS, s in NORM_SCALES, algorithm in ALGORITHMS
        @test check(geodesic(lift(Float32, N, n, s), algorithm)) < 1e-4
    end
end

@testset "the Geodesic retraction carries its algorithm" begin
    using GeometricOptimizers: Geodesic, retraction

    B = 60 * rand(StiefelLieAlgHorMatrix{Float64}, 20, 3)

    @test Geodesic().algorithm == ScaledSquaring()
    @test check(Geodesic()(B)) < 1e-12
    @test check(Geodesic(ProjectedSkew())(B)) < 1e-12
    @test check(Geodesic(TaylorSeries())(B)) > 1e-6       # the algorithm is actually consulted
    @test retraction(Geodesic(AugmentedPade()), B) == geodesic(B, AugmentedPade())

    # On a vector space the retraction is addition, so the algorithm has nothing to do.
    x = rand(3, 3)
    @test Geodesic(ProjectedSkew())(x) == x
end
