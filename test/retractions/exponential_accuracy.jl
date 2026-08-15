using Test
using LinearAlgebra: norm, opnorm, I
using GeometricOptimizers
using GeometricOptimizers: geodesic, check, rgrad, 𝔄, 𝔄exp, opnorm₁, Geodesic, retraction
using GeometricOptimizers: ScaledSquaring, AugmentedPade, ProjectedSkew, TaylorSeries
import Random

Random.seed!(1234)

# The regression net for bugs.md A1: `geodesic` silently left the manifold for a lift of norm ≳ 50,
# and nothing in the suite would have noticed, because every retraction test took a step of
# `Δ / 1000` and `check` existed for one of the two manifolds.

# `‖B̄‖` throughout is the Frobenius norm of the full `N × N` lift, which is what `bugs.md` and the
# CHANGELOG quote. With `N, n = 20, 3` and this seed the scales below give
#
#     Stiefel     0.66  2.88  5.93  18.3  39.4  64.0  180  384
#     Grassmann   0.53  3.22  5.37  15.9  35.5  69.3  197  336
#
# Each pass through the loops draws its own lift, so the two rows differ; the Stiefel row is the one
# the docstrings quote. Do NOT wrap the loop bodies below in a nested `@testset` to label them:
# `@testset` restores the global RNG to the same state on entry to every one of its bodies, so each
# nested set would draw an identical lift and the sweep would silently collapse to one matrix.
const NORM_SCALES = (0.1, 0.5, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0)

const ALGORITHMS = (ScaledSquaring(), AugmentedPade(), ProjectedSkew())

stiefel_lift(T, N, n, s) = T(s) * rand(StiefelLieAlgHorMatrix{T}, N, n)
grassmann_lift(T, N, n, s) = T(s) * rand(GrassmannLieAlgHorMatrix{T}, N, n)

const LIFTS = (("Stiefel", stiefel_lift), ("Grassmann", grassmann_lift))

@testset "every algorithm stays on the manifold at every lift norm" begin
    N, n = 20, 3
    for (_, lift) in LIFTS, s in NORM_SCALES
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
    for (_, lift) in LIFTS, s in NORM_SCALES
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
    for (_, lift) in LIFTS
        @test check(geodesic(lift(Float64, N, n, 1.0), TaylorSeries())) < 1e-12   # ‖B̄‖ ≈ 6
        @test check(geodesic(lift(Float64, N, n, 12.0), TaylorSeries())) > 1e-6   # ‖B̄‖ ≈ 64
        @test check(geodesic(lift(Float64, N, n, 60.0), TaylorSeries())) > 1e-6   # ‖B̄‖ ≈ 384
    end
end

@testset "degenerate shapes" begin
    for (_, lift) in LIFTS, (N, n) in ((5, 5), (5, 1), (4, 2)), algorithm in ALGORITHMS
        # `n == N` leaves `B.B` empty, and for Grassmann it makes the lift identically zero.
        @test check(geodesic(lift(Float64, N, n, 3.0), algorithm)) < 1e-12
    end

    for (_, lift) in LIFTS, algorithm in ALGORITHMS
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
    for (_, lift) in LIFTS, s in NORM_SCALES, algorithm in ALGORITHMS
        @test check(geodesic(lift(Float32, N, n, s), algorithm)) < 1e-4
    end
end

@testset "the Geodesic retraction carries its algorithm" begin
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

# The tangent-vector entry point used to hard-call `geodesic(B)`, so an algorithm could only be
# selected by reaching past it to a lift. These assertions are what say it is threaded through.
@testset "geodesic(Y, Δ, algorithm) selects the algorithm" begin
    N, n = 20, 3
    Y = rand(StiefelManifold{Float64}, N, n)
    Δ = rgrad(Y, rand(Float64, N, n))

    # `GlobalSection` completes `Y` with a random basis of its orthogonal complement, so two calls
    # take different sections and agree only to round-off — which is itself the statement that the
    # retracted point does not depend on the section.
    @test geodesic(Y, Δ) ≈ geodesic(Y, Δ, ScaledSquaring()) rtol = 1e-10

    for algorithm in ALGORITHMS
        @test check(geodesic(Y, 300 * Δ, algorithm)) < 1e-12
    end

    # A step this long is where the unscaled series comes apart, so it is also where the argument
    # demonstrably reaches the exponential rather than being dropped on the floor. Negated rather
    # than `> 1e-6`: at this step the series overflows and `check` is `NaN`, which fails every
    # ordered comparison, including the one that is trying to assert that it is bad.
    @test !(check(geodesic(Y, 300 * Δ, TaylorSeries())) < 1e-6)
end

# `ScaledSquaring` is the default because it is the only algorithm free of dense LAPACK, which makes
# it the only one that runs on a `KernelAbstractions` GPU backend. `LinearAlgebra.opnorm(X, 1)` is a
# scalar-indexing double loop and would give that up, so the 1-norm is taken as a reduction instead.
# GPU-ness itself is not testable without a GPU; what is testable is that the substitute is the same
# number, which is the part that could silently regress.
@testset "the scaling threshold is the 1-norm, taken as a reduction" begin
    Random.seed!(99)

    for T in (Float64, Float32), m in (1, 2, 6, 20)
        X = randn(T, m, m) * T(10)
        @test opnorm₁(X) ≈ opnorm(X, 1) rtol = 8 * eps(T)
        @test opnorm₁(X) isa T
    end

    # `opnorm` returns zero for a 0×0 argument and so must this, since `maximum` of an empty
    # reduction throws — an `n == 0` lift would otherwise take the scaling path into an exception.
    @test opnorm₁(zeros(Float64, 0, 0)) == 0.0

    # The threshold is what picks the number of halvings, so it has to hold across the sweep too.
    X = 𝔄(randn(6, 6))
    @test opnorm₁(X) ≈ opnorm(X, 1) rtol = 1e-12
end

# `𝔄exp`'s defining property, ``\mathbb{I} + B'\mathfrak{A}(B', B'')(B'')^T = \exp(B'(B'')^T)``,
# swept over shapes and both element types rather than asserted once. The docstrings' jldoctests
# check it for a single 10×2 `StiefelLieAlgHorMatrix` lift in `Float64`; this covers rectangular
# arguments down to 1×1 and pins the element type of the result, which a doctest printing `true`
# cannot.
#
# `𝔄exp` and this sweep both come from GeometricMachineLearning, which carried the one-line wrapper
# and tested it here. It was replicated GeometricOptimizers functionality and moved over when GML
# went onto this package (GeometricMachineLearning#230).
#
# No nested `@testset` in the loop — see the note on RNG state at the top of this file.
@testset "𝔄exp recovers the exponential across shapes and element types" begin
    for T in (Float32, Float64), N in 1:10, n in 1:N
        A = T(0.1) * rand(T, N, n)
        B = T(0.1) * rand(T, N, n)
        @test eltype(𝔄exp(A, B)) == T
        @test exp(A * B') ≈ 𝔄exp(A, B)
    end

    # The `algorithm` form forwards to `𝔄`, so it is defined exactly where `𝔄(X, algorithm)` is:
    # `TaylorSeries`, `ScaledSquaring` and `AugmentedPade`. `ProjectedSkew` is not among them — it is
    # a `geodesic`-level algorithm with its own branch there and no `𝔄` method — so `ALGORITHMS`,
    # which exists for the `geodesic` sweeps above and includes it, is not what to loop over here.
    for algorithm in (TaylorSeries(), ScaledSquaring(), AugmentedPade()), T in (Float32, Float64)
        A = T(0.1) * rand(T, 8, 3)
        B = T(0.1) * rand(T, 8, 3)
        @test eltype(𝔄exp(A, B, algorithm)) == T
        @test exp(A * B') ≈ 𝔄exp(A, B, algorithm)
    end
end

# The sweep above is broad in shape and element type and narrow in the one dimension that decides
# whether the default is right: `T(0.1) * rand(T, N, n)` keeps ‖AB'‖ around 0.1, where every
# algorithm is exact. The default has to hold where the unscaled series does not — the regime
# `geodesic`'s "The default changed in 0.2.0" warning is about — so it is asserted here directly.
#
# The four scales below draw lifts of ‖B̄‖ = 3.8, 36.3, 145.8 and 324.9, at which the unscaled series
# is off by 5e-16, 1e-7, 2e24 and 2e79 respectively; `𝔄exp` defaults to `ScaledSquaring`, which stays
# under 2e-14 throughout. Three of the four assertions below fail if that default moves back. Those
# are the figures the docstring and the CHANGELOG quote, which is why the seed is set here rather
# than inherited: this testset has to draw the same lifts however much runs before it.
@testset "𝔄exp defaults to an algorithm that survives a large argument" begin
    Random.seed!(1234)

    for scale in (1, 10, 50, 100)
        B = scale * rand(StiefelLieAlgHorMatrix, 10, 2)
        B̂, B̄ = GeometricOptimizers.lift_factors(B)
        reference = exp(Matrix(B))

        @test 𝔄exp(B̂, B̄) ≈ reference rtol = 1e-10
        # `geodesic` assembles the same product and defaults the same way, so the two agree; this is
        # what fails if either default moves without the other.
        @test 𝔄exp(B̂, B̄) ≈ Matrix(geodesic(B))
    end
end
