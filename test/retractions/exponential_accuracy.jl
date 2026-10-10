using Test
using JLArrays: JLArray
using GPUArraysCore: allowscalar
using LinearAlgebra: norm, opnorm, I
using GeometricOptimizers
using GeometricOptimizers: geodesic, check, rgrad, 𝔄, opnorm₁, unit_matrix, Geodesic,
                           retraction
using GeometricOptimizers: ScaledSquaring, NativePade, AugmentedPade, ProjectedSkew,
                           TaylorSeries
import Random

include("../helpers/eltypes.jl")
include("../helpers/manifold_tolerance.jl")

# The lifts are drawn from the global generator under this seed, not from a `Random.Xoshiro`: the
# `‖B̄‖` table below and the figures that the docstrings and the CHANGELOG quote (see the last
# testset) are the `Float64` draws of this seed, and a new generator would draw other lifts. Every
# `@testset` starts from the state this call leaves, and so does each of its `T` passes.
Random.seed!(1234)

# The regression net for bugs.md A1: `geodesic` silently left the manifold for a lift of norm ≳ 50,
# and nothing in the suite would have noticed, because every retraction test took a step of
# `Δ / 1000` and `check` existed for one of the two manifolds.

# `‖B̄‖` throughout is the Frobenius norm of the full `N × N` lift, which is what `bugs.md` and the
# CHANGELOG quote. With `N, n = 20, 3`, this seed and `T = Float64` the scales below give
#
#     Stiefel     0.66  2.88  5.93  18.3  39.4  64.0  180  384
#     Grassmann   0.53  3.22  5.37  15.9  35.5  69.3  197  336
#
# Each pass through the loops draws its own lift, so the two rows differ; the Stiefel row is the one
# the docstrings quote. Do NOT wrap the loop bodies below in a nested `@testset` to label them:
# `@testset` restores the global RNG to the same state on entry to every one of its bodies, so each
# nested set would draw an identical lift and the sweep would silently collapse to one matrix.
const NORM_SCALES = (0.1, 0.5, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0)

const ALGORITHMS = (ScaledSquaring(), NativePade(), AugmentedPade(), ProjectedSkew())

stiefel_lift(T, N, n, s) = T(s) * rand(StiefelLieAlgHorMatrix{T}, N, n)
grassmann_lift(T, N, n, s) = T(s) * rand(GrassmannLieAlgHorMatrix{T}, N, n)

const LIFTS = (("Stiefel", stiefel_lift), ("Grassmann", grassmann_lift))

# The bounds below were measured over this sweep under this seed and under nine further seeds, in
# both precisions, as multiples of `eps(T)`; the worst cases are the two precisions' alike:
#
#                                          Float32     Float64
#     check(Y)                             484eps      487eps     held to manifold_tolerance(T)
#     ‖Y - exp(B̄)‖ / ‖exp(B̄)‖              220eps      224eps     held to 1024eps(T)
#     ‖Y - Y_ScaledSquaring‖ / ‖Y‖         286eps      256eps     held to 1024eps(T)
#     ‖Y_NativePade - Y_AugmentedPade‖     112eps      111eps     held to  512eps(T)
#
# `check` is `ScaledSquaring`'s at the largest lift, its squarings each adding a rounding; the
# error against `exp` grows with `‖B̄‖` likewise, from about `10eps(T)` at the smallest scale.
# `ProjectedSkew` stays near `25eps(T)` at every lift, its orthogonality being structural. The
# reference `exp` is taken of the lift in `Float64` in both passes, so that the `Float32` pass
# measures its own error and not the reference's.
exponential_tolerance(::Type{T}) where {T} = 1024 * eps(T)

@testset "every algorithm stays on the manifold at every lift norm, $T" for T in REAL_ELTYPES
    N, n = 20, 3
    for (_, lift) in LIFTS, s in NORM_SCALES

        B = lift(T, N, n, s)
        reference = exp(Matrix{Float64}(Matrix(B)))

        for algorithm in ALGORITHMS
            Y = geodesic(B, algorithm)
            @test eltype(Y) == T
            @test check(Y) < manifold_tolerance(T)
            # ... and it is still the exponential map, not merely something orthogonal. A retraction
            # that re-orthonormalised its result would pass the line above and fail this one.
            @test norm(Matrix(Y) - reference) / norm(reference) < exponential_tolerance(T)
        end
    end
end

@testset "the algorithms agree with each other, $T" for T in REAL_ELTYPES
    N, n = 20, 3
    for (_, lift) in LIFTS, s in NORM_SCALES

        B = lift(T, N, n, s)
        Y = Matrix(geodesic(B, first(ALGORITHMS)))
        @test eltype(Y) == T

        for algorithm in ALGORITHMS[2:end]
            @test norm(Matrix(geodesic(B, algorithm)) - Y) / norm(Y) <
                  exponential_tolerance(T)
        end
    end
end

# `512eps(T)` is a factor 4.5 over the measured worst case. `NativePade`'s `[7/6]` approximant is
# exact through the `X¹²` term. With the `X⁶` term of its numerator dropped, the error here was
# measured at `1.6e-13` to `4.6e-13` (`720eps` to `2060eps`) at five of the sixteen `Float64` lifts,
# which this bound catches and a bound of `1e-10` would not; in `Float32` it is below round-off.
@testset "NativePade agrees with AugmentedPade across manifolds, $T" for T in REAL_ELTYPES
    N, n = 20, 3
    for (_, lift) in LIFTS, s in NORM_SCALES

        B = lift(T, N, n, s)
        reference = Matrix(geodesic(B, AugmentedPade()))
        result = Matrix(geodesic(B, NativePade()))
        @test eltype(result) == T
        relative_error = norm(result - reference) / norm(reference)
        @test relative_error < 512 * eps(T)
    end
end

# `NativePade`'s Newton--Schulz count is fixed at five, so its threshold is a ceiling and not a
# preference: past `θ ≈ 1` the inverse it computes stops being one and *nothing errors*. Measured
# worst relative error over 400 random 6×6 arguments of one-norm exactly `θ` is `6e-16` at `θ = 1`,
# `1.2e-10` at `θ = 3/2`, `1.1e-5` at `θ = 2` and `169` at `θ = 3`. `ScaledSquaring` has no such
# limit — its own docstring sweeps `θ` over `[0.125, 4]` and finds nothing to choose between — so the
# two constructors do *not* accept the same arguments, and that asymmetry is what this pins.
@testset "NativePade's threshold is bounded where ScaledSquaring's is not" begin
    @test NativePade(0.5).θ == 0.5
    @test NativePade(1 // 8).θ == 1 // 8
    @test NativePade().θ == ScaledSquaring().θ

    @test_throws AssertionError NativePade(0.0)
    @test_throws AssertionError NativePade(-0.5)
    @test_throws AssertionError NativePade(1.0)
    @test_throws AssertionError NativePade(4.0)
end

# The comparison that makes the point: the same values are fine for `ScaledSquaring`, and it really
# does stay accurate at all of them. Measured worst relative error over ten seeds: `47eps(T)` in
# `Float32` and `92eps(T)` in `Float64`, under `exponential_tolerance(T)`.
@testset "ScaledSquaring stays accurate at the thresholds NativePade refuses, $T" for T in REAL_ELTYPES
    B = T(30) * rand(StiefelLieAlgHorMatrix{T}, 20, 3)
    reference = exp(Matrix{Float64}(Matrix(B)))
    for θ in (1.0, 2.0, 4.0)
        Y = Matrix(geodesic(B, ScaledSquaring(θ)))
        @test eltype(Y) == T
        @test norm(Y - reference) / norm(reference) < exponential_tolerance(T)
    end
end

# This is the behaviour every version up to 0.2.0 had, and the reason it is kept: without it the
# table in `TaylorSeries`' docstring is unreproducible. Measured `check` on the `Float64` Stiefel
# lifts above:
#
#   ‖B̄‖  |  0.66      5.9       18       39        64       180        384
#   ----- | ------- -------- -------- -------- -------- --------- ----------
#   check | 4.5e-16  2.5e-15  4.0e-12  4.7e-06  2.1e+02   1.3e+67   1.3e+186
#
# Making the termination test relative to the partial sum rather than absolute — the obvious first
# fix — was measured not to change any of these. The loss is cancellation inside the sum, not the
# point at which the summation stops.
#
# "Off the manifold" is `check` at or above `manifold_tolerance(T)`, the bound every algorithm meets
# above. Measured over five seeds, the series' `check` is at most `19eps(T)` at `‖B̄‖ ≈ 6` in both
# precisions, at least `44` at `‖B̄‖ ≈ 64` in `Float64` and `1e17` in `Float32`, and `NaN` at
# `‖B̄‖ ≈ 384` in `Float32`, where the sum overflows. The two large-lift assertions are negated for
# that `NaN`, which fails every ordered comparison, including one asserting that it is bad.
@testset "the unscaled series is accurate only for a small lift, $T" for T in REAL_ELTYPES
    N, n = 20, 3
    for (_, lift) in LIFTS
        Y = geodesic(lift(T, N, n, 1.0), TaylorSeries())
        @test eltype(Y) == T
        @test check(Y) < manifold_tolerance(T)                               # ‖B̄‖ ≈ 6
        large = check(geodesic(lift(T, N, n, 12.0), TaylorSeries()))        # ‖B̄‖ ≈ 64
        larger = check(geodesic(lift(T, N, n, 60.0), TaylorSeries()))       # ‖B̄‖ ≈ 384
        @test !(large < manifold_tolerance(T))
        @test !(larger < manifold_tolerance(T))
    end
end

# Measured worst `check` over five seeds: `134eps(T)` in `Float32`, `27eps(T)` in `Float64`.
@testset "degenerate shapes, $T" for T in REAL_ELTYPES
    for (_, lift) in LIFTS, (N, n) in ((5, 5), (5, 1), (4, 2)), algorithm in ALGORITHMS
        # `n == N` leaves `B.B` empty, and for Grassmann it makes the lift identically zero.
        Y = geodesic(lift(T, N, n, 3.0), algorithm)
        @test eltype(Y) == T
        @test check(Y) < manifold_tolerance(T)
    end

    for (_, lift) in LIFTS, algorithm in ALGORITHMS
        # A zero lift has to give back the identity exactly, not to round-off — a scaling loop that
        # divided by a norm rather than testing it would produce a `NaN` here.
        Y = Matrix(geodesic(lift(T, 6, 2, 0.0), algorithm))
        @test eltype(Y) == T
        @test Y == I
    end
end

# Measured worst `check` at this lift over five seeds: `357eps(T)` for the default and `25eps(T)` for
# `ProjectedSkew` in `Float32`, `405eps(T)` and `39eps(T)` in `Float64`. The series is `NaN` in
# `Float32` and above `1e150` in `Float64`; the assertion on it is negated for the `NaN`.
@testset "the Geodesic retraction carries its algorithm, $T" for T in REAL_ELTYPES
    B = T(60) * rand(StiefelLieAlgHorMatrix{T}, 20, 3)

    @test Geodesic().algorithm == ScaledSquaring()
    Y = Geodesic()(B)
    @test eltype(Y) == T
    @test check(Y) < manifold_tolerance(T)
    @test check(Geodesic(ProjectedSkew())(B)) < manifold_tolerance(T)
    # the algorithm is actually consulted
    @test !(check(Geodesic(TaylorSeries())(B)) < manifold_tolerance(T))
    @test retraction(Geodesic(AugmentedPade()), B) == geodesic(B, AugmentedPade())

    # On a vector space the retraction is addition, so the algorithm has nothing to do.
    x = rand(T, 3, 3)
    @test Geodesic(ProjectedSkew())(x) == x
end

# The tangent-vector entry point used to hard-call `geodesic(B)`, so an algorithm could only be
# selected by reaching past it to a lift. These assertions are what say it is threaded through.
@testset "geodesic(Y, Δ, algorithm) selects the algorithm, $T" for T in REAL_ELTYPES
    N, n = 20, 3
    Y = rand(StiefelManifold{T}, N, n)
    Δ = rgrad(Y, rand(T, N, n))

    # `GlobalSection` completes `Y` with a random basis of its orthogonal complement, so two calls
    # take different sections and agree only to round-off — which is itself the statement that the
    # retracted point does not depend on the section. Measured over five seeds: at most `3.1eps(T)`
    # apart in `Float32` and `4.6eps(T)` in `Float64`; `64eps(T)` leaves a factor ten.
    result = geodesic(Y, Δ)
    @test eltype(result) == T
    @test result ≈ geodesic(Y, Δ, ScaledSquaring()) rtol = 64 * eps(T)

    # Measured worst `check` over five seeds: `1158eps(T)` in `Float32`, `1328eps(T)` in `Float64`.
    for algorithm in ALGORITHMS
        @test check(geodesic(Y, 300 * Δ, algorithm)) < manifold_tolerance(T)
    end

    # A step this long is where the unscaled series comes apart, so it is also where the argument
    # demonstrably reaches the exponential rather than being dropped on the floor. Negated rather
    # than a `>`: at this step the series overflows and `check` is `NaN`, which fails every
    # ordered comparison, including the one that is trying to assert that it is bad.
    @test !(check(geodesic(Y, 300 * Δ, TaylorSeries())) < manifold_tolerance(T))
end

# `ScaledSquaring` and `NativePade` are free of dense LAPACK and run on a `KernelAbstractions` GPU
# backend. `LinearAlgebra.opnorm(X, 1)` is a scalar-indexing double loop and would give that up, so
# the 1-norm is taken as a reduction instead.
# GPU-ness itself is not testable without a GPU; what is testable is that the substitute is the same
# number, which is the part that could silently regress.
#
# Both sum the same `m` absolute values per column, in an order that may differ, so they agree to the
# rounding of a sum of `m ≤ 20` positive terms; the bound is `8eps(T)`, and the measured worst over
# fifty `6 × 6` draws of `𝔄` is `1.0eps(T)` in `Float32` and `0` in `Float64`.
@testset "the scaling threshold is the 1-norm, taken as a reduction, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(99)

    for m in (1, 2, 6, 20)
        X = randn(rng, T, m, m) * T(10)
        @test opnorm₁(X) ≈ opnorm(X, 1) rtol = 8 * eps(T)
        @test opnorm₁(X) isa T
        @test eltype(opnorm₁(X)) == T
    end

    # `opnorm` returns zero for a 0×0 argument and so must this, since `maximum` of an empty
    # reduction throws — an `n == 0` lift would otherwise take the scaling path into an exception.
    @test opnorm₁(zeros(T, 0, 0)) == zero(T)

    # The threshold is what picks the number of halvings, so it has to hold across the sweep too.
    X = 𝔄(randn(rng, T, 6, 6))
    @test eltype(X) == T
    @test opnorm₁(X) ≈ opnorm(X, 1) rtol = 8 * eps(T)
end

# The property both portable algorithms rest on, and the one issue A19 doubted: no scalar indexing
# anywhere on the path. A `JLArray` is the reference `KernelAbstractions` backend that forbids it, so
# this is testable without a GPU.
#
# `allowscalar(false)` is not redundant. `GPUArraysCore`'s default is `ScalarDisallowed` only in a
# non-interactive session; from a REPL — which is how this file gets debugged — a scalar index would
# merely warn, and every assertion below would pass anyway. The setting is what makes this a test
# rather than a description. It is task-global and left set: everything after this point is an
# `Array`, which never consults it.
#
# Two arguments. A skew one of norm several hundred, whose `exp` is orthogonal in either precision.
# And a general one, not normal, as `𝔄`'s arguments `(B'')ᵀB'` are not; its scale 15 keeps `exp`
# finite in `Float32` and `‖X‖₁` above 100. The reference is `AugmentedPade` of the argument in
# `Float64`. Measured worst relative error over five seeds: `32eps(T)` in `Float32` and `191eps(T)`
# in `Float64` for the skew argument, `52eps(T)` and `78eps(T)` for the general one, under
# `exponential_tolerance(T)`.
@testset "ScaledSquaring and NativePade do not require scalar indexing, $T" for T in REAL_ELTYPES
    allowscalar(false)

    rng = Random.Xoshiro(52)
    entries = randn(rng, T, 8, 8)
    skew = T(50) * (entries - entries')
    general = T(15) * randn(rng, T, 8, 8)
    for dense in (skew, general)
        X = JLArray(dense)
        reference = 𝔄(Matrix{Float64}(dense), AugmentedPade())

        # ‖X‖₁ is above 100, so both take the scaling path with squarings on top of it, and both
        # build the 8×8 identity they need on the backend rather than through `Base.one`.
        @test opnorm₁(X) > 100
        for algorithm in (ScaledSquaring(), NativePade())
            result = 𝔄(X, algorithm)
            @test result isa JLArray
            @test eltype(result) == T
            @test Array(result) ≈ reference rtol = exponential_tolerance(T)
        end
    end

    # `Base.one(::AbstractMatrix)`, the scalar-indexed diagonal write A19 named, spelled out rather
    # than left implicit in the two calls above: this is the substitution they depend on.
    @test unit_matrix(JLArray(skew)) isa JLArray
    @test Array(unit_matrix(JLArray(skew))) == one(skew)
end

# `𝔄exp`'s defining property, ``\mathbb{I} + B'\mathfrak{A}(B', B'')(B'')^T = \exp(B'(B'')^T)``,
# swept over shapes and both element types rather than asserted once. The docstrings' jldoctests
# check it for a single 10×2 `StiefelLieAlgHorMatrix` lift in `Float64`; this covers rectangular
# arguments down to 1×1 and pins the element type of the result, which a doctest printing `true`
# cannot.
#
# `𝔄exp` is the oracle of these sweeps and nothing in the package calls it, so it is defined here.
# The sweep comes from GeometricMachineLearning, which carried the one-line wrapper and tested it
# (GeometricMachineLearning#230). The default is `ScaledSquaring`, as `geodesic`'s is.
#
# No nested `@testset` in the loop — see the note on RNG state at the top of this file.
function 𝔄exp(B̂::AbstractMatrix, B̄::AbstractMatrix, algorithm = ScaledSquaring())
    I + B̂ * 𝔄(B̂, B̄, algorithm) * B̄'
end

@testset "𝔄exp recovers the exponential across shapes, $T" for T in REAL_ELTYPES
    for N in 1:10, n in 1:N

        A = T(0.1) * rand(T, N, n)
        B = T(0.1) * rand(T, N, n)
        @test eltype(𝔄exp(A, B)) == T
        @test exp(A * B') ≈ 𝔄exp(A, B)
    end

    # The `algorithm` form forwards to `𝔄`, so it is defined exactly where `𝔄(X, algorithm)` is:
    # `TaylorSeries`, `ScaledSquaring`, `NativePade` and `AugmentedPade`. `ProjectedSkew` is not
    # among them — it is a `geodesic`-level algorithm with its own branch there and no `𝔄` method —
    # so `ALGORITHMS`, which exists for the `geodesic` sweeps above and includes it, is not what to
    # loop over here.
    for algorithm in (TaylorSeries(), ScaledSquaring(), NativePade(), AugmentedPade())
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
# than inherited: this testset has to draw the same lifts however much runs before it. They are the
# `Float64` pass's; the `Float32` pass draws lifts of ‖B̄‖ = 3.6, 36.8, 170.8 and 324.7 from the same
# seed, where the series is off by 1e-7, 4e2, `NaN` and `NaN`. `ScaledSquaring` stays under `58eps(T)`
# in `Float32` and `84eps(T)` in `Float64` against `exp` of the lift in `Float64`, under
# `exponential_tolerance(T)`, which the series misses at the three larger scales in both.
@testset "𝔄exp defaults to an algorithm that survives a large argument, $T" for T in REAL_ELTYPES
    Random.seed!(1234)

    for scale in (1, 10, 50, 100)
        B = T(scale) * rand(StiefelLieAlgHorMatrix{T}, 10, 2)
        B̂, B̄ = GeometricOptimizers.lift_factors(B)
        reference = exp(Matrix{Float64}(Matrix(B)))

        result = 𝔄exp(B̂, B̄)
        @test eltype(result) == T
        @test result ≈ reference rtol = exponential_tolerance(T)
        # `geodesic` assembles the same product and defaults the same way, so the two agree; this is
        # what fails if either default moves without the other.
        @test 𝔄exp(B̂, B̄) ≈ Matrix(geodesic(B))
    end
end
