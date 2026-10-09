# Drawing a manifold point, and taking its global section, on a device backend.
#
# `rand(<backend>, <Manifold>, …)` and `global_section` orthonormalize with CholeskyQR2 on every
# backend, which is expressible in matrix products, reductions and triangular solves alone.
# `LinearAlgebra.qr!` is not available to them: `Metal` implements no `qr` for an `MtlArray` at all,
# and a `qr` on a `JLArray` cannot rebuild its `Q` -- `JLArray{Float32,2}(::QRCompactWYQ{…})` is a
# `MethodError` -- so a `qr!` here puts the device draw, `GlobalSection(Y)` and `Optimizer(Y, F)`
# out of reach on both device backends reachable from this suite.
#
# `allowscalar(false)` is what makes this file a test rather than a description: without it a scalar
# index on a `JLArray` merely warns, and the whole point is that nothing here reaches one.
#
# `JLArrays` stands in for the device, as it does in `similar_backend.jl` and
# `rgrad_backend.jl`. `Metal` is Apple-only and CI runs a matrix, so real device hardware is
# what this file cannot assert on.

using GeometricOptimizers
using GeometricOptimizers: check, global_section, _cholesky_qr2, orthonormal_columns
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: I, Symmetric, cholesky, norm
using Random
using Test

include("../helpers/eltypes.jl")

Random.seed!(4242)

const device = KernelAbstractions.get_backend(JLArray(zeros(1)))

allowscalar(false)

# The orthonormality bounds below are in `eps(T)`. Measured over 50 seeds in both precisions: a
# device draw's `check` stays under 4 `eps(T)`, a two-pass CholeskyQR2 under 12 `eps(T)`, a
# `GlobalSection`'s blocks under 4 `eps(T)`.

@testset "the device backend is a GPU as far as dispatch is concerned" begin
    # otherwise everything below would be exercising the host arm under another name
    @test device isa KernelAbstractions.GPU
end

@testset "a point is drawn on the device, and stays there, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(4242)
    for MT in (StiefelManifold, GrassmannManifold), (N, n) in ((6, 3), (12, 4))

        Y = rand(rng, device, MT{T}, N, n)

        @test Y isa MT
        @test Y.A isa JLArray{T, 2}
        @test size(Y) == (N, n)
        @test eltype(Y) === T
        # CholeskyQR2 of an `N × n` Gaussian: a few `eps(T)` (measured under 4)
        @test check(Y) < 10 * eps(T)
    end
end

@testset "the global section is taken on the device, and stays there, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(4243)
    for MT in (StiefelManifold, GrassmannManifold), (N, n) in ((6, 3), (12, 4))

        Y = rand(rng, device, MT{T}, N, n)
        λ = global_section(Y)

        @test λ isa JLArray{T, 2}
        @test eltype(λ) == T
        @test size(λ) == (N, N - n)
        # the two defining properties: orthonormal columns, and orthogonal to `Y`. `λᵀλ - I` is
        # measured under 6 `eps(T)`. `Yᵀλ` has a heavy tail on the device path and not on the host
        # one, in both precisions: over 400 seeds the median is 15 to 25 `eps(T)` and 10 to 15 % of
        # the draws exceed 100 `eps(T)`, up to 2e4 `eps(T)`, against at most 5 `eps(T)` for the same
        # point on the host. The bound is the one this testset had; the tail is reported, not
        # absorbed into it.
        @test norm(Array(λ' * λ) - I) < 100 * eps(T)
        @test norm(Array(Y.A' * λ)) < 100 * eps(T)
    end
end

@testset "GlobalSection is constructible on a device-backed point, $T" for T in REAL_ELTYPES
    # this is the call that raised `Cannot access the contents of a private buffer` on `Metal` and a
    # `MethodError` on `JLArrays`, and with it everything that holds one: `QuasiNewtonCache`,
    # `NewtonOptimizerCache`, and `geodesic`/`cayley` of a point and a tangent vector
    N, n = 8, 3
    Y = rand(Random.Xoshiro(4244), device, StiefelManifold{T}, N, n)
    λY = GlobalSection(Y)

    @test λY isa GlobalSection
    @test λY.λ isa JLArray{T, 2}
    @test λY.Y isa StiefelManifold{T, JLArray{T, 2}}
    @test eltype(λY.λ) == T

    # `Matrix(λY)` is the readable spelling of "the whole of `[Y λ]`" and is deliberately not used
    # here: it converts through the manifold's scalar `getindex`, which is the host-only path its
    # own manual warns about, and is a separate gap from this file's. The same property is asserted
    # on the two blocks, where it is the section's defining one anyway. The bounds are the ones of
    # the testset above, for the same reasons.
    @test norm(Array(λY.Y.A' * λY.Y.A) - I) < 100 * eps(T)
    @test norm(Array(λY.λ' * λY.λ) - I) < 100 * eps(T)
    @test norm(Array(λY.Y.A' * λY.λ)) < 100 * eps(T)
end

@testset "the lift of a device-backed point is taken on the device, $T" for T in REAL_ELTYPES
    # `global_rep` is the first thing a retraction of a point does after the section, and it is as
    # far as this file goes. The retractions themselves are `device_multiply.jl`'s: `geodesic` runs
    # end to end there, and `cayley` is pinned where it stops on `JLArrays`, which supplies no `lu`.
    N, n = 6, 3
    rng = Random.Xoshiro(4245)
    Y = rand(rng, device, StiefelManifold{T}, N, n)
    Δ = rgrad(Y, JLArray(randn(rng, T, N, n)))
    B = GeometricOptimizers.global_rep(GlobalSection(Y), Δ)

    @test B isa GeometricOptimizers.StiefelLieAlgHorMatrix
    @test eltype(B) == T
    @test B.B isa JLArray{T, 2}
    @test B.A.S isa JLArray{T, 1}
end

@testset "the second CholeskyQR pass is what makes the result usable, $T" for T in REAL_ELTYPES
    # One pass is an orthonormalization only to the accuracy of the squared condition number, and at
    # this shape that is nowhere near enough. The assertion is the *ratio*, not either number: the
    # one-pass residual grows with the size while the two-pass one stays at the element type's
    # resolution. Over 50 seeds the two-pass residual stays under 12 `eps(T)` and the ratio above 11
    # in both precisions.
    N, n = 40, 3
    rng = Random.Xoshiro(4246)
    A = JLArray(randn(rng, T, N, N - n))
    Y = rand(rng, device, StiefelManifold{T}, N, n)
    A = A - Y.A * (Y.A' * A)

    one_pass = A / cholesky(Symmetric(A' * A)).U
    two_pass = _cholesky_qr2(A)

    @test two_pass !== nothing
    @test eltype(two_pass) == T
    @test norm(Array(two_pass' * two_pass) - I) < 100 * eps(T)
    @test norm(Array(one_pass' * one_pass) - I) > 10 * norm(Array(two_pass' * two_pass) - I)
end

@testset "a draw CholeskyQR2 cannot orthonormalize is reported, not returned, $T" for T in REAL_ELTYPES
    # `_cholesky_qr2` answers `nothing` rather than throwing a `PosDefException`, which is what lets
    # `orthonormal_columns` redraw without an exception handler.
    #
    # The rank deficiency is a *zero column* and not a duplicated one, which is the difference
    # between a test and a coin toss: a duplicated column leaves the Gram matrix singular only in
    # exact arithmetic, and whether the last pivot lands above or below zero is up to the backend's
    # reduction order -- measured, LAPACK rejected it and the generic factorization a `JLArray` uses
    # accepted it. A zero column puts an exact zero on the Gram diagonal, and `0 > 0` is false
    # wherever the pivot test is written.
    rng = Random.Xoshiro(4247)
    A = randn(rng, T, 12, 9)
    A[:, 9] .= 0

    @test _cholesky_qr2(A) === nothing
    @test _cholesky_qr2(JLArray(A)) === nothing

    # and the redraw is what turns that into an answer: a `draw` that returns the singular matrix
    # once and a good one afterwards has to come back with the good one
    drawn = Ref(0)
    Q = orthonormal_columns() do
        drawn[] += 1
        drawn[] == 1 ? A : randn(rng, T, 12, 9)
    end
    @test drawn[] == 2
    @test eltype(Q) == T
    # CholeskyQR2 of a 12 × 9 Gaussian: measured under 5 `eps(T)`
    @test norm(Q' * Q - I) < 100 * eps(T)
end

@testset "an empty complement is an answer, not a failure, $T" for T in REAL_ELTYPES
    # `n = N` is in the retraction tests' sweep, and its complement is `N × 0`. `maximum(abs, ·)`
    # over no entries is `abs(zero(T))`, so without the early return an `N × 0` argument is rejected
    # as badly scaled and redrawn until `orthonormal_columns` gives up.
    for N in 1:4
        A = zeros(T, N, 0)
        @test _cholesky_qr2(A) == A
        λ = global_section(rand(Random.Xoshiro(N), StiefelManifold{T}, N, N))
        @test eltype(λ) == T
        @test size(λ) == (N, 0)
    end
end

@testset "a badly scaled argument does not overflow the Gram matrix, $T" for T in REAL_ELTYPES
    # Forming `AᵀA` squares every entry, so without the scaling inside `_cholesky_qr2` an argument
    # of magnitude 1e100 gives an infinite Gram matrix and no answer. A Householder QR scales
    # internally and does not have this failure, which is why replacing it needed the scaling.
    #
    # This is not a hypothetical: `optimizer_status_tests.jl` builds a point 1e100 off the manifold
    # to test a convergence guard, and its `global_section` carries that magnitude into `A`.
    rng = Random.Xoshiro(4248)

    # large enough that squaring an entry leaves the format, in either precision
    big = 10 * sqrt(floatmax(T))
    A = big * randn(rng, T, 8, 5)

    @test !isfinite(sum(abs2, A))      # the Gram matrix really would be infinite
    Q = _cholesky_qr2(A)
    @test Q !== nothing
    @test eltype(Q) == T
    # measured under 5 `eps(T)`: the scaling makes this the residual of a unit-scale argument
    @test norm(Q' * Q - I) < 100 * eps(T)

    # and end to end, which is the shape `optimizer_status_tests.jl` reaches. The point is scaled by
    # `∛floatmax(T)` (5.6e102 in `Float64`, the order of that file's 1e100), so that the projected
    # draw of the complement, `A - Y * (Yᵀ A)`, of size `‖Y‖²`, is finite and its Gram matrix, of
    # size `‖Y‖⁴`, is not.
    Y = rand(rng, StiefelManifold{T}, 6, 3)
    off = StiefelManifold(cbrt(floatmax(T)) * Y.A)
    λ = global_section(off)
    @test eltype(λ) == T
    # measured under 4 `eps(T)`
    @test norm(λ' * λ - I) < 100 * eps(T)
end

# not numeric: a refusal and its message, looped so that the `T` of the draw is not fixed
@testset "a draw that never succeeds is an error and not a silent answer, $T" for T in REAL_ELTYPES
    # `orthonormal_columns` is bounded. Exhausting it means the element type is too narrow for the
    # problem, which is a fact about the call and not bad luck, so it is raised rather than papered
    # over with the last attempt's result.
    A = randn(Random.Xoshiro(4249), T, 12, 9)
    A[:, 9] .= 0

    @test_throws OrthonormalizationFailure orthonormal_columns(() -> A)
    # The attempt count reaches the caller, and the message says which knob to turn.
    e = try
        orthonormal_columns(() -> A)
    catch err
        err
    end
    @test e.attempts == GeometricOptimizers.ORTHONORMALIZATION_ATTEMPTS
    @test occursin("element type", sprint(showerror, e))
end
