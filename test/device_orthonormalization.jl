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
# `gradient_backend.jl`. `Metal` is Apple-only and CI runs a matrix, so real device hardware is
# what this file cannot assert on.

using GeometricOptimizers
using GeometricOptimizers: check, global_section, _cholesky_qr2, orthonormal_columns
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: I, Symmetric, cholesky, norm
using Random
using Test

Random.seed!(4242)

const T = Float32
const device = KernelAbstractions.get_backend(JLArray(zeros(T, 1)))

allowscalar(false)

@testset "the device backend is a GPU as far as dispatch is concerned" begin
    # otherwise everything below would be exercising the host arm under another name
    @test device isa KernelAbstractions.GPU
end

@testset "a point is drawn on the device, and stays there" begin
    for MT in (StiefelManifold, GrassmannManifold), (N, n) in ((6, 3), (12, 4))

        Y = rand(device, MT, N, n)

        @test Y isa MT
        @test Y.A isa JLArray{T, 2}
        @test size(Y) == (N, n)
        @test eltype(Y) === T
        @test check(Y) < 10 * eps(T)
    end
end

@testset "the global section is taken on the device, and stays there" begin
    for MT in (StiefelManifold, GrassmannManifold), (N, n) in ((6, 3), (12, 4))

        Y = rand(device, MT, N, n)
        λ = global_section(Y)

        @test λ isa JLArray{T, 2}
        @test size(λ) == (N, N - n)
        # the two defining properties: orthonormal columns, and orthogonal to `Y`
        @test norm(Array(λ' * λ) - I) < 100 * eps(T)
        @test norm(Array(Y.A' * λ)) < 100 * eps(T)
    end
end

@testset "GlobalSection is constructible on a device-backed point" begin
    # this is the call that raised `Cannot access the contents of a private buffer` on `Metal` and a
    # `MethodError` on `JLArrays`, and with it everything that holds one: `BFGSCache`, `DFPCache`,
    # `NewtonOptimizerCache`, and `geodesic`/`cayley` of a point and a tangent vector
    N, n = 8, 3
    Y = rand(device, StiefelManifold, N, n)
    λY = GlobalSection(Y)

    @test λY isa GlobalSection
    @test λY.λ isa JLArray{T, 2}
    @test λY.Y isa StiefelManifold{T, JLArray{T, 2}}

    # `Matrix(λY)` is the readable spelling of "the whole of `[Y λ]`" and is deliberately not used
    # here: it converts through the manifold's scalar `getindex`, which is the host-only path its
    # own manual warns about, and is a separate gap from this file's. The same property is asserted
    # on the two blocks, where it is the section's defining one anyway.
    @test norm(Array(λY.Y.A' * λY.Y.A) - I) < 100 * eps(T)
    @test norm(Array(λY.λ' * λY.λ) - I) < 100 * eps(T)
    @test norm(Array(λY.Y.A' * λY.λ)) < 100 * eps(T)
end

@testset "the lift of a device-backed point is taken on the device" begin
    # `global_rep` is the first thing a retraction of a point does after the section, and it is as
    # far as this file goes. The retractions themselves are `device_multiply.jl`'s: `geodesic` runs
    # end to end there, and `cayley` is pinned where it stops on `JLArrays`, which supplies no `lu`.
    N, n = 6, 3
    Y = rand(device, StiefelManifold, N, n)
    Δ = rgrad(Y, JLArray(rand(T, N, n)))
    B = GeometricOptimizers.global_rep(GlobalSection(Y), Δ)

    @test B isa GeometricOptimizers.StiefelLieAlgHorMatrix
    @test B.B isa JLArray{T, 2}
    @test B.A.S isa JLArray{T, 1}
end

@testset "the second CholeskyQR pass is what makes the result usable" begin
    # One pass is an orthonormalization only to the accuracy of the squared condition number, and at
    # this shape that is nowhere near enough. The assertion is the *ratio*, not either number: the
    # one-pass residual grows with the size while the two-pass one stays at the element type's
    # resolution.
    N, n = 40, 3
    A = JLArray(randn(T, N, N - n))
    Y = rand(device, StiefelManifold, N, n)
    A = A - Y.A * (Y.A' * A)

    one_pass = A / cholesky(Symmetric(A' * A)).U
    two_pass = _cholesky_qr2(A)

    @test two_pass !== nothing
    @test norm(Array(two_pass' * two_pass) - I) < 100 * eps(T)
    @test norm(Array(one_pass' * one_pass) - I) > 10 * norm(Array(two_pass' * two_pass) - I)
end

@testset "a draw CholeskyQR2 cannot orthonormalize is reported, not returned" begin
    # `_cholesky_qr2` answers `nothing` rather than throwing a `PosDefException`, which is what lets
    # `orthonormal_columns` redraw without an exception handler.
    #
    # The rank deficiency is a *zero column* and not a duplicated one, which is the difference
    # between a test and a coin toss: a duplicated column leaves the Gram matrix singular only in
    # exact arithmetic, and whether the last pivot lands above or below zero is up to the backend's
    # reduction order -- measured, LAPACK rejected it and the generic factorization a `JLArray` uses
    # accepted it. A zero column puts an exact zero on the Gram diagonal, and `0 > 0` is false
    # wherever the pivot test is written.
    A = randn(T, 12, 9)
    A[:, 9] .= 0

    @test _cholesky_qr2(A) === nothing
    @test _cholesky_qr2(JLArray(A)) === nothing

    # and the redraw is what turns that into an answer: a `draw` that returns the singular matrix
    # once and a good one afterwards has to come back with the good one
    drawn = Ref(0)
    Q = orthonormal_columns() do
        drawn[] += 1
        drawn[] == 1 ? A : randn(T, 12, 9)
    end
    @test drawn[] == 2
    @test norm(Q' * Q - I) < 100 * eps(T)
end

@testset "an empty complement is an answer, not a failure" begin
    # `n = N` is in the retraction tests' sweep, and its complement is `N × 0`. `maximum(abs, ·)`
    # over no entries is `abs(zero(T))`, so without the early return an `N × 0` argument is rejected
    # as badly scaled and redrawn until `orthonormal_columns` gives up.
    for T in (Float32, Float64), N in 1:4

        A = zeros(T, N, 0)
        @test _cholesky_qr2(A) == A
        @test size(global_section(rand(StiefelManifold{T}, N, N))) == (N, 0)
    end
end

@testset "a badly scaled argument does not overflow the Gram matrix" begin
    # Forming `AᵀA` squares every entry, so without the scaling inside `_cholesky_qr2` an argument
    # of magnitude 1e100 gives an infinite Gram matrix and no answer. A Householder QR scales
    # internally and does not have this failure, which is why replacing it needed the scaling.
    #
    # This is not a hypothetical: `optimizer_status_tests.jl` builds a point 1e100 off the manifold
    # to test a convergence guard, and its `global_section` carries that magnitude into `A`.
    for T in (Float32, Float64)
        # large enough that squaring an entry leaves the format: `Float32` tops out at 3.4e38 and
        # `Float64` at 1.8e308
        big = T(10)^(T === Float32 ? 30 : 200)
        A = big * randn(T, 8, 5)

        @test !isfinite(sum(abs2, A))      # the Gram matrix really would be infinite
        Q = _cholesky_qr2(A)
        @test Q !== nothing
        @test norm(Q' * Q - I) < 100 * eps(T)
    end

    # and end to end, which is the shape `optimizer_status_tests.jl` reaches
    Y = rand(StiefelManifold, 6, 3)
    off = StiefelManifold(1e100 * Y.A)
    λ = global_section(off)
    @test norm(λ' * λ - I) < 100 * eps(Float64)
end

@testset "a draw that never succeeds is an error and not a silent answer" begin
    # `orthonormal_columns` is bounded. Exhausting it means the element type is too narrow for the
    # problem, which is a fact about the call and not bad luck, so it is raised rather than papered
    # over with the last attempt's result.
    A = randn(T, 12, 9)
    A[:, 9] .= 0

    @test_throws ErrorException orthonormal_columns(() -> A)
end
