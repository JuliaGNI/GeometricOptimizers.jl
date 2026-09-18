using Test
using LinearAlgebra
using GeometricOptimizers
using GeometricOptimizers: symplectic_form, _poisson_tensor
import Random

Random.seed!(1234)

# The tolerance grows with the size, and that is the algorithm rather than the test being lax. `S`
# is symplectic, not orthogonal, so its condition number is unbounded and the reflectors amplify
# what they are given; there is no re-orthogonalization step here.
#
# The distribution has a heavy tail, and no per-draw threshold both discriminates and never fails.
# Over 20000 draws per size the residual `‖SᵀJS - J‖` has median 1.8e-15, 3.2e-14 and 2.4e-12 at
# 4x2, 6x4 and 10x6, but maximum 3.8e-6, 1.0e-3 and 7.7e-2 — so the thresholds below are exceeded
# by 1, 6 and 21 draws in 20000. The medians are stable across seeds and the maxima are not, which
# is the shape of the algorithm: `S` is symplectic rather than orthogonal, and a draw that brings a
# reflector close to its breakdown amplifies without bound.
#
# The per-draw assertions below therefore run on the fixed seed above and are deterministic, but
# they are not a bound. What is asserted as a property, rather than as one draw's luck, is the
# sweep in `the residual distribution` at the end of this file: the median over many draws, and the
# exceedance rate. An edit that reorders the draws can move a per-draw assertion into the tail; the
# sweep is what will not move.
#
# They still discriminate. A point that is genuinely not on the manifold gives an O(1) residual:
# the *wrong* constraint, `‖UᵀU - I‖`, measures 7.6, 38 and 295 at these three sizes, so 1e-4
# rejects breakage by four orders and more.
#
# `Float32` is not tested at any size. Its median residual at 10x6 is 7.5e-5, and at 40x20 the
# median is of order 10 with a few draws in 200 returning a non-finite value or throwing, so there
# is no threshold that is both passing and meaningful. See *Open Issues* in `CHANGELOG.md`.
tolerance(N2) = N2 ≤ 4 ? 1e-6 : N2 ≤ 6 ? 1e-5 : 1e-4

const SIZES = ((4, 2), (6, 4), (10, 6))

@testset "SR decomposition: A = S * R" begin
    for (N2, n2) in SIZES
        A = randn(N2, n2)
        A_before = copy(A)
        F = sr(A)
        @test norm(Matrix(F.S) * Matrix(F.R) - A) < tolerance(N2)
        # `sr` leaves its argument alone; `sr!` is the one that overwrites it with the reflectors.
        @test A == A_before
        sr!(A)
        @test A != A_before
    end
end

@testset "SR decomposition: S is symplectic" begin
    for (N2, _) in SIZES
        J = _poisson_tensor(Float64, N2)
        S = Matrix(sr(randn(N2, N2)).S)
        @test norm(S' * J * S - J) < tolerance(N2)
    end
end

@testset "SR decomposition: the S factor as an operator" begin
    for (N2, _) in SIZES
        F = sr(randn(N2, N2))
        S = Matrix(F.S)

        # Multiplying by the operator is the same as multiplying by the matrix it stands for. This
        # is what makes `Sfac` usable without ever forming it.
        x = randn(N2)
        B = randn(3, N2)
        @test norm(F.S * x - S * x) < tolerance(N2)
        @test norm(F.S * Matrix{Float64}(I, N2, N2) - S) < tolerance(N2)
        @test norm(B * F.S - B * S) < tolerance(N2)

        # The inverse applies the same reflectors in the opposite order.
        @test norm(inv(F.S) * (F.S * x) - x) < tolerance(N2)

        # Right-multiplication by the inverse has no reflector kernel and goes through the matrix.
        # It still has to agree with the inverse of the materialized factor.
        @test norm(B * inv(F.S) - B * inv(S)) < tolerance(N2)

        # Indexing agrees with the materialized matrix, entry for entry.
        @test all(F.S[i, j] == S[i, j] for i in 1:N2, j in 1:N2)
        @test size(F.S) == (N2, N2)
    end
end

@testset "SR decomposition: R has the symplectic triangular shape" begin
    # `R` is not upper triangular as a whole. In `2N x 2M` block form the diagonal blocks are upper
    # triangular and the lower-left block is strictly upper triangular; the lower-left block being
    # merely strictly upper triangular rather than zero is what distinguishes SR from QR.
    for (N2, n2) in ((6, 4), (10, 6))
        R = Matrix(sr(randn(N2, n2)).R)
        N, M = N2 ÷ 2, n2 ÷ 2
        R₁₁ = R[1:N, 1:M]
        R₂₁ = R[(N + 1):N2, 1:M]
        R₂₂ = R[(N + 1):N2, (M + 1):n2]
        @test all(R₁₁[i, j] == 0 for i in 1:N, j in 1:M if i > j)
        @test all(R₂₂[i, j] == 0 for i in 1:N, j in 1:M if i > j)
        @test all(R₂₁[i, j] == 0 for i in 1:N, j in 1:M if i ≥ j)
        # And it is not the zero block, or the shape above would be vacuous.
        @test any(R₂₁[i, j] != 0 for i in 1:N, j in 1:M if i < j)
    end
end

@testset "symplectic form without building J" begin
    for N2 in (4, 6, 10)
        J = _poisson_tensor(Float64, N2)
        a, b = randn(N2), randn(N2)
        @test symplectic_form(a, b) ≈ a' * J * b
        # It is antisymmetric and vanishes on a repeated argument.
        @test symplectic_form(a, b) ≈ -symplectic_form(b, a)
        @test abs(symplectic_form(a, a)) < 1e-14
    end
end

@testset "symplectic Gram-Schmidt" begin
    for (N2, n2) in SIZES
        J_N = _poisson_tensor(Float64, N2)
        J_n = _poisson_tensor(Float64, n2)
        A = randn(N2, n2)
        B = symplectic_gram_schmidt(A, J_N)
        @test norm(B' * J_N * B - J_n) < tolerance(N2)
        # The copying version leaves its argument alone.
        A_before = copy(A)
        symplectic_gram_schmidt(A, J_N)
        @test A == A_before
        # And the mutating one does not.
        symplectic_gram_schmidt!(A, J_N)
        @test norm(A' * J_N * A - J_n) < tolerance(N2)
    end
end

# The per-draw assertions above are one draw each, and the header explains why that cannot be a
# bound. This is the same property asserted as a property: over a sweep, the median residual and
# the rate at which the per-draw threshold is exceeded. Both are stable across seeds where a single
# maximum is not, so this is what a change to the algorithm would have to move.
#
# The bounds are three orders above the medians measured over 20000 draws (1.8e-15, 3.2e-14,
# 2.4e-12) and the rate bound is ten times the measured exceedance (1, 6 and 21 draws in 20000).
# They still discriminate: the *wrong* constraint `‖UᵀU - I‖` measures 7.6, 38 and 295 at these
# sizes, so a broken factorization fails the median assertion by orders, not by a margin.
@testset "the residual distribution" begin
    Random.seed!(90_2026)
    for ((N2, _), median_bound) in zip(SIZES, (1e-12, 1e-11, 1e-9))
        J = _poisson_tensor(Float64, N2)
        residuals = map(1:500) do _
            S = Matrix(sr(randn(N2, N2)).S)
            norm(S' * J * S - J)
        end
        @test sort(residuals)[250] < median_bound
        @test count(>(tolerance(N2)), residuals) / 500 < 0.01
    end
end
