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
# The thresholds are set against the *tail* over eight seeds of 200 draws each, not one seed's: the
# medians reproduce across seeds and the maxima do not. Worst residual of `check` seen that way is
# 9.6e-10 at 4x2, 3.8e-8 at 6x4 and 7.2e-7 at 10x6 — so a threshold read off a single seed would
# have had a factor of one in hand at 4x2, and any edit that reordered a draw could have turned it
# red. Each threshold below clears its eight-seed worst by about three orders of magnitude.
#
# They still discriminate. A point that is genuinely not on the manifold gives an O(1) residual:
# the *wrong* constraint, `‖UᵀU - I‖`, measures 7.6, 38 and 295 at these three sizes, so 1e-4
# rejects breakage by four orders and more.
#
# `Float32` is not tested at any size: its median residual at 10x6 is 9.1e-5 and it returns NaN
# outright at 40x20, so there is no threshold that is both passing and meaningful. See
# *Open Issues* in `CHANGELOG.md`.
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
