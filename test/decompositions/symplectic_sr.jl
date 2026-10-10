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
# Over 20000 draws per size the residual `‖SᵀJS - J‖` has median 1.8e-15, 3.0e-14 and 2.2e-12 at
# 4x2, 6x4 and 10x6, but maximum 3.8e-6, 1.0e-3 and 7.7e-2 — so the `Float64` thresholds below are
# exceeded by 0 to 4, 3 to 5 and 17 to 21 draws in 20000. The medians are stable across seeds and
# the maxima are not, which is the shape of the algorithm: `S` is symplectic rather than orthogonal,
# and a draw that brings a reflector close to its breakdown amplifies without bound.
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
# Every testset runs in `Float32` and `Float64`, at these three small sizes; at 20x10 and above the
# `Float32` median is of order 1e-2 to 10 and a few draws in 200 return a non-finite value or throw
# (A22 in `KNOWN_ISSUES.md`). The per-draw tolerance is `k √eps(T)`: 1e-6, 1e-5 and 1e-4 in
# `Float64`, and 0.023, 0.23 and 2.3 in `Float32`, which is still more than 100 times under the
# residual of the wrong constraint below.
include("../helpers/eltypes.jl")

tolerance(N2, ::Type{T}) where {T} = (N2 ≤ 4 ? 67 : N2 ≤ 6 ? 670 : 6700) * sqrt(eps(T))

const SIZES = ((4, 2), (6, 4), (10, 6))

@testset "SR decomposition: A = S * R, $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    for (N2, n2) in SIZES
        A = randn(T, N2, n2)
        A_before = copy(A)
        F = sr(A)
        SR = Matrix(F.S) * Matrix(F.R)
        @test eltype(SR) == T
        @test norm(SR - A) < tolerance(N2, T)
        # `sr` leaves its argument alone; `sr!` is the one that overwrites it with the reflectors.
        @test A == A_before
        sr!(A)
        @test A != A_before
    end
end

@testset "SR decomposition: S is symplectic, $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    for (N2, _) in SIZES
        J = _poisson_tensor(T, N2)
        S = Matrix(sr(randn(T, N2, N2)).S)
        @test eltype(S) == T
        @test norm(S' * J * S - J) < tolerance(N2, T)
    end
end

@testset "SR decomposition: the S factor as an operator, $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    for (N2, _) in SIZES
        F = sr(randn(T, N2, N2))
        S = Matrix(F.S)
        @test eltype(S) == T

        # Multiplying by the operator is the same as multiplying by the matrix it stands for. This
        # is what makes `Sfac` usable without ever forming it.
        x = randn(T, N2)
        B = randn(T, 3, N2)
        @test eltype(F.S * x) == T
        @test norm(F.S * x - S * x) < tolerance(N2, T)
        @test norm(F.S * Matrix{T}(I, N2, N2) - S) < tolerance(N2, T)
        @test norm(B * F.S - B * S) < tolerance(N2, T)

        # The inverse applies the same reflectors in the opposite order.
        @test norm(inv(F.S) * (F.S * x) - x) < tolerance(N2, T)

        # Right-multiplication by the inverse has its own reflector kernel, which applies the steps
        # from the last to the first with the two reflectors of a step swapped and each factor
        # negated. That is three things reversed at once, and getting any one of them wrong still
        # produces a plausible matrix, so it is checked against the inverse of the materialized
        # factor and by a round trip rather than against itself.
        @test norm(B * inv(F.S) - B * inv(S)) < tolerance(N2, T)
        @test norm((B * F.S) * inv(F.S) - B) < tolerance(N2, T)

        # A product of two operators materializes both, which is what keeps `S * inv(S)` from
        # being an ambiguous `MethodError` — `Sfac` is an `AbstractMatrix`, so without these the
        # left and right methods are equally specific. All four combinations are covered, because
        # one method on `(::Sfac, ::Sfac)` does not resolve it.
        # The assertion is against the product of the materialized factors, which is exact, and
        # deliberately not against the identity: two ill-conditioned kernels multiplied put
        # `S·S⁻¹ - I` at order 1e-8 on an unlucky draw, which is the conditioning of this
        # decomposition and not a property of the dispatch these lines exist to pin.
        Sinv = Matrix(inv(F.S))
        @test F.S * inv(F.S) == S * Sinv
        @test inv(F.S) * F.S == Sinv * S
        @test F.S * F.S == S * S
        @test inv(F.S) * inv(F.S) == Sinv * Sinv

        # Indexing agrees with the materialized matrix, entry for entry.
        @test all(F.S[i, j] == S[i, j] for i in 1:N2, j in 1:N2)
        @test size(F.S) == (N2, N2)

        # `S` has `2N` columns, so an operand with more rows than that is not a product it can
        # take. The reflector kernels address rows by index rather than by iterating the operand,
        # so without the check the extra rows pass through untouched and the result is a plausible
        # wrong answer rather than an error.
        @test_throws AssertionError F.S * randn(T, N2 + 2, 3)
        @test_throws AssertionError inv(F.S) * randn(T, N2 + 2, 3)
        @test_throws AssertionError F.S * randn(T, N2 + 2)
        @test_throws AssertionError inv(F.S) * randn(T, N2 + 2)
    end
end

@testset "SR decomposition: R has the symplectic triangular shape, $T" for T in REAL_ELTYPES
    # `R` is not upper triangular as a whole. In `2N x 2M` block form the diagonal blocks are upper
    # triangular and the lower-left block is strictly upper triangular; the lower-left block being
    # merely strictly upper triangular rather than zero is what distinguishes SR from QR.
    Random.seed!(1234)
    for (N2, n2) in ((6, 4), (10, 6))
        R = Matrix(sr(randn(T, N2, n2)).R)
        @test eltype(R) == T
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

@testset "symplectic form without building J, $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    for N2 in (4, 6, 10)
        J = _poisson_tensor(T, N2)
        a, b = randn(T, N2), randn(T, N2)
        @test symplectic_form(a, b) isa T
        @test symplectic_form(a, b) ≈ a' * J * b
        # It is antisymmetric and vanishes on a repeated argument: `N2` products and sums, each
        # rounded once.
        @test symplectic_form(a, b) ≈ -symplectic_form(b, a)
        @test abs(symplectic_form(a, a)) ≤ N2 * eps(T) * sum(abs2, a)
    end
end

@testset "symplectic Gram-Schmidt, $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    for (N2, n2) in SIZES
        J_N = _poisson_tensor(T, N2)
        J_n = _poisson_tensor(T, n2)
        A = randn(T, N2, n2)
        B = symplectic_gram_schmidt(A, J_N)
        @test eltype(B) == T
        @test norm(B' * J_N * B - J_n) < tolerance(N2, T)
        # The copying version leaves its argument alone.
        A_before = copy(A)
        symplectic_gram_schmidt(A, J_N)
        @test A == A_before
        # And the mutating one does not.
        symplectic_gram_schmidt!(A, J_N)
        @test norm(A' * J_N * A - J_n) < tolerance(N2, T)
    end
end

# The per-draw assertions above are one draw each, and the header explains why that cannot be a
# bound. This is the same property asserted as a property: over a sweep, the median residual and
# the rate at which the per-draw threshold is exceeded. Both are stable across seeds where a single
# maximum is not, so this is what a change to the algorithm would have to move.
#
# The median scales with `eps(T)`: over 20000 draws at each of three seeds it is 8.0 to 8.5, 136 to
# 141 and 9820 to 10400 eps at 4x2, 6x4 and 10x6, in `Float32` and in `Float64` alike (Julia
# 1.13.1). The sample median of 500 draws lies between 0.70 and 1.76 times `MEDIAN_IN_EPS` over the
# seeds 1 to 40, in both precisions and at all three sizes, so the median bound is twice
# `MEDIAN_IN_EPS`. A sign error in `ρ` of `symplectic_householder!`, or a reflector that is off a
# symplectic transvection by 64 eps, moves the median above it.
#
# The rate threshold is in eps too, `4.5e9`, `4.5e10` and `4.5e11` eps, which is the per-draw
# tolerance in `Float64`. Over the same draws it is exceeded by 0 to 4, 3 to 5 and 17 to 21 draws in
# 20000 in `Float64`, and by 0 to 4, 22 to 29 and 120 to 132 in `Float32`, where most of them return
# a non-finite value or throw (A22 in `KNOWN_ISSUES.md`). The rate bound of each precision is ten
# times its largest measured exceedance, and no less than 1 %.
const MEDIAN_IN_EPS = (8, 136, 10_000)
rate_bounds(::Type{Float64}) = (0.01, 0.01, 0.01)
rate_bounds(::Type{Float32}) = (0.01, 0.015, 0.07)

@testset "the residual distribution, $T" for T in REAL_ELTYPES
    Random.seed!(90_2026)
    for ((N2, _), median_in_eps, rate_bound) in zip(SIZES, MEDIAN_IN_EPS, rate_bounds(T))
        J = _poisson_tensor(T, N2)
        # A draw that raises the `DomainError` of `symplectic_householder!`'s `sqrt` (A22) is one of
        # the failures the rate counts, so it counts as an infinite residual; nothing else is caught.
        residuals = map(1:500) do _
            try
                S = Matrix(sr(randn(T, N2, N2)).S)
                norm(S' * J * S - J)
            catch err
                err isa DomainError || rethrow()
                T(Inf)
            end
        end
        @test eltype(residuals) == T
        @test sort(residuals)[250] < 2 * median_in_eps * eps(T)
        rate_threshold = tolerance(N2, Float64) / eps(Float64) * eps(T)
        rate = count(r -> !(r ≤ rate_threshold), residuals) / 500
        @test rate < rate_bound
    end
end

# A row vector on the left is the one shape `*(::AbstractMatrix, ::OwnedMatrix)` does not settle on
# its own: `LinearAlgebra` has its own method for that left operand, narrower there and wider on the
# right, so neither wins. The two row-vector methods in `src/ambiguities.jl` settle it.
#
# Both `S` and `inv(S)` run: they are separate types and so need, and have, separate kernels. This
# compares the operator against its own dense form rather than against the manifold, so the
# factorization's residual does not enter and the tolerance above is not needed.
@testset "a row vector times an Sfac, $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    for (N2, n2) in SIZES
        S = sr!(randn(T, N2, n2)).S
        v = rand(T, N2)

        for X in (S, inv(S))
            @test eltype(v' * X) == T
            @test v' * X ≈ v' * Matrix(X)
            @test transpose(v) * X ≈ transpose(v) * Matrix(X)
            @test size(v' * X) == (1, N2)
        end
    end
end

# `R` has no product kernel of its own, so against an owned matrix it is the plain operand and the
# backend guard asks for its backend. Each answer is compared with the one for `R`'s dense form.
@testset "the R factor against an owned matrix, $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    for (N2, n2) in SIZES
        R = sr(randn(T, N2, n2)).R
        Rsq = sr(randn(T, N2, N2)).R
        A = rand(SkewSymMatrix{T}, N2)

        @test eltype(A * R) == T
        @test A * R ≈ A * Matrix(R)
        @test Rsq * A ≈ Matrix(Rsq) * A
        @test A - Rsq ≈ A - Matrix(Rsq)
        @test Rsq + A ≈ Matrix(Rsq) + A
    end
end

# The reflectors and the symplectic form are bilinear, `aᵀJb`, so a complex operand is not
# conjugated. The factorization itself is of a real matrix; only the operand is complex. With
# `adjoint` in place of `transpose` the real part still agrees and the product is wrong by O(1).
@testset "a complex operand times a real Sfac, $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    for (N2, _) in SIZES
        F = sr(randn(T, N2, N2))
        B = randn(Complex{T}, 3, N2)
        v = randn(Complex{T}, N2)
        for X in (F.S, inv(F.S))
            M = Matrix(X)
            @test eltype(B * X) == Complex{T}
            @test norm(B * X - B * M) < tolerance(N2, T)
            @test norm(X * transpose(B) - M * transpose(B)) < tolerance(N2, T)
            @test norm(X * v - M * v) < tolerance(N2, T)
            @test norm(transpose(v) * X - transpose(v) * M) < tolerance(N2, T)
            @test norm(v' * X - v' * M) < tolerance(N2, T)
        end
        J = _poisson_tensor(T, N2)
        a, b = randn(Complex{T}, N2), randn(Complex{T}, N2)
        @test symplectic_form(a, b) ≈ transpose(a) * J * b
    end
end
