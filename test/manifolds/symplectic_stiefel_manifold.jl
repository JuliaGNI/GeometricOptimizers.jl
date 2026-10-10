using Test
using LinearAlgebra
using GeometricOptimizers
using GeometricOptimizers: _poisson_tensor, _similar, check
using KernelAbstractions: CPU, GPU
import Random

include("../helpers/eltypes.jl")

# `global_section` draws its completion from the global generator
Random.seed!(1234)

# Every bound below is `eps(T)` times a power of the condition number `κ = cond(U.A)` of the point,
# computed per point by an SVD. The SR decomposition `rand` draws through has no
# re-orthogonalization step and a symplectic factor is not orthogonal, so `κ` of a drawn point is
# unbounded: its median is 7, 130 and 1600 at the three sizes below, with a tail past `1e6` in both
# precisions. A bound in `eps(T)` alone either fails on the tail or is vacuous at the median; a
# bound in `eps(T) * κ^p` holds on the tail. Each power and factor is measured over 2000 points per
# size, in `Float32` and in `Float64`, and its comment gives the largest ratio seen. See also the
# note on tolerances in `test/decompositions/symplectic_sr.jl`.
#
# On a drawn point with a large `κ` such a bound is also too wide to see a wrong formula: at the
# `Float32` draws of this file, `κ` reaches `1.7e6`, and substituting the Frobenius product for the
# metric stays within `2eps(T) * κ^4`. The testsets that check the metric's algebra rather than the
# draw therefore take their point from `conditioned_point`, where `κ ≤ 4`.
const SIZES = ((4, 2), (6, 4), (10, 6))

# A point of the symplectic Stiefel manifold built without `sr!`, with `κ ≤ 4`: the orthosymplectic
# point `[A -B; B A]` of a complex Stiefel point `A + iB`, times the symplectic `diag(D, D⁻¹)` on
# the right, with the entries of `D` in `[1/2, 2]`. `D` is what makes `UᵀU ≠ I`, so that `P` in the
# metric is not the identity.
function conditioned_point(rng, T, N2, n2)
    N, n = N2 ÷ 2, n2 ÷ 2
    Q = Matrix(qr(randn(rng, Complex{T}, N, n)).Q)
    A, B = real(Q), imag(Q)
    d = exp2.(2 .* rand(rng, T, n) .- 1)
    SymplecticStiefelManifold(hcat(vcat(A, B) .* d', vcat(-B, A) ./ d'))
end

@testset "a random point lies on the manifold, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1234)
    for (N2, n2) in SIZES
        U = rand(rng, SymplecticStiefelManifold{T}, N2, n2)
        κ = cond(U.A)
        @test size(U) == (N2, n2)
        @test eltype(U) == T
        # measured `check(U) ≤ 6.1eps(T) * κ^2` but for one draw in 2000 at `6 × 4` and at
        # `10 × 6` in `Float32`
        @test check(U) < 8eps(T) * κ^2

        # `check` is the symplectic residual and not the generic orthonormality one. Without the
        # specific method this testset would be asserting the wrong constraint and would fail; the
        # assertion below states that the two are genuinely different quantities here. An
        # orthonormal draw puts this residual at a few `eps(T)`.
        @test norm(U.A' * U.A - I) > sqrt(eps(T))
    end
end

@testset "the element type is honoured, $T" for T in REAL_ELTYPES
    U = rand(Random.Xoshiro(1235), SymplecticStiefelManifold{T}, 6, 4)
    @test eltype(U) == T
end

@testset "the constructor rejects what is not a symplectic shape" begin
    @test_throws AssertionError SymplecticStiefelManifold(randn(5, 4))
    @test_throws AssertionError SymplecticStiefelManifold(randn(6, 3))
    @test_throws AssertionError SymplecticStiefelManifold(randn(4, 6))
end

@testset "rgrad lands in the tangent space, $T" for T in REAL_ELTYPES
    # The tangent space at `U` is `{Δ : Δᵀ𝕁U + Uᵀ𝕁Δ = 0}`, which is what makes the Riemannian
    # gradient a direction the retraction can follow. A gradient of the right *size* says nothing.
    rng = Random.Xoshiro(1236)
    for (N2, n2) in SIZES
        J = _poisson_tensor(T, N2)
        U = rand(rng, SymplecticStiefelManifold{T}, N2, n2)
        Δ = rgrad(U, randn(rng, T, N2, n2))
        @test size(Δ) == (N2, n2)
        @test eltype(Δ) == T
        # measured at most `1.13eps(T) * κ^2 * ‖Δ‖`
        @test norm(Δ' * J * U.A + U.A' * J * Δ) < 4eps(T) * cond(U.A)^2 * norm(Δ)
    end
end

# `metric` evaluates a regrouping of the expression its docstring writes first, in the `2n × 2n`
# factors rather than through the `2N × 2N` middle matrix. Symmetry and bilinearity below survive
# almost any slip in that regrouping; *the metric is the one `rgrad` is taken against* does catch
# one, but at a tolerance in `κ^4`. This writes the `2N × 2N` form out and asserts the two agree.
#
# **Not at machine precision.** `P = (UᵀU)⁻¹` has norm `κ`, and the two forms round differently
# through it, so the absolute difference is measured at most `0.63eps(T) * κ^3 * ‖Δ₁‖ ‖Δ₂‖`, on
# drawn and on conditioned points. A relative bound does not serve: the trace cancels, and the
# relative difference reaches `1.7` in `Float32` at `6 × 4` on a drawn point whose absolute
# difference is within the bound. Dropping the factor `1/2` exceeds the bound at least 8 times
# over, on every one of 2000 conditioned points per size in `Float32`.
@testset "the metric is the 2N × 2N expression it is a regrouping of, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1237)
    for (N2, n2) in SIZES
        U = conditioned_point(rng, T, N2, n2)
        Δ₁, Δ₂ = randn(rng, T, N2, n2), randn(rng, T, N2, n2)
        J = _poisson_tensor(T, N2)
        P = inv(U.A' * U.A)

        g = metric(U, Δ₁, Δ₂)
        @test eltype(g) == T
        @test abs(g -
                  tr(P * Δ₁' * (Matrix{T}(I, N2, N2) - J' * U.A * P * U.A' * J / 2) * Δ₂)) <
              2eps(T) * cond(U.A)^3 * norm(Δ₁) * norm(Δ₂)
    end
end

@testset "the metric is symmetric and bilinear, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1238)
    for (N2, n2) in SIZES
        U = conditioned_point(rng, T, N2, n2)
        Δ₁, Δ₂ = randn(rng, T, N2, n2), randn(rng, T, N2, n2)
        # each difference is measured at most `6.3eps(T) * κ^2` times the product of the norms
        # of its arguments, which `(‖Δ₁‖ + ‖Δ₂‖)^2` bounds; doubling `Δ₁` is exact
        atol = 16eps(T) * cond(U.A)^2 * (norm(Δ₁) + norm(Δ₂))^2
        @test eltype(metric(U, Δ₁, Δ₂)) == T
        @test isapprox(metric(U, Δ₁, Δ₂), metric(U, Δ₂, Δ₁); atol)
        @test isapprox(metric(U, 2 * Δ₁, Δ₂), 2 * metric(U, Δ₁, Δ₂); atol)
        @test isapprox(metric(U, Δ₁ + Δ₂, Δ₂), metric(U, Δ₁, Δ₂) + metric(U, Δ₂, Δ₂); atol)
    end
end

# Symmetry and bilinearity hold for *every* inner product, so the testset above pins nothing about
# which one this is: substituting the Frobenius product `tr(Δ₁ᵀΔ₂)` leaves all nine of its
# assertions passing. What identifies the metric is the property that defines the Riemannian
# gradient against it — `g_U(rgrad(U, ∇L), Δ) = tr(∇LᵀΔ)` for every tangent `Δ`. The difference
# passes through `P` twice and `rgrad` once, and is measured at most
# `1.1eps(T) * κ^4 * ‖∇L‖ ‖Δ‖`. The Frobenius substitution exceeds the bound below at least
# 1.8 times over on every one of 2000 conditioned points per size in `Float32`, and `1e9` times
# over in `Float64`.
@testset "the metric is the one rgrad is taken against, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1239)
    for (N2, n2) in SIZES
        U = conditioned_point(rng, T, N2, n2)
        ∇L = randn(rng, T, N2, n2)
        Δ = rgrad(U, randn(rng, T, N2, n2))
        g = metric(U, rgrad(U, ∇L), Δ)
        @test eltype(g) == T
        @test abs(g - tr(∇L' * Δ)) < 4eps(T) * cond(U.A)^4 * norm(∇L) * norm(Δ)

        # And it is positive definite on tangent directions, which a bilinear form need not be.
        @test metric(U, Δ, Δ) > 0
    end
end

@testset "global_section completes the point symplectically, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1240)
    for (N2, n2) in SIZES
        J = _poisson_tensor(T, N2)
        U = conditioned_point(rng, T, N2, n2)
        Λ = global_section(U)

        # The shape is the completion's, matching `global_section(::StiefelManifold)` and what
        # `GlobalSection` assumes — not the whole `2N x 2N` factor it is sliced from.
        @test size(Λ) == (N2, N2 - n2)
        @test eltype(Λ) == T

        # And it is a completion: symplectically orthogonal to the point. Asserting only the shape
        # would pass for any matrix of the right size, so the shape alone pins nothing. The
        # completion is made symplectic by `sr!`, whose factor is not orthogonal, so the residual
        # scales with `‖Λ‖²` and has a heavy tail: measured at most `2540eps(T) * ‖U‖² ‖Λ‖²` over
        # 2000 conditioned points per size in both precisions, median under `0.4`. A completion
        # that skips the projection out of `U` exceeds the bound on every such point in `Float64`,
        # and on 87 % (at `10 × 6`) to 99 % (at `4 × 2`) of them in `Float32`.
        @test norm(U.A' * J * Λ) < 4096eps(T) * norm(U.A)^2 * norm(Λ)^2

        # Together they span the whole space symplectically. Measured at most
        # `2330eps(T) * cond(full)^2` on the same points, median under `0.5`.
        N, n = N2 ÷ 2, n2 ÷ 2
        m = N - n
        full = hcat(U.A[:, 1:n], Λ[:, 1:m], U.A[:, (n + 1):(2 * n)], Λ[:, (m + 1):(2 * m)])
        @test norm(full' * J * full - J) < 4096eps(T) * cond(full)^2
    end
end

# `Base.copy(::Manifold)` builds `typeof(U)(…)`, i.e. the two-parameter constructor, and `_similar`
# and `GlobalSection` both go through it — which is the path every manifold optimizer takes. A type
# that declares an inner constructor loses that one unless it declares it too, so it is asserted
# here rather than left to the first optimizer that reaches for it.
@testset "the generic Manifold methods reach this type" begin
    U = rand(SymplecticStiefelManifold, 6, 4)
    @test copy(U) isa SymplecticStiefelManifold
    @test copy(U).A == U.A
    @test _similar(U) isa SymplecticStiefelManifold
    @test size(_similar(U)) == size(U)
end

# A row vector on the left is the one shape `*(::AbstractMatrix, ::SymplecticStiefelManifold)` does
# not settle on its own: `LinearAlgebra` has its own method for that left operand, narrower there
# and wider on the right, so neither wins. The two row-vector methods in `src/ambiguities.jl`
# settle it.
#
# How far the point is from the manifold does not enter: both sides of each assertion are the same
# point, one wrapped and one dense. `U` is rectangular, so a method that swapped or dropped an
# operand would not conform.
@testset "a row vector times a SymplecticStiefelManifold, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1241)
    for (N2, n2) in SIZES
        U = rand(rng, SymplecticStiefelManifold{T}, N2, n2)
        v = rand(rng, T, N2)

        @test v' * U ≈ v' * Matrix(U)
        @test transpose(v) * U ≈ transpose(v) * Matrix(U)
        @test size(v' * U) == (1, n2)
        @test eltype(v' * U) == T
    end
end

# The same standoff one wrapper in: `*(::AbstractMatrix, ::Adjoint{<:Symplectic…})` does not settle
# it either. `LinearAlgebra` has its own method for that left operand, narrower there and
# wider on the right, so neither wins. The two row-vector methods in `src/ambiguities.jl` settle
# it. `U'` is rectangular here, so a method that swapped or dropped an operand would not conform.
@testset "a row vector times the adjoint of a point, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1242)
    for (N2, n2) in SIZES
        U = rand(rng, SymplecticStiefelManifold{T}, N2, n2)
        v = rand(rng, T, n2)

        @test v' * U' ≈ v' * Matrix(U.A')
        @test transpose(v) * U' ≈ transpose(v) * Matrix(transpose(U.A))
        @test size(v' * U') == (1, N2)
        @test eltype(v' * U') == T
    end
end

# A `Float64` literal anywhere in the metric silently widens a `Float32` point's metric to
# `Float64`, which no `Float64` test can see.
@testset "the metric keeps the element type of the point, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1243)
    U = rand(rng, SymplecticStiefelManifold{T}, 6, 4)
    Δ = randn(rng, T, 6, 4)
    @test metric(U, Δ, Δ) isa T
    @test eltype(metric(U, Δ, Δ)) == T
end

# The entries of the Poisson tensor are `0` and `±1`, and the identities below are exact; the
# integer-valued data is what the assertions are about.
@testset "the canonical Poisson tensor, $T" for T in REAL_ELTYPES
    J = _poisson_tensor(T, 4)
    @test eltype(J) == T
    @test J == T[0 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0]
    @test J' == -J
    @test J * J == -Matrix{T}(I, 4, 4)
    @test eltype(_poisson_tensor(T, 6)) == T
    @test_throws AssertionError _poisson_tensor(T, 5)
end

# A stand-in device, because the refusal below is a property of `GPU` and not of any one vendor.
struct _StandInGPU <: GPU end

# The generic `Manifold{T}` draw in `abstract_manifold.jl` orthonormalises a Gaussian matrix with
# `qr`, which preserves the Euclidean form and not ``\mathbb{J}``. It matches this type as well, and
# the inner constructor asserts shape alone -- so a backend-taking draw that reached it would return
# a point off this manifold and nothing downstream would say so. The second assertion in each group
# is what discriminates the two draws: an orthonormal point puts `norm(U.A'U.A - I)` at machine
# precision, and a symplectic one does not. The bounds are those of the first testset.
@testset "a backend-taking draw is the symplectic draw, $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    for MT in (SymplecticStiefelManifold{T}, SymplecticStiefelManifold{T, Matrix{T}})
        U = rand(CPU(), MT, 6, 4)
        @test U isa SymplecticStiefelManifold
        @test eltype(U) == T
        @test check(U) < 8eps(T) * cond(U.A)^2
        @test norm(U.A' * U.A - I) > sqrt(eps(T))
    end

    # the rng-taking spelling reaches the same draw
    U = rand(Random.Xoshiro(1244), CPU(), SymplecticStiefelManifold{T}, 6, 4)
    @test eltype(U) == T
    @test check(U) < 8eps(T) * cond(U.A)^2
    @test norm(U.A' * U.A - I) > sqrt(eps(T))
end

# The bare type has no element type to loop over: `default_eltype(CPU())` fills it in, so this
# draw is in that one precision by design.
@testset "a backend-taking draw of the bare type is the symplectic draw" begin
    Random.seed!(1234)
    U = rand(CPU(), SymplecticStiefelManifold, 6, 4)
    @test U isa SymplecticStiefelManifold
    @test eltype(U) == Float64
    @test check(U) < 8eps(Float64) * cond(U.A)^2
    @test norm(U.A' * U.A - I) > sqrt(eps(Float64))
end

# `sr!` is a host factorization -- `_rand_symplectic_stiefel` calls `Matrix` on its factor -- so
# there is no device draw to fall back to. Refusing is the point: without these methods the generic
# `rand(::GPU, ...)` answers with an orthonormal point instead.
@testset "a device draw is refused rather than answered orthonormally" begin
    @test_throws ArgumentError rand(_StandInGPU(), SymplecticStiefelManifold{Float64}, 6, 4)
    @test_throws ArgumentError rand(_StandInGPU(), SymplecticStiefelManifold{Float32}, 6, 4)
    # the element type left open, which `default_eltype` fills in before dispatch arrives here
    @test_throws ArgumentError rand(_StandInGPU(), SymplecticStiefelManifold, 6, 4)
    @test_throws ArgumentError rand(
        Random.default_rng(), _StandInGPU(), SymplecticStiefelManifold{Float64}, 6, 4)
end
