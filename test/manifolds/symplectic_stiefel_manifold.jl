using Test
using LinearAlgebra
using GeometricOptimizers
using GeometricOptimizers: _poisson_tensor, _similar
using KernelAbstractions: CPU, GPU
import Random

Random.seed!(1234)

# See the note on tolerances in `test/decompositions/symplectic_sr.jl`: the residual grows with the
# size because the SR decomposition these points come from has no re-orthogonalization step, no
# per-draw threshold both discriminates and never fails, so the assertions below run on the fixed
# seed above and are deterministic rather than bounds, and `Float32` is out of reach at every size.
# The figures are in `CHANGELOG.md` under *Open Issues*.
tolerance(N2) = N2 ≤ 4 ? 1e-6 : N2 ≤ 6 ? 1e-5 : 1e-4

const SIZES = ((4, 2), (6, 4), (10, 6))

@testset "a random point lies on the manifold" begin
    for (N2, n2) in SIZES
        U = rand(SymplecticStiefelManifold, N2, n2)
        @test size(U) == (N2, n2)
        @test check(U) < tolerance(N2)

        # `check` is the symplectic residual and not the generic orthonormality one. Without the
        # specific method this testset would be asserting the wrong constraint and would fail; the
        # assertion below states that the two are genuinely different quantities here.
        @test norm(U.A' * U.A - I) > 1e-6
    end
end

@testset "the element type is honoured" begin
    U = rand(SymplecticStiefelManifold{Float64}, 6, 4)
    @test eltype(U) == Float64
    # `Float32` points are constructed, and deliberately not checked against the manifold: the
    # median residual at this size is 4.2e-6, which no useful threshold clears. See the table in
    # `CHANGELOG.md`, and read its medians — the maxima move with the draw order.
    @test eltype(rand(SymplecticStiefelManifold{Float32}, 6, 4)) == Float32
end

@testset "the constructor rejects what is not a symplectic shape" begin
    @test_throws AssertionError SymplecticStiefelManifold(randn(5, 4))
    @test_throws AssertionError SymplecticStiefelManifold(randn(6, 3))
    @test_throws AssertionError SymplecticStiefelManifold(randn(4, 6))
end

@testset "rgrad lands in the tangent space" begin
    # The tangent space at `U` is `{Δ : Δᵀ𝕁U + Uᵀ𝕁Δ = 0}`, which is what makes the Riemannian
    # gradient a direction the retraction can follow. A gradient of the right *size* says nothing.
    for (N2, n2) in SIZES
        J = _poisson_tensor(Float64, N2)
        U = rand(SymplecticStiefelManifold, N2, n2)
        Δ = rgrad(U, randn(N2, n2))
        @test size(Δ) == (N2, n2)
        @test norm(Δ' * J * U.A + U.A' * J * Δ) < tolerance(N2)
    end
end

@testset "the metric is symmetric and bilinear" begin
    for (N2, n2) in SIZES
        U = rand(SymplecticStiefelManifold, N2, n2)
        Δ₁, Δ₂ = randn(N2, n2), randn(N2, n2)
        @test isapprox(metric(U, Δ₁, Δ₂), metric(U, Δ₂, Δ₁); atol = tolerance(N2))
        @test isapprox(metric(U, 2 * Δ₁, Δ₂), 2 * metric(U, Δ₁, Δ₂); atol = tolerance(N2))
        @test isapprox(metric(U, Δ₁ + Δ₂, Δ₂), metric(U, Δ₁, Δ₂) + metric(U, Δ₂, Δ₂);
            atol = tolerance(N2))
    end
end

# Symmetry and bilinearity hold for *every* inner product, so the testset above pins nothing about
# which one this is: substituting the Frobenius product `tr(Δ₁ᵀΔ₂)` leaves all nine of its
# assertions passing. What identifies the metric is the property that defines the Riemannian
# gradient against it — `g_U(rgrad(U, ∇L), Δ) = tr(∇LᵀΔ)` for every tangent `Δ`. The relative
# residual has median 4.4e-16, 5.4e-15 and 8.7e-14 at the three sizes over 200 draws, against 20,
# 430 and 5300 for the Frobenius substitution, so this assertion separates them by orders.
@testset "the metric is the one rgrad is taken against" begin
    for (N2, n2) in SIZES
        U = rand(SymplecticStiefelManifold, N2, n2)
        ∇L = randn(N2, n2)
        Δ = rgrad(U, randn(N2, n2))
        @test isapprox(metric(U, rgrad(U, ∇L), Δ), tr(∇L' * Δ); rtol = tolerance(N2))

        # And it is positive definite on tangent directions, which a bilinear form need not be.
        @test metric(U, Δ, Δ) > 0
    end
end

@testset "global_section completes the point symplectically" begin
    for (N2, n2) in SIZES
        J = _poisson_tensor(Float64, N2)
        U = rand(SymplecticStiefelManifold, N2, n2)
        Λ = global_section(U)

        # The shape is the completion's, matching `global_section(::StiefelManifold)` and what
        # `GlobalSection` assumes — not the whole `2N x 2N` factor it is sliced from.
        @test size(Λ) == (N2, N2 - n2)

        # And it is a completion: symplectically orthogonal to the point. Asserting only the shape
        # would pass for any matrix of the right size, so the shape alone pins nothing.
        @test norm(U.A' * J * Λ) < tolerance(N2)

        # Together they span the whole space symplectically.
        N, n = N2 ÷ 2, n2 ÷ 2
        m = N - n
        full = hcat(U.A[:, 1:n], Λ[:, 1:m], U.A[:, (n + 1):(2 * n)], Λ[:, (m + 1):(2 * m)])
        @test norm(full' * J * full - J) < tolerance(N2)
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

# A `Float64` literal anywhere in the metric silently widens a `Float32` point's metric to
# `Float64`, which no `Float64` test can see.
@testset "the metric keeps the element type of the point" begin
    U = rand(SymplecticStiefelManifold{Float32}, 6, 4)
    Δ = ones(Float32, 6, 4)
    @test metric(U, Δ, Δ) isa Float32
end

@testset "the canonical Poisson tensor" begin
    J = _poisson_tensor(Float64, 4)
    @test J == [0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0; -1.0 0.0 0.0 0.0; 0.0 -1.0 0.0 0.0]
    @test J' == -J
    @test J * J == -Matrix{Float64}(I, 4, 4)
    @test eltype(_poisson_tensor(Float32, 6)) == Float32
    @test_throws AssertionError _poisson_tensor(Float64, 5)
end

# A stand-in device, because the refusal below is a property of `GPU` and not of any one vendor.
struct _StandInGPU <: GPU end

# The generic `Manifold{T}` draw in `abstract_manifold.jl` orthonormalises a Gaussian matrix with
# `qr`, which preserves the Euclidean form and not ``\mathbb{J}``. It matches this type as well, and
# the inner constructor asserts shape alone -- so a backend-taking draw that reached it would return
# a point off this manifold and nothing downstream would say so. The second assertion in each group
# is what discriminates the two draws: an orthonormal point puts `norm(U.A'U.A - I)` at machine
# precision, and a symplectic one does not.
@testset "a backend-taking draw is the symplectic draw" begin
    Random.seed!(1234)
    for MT in (SymplecticStiefelManifold,
        SymplecticStiefelManifold{Float64},
        SymplecticStiefelManifold{Float64, Matrix{Float64}})
        U = rand(CPU(), MT, 6, 4)
        @test U isa SymplecticStiefelManifold
        @test eltype(U) == Float64
        @test check(U) < tolerance(6)
        @test norm(U.A' * U.A - I) > 1e-6
    end

    # the rng-taking spelling reaches the same draw
    U = rand(CPU(), Random.default_rng(), SymplecticStiefelManifold{Float64}, 6, 4)
    @test check(U) < tolerance(6)
    @test norm(U.A' * U.A - I) > 1e-6
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
        _StandInGPU(), Random.default_rng(), SymplecticStiefelManifold{Float64}, 6, 4)
end
