using GeometricOptimizers
using GeometricOptimizers: apply_section, global_rep, global_rep!, gradient_rep!,
                           StiefelProjection, _similar, _fill!, _dot
using LinearAlgebra: Symmetric, cond, norm
using NeuralNetworkParameters: NetworkParameters
using Test
import Random

include("../helpers/eltypes.jl")

# `GlobalSection` draws its completion from the global generator
Random.seed!(123)

include("../helpers/grassmann_test_help.jl")
include("../helpers/symplectic_points.jl")

# The section's first `n` columns span `Y`, so projecting onto them gives `Y` back to the round-off of
# two products: measured at most `1.03eps(T)` in both precisions over 200 seeds of this sweep.
function grassmann_global_section(rng, N::Integer, n::Integer, T::DataType)
    Y = rand(rng, GrassmannManifold{T}, N, n)
    Q = Matrix(GlobalSection(Y))
    @test eltype(Q) == T
    πQ = Q[1:N, 1:n]
    norm(Y - πQ * πQ' * Y) / N / n < 4eps(T)
end

# This built a `GrassmannManifold`, so it was `grassmann_global_section` under a second name and the
# Stiefel section went untested. What it says now is the defining property of a section — `λ(Y)E` is
# `Y` again — which is what `GeometricMachineLearning`'s
# `test/optimizers/utils/global_sections.jl` asserted, and is where this comes from.
function stiefel_global_section(rng, N::Integer, n::Integer, T::DataType)
    Y = rand(rng, StiefelManifold{T}, N, n)
    λY = GlobalSection(Y)

    E = StiefelManifold(Matrix{T}(StiefelProjection(T, N, n)))
    Y₂ = apply_section(λY, E)

    @test typeof(Y₂) <: StiefelManifold
    @test eltype(Y₂) == T
    isapprox(Y₂, Y)
end

# `global_rep` maps `T_Y M → 𝔤ʰᵒʳ`, and applying the section to `BE` has to bring the lift back to
# the tangent vector it came from. Nothing here tested that the two are inverse to each other: the
# `Ω` tests next door cover only the first of the two isomorphisms `global_rep` composes.
function global_tangent_space_rep(rng, N::Integer, n::Integer, T::DataType)
    Y = rand(rng, StiefelManifold{T}, N, n)
    λY = GlobalSection(Y)

    Δ = rgrad(Y, rand(rng, T, N, n))
    B = global_rep(λY, Δ)
    @test eltype(B) == T
    BE = B * StiefelProjection(T, N, n)
    # abuse of notation: `BE` is a tangent vector and not a point, but `apply_section` is the same
    # left-multiplication by `λ(Y)` either way
    Δ₂ = typeof(Δ)(apply_section(λY, StiefelManifold(BE)))

    isapprox(Δ₂, Δ)
end

@testset "GlobalSection and global_rep, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for N in 3:5
        for n in 1:N
            @test stiefel_global_section(rng, N, n, T)
            @test grassmann_global_section(rng, N, n, T)
            @test global_tangent_space_rep(rng, N, n, T)
        end
    end
end

# The section's columns are orthogonal to `Y`, to rounding, on every draw. One projection of `Y` out
# of the Gaussian draw leaves a rounding error in the span of `Y`, and the orthonormalisation
# amplifies it by the condition number of the projected draw, which has a heavy tail. With one pass
# only, one draw in a few thousand gives `‖Yᵀλ‖` near `1e-2` in `Float32` and `1e-10` in `Float64`,
# so the property is asserted over many draws of one seeded run rather than over one. With the two
# passes `global_section` makes, the maximum over these draws, and over 20 more points with a tenth
# of the draws each, is under `2eps(T)` in both precisions.
@testset "the section is orthogonal to the point on every draw, $T" for T in REAL_ELTYPES
    Random.seed!(2024)
    shapes = ((6, 3, 20000), (50, 3, 2000))
    for (N, n, draws) in shapes, M in (StiefelManifold, GrassmannManifold)

        Y = rand(M{T}, N, n)
        orthogonality = maximum(_ -> norm(Y.A' * global_section(Y)), 1:draws)
        @test eltype(orthogonality) == T
        @test orthogonality < 8eps(T)
    end
end

# `apply_section!` with no workspace. Into a destination that shares no memory with an input it is
# two `mul!`s and allocates nothing at any `N`; the second adds its product into the first, so it
# rounds differently from the sum of two materialised products, by the rounding of one more addition
# per entry: `100eps(T)` relative. Into its own input -- `Y === Y₂`, which `update_section!` passes
# when it has no workspace -- the products are materialised, and the answer is the reference's to
# the bit. The reference is the sum of the two materialised products.
reference_apply_section(λY, A₂, n, N) = λY.Y * A₂[1:n, :] .+ λY.λ * A₂[(n + 1):N, :]

function _measured_apply_section!(Y, λY, Y₂)
    (apply_section!(Y, λY, Y₂); @allocated apply_section!(Y, λY, Y₂))
end

function section_fixture(M, ::Type{T}, N, n, columns) where {T}
    λY = GlobalSection(rand(Random.Xoshiro(N), M{T}, N, n))
    Y₂ = rand(Random.Xoshiro(N + 1), M{T}, N, columns)
    (λY = λY, Y₂ = Y₂, Y = M(zeros(T, N, columns)))
end

@testset "apply_section! into a separate destination allocates nothing, $(nameof(M)){$T}" for T in REAL_ELTYPES,
    M in (StiefelManifold, GrassmannManifold)

    for N in (40, 400), columns in (3, N)

        f = section_fixture(M, T, N, 3, columns)
        @test _measured_apply_section!(f.Y, f.λY, f.Y₂) == 0
        @test (@inferred apply_section!(f.Y, f.λY, f.Y₂)) === f.Y
    end
end

@testset "apply_section! agrees with the materialised products, $(nameof(M)){$T}" for T in REAL_ELTYPES,
    M in (StiefelManifold, GrassmannManifold)

    for N in (40, 400), columns in (3, N)

        f = section_fixture(M, T, N, 3, columns)
        reference = reference_apply_section(f.λY, f.Y₂.A, 3, N)

        @test apply_section!(f.Y, f.λY, f.Y₂) === f.Y
        @test eltype(f.Y) == T
        @test isapprox(f.Y.A, reference; rtol = 100eps(T))

        # the alias `update_section!` passes
        aliased = M(copy(f.Y₂.A))
        apply_section!(aliased, f.λY, aliased)
        @test aliased.A == reference
    end
end

# A destination that is the section's own frame `λY.Y` is an input of the first product, so it
# takes the materialised branch too.
@testset "apply_section! into the section's frame, $(nameof(M)){$T}" for T in REAL_ELTYPES,
    M in (StiefelManifold, GrassmannManifold)

    for N in (40, 400)
        f = section_fixture(M, T, N, 3, 3)
        reference = reference_apply_section(f.λY, f.Y₂.A, 3, N)

        @test apply_section!(f.λY.Y, f.λY, f.Y₂) === f.λY.Y
        @test eltype(f.λY.Y) == T
        @test f.λY.Y.A == reference
    end
end

# `global_rep!` writes the lift into a destination that a cache fills with `NaN` at construction, so
# every entry has to be overwritten. The reference is the formula of the `global_rep` docstrings,
# written out with the lift constructors.
function reference_lift(λY::GlobalSection{T, <:StiefelManifold}, Δ) where {T}
    StiefelLieAlgHorMatrix(SkewSymMatrix(λY.Y.A' * Δ), λY.λ' * Δ, size(λY.Y)...)
end
function reference_lift(λY::GlobalSection{T, <:GrassmannManifold}, Δ) where {T}
    GrassmannLieAlgHorMatrix(λY.λ' * Δ, size(λY.Y)...)
end

@testset "global_rep! overwrites a NaN-filled lift, $(nameof(M)){$T}" for T in REAL_ELTYPES,
    M in (StiefelManifold, GrassmannManifold)

    rng = Random.Xoshiro(9)
    for (N, n) in ((5, 2), (40, 3))
        Y = rand(rng, M{T}, N, n)
        λY = GlobalSection(Y)
        Δ = rgrad(Y, randn(rng, T, N, n))
        D = fill!(similar(reference_lift(λY, Δ)), T(NaN))
        @test global_rep!(D, λY, Δ) === D
        @test D ≈ reference_lift(λY, Δ)
        @test global_rep(λY, Δ) == D
    end
end

# The container method pairs the trees by key and reaches the leaves at any depth: a lift for a
# manifold leaf and a copy for a vector leaf.
@testset "global_rep! on a parameter set is global_rep, $T" for T in REAL_ELTYPES
    Random.seed!(10)
    rng = Random.Xoshiro(10)
    ps = NetworkParameters((w = rand(rng, StiefelManifold{T}, 6, 3),
        v = (g = rand(rng, GrassmannManifold{T}, 5, 2), b = randn(rng, T, 4))))
    gx = NetworkParameters((w = randn(rng, T, 6, 3),
        v = (g = randn(rng, T, 5, 2), b = randn(rng, T, 4))))
    λ = GlobalSection(ps)
    reference = global_rep(λ, gx)
    D = _fill!(_similar(reference), T(NaN))
    @test global_rep!(D, λ, gx) === D
    @test D == reference
end

# The tests of `GlobalSection and global_rep` above, for the symplectic Stiefel manifold: `λ(U)E` is
# `U`, the section is symplectic in the adapted basis, and `global_rep` is inverted by applying the
# section to `BE`.
function symplectic_global_section(rng, N2, n2, T)
    U = conditioned_point(rng, T, N2, n2)
    λU = GlobalSection(U)

    E = SymplecticStiefelManifold(Matrix{T}(StiefelProjection(T, N2, n2)))
    U₂ = apply_section(λU, E)
    @test U₂ isa SymplecticStiefelManifold{T}
    M = Matrix(λU)
    @test eltype(M) == T
    @test norm(M' * poisson(T, N2) * M - adapted_poisson(T, N2, n2)) < 64eps(T) * norm(M)^2
    isapprox(U₂, U)
end

function symplectic_global_tangent_space_rep(rng, N2, n2, T)
    U = conditioned_point(rng, T, N2, n2)
    λU = GlobalSection(U)

    Δ = rgrad(U, rand(rng, T, N2, n2))
    B = global_rep(λU, Δ)
    @test B isa SymplecticLieAlgHorMatrix{T}
    BE = B * StiefelProjection(T, N2, n2)
    # the same abuse of notation as in `global_tangent_space_rep`
    Δ₂ = apply_section(λU, SymplecticStiefelManifold(BE))

    isapprox(Δ₂.A, Δ)
end

const SYMPLECTIC_SIZES = ((4, 2), (6, 2), (6, 4), (8, 4), (10, 4), (6, 6))

@testset "GlobalSection and global_rep of a SymplecticStiefelManifold, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for (N2, n2) in SYMPLECTIC_SIZES
        @test symplectic_global_section(rng, N2, n2, T)
        @test symplectic_global_tangent_space_rep(rng, N2, n2, T)
    end
end

# `apply_section!` into a separate destination is the two `mul!`s of the Stiefel method, and into its
# own input the materialised products.
@testset "apply_section! on a SymplecticStiefelManifold, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(124)
    λU = GlobalSection(conditioned_point(rng, T, 10, 4))
    U₂ = conditioned_point(rng, T, 10, 4)
    reference = reference_apply_section(λU, U₂.A, 4, 10)

    U = SymplecticStiefelManifold(zeros(T, 10, 4))
    @test apply_section!(U, λU, U₂) === U
    @test isapprox(U.A, reference; rtol = 100eps(T))
    aliased = SymplecticStiefelManifold(copy(U₂.A))
    apply_section!(aliased, λU, aliased)
    @test aliased.A == reference
end

# The reference of the `NaN` test above, for the symplectic lift: the blocks `U⁺Δ` and `(Uˢ)⁺Δ` with
# the Poisson tensors written out.
function reference_lift(λY::GlobalSection{T, <:SymplecticStiefelManifold}, Δ) where {T}
    N2, n2 = size(λY.Y)
    J = poisson(T, N2)
    𝒜 = poisson(T, n2)' * λY.Y.A' * J * Δ
    ℬ = poisson(T, N2 - n2)' * λY.λ' * J * Δ
    D = zeros(T, N2, N2)
    D[1:n2, 1:n2] .= 𝒜
    D[(n2 + 1):N2, 1:n2] .= ℬ
    SymplecticLieAlgHorMatrix(D, n2)
end

@testset "global_rep! overwrites a NaN-filled lift, SymplecticStiefelManifold{$T}" for T in REAL_ELTYPES
    rng = Random.Xoshiro(9)
    for (N2, n2) in ((6, 2), (10, 4))
        U = conditioned_point(rng, T, N2, n2)
        λU = GlobalSection(U)
        Δ = rgrad(U, randn(rng, T, N2, n2))
        D = fill!(similar(reference_lift(λU, Δ)), T(NaN))
        @test global_rep!(D, λU, Δ) === D
        @test D ≈ reference_lift(λU, Δ)
        @test global_rep(λU, Δ) == D
    end
end

# The lift the optimizers hold as the gradient pairs with every lift `B` under `_dot` as the
# derivative of the loss along `λ(U)BE` does: `⟨∇L, λ(U)BE⟩`. For the Stiefel and the Grassmann
# manifold this is `global_rep` of the Riemannian gradient; for the symplectic Stiefel manifold it is
# not, and `global_rep` of the Riemannian gradient pairs to something else, which the last assertion
# states.
@testset "gradient_rep! pairs as the derivative of the loss, $(nameof(M)){$T}" for T in REAL_ELTYPES,
    M in (StiefelManifold, GrassmannManifold, SymplecticStiefelManifold)

    rng = Random.Xoshiro(11)
    for _ in 1:5
        Y = M === SymplecticStiefelManifold ? conditioned_point(rng, T, 10, 4) :
            rand(rng, M{T}, 10, 4)
        λY = GlobalSection(Y)
        ∇L = randn(rng, T, 10, 4)
        Δ = rgrad(Y, ∇L)
        D = fill!(similar(global_rep(λY, Δ)), T(NaN))
        @test gradient_rep!(D, λY, Δ) === D

        B = rand(rng, typeof(D).name.wrapper{T}, 10, 4)
        derivative = sum(∇L .* (Matrix(λY) * (B * StiefelProjection(T, 10, 4))))
        @test _dot(D, B) ≈ derivative rtol = 64eps(T) * 10
        if M === SymplecticStiefelManifold
            @test !isapprox(_dot(global_rep(λY, Δ), B), derivative; rtol = sqrt(eps(T)))
        else
            @test D ≈ global_rep(λY, Δ)
        end
    end
end

@testset "global_rep! and gradient_rep! on a parameter set with a symplectic leaf, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(10)
    ps = NetworkParameters((u = conditioned_point(rng, T, 6, 2),
        v = (w = rand(rng, StiefelManifold{T}, 5, 2), b = randn(rng, T, 4))))
    gx = NetworkParameters((u = randn(rng, T, 6, 2),
        v = (w = randn(rng, T, 5, 2), b = randn(rng, T, 4))))
    λ = GlobalSection(ps)
    reference = global_rep(λ, gx)
    @test reference.u isa SymplecticLieAlgHorMatrix{T}
    D = _fill!(_similar(reference), T(NaN))
    @test global_rep!(D, λ, gx) === D
    @test D == reference

    @test gradient_rep!(D, λ, gx) === D
    @test D.u == gradient_rep!(similar(reference.u), λ.u, gx.u)
    @test D.v.w == reference.v.w
    @test D.v.b == reference.v.b

    # the descent of a first-order method is `global_rep` on every leaf
    @test gradient_rep!(D, λ, gx, GradientMethod()) == reference
end

# The gradient each method holds, `gradient_rep!(D, λY, Δ, method)`, against what defines it rather
# than against the code that computes it. `Δ = rgrad(U, ∇L)` and `⟨∇L, V⟩` is the derivative of the
# loss along a tangent vector `V`.
#
# - A method other than `GradientMethod`, `MomentumMethod` and `Adam` holds the derivative with
#   respect to the stored entries: the three-argument form.
# - `GradientMethod` holds the lift whose tangent vector `λ(U)DE` is the Riesz representative of `∇L`
#   in metric (3.16), `g_U(λ(U)DE, V) = ⟨∇L, V⟩` for every tangent `V`. It is `Δ` for every section.
# - The derivative with respect to the stored entries, read as a lift, is not the Riesz
#   representative, and a change of the section by `diag(I, S)`, which leaves the point where it is,
#   changes its tangent vector: the two controls.
#
# The pairings are measured against `‖∇L‖‖V‖`, because a pairing of two random matrices can cancel.
# The section change is `S = exp(𝕁H)`, `H` symmetric, which is symplectic; the round-off of a
# section-independent tangent vector grows with `cond(S)`, measured at most `0.3eps(T)cond(S)`
# in both precisions over 20 draws.
@testset "the gradient of each descent on a SymplecticStiefelManifold{$T}" for T in REAL_ELTYPES
    rng = Random.Xoshiro(12)
    E = StiefelProjection(T, 10, 4)
    metric_methods = (GradientMethod(), MomentumMethod(), Adam())
    for _ in 1:10
        U = conditioned_point(rng, T, 10, 4)
        λU = GlobalSection(U)
        ∇L = randn(rng, T, 10, 4)
        Δ = rgrad(U, ∇L)
        lift() = fill!(similar(global_rep(λU, Δ)), T(NaN))
        flat = gradient_rep!(lift(), λU, Δ)
        for method in (BFGS(), DFP())
            @test gradient_rep!(lift(), λU, Δ, method) == flat
        end

        V = rgrad(U, randn(rng, T, 10, 4))
        scale = norm(∇L) * norm(V)
        for method in metric_methods
            D = lift()
            @test gradient_rep!(D, λU, Δ, method) === D
            @test D == global_rep(λU, Δ)
            @test isapprox(metric(U, Matrix(λU) * (D * E), V), sum(∇L .* V);
                atol = 64eps(T) * scale)
        end

        @test !isapprox(metric(U, Matrix(λU) * (flat * E), V), sum(∇L .* V);
            atol = sqrt(eps(T)) * scale)

        λ₂ = GlobalSection(U)
        S = exp(poisson(T, 6) * T.(Symmetric(randn(rng, 6, 6))))
        λ₂.λ .= λ₂.λ * S
        tangent(λ, method) = Matrix(λ) * (gradient_rep!(lift(), λ, Δ, method) * E)
        Δ₁ = tangent(λU, GradientMethod())
        @test isapprox(tangent(λ₂, GradientMethod()), Δ₁; rtol = 8eps(T) * cond(S))
        @test isapprox(Δ₁, Δ; rtol = 64eps(T))
        @test !isapprox(tangent(λ₂, BFGS()), tangent(λU, BFGS()); rtol = T(0.1))
    end
end
