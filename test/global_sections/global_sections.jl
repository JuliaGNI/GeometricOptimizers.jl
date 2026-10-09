using GeometricOptimizers
using GeometricOptimizers: apply_section, global_rep, StiefelProjection
using LinearAlgebra: norm
using Test
import Random

include("../helpers/eltypes.jl")

# `GlobalSection` draws its completion from the global generator
Random.seed!(123)

include("../helpers/grassmann_test_help.jl")

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
