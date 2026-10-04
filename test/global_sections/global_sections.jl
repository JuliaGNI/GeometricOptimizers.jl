using GeometricOptimizers
using GeometricOptimizers: apply_section, global_rep, StiefelProjection
using LinearAlgebra: norm
using Test
import Random

Random.seed!(123)

include("../helpers/grassmann_test_help.jl")

function grassmann_global_section(N::Integer, n::Integer, T::DataType)
    Y = rand(GrassmannManifold{T}, N, n)
    Q = Matrix(GlobalSection(Y))
    πQ = Q[1:N, 1:n]
    norm(Y - πQ * πQ' * Y) / N / n < eps(T)
end

# This built a `GrassmannManifold`, so it was `grassmann_global_section` under a second name and the
# Stiefel section went untested. What it says now is the defining property of a section — `λ(Y)E` is
# `Y` again — which is what `GeometricMachineLearning`'s
# `test/optimizers/utils/global_sections.jl` asserted, and is where this comes from.
function stiefel_global_section(N::Integer, n::Integer, T::DataType)
    Y = rand(StiefelManifold{T}, N, n)
    λY = GlobalSection(Y)

    E = StiefelManifold(Matrix{T}(StiefelProjection(T, N, n)))
    Y₂ = apply_section(λY, E)

    @test typeof(Y₂) <: StiefelManifold
    isapprox(Y₂, Y)
end

# `global_rep` maps `T_Y M → 𝔤ʰᵒʳ`, and applying the section to `BE` has to bring the lift back to
# the tangent vector it came from. Nothing here tested that the two are inverse to each other: the
# `Ω` tests next door cover only the first of the two isomorphisms `global_rep` composes.
function global_tangent_space_rep(N::Integer, n::Integer, T::DataType)
    Y = rand(StiefelManifold{T}, N, n)
    λY = GlobalSection(Y)

    Δ = rgrad(Y, rand(T, N, n))
    B = global_rep(λY, Δ)
    BE = B * StiefelProjection(T, N, n)
    # abuse of notation: `BE` is a tangent vector and not a point, but `apply_section` is the same
    # left-multiplication by `λ(Y)` either way
    Δ₂ = typeof(Δ)(apply_section(λY, StiefelManifold(BE)))

    isapprox(Δ₂, Δ)
end

T = Float32

for N in 3:5
    for n in 1:N
        @test stiefel_global_section(N, n, T)
        @test grassmann_global_section(N, n, T)
        @test global_tangent_space_rep(N, n, T)
    end
end

# The section's columns are orthogonal to `Y`, to rounding, on every draw. One projection of `Y` out
# of the Gaussian draw leaves a rounding error in the span of `Y`, and the orthonormalisation
# amplifies it by the condition number of the projected draw, which has a heavy tail. With one pass
# only, one draw in a few thousand gives `‖Yᵀλ‖` near `1e-2` in `Float32` and `1e-10` in `Float64`,
# so the property is asserted over many draws of one seeded run rather than over one.
@testset "the section is orthogonal to the point on every draw" begin
    Random.seed!(2024)
    tolerances = ((Float32, 1.0f-5), (Float64, 1e-13))
    shapes = ((6, 3, 20000), (50, 3, 2000))
    for (T, tol) in tolerances, (N, n, draws) in shapes,
        M in (StiefelManifold, GrassmannManifold)
        Y = rand(M{T}, N, n)
        @test maximum(_ -> norm(Y.A' * global_section(Y)), 1:draws) < tol
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

@testset "apply_section! into a separate destination allocates nothing, $(nameof(M)){$T}" for M in (StiefelManifold,
        GrassmannManifold),
    T in (Float32, Float64)

    for N in (40, 400), columns in (3, N)

        f = section_fixture(M, T, N, 3, columns)
        @test _measured_apply_section!(f.Y, f.λY, f.Y₂) == 0
        @test (@inferred apply_section!(f.Y, f.λY, f.Y₂)) === f.Y
    end
end

@testset "apply_section! agrees with the materialised products, $(nameof(M)){$T}" for M in (
        StiefelManifold,
        GrassmannManifold),
    T in (Float32, Float64)

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
@testset "apply_section! into the section's frame, $(nameof(M)){$T}" for M in (StiefelManifold,
        GrassmannManifold),
    T in (Float32, Float64)

    for N in (40, 400)
        f = section_fixture(M, T, N, 3, 3)
        reference = reference_apply_section(f.λY, f.Y₂.A, 3, N)

        @test apply_section!(f.λY.Y, f.λY, f.Y₂) === f.λY.Y
        @test eltype(f.λY.Y) == T
        @test f.λY.Y.A == reference
    end
end
