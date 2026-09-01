using GeometricOptimizers
using GeometricOptimizers: apply_section, global_rep, StiefelProjection
using LinearAlgebra: norm
using Test
import Random

Random.seed!(123)

include("../grassmann_test_help.jl")

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
