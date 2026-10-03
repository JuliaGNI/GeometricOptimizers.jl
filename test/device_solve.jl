# `solve!` on a device, against a host twin built from the same numbers.
#
# The sweep is `scripts/device_solve.jl`, and `devices/metal.jl` runs the same one on Metal: `JLArrays`
# with `allowscalar(false)` stands in for the device here. A run that reads an iterate one entry at a
# time raises `Scalar indexing is disallowed`; a run whose result leaves the device, is not finite,
# takes another number of iterations or disagrees with the host twin is named by the check it fails.
#
# The retraction is `Geodesic()`: `Cayley()` inverts a matrix with `LinearAlgebra.inv`, and `JLArrays`
# supplies no `lu` for it (`device_multiply.jl`). Metal runs both.

using JLArrays: JLArray
using Test

include(joinpath(@__DIR__, "..", "scripts", "device_solve.jl"))

# The host twin above runs the same `update_section!` as the device, so a wrong update of a
# horizontal lift passes there on both twins. Here the update is checked against the sum of the
# dense matrices, which does not call it.
using GeometricOptimizers: GlobalSection, update_section!
using NeuralNetworkParameters: mapstorage
import Random

@testset "update_section! of a horizontal lift adds the step, $(nameof(L)), $T" for T in (
        Float32, Float64),
    L in (StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix)

    rng = Random.Xoshiro(3)
    Y₀, B = rand(rng, L{T}, 8, 3), rand(rng, L{T}, 8, 3)
    expected = Matrix(Y₀) + Matrix(B)
    for todevice in (identity, JLArray)
        Λᵗ = GlobalSection(mapstorage(todevice, zero(Y₀)))
        Λ⁽ᵗ⁻¹⁾ = GlobalSection(mapstorage(todevice, Y₀))
        @test update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, mapstorage(todevice, B), Geodesic()) === Λᵗ
        @test Matrix(mapstorage(Array, Λᵗ.Y)) == expected
        @test Matrix(mapstorage(Array, Λ⁽ᵗ⁻¹⁾.Y)) == Matrix(Y₀)
    end
end

@testset "solve! runs on the device and matches the host twin, $T" for T in (Float32, Float64)
    for (name, status) in device_solve(JLArray, T)
        @testset "$name" begin
            @test status === :pass
        end
    end
end
