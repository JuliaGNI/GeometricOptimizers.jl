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

@testset "solve! runs on the device and matches the host twin, $T" for T in (Float32, Float64)
    for (name, status) in device_solve(JLArray, T)
        @testset "$name" begin
            @test status === :pass
        end
    end
end
