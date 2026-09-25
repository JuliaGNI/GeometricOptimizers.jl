# Every product and sum among this package's own matrix types, and between one of them and a plain
# array, on a device, each against a host twin built from the same numbers.
#
# The sweep is `scripts/device_products.jl`, and `metal.jl` runs the same one on Metal: `JLArrays` with
# `allowscalar(false)` stands in for the device here, as it does in `device_multiply.jl`. A row that
# reads an owned matrix one entry at a time raises `Scalar indexing is disallowed`, and a row whose
# answer comes back on the host or disagrees with the host twin is `:wrong`.

using JLArrays: JLArray
using Test

include(joinpath(@__DIR__, "..", "scripts", "device_products.jl"))

# `cayley` inverts a matrix with `LinearAlgebra.inv`, and `JLArrays` supplies no `lu` for it. That is
# the reference backend's gap, pinned in `device_multiply.jl`; Metal has an `lu`, and the rows pass
# there.
const JLARRAYS_GAPS = ("cayley(StiefelManifold, Δ)", "cayley(GrassmannManifold, Δ)")

# `geodesic(B, ProjectedSkew())` calls the backend's own `qr` and `eigen`. `JLArrays` has neither, so
# under `allowscalar(false)` it raises inside the generic `qr`, and that is where Metal stops too.
# With scalar indexing allowed the generic ones run, and a result that is still a `JLArray` is what
# says that no step copies to the host — the part of the method this package controls. `(6, 4)` has
# `2n > N`, where the thin `Q` has `N` columns and not `2n`.
const PROJECTED_SKEW_SHAPES = ((6, 3), (6, 4))

@testset "ProjectedSkew keeps every step on the lift's backend, $N × $n" for (N, n) in PROJECTED_SKEW_SHAPES
    rng = Random.Xoshiro(4)
    host = StiefelLieAlgHorMatrix(rand(rng, SkewSymMatrix{T}, n), randn(rng, T, N - n, n), N, n)
    device = todev(JLArray, host)

    @test_throws "Scalar indexing is disallowed" geodesic(device, GeometricOptimizers.ProjectedSkew())

    Y = allowscalar(() -> geodesic(device, GeometricOptimizers.ProjectedSkew()))
    @test Y.A isa JLArray{T, 2}
    @test Array(Y.A) ≈ geodesic(host, GeometricOptimizers.ProjectedSkew()).A
end

@testset "every product and sum runs on the device and matches the host" begin
    for (name, status) in device_products(JLArray)
        @testset "$name" begin
            if name in JLARRAYS_GAPS
                @test occursin("Scalar indexing is disallowed", string(status))
            else
                @test status === :pass
            end
        end
    end
end
