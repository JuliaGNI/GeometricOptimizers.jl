# Every product and sum among this package's own matrix types, and between one of them and a plain
# array, on a device, each against a host twin built from the same numbers.
#
# The sweep is `scripts/device_products.jl`, and `metal.jl` runs the same one on Metal: `JLArrays` with
# `allowscalar(false)` stands in for the device here, as it does in `device_multiply.jl`. A row that
# reads an owned matrix one entry at a time raises `Scalar indexing is disallowed`, and a row whose
# answer comes back on the host or disagrees with the host twin is `:wrong`.

using JLArrays: JLArray
using Test

include(joinpath(@__DIR__, "..", "..", "scripts", "device_products.jl"))
include("../helpers/eltypes.jl")
include("../helpers/jlarray_cholesky.jl")

# `cayley` inverts a matrix with `LinearAlgebra.inv`, and `JLArrays` supplies no `lu` for it. That is
# the reference backend's gap, pinned in `device_multiply.jl`; Metal has an `lu`, and the rows pass
# there.
const JLARRAYS_GAPS = ("cayley(StiefelManifold, Δ)", "cayley(GrassmannManifold, Δ)")

# `geodesic(B, ProjectedSkew())` calls the backend's own `qr` and `eigen`. `JLArrays` has neither: it
# refuses to hand a `JLArray` to LAPACK, so the `qr` raises, with scalar indexing allowed or not, and
# that is where Metal stops too. `(6, 4)` has `2n > N`, where the thin `Q` has `N` columns and not
# `2n`. That no step of the method copies to the host is untested (K55 in `KNOWN_ISSUES.md`).
const PROJECTED_SKEW_SHAPES = ((6, 3), (6, 4))

# `todev` of a lift moves its storage as it is, so the device lift is in `T` too.
@testset "ProjectedSkew raises in the backend's qr, $T, $N × $n" for T in REAL_ELTYPES,
    (N, n) in PROJECTED_SKEW_SHAPES

    rng = Random.Xoshiro(4)
    host = StiefelLieAlgHorMatrix(rand(rng, SkewSymMatrix{T}, n), randn(rng, T, N - n, n), N, n)
    device = todev(JLArray, host)

    @test_throws "Illegal conversion of a JLArray to a Ptr" geodesic(
        device, GeometricOptimizers.ProjectedSkew())
end

# A row passes when its result is in the element type of its host twin, on the device, and agrees
# with the twin; `metal.jl` runs the `Float32` sweep only, because Metal has no `Float64`.
@testset "every product and sum runs on the device and matches the host, $T" for T in REAL_ELTYPES
    for (name, status) in device_products(JLArray, T)
        @testset "$name" begin
            if name in JLARRAYS_GAPS
                @test occursin("Scalar indexing is disallowed", string(status))
            else
                @test status === :pass
            end
        end
    end
end
