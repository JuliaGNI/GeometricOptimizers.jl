# The device sweep on Metal: construction, the manifold operations, every product and sum, the
# retractions and `solve!`, each against a host twin, under `allowscalar(false)`. It prints
# PASS or FAIL per row and returns the rows.
#
# Run it by hand through a Kaimon session: Metal is unreachable from a sandboxed shell, where
# `Metal.functional()` is `false`. Activate a scratch environment that develops this tree first, and
# restart the session after the last edit of `src/`:
#
#     import Pkg
#     Pkg.activate(; temp = true)
#     Pkg.develop(path = "<this repository>")
#     Pkg.add(["Metal", "JLArrays", "GPUArraysCore", "KernelAbstractions", "AbstractNeuralNetworks",
#         "NeuralNetworkParameters"])
#     include("<this repository>/scripts/metal_check.jl")
#     metal_check()
#
# `Float32` throughout, because Metal has no `Float64`. The rows are those of
# `scripts/device_manifolds.jl`, `scripts/device_products.jl` and `scripts/device_solve.jl`, with
# `solve!` run once under `Cayley()` and once under `Geodesic()`. `test/devices/metal.jl` asserts the
# same rows in the `metal` test group. Metal draws its random numbers from a generator of its own, so
# the `Adam` rows of a manifold run are compared by a property; see `UNMATCHED_ADAM_RTOL` in
# `device_solve.jl`.
#
# `device_check(todevice)` is the same sweep for another array type: `device_check(JLArray)` runs it
# on the reference backend, where the two `cayley` rows of `device_products` and the ten manifold
# rows of `solve!` under `Cayley()` fail for want of an `lu`.

using GeometricOptimizers
using Metal: Metal, MtlArray

include(joinpath(@__DIR__, "device_products.jl"))
include(joinpath(@__DIR__, "device_manifolds.jl"))
include(joinpath(@__DIR__, "device_solve.jl"))

function device_check(todevice; adam_rtol = UNMATCHED_ADAM_RTOL, matched_rng = false)
    rows = device_manifolds(todevice)
    append!(rows, device_products(todevice))
    for retraction in (Cayley(), Geodesic())
        for (name, status) in device_solve(todevice, Float32; retraction = retraction,
            matched_rng = matched_rng, adam_rtol = adam_rtol)
            push!(rows, "$name, $(nameof(typeof(retraction)))" => status)
        end
    end
    for (name, status) in rows
        println(status === :pass ? "PASS  " : "FAIL  ", name,
            status === :pass ? "" : "  ($(status))")
    end
    rows
end

metal_check(; kwargs...) = device_check(MtlArray; kwargs...)
