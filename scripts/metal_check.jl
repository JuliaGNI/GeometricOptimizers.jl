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
# `scripts/device_products.jl` and `scripts/device_solve.jl`, with `solve!` run once under `Cayley()`
# and once under `Geodesic()`, and the rows below. Metal draws its random numbers from a generator of
# its own, so the `Adam` rows of a manifold run are compared by a property; see `UNMATCHED_ADAM_RTOL`
# in `device_solve.jl`.
#
# `device_check(todevice)` is the same sweep for another array type: `device_check(JLArray)` runs it
# on the reference backend, where the two `cayley` rows of `device_products` and the eight manifold
# rows of `solve!` under `Cayley()` fail for want of an `lu`.

using GeometricOptimizers
using GeometricOptimizers: check, metric, rgrad, NativePade, ScaledSquaring
using GPUArraysCore: allowscalar
using KernelAbstractions: get_backend
using LinearAlgebra: qr!
using Metal: Metal, MtlArray
using Random

include(joinpath(@__DIR__, "device_products.jl"))
include(joinpath(@__DIR__, "device_solve.jl"))

function check_row(name, f)
    status = try
        f() ? :pass : :wrong
    catch err
        first(split(sprint(showerror, err), '\n'))
    end
    name => status
end

# A point drawn on the device, the manifold operations at it against the same operations on its host
# copy, and the two exponential algorithms of `geodesic` on a lift whose norm needs squarings.
function manifold_rows(todevice; seed = 1234)
    allowscalar(false)
    backend = get_backend(todevice(zeros(Float32, 1)))
    rng = Random.Xoshiro(seed)
    rows = Pair{String, Any}[]
    for M in (StiefelManifold, GrassmannManifold)
        name = string(nameof(M))
        push!(rows,
            check_row("rand(backend, $name{Float32}, 20, 3)",
                () -> let Y = rand(backend, M{Float32}, 20, 3)
                    get_backend(parent(Y)) == backend && check(Y) < 1.0f-5
                end))
        Y = M(Matrix(qr!(randn(rng, Float32, 6, 6)).Q)[:, 1:3])
        dY = M(todevice(parent(Y)))
        ∇L, Δ₁, Δ₂ = randn(rng, Float32, 6, 3), randn(rng, Float32, 6, 3),
        randn(rng, Float32, 6, 3)
        push!(rows,
            check_row("rgrad($name, ∇L)",
                () -> let r = rgrad(dY, todevice(∇L))
                    get_backend(r) == backend && Array(r) ≈ rgrad(Y, ∇L)
                end))
        push!(rows,
            check_row("rgrad($name, host ∇L) refused",
                () -> try
                    rgrad(dY, ∇L)
                    false
                catch err
                    err isa ArgumentError &&
                        occursin("mixed backends", sprint(showerror, err))
                end))
        push!(rows,
            check_row("metric($name, Δ₁, Δ₂)",
                () -> metric(dY, todevice(Δ₁), todevice(Δ₂)) ≈ metric(Y, Δ₁, Δ₂)))
    end
    for algorithm in (ScaledSquaring(), NativePade())
        push!(rows,
            check_row("geodesic(60 * lift, $(nameof(typeof(algorithm))))",
                () -> let B = 60 * rand(rng, StiefelLieAlgHorMatrix{Float32}, 20, 3)
                    Y = geodesic(todev(todevice, B), algorithm)
                    get_backend(parent(Y)) == backend &&
                        Array(parent(Y)) ≈ parent(geodesic(B, algorithm))
                end))
    end
    rows
end

function device_check(todevice; adam_rtol = UNMATCHED_ADAM_RTOL, matched_rng = false)
    rows = manifold_rows(todevice)
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
