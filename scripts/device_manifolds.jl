# The manifold operations on a device backend: a point drawn on the device, the operations at a point
# against the same operations on its host copy, and the two exponential algorithms of `geodesic` on a
# lift whose norm needs squarings.
#
# `device_manifolds(todevice)` takes the function that moves a host array onto the device and returns
# one row per call: its name, `:pass`, `:wrong` (it ran and disagrees with the host twin, or its
# result left the device) or the first line of the error it raised. Scalar indexing has to be off
# for the rows to mean anything, and the function turns it off.
#
#     using Metal; device_manifolds(MtlArray)  # what `test/devices/metal.jl` asserts
#
# Include `device_products.jl` first: it defines `todev`.

using GeometricOptimizers
using GeometricOptimizers: check, metric, rgrad, NativePade, ScaledSquaring
using GPUArraysCore: allowscalar
using KernelAbstractions: get_backend
using LinearAlgebra: qr!
using Random

function check_row(name, f)
    status = try
        f() ? :pass : :wrong
    catch err
        first(split(sprint(showerror, err), '\n'))
    end
    name => status
end

function device_manifolds(todevice; seed = 1234)
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
