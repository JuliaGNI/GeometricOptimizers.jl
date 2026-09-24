# How far a `Float32` geodesic leaves the manifold across random global sections, on the host and
# on a JLArray, and whether `Random.seed!` makes the section draw repeatable on a JLArray.
#
# `geodesic(Y, Δ)` draws its global section from the global generator. For a `6 × 3` Stiefel and
# Grassmann point and a step `rgrad(Y, ·) / 10`, the script retracts 20000 times on each backend and
# prints the largest `check` of the result and the number of results above the `1f-4` that
# `scripts/device_products.jl` asserts, and for the JLArray also the median and the 99.9th
# percentile. It also retracts twice with the same seed on a JLArray and says whether the two
# results are equal, which is what that sweep relies on when it seeds its retraction rows.
#
# None exceeds `1f-4`: 0 of 20000 on either backend, the largest `check` `6.2e-7` for the Stiefel
# and `2.8e-7` for the Grassmann point.
#
# Run in a cold process, in an environment that develops this checkout and adds JLArrays and
# GPUArraysCore:
#
#     julia --startup-file=no --project=<env> scripts/geodesic_section_tail.jl

using GeometricOptimizers
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra
using Random
using Statistics

allowscalar(false)
sig(x) = round(x; sigdigits = 2)

const T = Float32
rng = Xoshiro(7)
for M in (StiefelManifold, GrassmannManifold)
    Y = M(Matrix(qr!(randn(rng, T, 6, 6)).Q)[:, 1:3])
    Δ = rgrad(Y, randn(rng, T, 6, 3)) / 10
    dY, dΔ = M(JLArray(Y.A)), JLArray(Δ)
    Random.seed!(11)
    a = Array(geodesic(dY, dΔ).A)
    Random.seed!(11)
    b = Array(geodesic(dY, dΔ).A)
    device = [GeometricOptimizers.check(geodesic(dY, dΔ)) for _ in 1:20000]
    host = [GeometricOptimizers.check(geodesic(Y, Δ)) for _ in 1:20000]
    println(nameof(M), ": seed repeatable = ", a == b,
        " | device: median ", sig(median(device)), " q999 ", sig(quantile(device, 0.999)),
        " max ", sig(maximum(device)), " above 1f-4: ", count(>(1.0f-4), device),
        " | host: max ", sig(maximum(host)), " above 1f-4: ", count(>(1.0f-4), host))
end
