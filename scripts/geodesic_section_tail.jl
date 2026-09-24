# How far a `Float32` geodesic leaves the manifold across random global sections, on the host and
# on a JLArray, and whether `Random.seed!` makes the section draw repeatable on a JLArray.
#
# `geodesic(Y, Δ)` draws its global section from the global generator. For a `6 × 3` Stiefel and
# Grassmann point and a step `rgrad(Y, ·) / 10`, the script retracts 20000 times on each backend and
# prints the median, the 99.9th percentile and the largest `check` of the result, and the number of
# results above the `1f-4` that `scripts/device_products.jl` asserts. It also retracts twice with
# the same seed on a JLArray and says whether the two results are equal, which is what that sweep
# relies on when it seeds its retraction rows.
#
# With `global_section` projecting and orthonormalising once, one or two retractions in a thousand
# exceeded `1f-4`, on the host and on a JLArray alike, and the largest `check` reached `8.5e-3`;
# the CHANGELOG entry that fixes this quotes those figures. With the second pass none does: 0 of
# 20000 on either backend, the largest `check` `6.2e-7` for the Stiefel and `2.8e-7` for the
# Grassmann point.
#
# Run in a cold process:
#
#     julia --startup-file=no --project=<env with GeometricOptimizers, JLArrays and GPUArraysCore> \
#         scripts/geodesic_section_tail.jl

using GeometricOptimizers
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra
using Random
using Statistics

allowscalar(false)

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
        " | device: median ", median(device), " q999 ", quantile(device, 0.999),
        " max ", maximum(device), " above 1e-4: ", count(>(1.0f-4), device),
        " | host: max ", maximum(host), " above 1e-4: ", count(>(1.0f-4), host))
end
