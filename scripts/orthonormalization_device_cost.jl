# The device cost of the CholeskyQR2 orthonormalization, `_cholesky_qr2`, on Metal, against the two
# host alternatives: a Householder `qr!` of a host matrix, and the round trip that downloads a device
# matrix, factorizes it on the host and uploads `Q`.
#
# The shape is the one `global_section` factorizes, `N × (N - n)` with `n = 3`, at `N = 20` and
# `N = 400`, in `Float32`, because Metal has no `Float64`. Each figure is the median of five timed
# runs after one warm-up, and a device run ends in `Metal.synchronize()` inside the timed region, so
# the time is the computation's and not the launch's.
#
# Run it by hand through a cold Kaimon session, in a scratch environment that develops this tree
# (Metal is unreachable from a sandboxed shell), and quote the machine and the versions it prints:
#
#     include("<this repository>/scripts/orthonormalization_device_cost.jl")
#     orthonormalization_device_cost()

using GeometricOptimizers: _cholesky_qr2
using LinearAlgebra: qr!
using Metal: Metal, MtlArray
using Random
using Statistics: median

const RUNS = 5

function timed(f)
    f()
    median([@elapsed(f()) for _ in 1:RUNS])
end

host_qr(A) = Matrix(qr!(copy(A)).Q)

function orthonormalization_device_cost(; sizes = (20, 400), n = 3, seed = 1234)
    rng = Random.Xoshiro(seed)
    rows = map(sizes) do N
        A = randn(rng, Float32, N, N - n)
        dA = MtlArray(A)
        device = timed(() -> (_cholesky_qr2(dA); Metal.synchronize()))
        host = timed(() -> host_qr(A))
        round_trip = timed(() -> (MtlArray(host_qr(Array(dA))); Metal.synchronize()))
        (N = N, cholesky_qr2_device = device, qr_host = host, qr_round_trip = round_trip)
    end
    (machine = Sys.cpu_info()[1].model, julia = VERSION, metal = pkgversion(Metal),
        device = string(Metal.device()), rows = rows)
end
