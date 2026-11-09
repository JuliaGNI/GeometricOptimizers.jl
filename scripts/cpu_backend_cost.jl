# The hot paths on the `CPU()` backend: the kernel-backed products and projections of the
# vector-storage matrices, the allocators that write through a kernel, and one training step.
#
#     julia --startup-file=no --check-bounds=auto --project=<env> scripts/cpu_backend_cost.jl
#
# `<env>` holds this package and `NeuralNetworkParameters`. Each row is the minimum `@allocated`
# over 5 calls and the median time over 200 samples of at least 20 µs each, after two warm-up
# calls. BLAS runs on one thread. On KernelAbstractions 0.10 `CPU()` is the PoCL backend, so a
# kernel launch costs a few µs; the rows show where that lands.
using GeometricOptimizers
using GeometricOptimizers: map_to_Skew
using KernelAbstractions: KernelAbstractions, CPU
using LinearAlgebra: BLAS, mul!
using Random: Xoshiro

BLAS.set_num_threads(1)
println("KernelAbstractions ", pkgversion(KernelAbstractions), ", Julia ", VERSION)

median_of(v) = sort(v)[cld(length(v), 2)]

function repeated(f, c)
    for _ in 1:c
        f()
    end
end

function measure(f)
    f()
    f()
    bytes = minimum(@allocated(f()) for _ in 1:5)
    c = max(1, ceil(Int, 2.0e-5 / max(@elapsed(f()), 1.0e-8)))
    samples = [@elapsed(repeated(f, c)) / c for _ in 1:200]
    bytes, median_of(samples)
end

function row(name, f)
    b, t = measure(f)
    println(rpad(name, 44), lpad(b, 9), " B ", lpad(round(t * 1e6; digits = 2), 10), " µs")
end

function parameters(::Type{T}) where {T}
    rng = Xoshiro(1)
    NetworkParameters((L1 = (weight = rand(rng, StiefelManifold{T}, 64, 8),),
        L2 = (W = randn(rng, T, 64, 64), b = zeros(T, 64)),
        L3 = (W = randn(rng, T, 64, 64), b = zeros(T, 64))))
end

function gradient(x::NetworkParameters{T}) where {T}
    rng = Xoshiro(2)
    NetworkParameters((L1 = (weight = rand(rng, T, 64, 8) ./ 100,),
        L2 = (W = rand(rng, T, 64, 64) ./ 100, b = rand(rng, T, 64) ./ 100),
        L3 = (W = rand(rng, T, 64, 64) ./ 100, b = rand(rng, T, 64) ./ 100)))
end

for T in (Float64, Float32)
    for n in (8, 64)
        A = randn(Xoshiro(3), T, n, n)
        B = randn(Xoshiro(4), T, n, 8)
        S = SkewSymMatrix(A)
        Y = SymmetricMatrix(A)
        L = StrictlyLowerTriangular(A)
        C = zeros(T, n, 8)
        row("map_to_Skew $n×$n $T", () -> map_to_Skew(A))
        row("SkewSym($n) * B($n×8) $T", () -> S * B)
        row("mul!(C, SkewSym($n), B) $T", () -> mul!(C, S, B))
        row("Symmetric($n) * B $T", () -> Y * B)
        row("StrictlyLower($n) * B $T", () -> L * B)
        row("rand Stiefel $n×4 $T", () -> rand(Xoshiro(1), StiefelManifold{T}, n, 4))
        row("StiefelProjection($n, 4) $T", () -> StiefelProjection(CPU(), T, n, 4))
    end
    for alg in (GradientMethod(), MomentumMethod(), Adam())
        x = parameters(T)
        dp = gradient(x)
        opt = TrainingOptimizer(x; algorithm = alg)
        row("step $(nameof(typeof(alg))) Stiefel+2 Dense $T", () -> optimization_step!(x, opt, dp))
    end
end
