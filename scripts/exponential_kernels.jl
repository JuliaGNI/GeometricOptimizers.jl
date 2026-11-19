# Compares ways of evaluating `𝔄` inside scaling and modified squaring, which is what issues #52, #63
# and #67 ask: which kernel (Taylor or Padé), which way to apply the Padé denominator (Newton–Schulz
# or an LU solve), which threshold `θ`, and whether to balance the argument before choosing the
# number of halvings.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/exponential_kernels.jl
#
# Two input sets, both of the arguments `X = (B'')ᵀB'` that `𝔄` is handed:
#
#   * the synthetic `sweep` of `retraction_sweep.jl`, eight lifts at each of `St(20, 3)`,
#     `St(200, 10)` and `St(500, 50)`, the last for the cost only;
#   * the arguments recorded while a symplectic autoencoder is trained on the linear wave equation,
#     at `2n = 4, 10, 20, 30`, which `linear_wave_lifts.jl` wrote to `scripts/data/`.
#
# Every variant runs the same framework, `scaled_squaring!` below; it differs from the package's
# `𝔄!(ws, X, algorithm)` only where a variant asks it to, and the script checks that its balanced
# Taylor and Padé+NS variants agree with `ScaledSquaring` and `NativePade` to the bit. The reference
# is `AugmentedPade` on the sweep, and the series summed in 256-bit arithmetic on the recorded
# arguments.
#
# The timings are a `minimum` over repetitions with a single BLAS thread, and machine-dependent; the
# errors, halving counts and allocations are not.

using GeometricOptimizers
using GeometricOptimizers: 𝔄, 𝔄!, ScaledSquaring, NativePade, AugmentedPade, lift_factors,
                           opnorm₁, unit_matrix, RetractionWorkspace, _taylor_𝔄!,
                           _native_pade_polynomials!, _scaled_kernel!
using KernelAbstractions: CPU
using LinearAlgebra
using LinearAlgebra: LAPACK
using SimpleSolvers: SimpleSolvers, LinearSolver, LU, factorize!
using Printf
import Random

include(joinpath(@__DIR__, "retraction_sweep.jl"))

# ---------------------------------------------------------------------------------------------
# The kernels: what is evaluated at the scaled argument `Y`, with `‖Y‖₁ ≤ θ`, into `sc.𝔄X`.

abstract type Kernel end
struct Taylor <: Kernel end              # `ScaledSquaring`'s kernel
struct PadeNewtonSchulz <: Kernel end    # `NativePade`'s kernel
struct PadeLAPACK <: Kernel end          # `[6/6]` with `q₆ \ p₆` by LAPACK's `getrf!`/`getrs!`
struct PadeSimpleLU <: Kernel end        # `[6/6]` with `q₆ \ p₆` by `SimpleSolvers.LU`, per column

kernel!(sc, Y, ::Taylor) = _scaled_kernel!(sc, Y, ScaledSquaring())
kernel!(sc, Y, ::PadeNewtonSchulz) = _scaled_kernel!(sc, Y, NativePade())

function pade!(sc, Y)
    _native_pade_polynomials!(sc.s₆, sc.s₇, Y, sc.𝕀_small2, sc.s₂, sc.s₃, sc.s₄, sc.s₅)
end

function kernel!(sc, Y, ::PadeLAPACK)
    p, q = pade!(sc, Y)
    LAPACK.getrf!(q, sc.ipiv)
    LAPACK.getrs!('N', q, sc.ipiv, p)
    copyto!(sc.𝔄X, p)
end

function kernel!(sc, Y, ::PadeSimpleLU)
    p, q = pade!(sc, Y)
    factorize!(sc.lu, q)
    for j in axes(p, 2)
        LinearAlgebra.ldiv!(view(sc.𝔄X, :, j), sc.lu, view(p, :, j))
    end
    sc.𝔄X
end

name(::Taylor) = "Taylor"
name(::PadeNewtonSchulz) = "Padé+NS"
name(::PadeLAPACK) = "Padé+LAPACK"
name(::PadeSimpleLU) = "Padé+SimpleLU"

function scratch(::Type{T}, m::Integer) where {T}
    fresh() = zeros(T, m, m)
    (𝔄X = fresh(), 𝕀_small2 = unit_matrix(CPU(), T, m),
        X′ = fresh(), colsum = zeros(T, 1, m),
        s₁ = fresh(), s₂ = fresh(), s₃ = fresh(), s₄ = fresh(), s₅ = fresh(), s₆ = fresh(),
        s₇ = fresh(), s₈ = fresh(), ipiv = zeros(LinearAlgebra.BlasInt, m),
        lu = LinearSolver(LU(; static = false), zeros(T, m)))
end

# ---------------------------------------------------------------------------------------------
# The framework.
#
# `balance = true` replaces `X` by the similar matrix `X′ = D⁻¹XD`, `D = diag(𝕀, t𝕀)`, before the
# halvings are counted, with `t` the power of two that comes closest to giving the two off-diagonal
# `n × n` blocks of `X′` the same one-norm, and returns `D𝔄(X′)D⁻¹`, which is `𝔄(X)` because `𝔄` is a
# power series. The reduced argument is `[A/2 𝕀; C A/2]` with `‖C‖ ≈ ‖B̄‖²`, so `‖X‖₁` is quadratic in
# `‖B̄‖` while `‖X′‖₁` is linear in it, and an argument always has `‖X‖₁ ≥ 1` from the identity block
# however small the lift. `balance = true` is what the package's `𝔄!` does, and `balance = false` is
# the baseline without it.

function block_norms!(colsum, X)
    m = size(X, 1)
    n = m ÷ 2
    @views a = maximum(sum!(abs, colsum[:, (n + 1):m], X[1:n, (n + 1):m]))
    @views b = maximum(sum!(abs, colsum[:, 1:n], X[(n + 1):m, 1:n]))
    a, b
end

function scaled_squaring!(
        sc, X::AbstractMatrix{T}, θ, kernel::Kernel; balance::Bool) where {T}
    m = size(X, 1)
    n = m ÷ 2
    X′ = sc.X′
    copyto!(X′, X)
    t = one(T)
    if balance && iseven(m)
        a, b = block_norms!(sc.colsum, X)
        t = a > 0 && b > 0 ? exp2(round((log2(b) - log2(a)) / 2)) : one(T)
        t = isfinite(t) ? t : one(T)
        @views X′[1:n, (n + 1):m] .*= t
        @views X′[(n + 1):m, 1:n] ./= t
    end
    nrm = maximum(sum!(abs, sc.colsum, X′))
    s = nrm > θ ? ceil(Int, log2(nrm / θ)) : 0
    scale = T(2)^s

    sc.s₁ .= X′ ./ scale
    W = kernel!(sc, sc.s₁, kernel)
    W ./= scale
    for _ in 1:s
        mul!(sc.s₂, W, X′)
        mul!(sc.s₃, sc.s₂, W)
        W .= 2 .* W .+ sc.s₃
    end
    if t != one(T)
        @views W[1:n, (n + 1):m] ./= t
        @views W[(n + 1):m, 1:n] .*= t
    end
    W, s
end

# ---------------------------------------------------------------------------------------------
# The inputs.

reduced(B) = ((B̂, B̄) = lift_factors(B); B̄' * B̂)

function recorded(n::Integer)
    path = joinpath(@__DIR__, "data", @sprintf("linear_wave_lifts_n%02d.csv", n))
    map(eachline(path)) do line
        fields = split(line, ',')
        m = parse(Int, fields[2])
        reshape(parse.(Float64, fields[3:end]), m, m)
    end
end

const KERNELS = (Taylor(), PadeNewtonSchulz(), PadeLAPACK(), PadeSimpleLU())
const RECORDED_N = (2, 5, 10, 15)

function best(f, repetitions::Integer = 50)
    f()
    minimum(1:repetitions) do _
        t₀ = time_ns()
        f()
        (time_ns() - t₀) / 1e3
    end
end

relerr(W, reference) = norm(W - reference) / norm(reference)

# ---------------------------------------------------------------------------------------------

function agreement_with_package()
    println("== the balanced variants against 𝔄!(ws, X, algorithm) ==")
    for B in sweep(Float64, 200, 10),
        (k, algorithm) in ((Taylor(), ScaledSquaring()),
            (PadeNewtonSchulz(), NativePade()))

        X = reduced(B)
        ws = RetractionWorkspace(CPU(), Float64, 200, 10)
        W, _ = scaled_squaring!(scratch(Float64, 20), X, 0.5, k; balance = true)
        W == 𝔄!(ws, X, algorithm) || error("the framework here differs from the package")
    end
    println("bitwise equal on all eight lifts, for both kernels")
end

function recorded_summary()
    println("\n== the recorded linear-wave arguments ==")
    @printf("%4s %7s %20s %20s %20s %8s %8s\n", "2n", "sampled", "‖X‖₁", "‖C‖₁",
        "‖X′‖₁, balanced", "s", "s, bal.")
    for n in RECORDED_N
        Xs = recorded(n)
        sc = scratch(Float64, 2n)
        span(f) = (r = extrema(f, Xs); @sprintf("%.2e – %.2e", r...))
        balanced_norm(X) = (scaled_squaring!(sc, X, 0.5, Taylor(); balance = true);
            opnorm₁(sc.X′))
        s = extrema(X -> scaled_squaring!(sc, X, 0.5, Taylor(); balance = false)[2], Xs)
        sb = extrema(X -> scaled_squaring!(sc, X, 0.5, Taylor(); balance = true)[2], Xs)
        @printf("%4d %7d %20s %20s %20s %8s %8s\n", 2n, length(Xs), span(opnorm₁),
            span(X -> opnorm₁(X[(n + 1):end, 1:n])), span(balanced_norm),
            "$(s[1])–$(s[2])", "$(sb[1])–$(sb[2])")
    end
end

function accuracy_tables(::Type{T}) where {T}
    for (N, n) in ((20, 3), (200, 10))
        println("\n== $T, relative error against AugmentedPade (Float64), sweep at St($N, $n), θ = 0.5 ==")
        @printf("%9s %4s %4s", "‖B̄‖", "s", "bal")
        foreach(k -> @printf("%15s%15s", name(k), name(k) * ",b"), KERNELS)
        println()
        sc = scratch(T, 2n)
        for B in sweep(Float64, N, n)
            X = reduced(B)
            reference = 𝔄(X, AugmentedPade())
            XT = T.(X)
            s = scaled_squaring!(sc, XT, 0.5, Taylor(); balance = false)[2]
            sb = scaled_squaring!(sc, XT, 0.5, Taylor(); balance = true)[2]
            @printf("%9.2f %4d %4d", norm(Matrix(B)), s, sb)
            for k in KERNELS, balance in (false, true)

                W, _ = scaled_squaring!(sc, XT, 0.5, k; balance = balance)
                @printf("%15.2e", relerr(Float64.(W), reference))
            end
            println()
        end
    end
end

# The orthogonality of the retracted point, `‖YᵀY - 𝕀‖` for `Y = (𝕀 + B̂WB̄ᵀ)E`, which is what `check`
# measures for a Stiefel point.
function orthogonality_table()
    println("\n== ‖YᵀY - 𝕀‖ of the retracted point, sweep at St(200, 10), θ = 0.5 ==")
    @printf("%9s", "‖B̄‖")
    foreach(k -> @printf("%15s%15s", name(k), name(k) * ",b"), KERNELS)
    println()
    sc = scratch(Float64, 20)
    for B in sweep(Float64, 200, 10)
        B̂, B̄ = lift_factors(B)
        X = B̄' * B̂
        E = Matrix{Float64}(I, 200, 10)
        @printf("%9.2f", norm(Matrix(B)))
        for k in KERNELS, balance in (false, true)

            W, _ = scaled_squaring!(sc, X, 0.5, k; balance = balance)
            Y = E + B̂ * (W * (B̄' * E))
            @printf("%15.2e", norm(Y' * Y - I))
        end
        println()
    end
end

# The series of `𝔄` summed in 256-bit arithmetic until a term is below `10⁻⁶⁰`. On the recorded
# arguments, whose norm is one and whose powers decay from the second on, its own error is far below
# `Float64` round-off, which `AugmentedPade`'s is not: there every variant lands on the same
# `5.7e-14`, which is the reference's error and not theirs.
function series_reference(X::AbstractMatrix)
    setprecision(BigFloat, 256) do
        A = big.(X)
        term = Matrix{BigFloat}(I, size(A)...)
        sum = copy(term)
        k = 2
        while norm(term) > big"1e-60"
            term = term * A / k
            sum += term
            k += 1
        end
        Float64.(sum)
    end
end

function recorded_accuracy()
    println("\n== recorded arguments: worst relative error against the 256-bit series, θ = 0.5 ==")
    @printf("%4s", "2n")
    foreach(k -> @printf("%15s%15s", name(k), name(k) * ",b"), KERNELS)
    @printf("%15s\n", "AugmentedPade")
    for n in RECORDED_N
        Xs = recorded(n)
        references = map(series_reference, Xs)
        sc = scratch(Float64, 2n)
        @printf("%4d", 2n)
        for k in KERNELS, balance in (false, true)

            worst = maximum(zip(Xs, references)) do (X, reference)
                relerr(copy(scaled_squaring!(sc, X, 0.5, k; balance = balance)[1]), reference)
            end
            @printf("%15.2e", worst)
        end
        @printf("%15.2e\n",
            maximum(((X, r),) -> relerr(𝔄(X, AugmentedPade()), r),
            zip(Xs, references)))
    end
end

function cost_tables()
    BLAS.set_num_threads(1)
    println("\n== cost of one 𝔄, μs, minimum of 50, 1 BLAS thread, θ = 0.5 ==")
    @printf("%-22s", "input")
    foreach(k -> @printf("%15s%15s", name(k), name(k) * ",b"), KERNELS)
    println()
    inputs = Any[]
    for (N, n) in ((20, 3), (200, 10), (500, 50))
        Bs = sweep(Float64, N, n)
        push!(inputs, (
            "St($N,$n) ‖B̄‖≈$(round(Int, norm(Matrix(Bs[2]))))", [reduced(Bs[2])]))
        push!(inputs, (
            "St($N,$n) ‖B̄‖≈$(round(Int, norm(Matrix(Bs[6]))))", [reduced(Bs[6])]))
    end
    for n in RECORDED_N
        push!(inputs, ("linear wave, 2n = $(2n)", recorded(n)))
    end
    for (label, Xs) in inputs
        sc = scratch(Float64, size(first(Xs), 1))
        @printf("%-22s", label)
        for k in KERNELS, balance in (false, true)
            # the mean over the recorded arguments of the best time of each
            t = sum(X -> best(() -> scaled_squaring!(sc, X, 0.5, k; balance = balance)), Xs) /
                length(Xs)
            @printf("%15.2f", t)
        end
        println()
    end

    println("\n== allocations of one 𝔄 in a preallocated scratch, bytes ==")
    sc = scratch(Float64, 20)
    X = reduced(sweep(Float64, 200, 10)[6])
    for k in KERNELS, balance in (false, true)

        f = () -> scaled_squaring!(sc, X, 0.5, k; balance = balance)
        f()
        @printf("%-16s balance = %-5s %8d\n", name(k), balance, @allocated f())
    end
end

# How large a `θ` each kernel tolerates, and what it buys: the number of halvings, the error and the
# time, with balancing. Newton–Schulz is left out above `1/2`, where `NativePade` refuses it.
function theta_table()
    println("\n== θ, balanced, St(200, 10) at ‖B̄‖ ≈ 1066: halvings, error, μs ==")
    B = sweep(Float64, 200, 10)[6]
    X = reduced(B)
    reference = 𝔄(X, AugmentedPade())
    sc = scratch(Float64, 20)
    BLAS.set_num_threads(1)
    for k in (Taylor(), PadeLAPACK(), PadeSimpleLU()), θ in (0.25, 0.5, 1.0, 2.0, 4.0)

        W, s = scaled_squaring!(sc, X, θ, k; balance = true)
        e = relerr(copy(W), reference)
        t = best(() -> scaled_squaring!(sc, X, θ, k; balance = true))
        @printf("%-15s θ = %4.2f   s = %2d   error %9.2e   %8.2f μs\n", name(k), θ, s, e, t)
    end
end

# Issue #67: the denominator solve itself, on the scaled arguments the kernel actually sees.
function denominator_table()
    println("\n== the Padé denominator at the scaled argument Y, balanced, θ = 0.5 ==")
    @printf("%9s %10s %10s %13s %13s %13s\n", "‖B̄‖", "‖Y‖₁", "cond₁(q₆)", "NS residual",
        "LAPACK resid.", "SimpleLU res.")
    for B in sweep(Float64, 200, 10)
        X = reduced(B)
        sc = scratch(Float64, 20)
        scaled_squaring!(sc, X, 0.5, Taylor(); balance = true)
        Y = copy(sc.s₁)
        p, q = map(copy, pade!(sc, Y))
        residual(W) = opnorm(q * W - p, 1) / opnorm(p, 1)
        r = map((PadeNewtonSchulz(), PadeLAPACK(), PadeSimpleLU())) do k
            residual(copy(kernel!(sc, Y, k)))
        end
        @printf("%9.2f %10.3f %10.3f %13.2e %13.2e %13.2e\n",
            norm(Matrix(B)), opnorm(Y, 1),
            cond(q, 1), r...)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    agreement_with_package()
    recorded_summary()
    accuracy_tables(Float64)
    accuracy_tables(Float32)
    orthogonality_table()
    recorded_accuracy()
    denominator_table()
    theta_table()
    cost_tables()
end
