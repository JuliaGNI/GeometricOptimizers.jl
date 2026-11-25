# Is the metric block-diagonal in `(𝒜, ℬ)` at `global_section`'s own section? `global_section` takes
# the completion `Uˢ` in the symplectic complement, `UᵀJUˢ = 𝕆`, which leaves `UᵀUˢ` free, and the
# metric pairs `Uˢℬ` with `U𝒜` as `tr(Pℬᵀ(Uˢ)ᵀU𝒜)`, `P = (UᵀU)⁻¹`.
#
# 1. Over 20 random points each, the `𝒜` part of `M` times the first unit `ℬ` is up to 4.8, 70 and 275
#    times its `ℬ` part at `(2N, 2n) = (4, 2), (12, 4), (40, 6)`, so the metric is not block-diagonal
#    there.
# 2. At 20 orthosymplectic points `U = [Re W  -Im W; Im W  Re W]`, `W` with orthonormal complex
#    columns, each, `‖UᵀUˢ‖` is below 1e-15 and the metric is `½tr(𝒜ᵀ𝒜) + tr(ℬᵀℬ)` on the lift to a
#    relative 1.1e-15, for `(2N, 2n) = (4, 2), (12, 4), (40, 6)`.
#
#   julia --startup-file=no --project=test scripts/symplectic_metric_coupling.jl
using GeometricOptimizers
using LinearAlgebra: I, cond, norm, opnorm, qr
import Random
const GO = GeometricOptimizers
const T = Float64

random_points = map(((4, 2), (12, 4), (40, 6))) do (N2, n2)
    worst = zero(T)
    κ = zero(T)
    coupling = zero(T)
    for seed in 1:20
        Random.seed!(seed)
        U = rand(SymplecticStiefelManifold{T}, N2, n2)
        λ = GlobalSection(U)
        worst = max(worst, opnorm(U.A' * λ.λ) / (opnorm(U.A) * opnorm(λ.λ)))
        κ = max(κ, cond(U.A))
        E = GO.zero_tangent(U)
        E.B[1] = 1
        D = GO.zero_tangent(U)
        GO._metric_apply!(D, λ, E)
        c = max(maximum(abs, D.A₁), maximum(abs, D.B₁.S), maximum(abs, D.C₁.S)) /
            maximum(abs, D.B)
        coupling = max(coupling, c)
    end
    ((N2, n2), "max ‖UᵀUˢ‖/(‖U‖‖Uˢ‖)", worst, "max cond(U)", κ,
        "max 𝒜-part over ℬ-part of M·e₁", coupling)
end
foreach(println, random_points)

poisson(N) = [zeros(T, N, N) I; -I zeros(T, N, N)]

orthosymplectic_points = map(((2, 1), (6, 2), (20, 3))) do (N, n)
    residual, coupling, form = zero(T), zero(T), zero(T)
    for seed in 1:20
        rng = Random.Xoshiro(seed)
        W = Matrix(qr(randn(rng, Complex{T}, N, n)).Q)[:, 1:n]
        U = [real(W) -imag(W); imag(W) real(W)]
        symplectic = opnorm(U' * poisson(N) * U - poisson(n))
        orthonormal = opnorm(U' * U - I)
        residual = max(residual, symplectic, orthonormal)
        Y = SymplecticStiefelManifold(U)
        λ = GlobalSection(Y)
        coupling = max(coupling, opnorm(U' * λ.λ))
        for _ in 1:5
            Δ = rgrad(Y, randn(rng, T, 2N, 2n))
            D = Matrix(global_rep(λ, Δ))
            𝒜, ℬ = D[1:(2n), 1:(2n)], D[(2n + 1):end, 1:(2n)]
            g = metric(Y, Δ, Δ)
            form = max(form, abs(g - (norm(𝒜)^2 / 2 + norm(ℬ)^2)) / g)
        end
    end
    ((2N, 2n), "orthosymplectic residual", residual, "max ‖UᵀUˢ‖", coupling,
        "max relative error of ½‖𝒜‖² + ‖ℬ‖²", form)
end
foreach(println, orthosymplectic_points)
