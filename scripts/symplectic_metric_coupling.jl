# Is the metric block-diagonal in `(𝒜, ℬ)` at `global_section`'s own section? `global_section` takes
# the completion `Uˢ` in the symplectic complement, `UᵀJUˢ = 𝕆`, which leaves `UᵀUˢ` free, and the
# metric pairs `Uˢℬ` with `U𝒜` as `tr(Pℬᵀ(Uˢ)ᵀU𝒜)`, `P = (UᵀU)⁻¹`. Over 20 random points each, the
# `𝒜` part of `M` times the first unit `ℬ` is up to 4.8, 70 and 275 times its `ℬ` part at
# `(2N, 2n) = (4, 2), (12, 4), (40, 6)`, so the metric is not block-diagonal there (K64).
#
#   julia --startup-file=no --project=test scripts/symplectic_metric_coupling.jl
using GeometricOptimizers
using LinearAlgebra: opnorm, cond
import Random
const GO = GeometricOptimizers
const T = Float64
rows = map(((4, 2), (12, 4), (40, 6))) do (N2, n2)
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
foreach(println, rows)
