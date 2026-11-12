# The choice of the optimizer's inner product on the symplectic Stiefel manifold, checked in Julia.
# Parts (i) to (v) are standalone: no package code, so they check the mathematics and not an
# implementation. Part (vi) checks the package's `global_rep` and `gradient_rep!` against them.
#
# Run: julia --startup-file=no --project=. scripts/symplectic_gradient_options.jl
#
# Results at 10 × 4, Julia 1.13, 2026-11-10:
#   (i)   metric (3.16) in the coordinates (𝒜, ℬ) of the canonical section: the same form at the two
#         orthosymplectic points (6e-16), 150–210 % different at the others; no 𝒜–ℬ coupling (≤ 1e-16)
#   (ii)  the canonical section is unique up to an orthosymplectic change: ‖SᵀS − I‖ = 1.5e-15
#   (iii) the retraction λ exp(tB) E does not depend on the section: 3e-15 at cond(S) = 85, while ‖ℬ‖
#         changes from 5.9 to 16.4
#   (iv)  at an orthosymplectic point the metric is ½tr(𝒜ᵀ𝒜) + tr(ℬᵀℬ) to 6e-16, and neither of the
#         two other weightings (0.48, 0.44)
#   (v)   the lift is in sp for 𝕁̄ (4e-15), its blocks are -ℬ⁺ and 𝕆 exactly, λXE = Δ (5e-16), and
#         Δ ↦ X has rank 34 = 4Nn - n(2n - 1)
#   (vi)  the package's global_rep is the lift (3e-17), its flat gradient_rep! the derivative with
#         respect to the stored entries (2e-16), MetricDescent the Riesz representative in metric
#         (3.16) (9e-17), and _metric_apply! the Gram matrix of metric (3.16) in the stored entries
#         (2e-16), whose inverse BFGS and DFP start from; the control, the Riesz representative in ½tr(XᵀY)
#         of the dense lifts, is off by 0.25 in metric (3.16)
using LinearAlgebra, Random
using LinearAlgebra: rank

poisson(n) = [zeros(n ÷ 2, n ÷ 2) I; -I zeros(n ÷ 2, n ÷ 2)]
splus(X) = poisson(size(X, 2))' * transpose(X) * poisson(size(X, 1))   # X⁺ = 𝕁ᵀXᵀ𝕁

# a point with κ ≤ 4, as `conditioned_point` of the tests; `spread` widens the diagonal factor
function point(rng, N2, n2; spread = 1.0)
    N, n = N2 ÷ 2, n2 ÷ 2
    Q = Matrix(qr(randn(rng, ComplexF64, N, n)).Q)
    A, B = real(Q), imag(Q)
    d = exp2.(spread .* (2 .* rand(rng, n) .- 1))
    hcat(vcat(A, B) .* d', vcat(-B, A) ./ d')
end

# the completion of `global_section`, with an optional rotation `O` of the QR basis
function completion(U; O = I)
    N2, n2 = size(U)
    m = (N2 - n2) ÷ 2
    J = poisson(N2)
    Q = (qr(J' * U).Q * Matrix(I, N2, N2))[:, (n2 + 1):N2] * O
    S = Q' * J * Q
    S = (S - S') / 2
    σ, V = eigen(Hermitian(im .* S))
    sc = transpose(sqrt.(2 ./ σ[(m + 1):(2m)]))
    Q * hcat(imag.(V[:, (m + 1):(2m)]) .* sc, real.(V[:, (m + 1):(2m)]) .* sc)
end

# metric (3.16) of Bendokat and Zimmermann, as the package implements it
function gmetric(U, Δ₁, Δ₂)
    P = inv(U' * U)
    X = poisson(size(U, 1))' * U
    tr(P * (Δ₁' * Δ₂ - (Δ₁' * X) * P * (X' * Δ₂) / 2))
end

# a basis of the coordinates (𝒜, ℬ): 𝒜 = 𝕁ₙ Sym, ℬ free
function coordinate_basis(N2, n2)
    basis = Tuple{Matrix{Float64}, Matrix{Float64}}[]
    for i in 1:n2, j in i:n2

        S = zeros(n2, n2)
        S[i, j] = S[j, i] = 1
        push!(basis, (poisson(n2) * S, zeros(N2 - n2, n2)))
    end
    for k in 1:((N2 - n2) * n2)
        B = zeros(N2 - n2, n2)
        B[k] = 1
        push!(basis, (zeros(n2, n2), B))
    end
    basis
end

function gram(U, Us, basis)
    [gmetric(U, U * a + Us * b, U * c + Us * d) for (a, b) in basis, (c, d) in basis]
end

rng = Xoshiro(7)
N2, n2 = 10, 4
basis = coordinate_basis(N2, n2)
nA = n2 * (n2 + 1) ÷ 2
println("dimension of the coordinates: ", length(basis))

# (i) Is metric (3.16), written in the coordinates of the canonical section, the same quadratic form
#     at every point? If it were, a fixed `_dot` would reproduce it.
Us = [point(rng, N2, n2; spread = s) for s in (0.0, 0.0, 1.0, 1.0, 2.0)]
Gs = [gram(U, completion(U), basis) for U in Us]
for (k, Gk) in enumerate(Gs)
    println("(i) point $k  cond(U) = ", round(cond(Us[k]); sigdigits = 3),
        "  ‖G - G₁‖/‖G₁‖ = ", round(norm(Gk - Gs[1]) / norm(Gs[1]); sigdigits = 3),
        "  𝒜ℬ coupling ‖G₁₂‖/‖G‖ = ",
        round(norm(Gk[1:nA, (nA + 1):end]) / norm(Gk); sigdigits = 3))
end

# (ii) The canonical completion is defined up to an orthosymplectic change: a different QR basis
#      gives a completion related by an S with SᵀS = I, so ‖ℬ‖_F is the same.
let U = Us[3]
    Us₁ = completion(U)
    O = Matrix(qr(randn(rng, N2 - n2, N2 - n2)).Q)
    Us₂ = completion(U; O = O)
    S = poisson(N2 - n2)' * Us₁' * poisson(N2) * Us₂
    println("(ii) relating S: ‖SᵀS - I‖ = ", norm(S' * S - I), "  ‖SᵀJS - J‖ = ",
        norm(S' * poisson(N2 - n2) * S - poisson(N2 - n2)))
end

# (iii) The retraction curve λ exp(tB) E of an h-horizontal lift does not depend on the section:
#       λ → λh with h = diag(I, S), S ∈ Sp, sends B → h⁻¹Bh, and λh exp(t h⁻¹Bh) E = λ exp(tB) E.
let U = Us[3]
    Ucomp = completion(U)
    λ = hcat(U, Ucomp)
    Δ = U * (poisson(n2) * Symmetric(randn(rng, n2, n2))) + Ucomp * randn(rng, N2 - n2, n2)
    Jbar = cat(poisson(n2), poisson(N2 - n2); dims = (1, 2))
    function lift(λ)
        C = Jbar' * λ' * poisson(N2) * Δ
        [C[1:n2, :] -splus(C[(n2 + 1):end, :]); C[(n2 + 1):end, :] zeros(N2 - n2, N2 - n2)]
    end
    S = exp(poisson(N2 - n2) * Matrix(Symmetric(randn(rng, N2 - n2, N2 - n2))))
    λ₂ = hcat(U, Ucomp * S)
    E = Matrix(I, N2, n2)
    c₁ = λ * exp(0.7 * lift(λ)) * E
    c₂ = λ₂ * exp(0.7 * lift(λ₂)) * E
    println("(iii) cond(S) = ", round(cond(S); sigdigits = 3),
        "  ‖curve₁ - curve₂‖/‖curve₁‖ = ", norm(c₁ - c₂) / norm(c₁))
    # the coordinates of the lift do depend on the section: ‖ℬ‖ changes with S
    println("(iii) ‖ℬ‖ at the two sections: ", norm(lift(λ)[(n2 + 1):end, 1:n2]), "  ",
        norm(lift(λ₂)[(n2 + 1):end, 1:n2]))
end

# (iv) At an orthosymplectic point, which fixed form is metric (3.16) in canonical coordinates?
let U = Us[1], Ucomp = completion(Us[1])
    G₁ = gram(U, Ucomp, basis)
    for (name, F) in (("½tr𝒜ᵀ𝒜 + trℬᵀℬ", (a, b, c, d) -> tr(a' * c) / 2 + tr(b' * d)),
        ("tr𝒜ᵀ𝒜 + trℬᵀℬ", (a, b, c, d) -> tr(a' * c) + tr(b' * d)),
        ("½tr𝒜ᵀ𝒜 + ½trℬᵀℬ", (a, b, c, d) -> (tr(a' * c) + tr(b' * d)) / 2))
        Fm = [F(a, b, c, d) for (a, b) in basis, (c, d) in basis]
        println("(iv) ", name, ": ‖G₁ - F‖/‖G₁‖ = ", norm(G₁ - Fm) / norm(G₁))
    end
end

# The lift `X = [𝒜 -ℬ⁺; ℬ 0]` of a tangent vector `Δ` at the section `λ = [U Uˢ]`, with
# `[𝒜; ℬ] = 𝕁̄ᵀλᵀ𝕁Δ` in the adapted basis `𝕁̄ = diag(𝕁₂ₙ, 𝕁₂N₋₂ₙ)`
Jbar = cat(poisson(n2), poisson(N2 - n2); dims = (1, 2))
function lift(λ, Δ)
    C = Jbar' * λ' * poisson(N2) * Δ
    [C[1:n2, :] -splus(C[(n2 + 1):end, :]); C[(n2 + 1):end, :] zeros(N2 - n2, N2 - n2)]
end
function tangent_vector(rng, U, Ucomp)
    U * (poisson(n2) * Symmetric(randn(rng, n2, n2))) +
    Ucomp * randn(rng, N2 - n2, n2)
end

# (v) The block relations of `eq:lg-spst-ghor` of the book, below (3.4) of Bendokat and Zimmermann:
#     the lift is in `sp` for `𝕁̄`, its upper right block is `-ℬ⁺`, its lower right block is zero, so
#     it is `h`-orthogonal to the vertical space `{diag(𝕆, 𝕁H)}` under `tr(XY)`, and `λXE = Δ`. The
#     space has dimension `4Nn - n(2n - 1)`, which is the rank of `Δ ↦ X` on the tangent space.
let U = Us[3], Ucomp = completion(Us[3])
    λ = hcat(U, Ucomp)
    E = Matrix(I, N2, n2)
    Δ = tangent_vector(rng, U, Ucomp)
    X = lift(λ, Δ)
    V = cat(zeros(n2, n2), poisson(N2 - n2) * Symmetric(randn(rng, N2 - n2, N2 - n2)); dims = (
        1, 2))
    println("(v) ‖Xᵀ𝕁̄ + 𝕁̄X‖ = ", norm(X' * Jbar + Jbar * X),
        "  ‖X₁₂ + ℬ⁺‖ = ", norm(X[1:n2, (n2 + 1):end] + splus(X[(n2 + 1):end, 1:n2])),
        "  ‖X₂₂‖ = ", norm(X[(n2 + 1):end, (n2 + 1):end]),
        "  |tr(XV)| = ", abs(tr(X * V)),
        "  ‖λXE - Δ‖/‖Δ‖ = ", norm(λ * X * E - Δ) / norm(Δ))
    N, n = N2 ÷ 2, n2 ÷ 2
    images = [vec(lift(λ, U * a + Ucomp * b)) for (a, b) in basis]
    println("(v) rank of Δ ↦ X on the tangent space: ", rank(reduce(hcat, images)),
        "  4Nn - n(2n - 1) = ", 4N * n - n * (2n - 1))
end

# (vi) The package against the parts above, at a conditioned point and the package's own section:
#      `global_rep` is the lift above; the flat gradient `gradient_rep!` is the derivative of
#      `B ↦ ⟨∇L, λBE⟩` with respect to each stored entry of `B`; `MetricDescent` is `global_rep` of
#      `rgrad`, whose tangent vector is the Riesz representative of `∇L` in metric (3.16);
#      `_metric_apply!` is the Gram matrix `g(λBᵢE, λBⱼE)` of metric (3.16) in the stored entries,
#      whose inverse `BFGS` and `DFP` start from. The control is the Riesz representative of the
#      derivative in `½tr(XᵀY)` of the dense lifts, the solution of that inner product's Gram system,
#      which is not the Riemannian gradient.
using GeometricOptimizers
using GeometricOptimizers: gradient_rep!, StiefelProjection, _metric_apply!

storage(D) = vcat(vec(D.A₁), parent(D.B₁), parent(D.C₁), vec(D.B))

let U = SymplecticStiefelManifold(point(rng, N2, n2; spread = 1.0))
    λU = GlobalSection(U)
    λ = Matrix(λU)
    E = Matrix(I, N2, n2)
    ∇L = randn(rng, N2, n2)
    Δ = rgrad(U, ∇L)
    println("(vi) ‖λᵀ𝕁λ - 𝕁̄‖ = ", norm(λ' * poisson(N2) * λ - Jbar),
        "  ‖global_rep - lift‖/‖lift‖ = ",
        norm(Matrix(global_rep(λU, Δ)) - lift(λ, Δ)) / norm(lift(λ, Δ)))

    k = n2 ÷ 2
    stored = k^2 + k * (k + 1) + (N2 - n2) * n2
    Bs = [Matrix(SymplecticLieAlgHorMatrix(Float64.(1:stored .== i), N2, n2))
          for i in 1:stored]
    flat_ref = [sum(∇L .* (λ * B * E)) for B in Bs]
    flat = storage(gradient_rep!(global_rep(λU, Δ), λU, Δ))
    println("(vi) flat gradient: ‖package - reference‖/‖reference‖ = ",
        norm(flat - flat_ref) / norm(flat_ref))

    G = [tr(B₁' * B₂) / 2 for B₁ in Bs, B₂ in Bs]
    section_ref = G \ flat_ref

    lifts = [SymplecticLieAlgHorMatrix(Float64.(1:stored .== i), N2, n2) for i in 1:stored]
    M_ref = [metric(U, λ * B₁ * E, λ * B₂ * E) for B₁ in Bs, B₂ in Bs]
    M = reduce(hcat, [storage(_metric_apply!(global_rep(λU, Δ), λU, B)) for B in lifts])
    println("(vi) _metric_apply!: ‖package - Gram matrix of (3.16)‖/‖Gram matrix‖ = ",
        norm(M - M_ref) / norm(M_ref))

    Dm = gradient_rep!(global_rep(λU, Δ), λU, Δ, GradientMethod())
    W = tangent_vector(rng, U.A, λ[:, (n2 + 1):end])
    println("(vi) MetricDescent: |g(λDE, W) - ⟨∇L, W⟩|/(‖∇L‖‖W‖) = ",
        abs(metric(U, λ * (Dm * StiefelProjection(Float64, N2, n2)), W) - sum(∇L .* W)) /
        (norm(∇L) * norm(W)))
    Ds = SymplecticLieAlgHorMatrix(section_ref, N2, n2)
    println("(vi) control, Riesz representative in ½tr(XᵀY): |g(λDE, W) - ⟨∇L, W⟩|/(‖∇L‖‖W‖) = ",
        abs(metric(U, λ * (Ds * StiefelProjection(Float64, N2, n2)), W) - sum(∇L .* W)) /
        (norm(∇L) * norm(W)))
end
