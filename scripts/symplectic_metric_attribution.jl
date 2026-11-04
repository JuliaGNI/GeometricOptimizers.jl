# Which published metric `rgrad` and `metric` of `SymplecticStiefelManifold` implement.
#
#   julia --startup-file=no --project=. scripts/symplectic_metric_attribution.jl
#
# Bendokat and Zimmermann, "The real symplectic Stiefel and Grassmann manifolds", (3.16) and (3.17),
# are transcribed below, and the package must match them at a generic point. The canonical-like metric
# of Gao, Son, Absil and Stykel, "Riemannian optimization on the symplectic Stiefel manifold", is
# `tr(Z₁ᵀ B_ρ Z₂)` with `B_ρ` from their (4.2) and Proposition 4.1, under each of their two
# orthonormalization conditions. `g_ρ(Z, Z) = c₁/ρ + c₂`, so one tangent vector fixes the one `ρ` that
# matches the package on it. The script shows that two tangent vectors fix two different `ρ` at a
# generic point, so that no `ρ` gives this metric. At the base point `E` the metric is `g₂`, which is
# why a check at `E` alone cannot tell the two apart.

using GeometricOptimizers
using GeometricOptimizers: check
using LinearAlgebra
using Random

const N, n = 5, 2
J(k) = [zeros(k, k) I(k); -I(k) zeros(k, k)]
const J₁, J₂ = J(N), J(n)
const COLUMNS = [1:n; (N + 1):(N + n)]

tangent(rng, U) = (S = randn(rng, 2N, 2N); (S + S') * J₁ * U)

g_316(U, a, b) = (P = inv(U' * U); tr(a' * (I - J₁' * U * P * U' * J₁ / 2) * b * P))
grad_317(U, G) = G * (U' * U) + J₁ * U * (G' * J₁ * U)

# The term of `B_ρ` that does not depend on `ρ`, under condition (I) and under condition (II).
term_I(U) = -(J₁ * U * J₂ * U' * J₁' - J₁)^2
term_II(U) = I - U * inv(U' * U) * U'
g_ρ(U, term, ρ, a, b) = tr(a' * (J₁ * U * U' * J₁' / ρ + term(U)) * b)

function matching_ρ(U, term, a)
    c₂ = tr(a' * term(U) * a)
    tr(a' * J₁ * U * U' * J₁' * a) / (metric(U, a, a) - c₂)
end

rng = Xoshiro(1801)
S = randn(rng, 2N, 2N)
U = SymplecticStiefelManifold((exp(0.3 * J₁ * (S + S') / 2))[:, COLUMNS])
A = Matrix(U.A)
G = randn(rng, 2N, 2n)
a, b, c = tangent(rng, A), tangent(rng, A), tangent(rng, A)

@assert check(U) < 1.0e-12
@assert norm(rgrad(U, G) - grad_317(A, G)) < 1.0e-11 * norm(G)
@assert isapprox(metric(U, a, b), g_316(A, a, b); rtol = 1.0e-11)
@assert isapprox(metric(U, rgrad(U, G), c), tr(G' * c); rtol = 1.0e-9)

for (name, term) in ("condition (I)" => term_I, "condition (II)" => term_II)
    ρ₁, ρ₂ = matching_ρ(U, term, a), matching_ρ(U, term, b)
    println(name, ": the ρ that match two tangent vectors at a generic point: ",
        ρ₁, " and ", ρ₂)
    @assert abs(ρ₁ - ρ₂) > 0.05
    println(name, ": g_ρ(rgrad, Z) - tr(GᵀZ) at ρ = 2: ",
        g_ρ(A, term, 2, rgrad(U, G), c) - tr(G' * c))
    @assert abs(g_ρ(A, term, 2, rgrad(U, G), c) - tr(G' * c)) > 1.0e-3
end

E = Matrix(1.0I, 2N, 2N)[:, COLUMNS]
x, y = tangent(rng, E), tangent(rng, E)
for term in (term_I, term_II)
    @assert isapprox(g_ρ(E, term, 2, x, y), g_316(E, x, y); rtol = 1.0e-11)
end
println("at E, g₂ is (3.16) under both conditions")
