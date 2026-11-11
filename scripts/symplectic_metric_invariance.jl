# Why no inner product on the symplectic g^hor reproduces a Riemannian metric through every section,
# checked without the package. The vertical space at a point is the null space of Z ↦ ZU in sp(2N),
# and its complements for ½tr(XᵀY) and for the bi-invariant h of Bendokat and Zimmermann (2.1) are
# computed numerically; only (3.7), (3.16) and (3.18) of that paper are typed in.
#
# Run: julia --startup-file=no scripts/symplectic_metric_invariance.jl
#
# Results, Julia 1.13.3, seed 7, N = 5, n = 2:
#   (b) at E the h-complement and the ½tr-complement are one g^hor: 0.0 both ways
#   (c) the ½tr-complement at a generic U, conjugated back by λ, is not in g^hor (0.82); the
#       h-complement is (4e-15)
#   (a) ½tr of the ½tr-horizontal lift is (3.16) (−98.05 both); (3.7) is the h-horizontal lift
#       (1.5e-13), and its ½tr is not (3.16) (580 against 157)
#       ½tr of the stored lift λ⁻¹Ω̃λ depends on the section: 32.6 against 8916 at cond(S) = 1103
#   (f) Ad_h with S = diag(tI, I/t) scales a B-block entry by 1/t: 1, 0.1, 0.01, so no positive
#       definite form on g^hor is Ad_H-invariant
#   (g) h (3.8) is indefinite on T_E: −2 and +2
#   (h) exp(tΩ̃)U agrees with the geodesic (3.18) of (3.16) to first order only: ‖diff‖/t² = 295
#   (d) at an orthosymplectic point the conjugated ½tr-complement is g^hor: 1e-15
#   (e) an orthosymplectic Q is an isometry of (3.16) (ratio 1.000), a generic symplectic L is not
#       (9.68)
#   the ω-complement of a generic U is not J-invariant (1.7), so it has no orthonormal symplectic
#   basis; `global_section` normalises an orthonormal basis of it instead
using LinearAlgebra, Random
Random.seed!(7)

J(m) = [zeros(m, m) I(m); -I(m) zeros(m, m)]
plus(U) = J(size(U, 2) ÷ 2)' * U' * J(size(U, 1) ÷ 2)          # U⁺
ham(m) = (S = randn(2m, 2m); J(m)' * (S + S'))

N, n = 5, 2
idx = [1:n; (N + 1):(N + n)];
cidx = setdiff(1:(2N), idx);
E = Matrix{Float64}(I, 2N, 2N)[:, idx]

# basis of sp(2N): J'S for the symmetric unit matrices S
basis = Matrix{Float64}[]
for i in 1:(2N), j in i:(2N)

    S = zeros(2N, 2N)
    S[i, j] = 1
    S[j, i] = 1
    push!(basis, J(N)' * S)
end
d = length(basis);
@assert d == N * (2N + 1)
mat(c) = sum(c[i] * basis[i] for i in 1:d)
Bm = reduce(hcat, vec.(basis))
coords(Z) = Bm \ vec(Z)
tr½(X, Y) = tr(X' * Y) / 2
hf(X, Y) = -tr(X * Y) / 2                                       # bi-invariant h, = ½tr(X⁺Y) on sp
Gtr = [tr½(basis[i], basis[j]) for i in 1:d, j in 1:d]
Gh = [hf(basis[i], basis[j]) for i in 1:d, j in 1:d]

vertical(U) = nullspace(reduce(hcat, [vec(b * U) for b in basis]))   # coordinates of {Z : ZU = 0}
complement(V, G) = nullspace(Matrix((G * V)'))
function resid(A, B)
    Q = Matrix(qr(B).Q)[:, 1:size(B, 2)]
    maximum(norm(A[:, k] - Q * (Q' * A[:, k])) / norm(A[:, k]) for k in axes(A, 2))
end
function lift(H, U, Δ)
    mat(H * ((reduce(hcat, [vec(mat(H[:, k]) * U) for k in axes(H, 2)])) \ vec(Δ)))
end

VE = vertical(E)
HorE_h = complement(VE, Gh);
HorE_tr = complement(VE, Gtr);
println(
    "dim g, ver_E, hor_E:                      ", d, " ", size(VE, 2), " ", size(HorE_h, 2))
println("(b) at E, h-complement = ½tr-complement:   ", resid(HorE_h, HorE_tr), " ",
    resid(HorE_tr, HorE_h))

M = exp(0.6 * ham(N));
U = M * E
Ad(Λ, H) = reduce(hcat, [coords(Λ \ mat(H[:, k]) * Λ) for k in axes(H, 2)])
VU = vertical(U)
HorU_tr = complement(VU, Gtr);
HorU_h = complement(VU, Gh);
println("(c) λ⁻¹ hor_U(½tr) λ ⊂ hor_E:              ", resid(Ad(M, HorU_tr), HorE_h))
println("    λ⁻¹ hor_U(h) λ ⊂ hor_E:                ", resid(Ad(M, HorU_h), HorE_h))

tangent(U, Λ) = U * ham(n) + Λ[:, cidx] * randn(2N - 2n, 2n)
Δ1, Δ2 = tangent(U, M), tangent(U, M)
g316(U, A, B) = (G = inv(U' * U); tr(A' * (I - J(N)' * U * G * U' * J(N) / 2) * B * G))
Ωbar(Δ) = lift(HorU_tr, U, Δ)
println("(a) ½tr(Ω̄1, Ω̄2) vs (3.16):                 ", tr½(Ωbar(Δ1), Ωbar(Δ2)), "  ",
    g316(U, Δ1, Δ2))
Ωt(U, Δ) = (I - U * plus(U) / 2) * Δ * plus(U) - U * plus(Δ) * (I - U * plus(U) / 2)
println("    (3.7) Ω̃ = numeric h-lift:              ", norm(Ωt(U, Δ1) -
                                                            lift(HorU_h, U, Δ1)))
println("    ½tr(Ω̃,Ω̃) vs (3.16):                    ", tr½(Ωt(U, Δ1), Ωt(U, Δ1)), "  ",
    g316(U, Δ1, Δ1))

# section dependence of ½tr of the stored lift λ⁻¹Ω̃λ
stab(S) = (h = Matrix{Float64}(I, 2N, 2N); h[cidx, cidx] = S; h)
S = exp(0.8 * ham(N - n));
h = stab(S)
sec(Λ) = (B = Λ \ Ωt(U, Δ1) * Λ; tr½(B, B))
println(
    "    ½tr through λ, λh (cond S = ", round(cond(S); digits = 1), "):    ", sec(M), "  ",
    sec(M * h))

# lemma: Ad_h with S = diag(tI, I/t) scales a B-block entry by 1/t
Bx = zeros(2N, 2N);
Bx[cidx[1], 1] = 1;
Bx = Bx - J(N)' * Bx' * J(N)
for t in (1.0, 10.0, 100.0)
    ht = stab(Matrix(Diagonal([fill(t, N - n); fill(1 / t, N - n)])))
    @assert norm(ht' * J(N) * ht - J(N)) < 1e-9 && norm(ht * E - E) == 0
    println("(f) t = ", rpad(t, 5), " ‖B-block of h⁻¹Bh‖ = ", norm((ht \ Bx * ht)[cidx, idx]))
end

# (g) h-metric (3.8) on T_E
A1 = [I(n) zeros(n, n); zeros(n, n) -I(n)] * 1.0;
A2 = J(n) * 1.0
h38(A, B) = tr(plus(A) * (I - E * plus(E) / 2) * B)
println("(g) h(EA, EA), A = diag(I,-I) and J:       ", h38(E * A1, E * A1), "  ",
    h38(E * A2, E * A2))

# (h) exp(tΩ̃)U against (3.18)
O = Ωbar(Δ1)
for t in (1e-2, 1e-3)
    γh = exp(t * Ωt(U, Δ1)) * U
    γg = exp(t * (O - O')) * exp(t * O') * U
    println("(h) t = ", t, "  ‖diff‖/t² = ", norm(γh - γg) / t^2, "   ‖diff‖/t = ",
        norm(γh - γg) / t)
end

# orthosymplectic point: ½tr-complement transported equals hor_E
Q = exp(0.6 * (A = ham(N); (A - A') / 2));
@assert norm(Q' * Q - I) < 1e-10
println("(d) orthosymplectic: λ⁻¹ hor(½tr) λ ⊂ hor_E: ",
    resid(Ad(Q, complement(vertical(Q * E), Gtr)), HorE_h))

# (e) left action and (3.16)
L = exp(0.6 * ham(N))
println("(e) g(QU)(QΔ)/g(U)(Δ), g(LU)(LΔ)/g(U)(Δ): ",
    g316(Q * U, Q * Δ1, Q * Δ1) / g316(U, Δ1, Δ1), "  ",
    g316(L * U, L * Δ1, L * Δ1) / g316(U, Δ1, Δ1))

# The ω-complement W of span U has an orthonormal symplectic basis only if W is J-invariant.
W = nullspace(Matrix(U' * J(N)))
PW = W * W'
println("ω-complement J-invariant? ‖(I-P_W) J W‖ = ", norm((I - PW) * J(N) * W))
