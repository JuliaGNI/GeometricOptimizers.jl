# The flat gradient of a symplectic lift, `gradient_rep!`: the package's path from the Riemannian
# gradient Δ = rgrad(U, ∇L), which applies (UᵀU)⁻¹ through a Cholesky factor, against a direct
# product from the Euclidean gradient ∇L, which the optimizer protocol does not hand it. See K59.
#
# Run: julia --startup-file=no --project=. scripts/symplectic_gradient_rep_direct.jl
#
# Results, Julia 1.13.3, load average about 5, 2026-11-11:
#   accuracy at 10 × 4, median relative error of 20 draws against the direct product in BigFloat
#   (the Cholesky column includes the rounding of Δ itself):
#     Float64  κ(U) = 2.2 / 24 / 576   Cholesky 3.4e-16 / 7.4e-16 / 7.2e-15   direct 1.3e-16 / 1.2e-16 / 1.0e-16
#     Float32  κ(U) = 2.2 / 24 / 576   Cholesky 1.5e-7 / 3.1e-7 / 2.3e-6      direct 5.4e-8 / 5.0e-8 / 5.0e-8
#   time per call, Float64, minimum of 200 calls:
#     10 × 4      Cholesky 1.0e-6 s   direct 2.5e-7 s   rgrad 4.6e-7 s
#     100 × 10    Cholesky 1.3e-5 s   direct 5.7e-6 s   rgrad 1.5e-5 s
#     1000 × 20   Cholesky 6.0e-4 s   direct 5.1e-4 s   rgrad 1.2e-3 s
#     4000 × 40   Cholesky 1.7e-2 s   direct 1.6e-2 s   rgrad 3.2e-2 s
using GeometricOptimizers, LinearAlgebra, Random, Printf
using GeometricOptimizers: gradient_rep!, global_rep, symmetric_storage_gradient_kernel!
import KernelAbstractions

function sym_storage_gradient!(S::AbstractVector{BigFloat}, G)
    n = size(G, 1)
    for i in 1:n
        for j in 1:i
            S[i * (i - 1) ÷ 2 + j] = i == j ? G[i, i] : G[i, j] + G[j, i]
        end
    end
    S
end

function sym_storage_gradient!(S, G)
    n = size(G, 1)
    backend = KernelAbstractions.get_backend(G)
    symmetric_storage_gradient_kernel!(backend)(S, G; ndrange = (n, n))
    S
end

# the stored blocks of the flat gradient from X = UᵀG and Y = (Uˢ)ᵀG, G a Frobenius representative
function from_products!(D::SymplecticLieAlgHorMatrix{T}, X, Y) where {T}
    k = D.n ÷ 2
    @views begin
        D.A₁ .= X[1:k, 1:k] .- transpose(X[(k + 1):(2k), (k + 1):(2k)])
        sym_storage_gradient!(D.B₁.S, X[1:k, (k + 1):(2k)])
        sym_storage_gradient!(D.C₁.S, X[(k + 1):(2k), 1:k])
    end
    D.B .= Y
    D
end

direct!(D, λY, ∇L) = from_products!(D, λY.Y.A' * ∇L, λY.λ' * ∇L)

storage(D) = vcat(vec(D.A₁), parent(D.B₁), parent(D.C₁), vec(D.B))

# `[A -B; B A] diag(d, 1/d)` for a complex Stiefel point `A + iB`; `spread` widens `d`, and `κ(U)`
function point(rng, T, N2, n2, spread)
    N, n = N2 ÷ 2, n2 ÷ 2
    Q = Matrix(qr(randn(rng, Complex{Float64}, N, n)).Q)
    A, B = real(Q), imag(Q)
    d = exp2.(spread .* (2 .* rand(rng, n) .- 1))
    SymplecticStiefelManifold(T.(hcat(vcat(A, B) .* d', vcat(-B, A) ./ d')))
end

println("accuracy, 10 × 4, relative error against the direct product in BigFloat, median of 20")
for T in (Float64, Float32), spread in (1.0, 4.0, 8.0)

    rng = Xoshiro(1)
    e_chol, e_direct, κs = Float64[], Float64[], Float64[]
    for _ in 1:20
        U = point(rng, T, 10, 4, spread)
        λY = GlobalSection(U)
        ∇L = T.(randn(rng, 10, 4))
        Δ = rgrad(U, ∇L)
        Dc = gradient_rep!(global_rep(λY, Δ), λY, Δ)
        Dd = direct!(global_rep(λY, Δ), λY, ∇L)
        D₀ = zeros(SymplecticLieAlgHorMatrix{BigFloat}, 10, 4)
        Dref = from_products!(D₀, big.(U.A)' * big.(∇L), big.(λY.λ)' * big.(∇L))
        ref = Float64.(storage(Dref))
        push!(e_chol, norm(Float64.(storage(Dc)) - ref) / norm(ref))
        push!(e_direct, norm(Float64.(storage(Dd)) - ref) / norm(ref))
        push!(κs, cond(Float64.(U.A)))
    end
    m(x) = sort(x)[10]
    @printf("  %-7s κ(U) median %8.2e   Cholesky from Δ %8.2e   direct from ∇L %8.2e\n",
        T, m(κs), m(e_chol), m(e_direct))
end

println("time per call, Float64, minimum of 200 calls")
for (N2, n2) in ((10, 4), (100, 10), (1000, 20), (4000, 40))
    rng = Xoshiro(2)
    U = point(rng, Float64, N2, n2, 1.0)
    λY = GlobalSection(U)
    ∇L = randn(rng, N2, n2)
    Δ = rgrad(U, ∇L)
    D = global_rep(λY, Δ)
    gradient_rep!(D, λY, Δ)
    direct!(D, λY, ∇L)
    rgrad(U, ∇L)
    tc = minimum(@elapsed(gradient_rep!(D, λY, Δ)) for _ in 1:200)
    td = minimum(@elapsed(direct!(D, λY, ∇L)) for _ in 1:200)
    tr = minimum(@elapsed(rgrad(U, ∇L)) for _ in 1:200)
    @printf("  %5d × %-3d  Cholesky from Δ %9.2e s   direct from ∇L %9.2e s   rgrad %9.2e s\n",
        N2, n2, tc, td, tr)
end
