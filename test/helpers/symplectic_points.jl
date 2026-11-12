# Points of the symplectic Stiefel manifold for the tests that check algebra rather than the draw,
# and the Poisson tensors the symplectic lift is checked against.
using LinearAlgebra: I, qr

# A point of the symplectic Stiefel manifold built without `sr!`, with `κ ≤ 4`: the orthosymplectic
# point `[A -B; B A]` of a complex Stiefel point `A + iB`, times the symplectic `diag(D, D⁻¹)` on
# the right, with the entries of `D` in `[1/2, 2]`. `D` is what makes `UᵀU ≠ I`, so that `P` in the
# metric is not the identity.
function conditioned_point(rng, T, N2, n2)
    N, n = N2 ÷ 2, n2 ÷ 2
    Q = Matrix(qr(randn(rng, Complex{T}, N, n)).Q)
    A, B = real(Q), imag(Q)
    d = exp2.(2 .* rand(rng, T, n) .- 1)
    SymplecticStiefelManifold(hcat(vcat(A, B) .* d', vcat(-B, A) ./ d'))
end

# The orthosymplectic point `[A -B; B A]` of a complex Stiefel point `A + iB`: `UᵀU = I`.
function orthosymplectic_point(rng, T, N2, n2)
    N, n = N2 ÷ 2, n2 ÷ 2
    Q = Matrix(qr(randn(rng, Complex{T}, N, n)).Q)
    A, B = real(Q), imag(Q)
    SymplecticStiefelManifold(hcat(vcat(A, B), vcat(-B, A)))
end

poisson(::Type{T}, n) where {T} = [zeros(T, n ÷ 2, n ÷ 2) I; -I zeros(T, n ÷ 2, n ÷ 2)]

# the Poisson tensor `𝕁̄ = diag(𝕁ₙ, 𝕁_{N-n})` of the basis a symplectic section is stored in
function adapted_poisson(::Type{T}, N, n) where {T}
    cat(poisson(T, n), poisson(T, N - n); dims = (1, 2))
end
