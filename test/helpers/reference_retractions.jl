# The allocating `𝔄` algorithms and the `Cayley` differential as `src/` wrote them before they were
# given in-place forms, copied verbatim apart from the names. The in-place forms promise the same
# answer to the bit wherever they keep the product order, and a test that compared them with the
# allocating methods of `src/` would compare one implementation with itself once those delegate to
# the in-place forms. These do not move.

using GeometricOptimizers: ScaledSquaring, NativePade, AugmentedPade, unit_matrix, opnorm₁,
                           lift_factors, lift_from_columns, StiefelProjection
using LinearAlgebra: LinearAlgebra, norm

function reference_𝔄(A::AbstractMatrix)
    T = eltype(A)
    term = unit_matrix(A)
    next = zero(A)
    𝔄A = copy(term)
    n = 2
    while norm(term) > eps(real(T))
        LinearAlgebra.mul!(next, term, A, T(inv(n)), zero(T))
        term, next = next, term
        𝔄A .+= term
        n += 1
    end
    𝔄A
end

reference_scaled_kernel(X::AbstractMatrix, ::ScaledSquaring) = reference_𝔄(X)

function reference_𝔄(X::AbstractMatrix, algorithm::Union{ScaledSquaring, NativePade})
    nrm = opnorm₁(X)
    s = nrm > algorithm.θ ? ceil(Int, log2(nrm / algorithm.θ)) : 0
    scale = eltype(X)(2)^s

    W = reference_scaled_kernel(X / scale, algorithm) / scale
    for _ in 1:s
        W = 2 * W + W * X * W
    end

    W
end

function reference_native_pade_polynomials(X::AbstractMatrix, 𝕀::AbstractMatrix)
    T = eltype(X)
    X² = X * X
    X⁴ = X² * X²

    p = 𝕀 + T(1 // 26) * X +
        X² * (T(5 // 156) * 𝕀 + T(1 // 858) * X) +
        X⁴ * (T(1 // 5720) * 𝕀 + T(1 // 205920) * X + T(1 // 8648640) * X²)
    q = 𝕀 - T(6 // 13) * X +
        X² * (T(5 // 52) * 𝕀 - T(5 // 429) * X) +
        X⁴ * (T(1 // 1144) * 𝕀 - T(1 // 25740) * X + T(1 // 1235520) * X²)

    p, q
end

function reference_scaled_kernel(X::AbstractMatrix, ::NativePade)
    𝕀 = unit_matrix(X)
    p, q = reference_native_pade_polynomials(X, 𝕀)

    q⁻¹ = 2 * 𝕀 - q
    for _ in 1:4
        q⁻¹ = q⁻¹ * (2 * 𝕀 - q * q⁻¹)
    end

    q⁻¹ * p
end

function reference_𝔄(X::AbstractMatrix, ::AugmentedPade)
    m = size(X, 1)
    T = eltype(X)
    augmented = [X one(X); zeros(T, m, m) zeros(T, m, m)]

    exp(augmented)[1:m, (m + 1):(2m)]
end

function reference_cayley_differential(B, α)
    T = eltype(B)
    iszero(α) && return B

    a = T(α) / 2
    B̂, B̄ = lift_factors(B)
    E = StiefelProjection(B)
    G = B̄' * B̂
    𝕀 = one(G)

    w₁ = E + B̂ * ((𝕀 - a * G) \ (a * (B̄' * E)))      # M E
    w₂ = B̂ * (B̄' * w₁)                                # B̄ M E
    V = w₂ - B̂ * ((𝕀 + a * G) \ (a * (B̄' * w₂)))     # Mᵀ B̄ M E

    lift_from_columns(B, V)
end
