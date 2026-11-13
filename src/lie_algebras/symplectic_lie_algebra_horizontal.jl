@doc raw"""
    SymplecticLieAlgHorMatrix(A₁, B₁::SymmetricMatrix, C₁::SymmetricMatrix, B, N, n)

The horizontal component ``\mathfrak{g}^\mathrm{hor}`` of the Lie algebra of the symplectic group,
for the [`SymplecticStiefelManifold`](@ref) of ``2N\times2n`` matrices. The fields `N` and `n` hold
the sizes of a point, ``2N`` and ``2n``, as they do for the other lifts.

An element is the ``2N\times2N`` matrix

```math
\begin{pmatrix}
\mathcal{A} & -\mathcal{B}^+ \\ \mathcal{B} & \mathbb{O}
\end{pmatrix},
\qquad
\mathcal{A} = \begin{pmatrix} A_1 & B_1 \\ C_1 & -A_1^T \end{pmatrix},
```

where ``A_1`` is an arbitrary ``n\times{}n`` matrix and ``B_1`` and ``C_1`` are symmetric, so that
``\mathcal{A}\in\mathfrak{sp}(2n)`` is Hamiltonian, ``\mathcal{B}`` is an arbitrary
``(2N - 2n)\times2n`` matrix, and
``\mathcal{B}^+ = \mathbb{J}_{2n}^T\mathcal{B}^T\mathbb{J}_{2N - 2n}`` is its symplectic adjoint.
This is the [`StiefelLieAlgHorMatrix`](@ref) with the transpose replaced by the symplectic adjoint
and the skew-symmetric block by a Hamiltonian one.

Also see [`StiefelLieAlgHorMatrix`](@ref) and [`GrassmannLieAlgHorMatrix`](@ref).

# Extended help

This is the horizontal space that [bendokat2021real](@cite) give below their (3.4): the complement
of the vertical space with respect to the bi-invariant pseudo-Riemannian metric ``h`` of their
(2.1). It is the only one of their two horizontal spaces that is the same subspace of the Lie
algebra at every point. The horizontal space (3.13) of the Riemannian metric ``g`` depends on the
point, so it has no global representation of this kind. The metric the gradient is taken in is a
separate matter; see [`GeometricOptimizers.gradient_rep!`](@ref).

[bendokat2021real](@cite) order the columns of the symplectic group so that ``E`` picks the columns
``1`` to ``n`` and ``N + 1`` to ``N + n``. In that order ``\mathcal{B}`` above is
``\begin{pmatrix} A_3 & B_2 \\ C_2 & -A_2 \end{pmatrix}``. A [`GlobalSection`](@ref) of this
package stores ``\lambda(U) = [U\ \ U^s]``, the point first and its symplectic completion
[`global_section`](@ref) after it, as it does for the [`StiefelManifold`](@ref). So ``E`` is the
first ``2n`` columns, as [`StiefelProjection`](@ref) gives it. In this basis the Poisson tensor is

```math
\bar{\mathbb{J}} = \begin{pmatrix} \mathbb{J}_{2n} & \mathbb{O} \\ \mathbb{O} & \mathbb{J}_{2N - 2n} \end{pmatrix},
```

the section satisfies ``\lambda(U)^T\mathbb{J}_{2N}\lambda(U) = \bar{\mathbb{J}}``, the lift satisfies
``\Omega^T\bar{\mathbb{J}} + \bar{\mathbb{J}}\Omega = \mathbb{O}``, and a retraction of it is a
matrix ``R`` with ``R^T\bar{\mathbb{J}}R = \bar{\mathbb{J}}``. So ``\lambda(U)R`` is a section again
and ``\lambda(U)RE`` a point of the manifold. A retraction itself, which [`cayley`](@ref) and
[`geodesic`](@ref) wrap in a [`SymplecticStiefelManifold`](@ref), preserves ``\bar{\mathbb{J}}`` and
not ``\mathbb{J}_{2N}``, so [`check`](@ref) does not apply to it.
"""
mutable struct SymplecticLieAlgHorMatrix{
    T, AT <: AbstractMatrix{T}, ST <: SymmetricMatrix{T},
    BT <: AbstractMatrix{T}} <: AbstractLieAlgHorMatrix{T}
    A₁::AT
    B₁::ST
    C₁::ST
    B::BT
    N::Int
    n::Int

    # `N` and `n` follow from the blocks; the constructor takes them and asserts that they agree.
    function SymplecticLieAlgHorMatrix(A₁::AbstractMatrix{T}, B₁::SymmetricMatrix{T},
            C₁::SymmetricMatrix{T}, B::AbstractMatrix{T}, N::Integer, n::Integer) where {T}
        @assert iseven(N) && iseven(n)
        @assert size(A₁) == (n ÷ 2, n ÷ 2) && B₁.n == C₁.n == n ÷ 2
        @assert size(B) == (N - n, n)

        new{T, typeof(A₁), typeof(B₁), typeof(B)}(A₁, B₁, C₁, B, N, n)
    end
end

@doc raw"""
    SymplecticLieAlgHorMatrix(D::AbstractMatrix, n::Integer)

Take a big matrix as input and build an instance of `SymplecticLieAlgHorMatrix`.

`D` is ``2N\times2N`` and `n` is ``2n``, the number of columns of a point. The projection is

```math
\begin{pmatrix}
A & B_1  \\
B_2 & D
\end{pmatrix} \mapsto
\begin{pmatrix}
\mathrm{ham}(A) & -B_2^+ \\
B_2 & \mathbb{O}
\end{pmatrix},
```

with ``\mathrm{ham}(A) = \frac{1}{2}(A - A^+)`` the projection onto the Hamiltonian matrices, the
counterpart of the skew-symmetrization of [`StiefelLieAlgHorMatrix(::AbstractMatrix, ::Integer)`](@ref).
"""
function SymplecticLieAlgHorMatrix(D::AbstractMatrix{T}, n::Integer) where {T}
    N = size(D, 1)
    @assert N ≥ n
    @assert iseven(n)

    k = n ÷ 2
    backend = KernelAbstractions.get_backend(D)
    lift = SymplecticLieAlgHorMatrix(_zeros(backend, T, k, k),
        zeros(backend, SymmetricMatrix{T}, k), zeros(backend, SymmetricMatrix{T}, k),
        _zeros(backend, T, N - n, n), N, n)
    @views _hamiltonian_projection!(lift, D[1:n, 1:n])
    @views lift.B .= D[(n + 1):N, 1:n]
    lift
end

function SymplecticLieAlgHorMatrix(V::AbstractVector, N::Int, n::Int)
    k = n ÷ 2
    s = k * (k + 1) ÷ 2
    @assert length(V) == k^2 + 2s + (N - n) * n
    SymplecticLieAlgHorMatrix(
        reshape(@view(V[1:(k ^ 2)]), k, k),
        SymmetricMatrix(@view(V[(k ^ 2 + 1):(k ^ 2 + s)]), k),
        SymmetricMatrix(@view(V[(k ^ 2 + s + 1):(k ^ 2 + 2s)]), k),
        reshape(@view(V[(k ^ 2 + 2s + 1):end]), N - n, n),
        N,
        n
    )
end

Base.parent(A::SymplecticLieAlgHorMatrix) = (A.A₁, A.B₁, A.C₁, A.B)
Base.size(A::SymplecticLieAlgHorMatrix) = (A.N, A.N)

manifold_type(::SymplecticLieAlgHorMatrix) = SymplecticStiefelManifold

function KernelAbstractions.get_backend(B::SymplecticLieAlgHorMatrix)
    KernelAbstractions.get_backend(B.B)
end

# The entry `(i, j)` of the Hamiltonian block `𝒜`.
function _hamiltonian_entry(A::SymplecticLieAlgHorMatrix, i, j)
    k = A.n ÷ 2
    if i ≤ k
        return j ≤ k ? A.A₁[i, j] : A.B₁[i, j - k]
    end
    j ≤ k ? A.C₁[i - k, j] : -A.A₁[j - k, i - k]
end

# The entry `(i, j)` of `ℬ⁺ = 𝕁ₙᵀℬᵀ𝕁_{N-n}`, which is the block matrix
# `[ℬ₂₂ᵀ -ℬ₁₂ᵀ; -ℬ₂₁ᵀ ℬ₁₁ᵀ]` of the four blocks of `ℬ`. `transpose` and not `adjoint`, as for the
# Stiefel lift.
function _symplectic_adjoint_entry(B::AbstractMatrix, i, j)
    k, m = size(B, 2) ÷ 2, size(B, 1) ÷ 2
    if i ≤ k
        return j ≤ m ? B[m + j, k + i] : -B[j - m, k + i]
    end
    j ≤ m ? -B[m + j, i - k] : B[j - m, i - k]
end

# The bounds check comes first for the reason `getindex(::StiefelLieAlgHorMatrix, …)` gives.
function Base.getindex(
        A::SymplecticLieAlgHorMatrix{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    if i ≤ A.n
        j ≤ A.n && return _hamiltonian_entry(A, i, j)
        return -_symplectic_adjoint_entry(A.B, i, j - A.n)
    end
    j ≤ A.n && return A.B[i - A.n, j]
    zero(T)
end

# `ℬ⁺` scaled by `s`, written into `D` block by block. Four broadcasts of a transposed block, so it
# runs wherever `ℬ` is, and allocates nothing.
function _symplectic_adjoint!(D::AbstractMatrix, B::AbstractMatrix, s = one(eltype(B)))
    k, m = size(B, 2) ÷ 2, size(B, 1) ÷ 2
    @assert size(D) == (2k, 2m)
    @views begin
        D[1:k, 1:m] .= s .* transpose(B[(m + 1):(2m), (k + 1):(2k)])
        D[1:k, (m + 1):(2m)] .= .-s .* transpose(B[1:m, (k + 1):(2k)])
        D[(k + 1):(2k), 1:m] .= .-s .* transpose(B[(m + 1):(2m), 1:k])
        D[(k + 1):(2k), (m + 1):(2m)] .= s .* transpose(B[1:m, 1:k])
    end
    D
end

function _symplectic_adjoint(B::AbstractMatrix{T}) where {T}
    D = KernelAbstractions.allocate(KernelAbstractions.get_backend(B), T, size(B, 2), size(B, 1))
    _symplectic_adjoint!(D, B)
end

# The dense `𝒜`, written into `C`. A `SymmetricMatrix` block is read through `getindex` on the host,
# which allocates nothing, and through a product with the identity elsewhere, as `_dense_skew!` does
# for the Stiefel lift.
function _hamiltonian_block!(C::AbstractMatrix{T}, B::SymplecticLieAlgHorMatrix{T}) where {T}
    k = B.n ÷ 2
    @views begin
        C[1:k, 1:k] .= B.A₁
        C[(k + 1):(2k), (k + 1):(2k)] .= .-transpose(B.A₁)
        _dense_sym!(C[1:k, (k + 1):(2k)], B.B₁)
        _dense_sym!(C[(k + 1):(2k), 1:k], B.C₁)
    end
    C
end

function _hamiltonian_block(B::SymplecticLieAlgHorMatrix{T}) where {T}
    backend = KernelAbstractions.get_backend(B)
    _hamiltonian_block!(KernelAbstractions.allocate(backend, T, B.n, B.n), B)
end

function _dense_sym!(C::AbstractMatrix{T}, S::SymmetricMatrix{T, Vector{T}}) where {T}
    (C .= S)
end
function _dense_sym!(C::AbstractMatrix{T}, S::SymmetricMatrix{T}) where {T}
    C .= S * unit_matrix(KernelAbstractions.get_backend(S.S), T, S.n)
end

# The Hamiltonian projection `(Z - Z⁺)/2` of the `n × n` matrix `Z`, written into the blocks
# `A₁`, `B₁` and `C₁` of `D`: `A₁ = (Z₁₁ - Z₂₂ᵀ)/2`, and `B₁` and `C₁` the symmetric parts of
# `Z₁₂` and `Z₂₁`. Of a Hamiltonian `Z` this is `Z` itself.
function _hamiltonian_projection!(D::SymplecticLieAlgHorMatrix{T}, Z::AbstractMatrix{T}) where {T}
    k = D.n ÷ 2
    @assert size(Z) == (2k, 2k)
    @views begin
        D.A₁ .= T(0.5) .* (Z[1:k, 1:k] .- transpose(Z[(k + 1):(2k), (k + 1):(2k)]))
        _map_to_S!(D.B₁, Z[1:k, (k + 1):(2k)])
        _map_to_S!(D.C₁, Z[(k + 1):(2k), 1:k])
    end
    D
end

# The first `n` rows of `B * C`, `𝒜C₁ - ℬ⁺C₂`; see the method for the Stiefel lift.
function _hor_top_rows(B::SymplecticLieAlgHorMatrix, C₁, C₂)
    _hamiltonian_block(B) * C₁ - _symplectic_adjoint(B.B) * C₂
end

# The product with the transpose, `BᵀC = [𝒜ᵀC₁ + ℬᵀC₂; -(ℬ⁺)ᵀC₁]`, block by block as `B * C` is.
# The transposed blocks are `permutedims` copies, which a device product serves; they are `n × n`,
# `n × (N - n)` and `(N - n) × n`, each as large as a dense block that `B * C` builds.
function _transpose_lmul_into!(D::AbstractMatrix{T}, B::SymplecticLieAlgHorMatrix{T},
        C::AbstractMatrix{T}) where {T}
    @assert B.N == size(C, 1) == size(D, 1)
    @assert size(C, 2) == size(D, 2)

    C₁ = @view C[1:(B.n), :]
    C₂ = @view C[(B.n + 1):(B.N), :]
    @views D[1:(B.n), :] .= permutedims(_hamiltonian_block(B)) * C₁ .+ permutedims(B.B) * C₂
    @views D[(B.n + 1):(B.N), :] .= .-(permutedims(_symplectic_adjoint(B.B)) * C₁)
    D
end

function _transpose_lmul(B::SymplecticLieAlgHorMatrix{T}, C::AbstractMatrix{T}) where {T}
    backend = KernelAbstractions.get_backend(B)
    _transpose_lmul_into!(KernelAbstractions.allocate(backend, T, B.N, size(C, 2)), B, C)
end

# The product from the right, `CB = (BᵀCᵀ)ᵀ`. The method on `AbstractLieAlgHorMatrix` goes through
# the identity `Bᵀ = -B`, which a Hamiltonian lift does not satisfy. `permutedims` and not a view of
# `C`, for the reason given at `_rmul(::AbstractMatrix, ::AbstractLieAlgHorMatrix)`: `C` can be the
# adjoint of a device array, as `Y' * B` passes it on.
function _rmul(C::AbstractMatrix{T}, B::SymplecticLieAlgHorMatrix{T}) where {T}
    permutedims(_transpose_lmul(B, permutedims(C)))
end

function _rmul(x::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}},
        B::SymplecticLieAlgHorMatrix{T}) where {T}
    transpose(vec(_transpose_lmul(B, reshape(transpose(x), length(x), 1))))
end

function Base.zeros(backend::KernelAbstractions.Backend,
        ::Type{SymplecticLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    k = n ÷ 2
    SymplecticLieAlgHorMatrix(
        _zeros(backend, T, k, k), zeros(backend, SymmetricMatrix{T}, k),
        zeros(backend, SymmetricMatrix{T}, k), _zeros(backend, T, N - n, n), N, n)
end

# The blocks are drawn in the order `parent` returns them, on every backend, so that a seeded draw
# names the same matrix whether or not the call names the backend.
function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{SymplecticLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    k = n ÷ 2
    A₁ = _rand(rng, backend, T, k, k)
    B₁ = rand(rng, backend, SymmetricMatrix{T}, k)
    C₁ = rand(rng, backend, SymmetricMatrix{T}, k)
    SymplecticLieAlgHorMatrix(A₁, B₁, C₁, _rand(rng, backend, T, N - n, n), N, n)
end
