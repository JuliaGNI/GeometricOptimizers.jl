@doc raw"""
    BiorthogonalLieAlgHorMatrix(A, B, C, N, n)

The horizontal component ``\mathfrak{g}^\mathrm{hor}`` of the Lie algebra ``\mathfrak{gl}(N)`` for the
[`BiorthogonalManifold`](@ref) of pairs of ``N\times{}n`` matrices. An element is the ``N\times{}N``
matrix

```math
\begin{pmatrix}
A & C \\ B & \mathbb{O}
\end{pmatrix},
```

with ``A\in\mathbb{R}^{n\times{}n}``, ``B\in\mathbb{R}^{(N-n)\times{}n}`` and
``C\in\mathbb{R}^{n\times(N-n)}`` arbitrary. Its dimension is ``n^2 + 2(N-n)n = 2Nn - n^2``, the
dimension of the manifold.

This is the [`StiefelLieAlgHorMatrix`](@ref) with the two constraints of
``\mathfrak{so}(N)`` dropped: ``A`` need not be skew-symmetric, and the upper right block is a block of
its own and not ``-B^T``. Also see [`SymplecticLieAlgHorMatrix`](@ref).

# Extended help

The general linear group acts on the pairs by ``g\cdot(\Phi, \Psi) = (g\Phi, g^{-T}\Psi)``, and an
element ``X`` of its Lie algebra by ``X\cdot(\Phi, \Psi) = (X\Phi, -X^T\Psi)``. At the distinct point
``(E, E)``, with ``E`` the first ``n`` columns of the identity ([`StiefelProjection`](@ref)), this is
``(XE, -X^TE)``: the first ``n`` columns and the first ``n`` rows of ``X``. The vertical space, on
which both vanish, is the Lie algebra of the isotropy group ``GL(N - n)``, the matrices that are zero
outside their lower right block, and its complement above is the same subspace of ``\mathfrak{gl}(N)``
at every point. [`global_rep`](@ref) maps a tangent vector into it, see there.

Under ``\mathfrak{gl}(N)\to\mathfrak{sp}(2N)``, ``X\mapsto\mathrm{diag}(X, -X^T)``, this lift is the part of a
[`SymplecticLieAlgHorMatrix`](@ref) with ``B_1 = C_1 = \mathbb{O}``, ``A_1 = A`` and the block-diagonal
``\mathcal{B} = \mathrm{diag}(B, -C^T)``.
"""
mutable struct BiorthogonalLieAlgHorMatrix{T, AT <: AbstractMatrix{T}, BT <: AbstractMatrix{T},
    CT <: AbstractMatrix{T}} <: AbstractLieAlgHorMatrix{T}
    A::AT
    B::BT
    C::CT
    N::Int
    n::Int

    # `N` and `n` follow from the blocks; the constructor takes them and asserts that they agree.
    function BiorthogonalLieAlgHorMatrix(A::AbstractMatrix{T}, B::AbstractMatrix{T},
            C::AbstractMatrix{T}, N::Integer, n::Integer) where {T}
        @assert size(A) == (n, n)
        @assert size(B) == (N - n, n)
        @assert size(C) == (n, N - n)

        new{T, typeof(A), typeof(B), typeof(C)}(A, B, C, N, n)
    end
end

@doc raw"""
    BiorthogonalLieAlgHorMatrix(D::AbstractMatrix, n::Integer)

Take a big ``N\times{}N`` matrix and build its horizontal projection:

```math
\begin{pmatrix}
A & C  \\
B & D_{22}
\end{pmatrix} \mapsto
\begin{pmatrix}
A & C \\
B & \mathbb{O}
\end{pmatrix}.
```

The vertical space is the lower right block, so the projection drops it and keeps the rest as it is.
"""
function BiorthogonalLieAlgHorMatrix(D::AbstractMatrix, n::Integer)
    N = size(D, 1)
    @assert N ≥ n
    @views BiorthogonalLieAlgHorMatrix(D[1:n, 1:n], D[(n + 1):N, 1:n], D[1:n, (n + 1):N], N, n)
end

function BiorthogonalLieAlgHorMatrix(V::AbstractVector, N::Int, n::Int)
    @assert length(V) == n^2 + 2 * (N - n) * n
    m = (N - n) * n
    BiorthogonalLieAlgHorMatrix(
        reshape(@view(V[1:(n ^ 2)]), n, n),
        reshape(@view(V[(n ^ 2 + 1):(n ^ 2 + m)]), N - n, n),
        reshape(@view(V[(n ^ 2 + m + 1):end]), n, N - n),
        N,
        n
    )
end

Base.parent(A::BiorthogonalLieAlgHorMatrix) = (A.A, A.B, A.C)
Base.size(A::BiorthogonalLieAlgHorMatrix) = (A.N, A.N)

manifold_type(::BiorthogonalLieAlgHorMatrix) = BiorthogonalManifold

function KernelAbstractions.get_backend(B::BiorthogonalLieAlgHorMatrix)
    KernelAbstractions.get_backend(B.B)
end

# The bounds check comes first for the reason `getindex(::StiefelLieAlgHorMatrix, …)` gives.
function Base.getindex(A::BiorthogonalLieAlgHorMatrix{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    if i ≤ A.n
        j ≤ A.n && return A.A[i, j]
        return A.C[i, j - A.n]
    end
    j ≤ A.n && return A.B[i - A.n, j]
    zero(T)
end

# The first `n` rows of `B * C`, `AC₁ + CC₂`; see the method for the Stiefel lift.
_hor_top_rows(B::BiorthogonalLieAlgHorMatrix, C₁, C₂) = B.A * C₁ + B.C * C₂

# The product from the right, `[C₁ C₂]B = [C₁A + C₂B  C₁C]`. The method on `AbstractLieAlgHorMatrix`
# goes through the identity `Bᵀ = -B`, which this lift does not satisfy.
function _rmul(C::AbstractMatrix{T}, B::BiorthogonalLieAlgHorMatrix{T}) where {T}
    @assert size(C, 2) == B.N
    @views C₁, C₂ = C[:, 1:(B.n)], C[:, (B.n + 1):(B.N)]
    hcat(C₁ * B.A + C₂ * B.B, C₁ * B.C)
end

function _rmul(x::Union{Adjoint{T, <:AbstractVector}, Transpose{T, <:AbstractVector}},
        B::BiorthogonalLieAlgHorMatrix{T}) where {T}
    transpose(vec(_rmul(reshape(collect(x), 1, length(x)), B)))
end

function Base.zeros(backend::KernelAbstractions.Backend,
        ::Type{BiorthogonalLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    BiorthogonalLieAlgHorMatrix(_zeros(backend, T, n, n), _zeros(backend, T, N - n, n),
        _zeros(backend, T, n, N - n), N, n)
end

# The blocks are drawn in the order `parent` returns them, as for the other lifts.
function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{BiorthogonalLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    A = _rand(rng, backend, T, n, n)
    B = _rand(rng, backend, T, N - n, n)
    BiorthogonalLieAlgHorMatrix(A, B, _rand(rng, backend, T, n, N - n), N, n)
end
