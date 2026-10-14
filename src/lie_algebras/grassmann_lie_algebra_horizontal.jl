@doc raw"""
    GrassmannLieAlgHorMatrix(B::AbstractMatrix, N::Integer, n::Integer)

Build an instance of `GrassmannLieAlgHorMatrix` based on an arbitrary matrix `B` of size ``(N-n)\times{}n``.

`GrassmannLieAlgHorMatrix` is the *horizontal component of the Lie algebra of skew-symmetric matrices* (with respect to the canonical metric).

# Extended help

The projection here is: ``\pi:S \to SE/\sim`` where 
```math
E = \begin{bmatrix} \mathbb{I}_{n} \\ \mathbb{O}_{(N-n)\times{}n}  \end{bmatrix},
```

and the equivalence relation is 

```math
V_1 \sim V_2 \iff \exists A\in\mathcal{S}_\mathrm{skew}(n) \text{ such that } V_2 = V_1 + \begin{bmatrix} A \\ \mathbb{O} \end{bmatrix}
```

An element of GrassmannLieAlgMatrix takes the form: 
```math
\begin{pmatrix}
\bar{\mathbb{O}} & B^T \\ B & \mathbb{O}
\end{pmatrix},
```
where ``\bar{\mathbb{O}}\in\mathbb{R}^{n\times{}n}`` and ``\mathbb{O}\in\mathbb{R}^{(N - n)\times(N-n)}.``
"""
mutable struct GrassmannLieAlgHorMatrix{T, ST <: AbstractMatrix{T}} <:
               AbstractLieAlgHorMatrix{T}
    B::ST
    N::Int
    n::Int

    # `N` and `n` follow from `B`; the constructor takes them and asserts that they agree.
    function GrassmannLieAlgHorMatrix(B::AbstractMatrix{T}, N::Integer, n::Integer) where {T}
        @assert n == size(B, 2)
        @assert N == size(B, 1) + n

        new{T, typeof(B)}(B, N, n)
    end
end

@doc raw"""
    GrassmannLieAlgHorMatrix(D::AbstractMatrix, n::Integer)

Take a big matrix as input and build an instance of `GrassmannLieAlgHorMatrix`.

The integer ``N`` in ``Gr(n, N)`` here is the number of rows of `D`.

# Extended help

If the constructor is called with a big ``N\times{}N`` matrix, then the projection is performed the following way: 

```math
\begin{pmatrix}
A & B_1  \\
B_2 & D
\end{pmatrix} \mapsto 
\begin{pmatrix}
\bar{\mathbb{O}} & -B_2^T \\ 
B_2 & \mathbb{O}
\end{pmatrix}.
```

This can also be seen as the operation:
```math
D \mapsto \Omega(E, DE - EE^TDE),
```

where ``\Omega`` is the horizontal lift [`GeometricOptimizers.Ω`](@ref).
"""
function GrassmannLieAlgHorMatrix(D::AbstractMatrix, n::Integer)
    N = size(D, 1)
    @assert N ≥ n

    @views B = D[(n + 1):N, 1:n]
    GrassmannLieAlgHorMatrix(B, N, n)
end

Base.parent(A::GrassmannLieAlgHorMatrix) = (A.B,)
Base.size(A::GrassmannLieAlgHorMatrix) = (A.N, A.N)

manifold_type(::GrassmannLieAlgHorMatrix) = GrassmannManifold

function KernelAbstractions.get_backend(B::GrassmannLieAlgHorMatrix)
    KernelAbstractions.get_backend(B.B)
end

# The bounds check comes first for the reason `getindex(::StrictlyLowerTriangular, …)` gives.
function Base.getindex(A::GrassmannLieAlgHorMatrix{T}, i::Integer, j::Integer) where {T}
    @boundscheck checkbounds(A, i, j)
    if i ≤ A.n
        if j ≤ A.n
            return zero(T)
        end
        return -A.B[j - A.n, i]
    end
    if j ≤ A.n
        return A.B[i - A.n, j]
    end
    return zero(T)
end

# The first `n` rows of `B * C` — see the comment on the `StiefelLieAlgHorMatrix` method of this
# function for the `transpose` and for where the minus sits. This lift's `(1, 1)` block is the zero
# matrix, so only the `-Bᵀ` block contributes.
#
# `C₁` goes unused here. It is taken so that both methods share one signature, which is what lets
# `*(::AbstractLieAlgHorMatrix, ::AbstractMatrix)` be written once for the two lifts.
_hor_top_rows(B::GrassmannLieAlgHorMatrix, C₁, C₂) = -(transpose(B.B) * C₂)

function Base.zeros(backend::KernelAbstractions.Backend,
        ::Type{GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    GrassmannLieAlgHorMatrix(_zeros(backend, T, N - n, n), N, n)
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    GrassmannLieAlgHorMatrix(_rand(rng, backend, T, N - n, n), N, n)
end
