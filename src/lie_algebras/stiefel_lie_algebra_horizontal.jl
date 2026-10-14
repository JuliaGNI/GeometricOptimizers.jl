@doc raw"""
    StiefelLieAlgHorMatrix(A::SkewSymMatrix, B::AbstractMatrix, N::Integer, n::Integer)

Build an instance of `StiefelLieAlgHorMatrix` based on a skew-symmetric matrix `A` and an arbitrary matrix `B`.

An element of StiefelLieAlgMatrix takes the form:
```math
\begin{pmatrix}
A & B^T \\ B & \mathbb{O}
\end{pmatrix},
```
where ``A`` is skew-symmetric (this is [`SkewSymMatrix`](@ref) in `GeometricOptimizers`).

Also see [`GrassmannLieAlgHorMatrix`](@ref).

# Extended help

`StiefelLieAlgHorMatrix` is the *horizontal component of the Lie algebra of skew-symmetric matrices* (with respect to the canonical metric).

The projection here is: ``\pi:S \to SE`` where
```math
E = \begin{bmatrix} \mathbb{I}_{n} \\ \mathbb{O}_{(N-n)\times{}n}  \end{bmatrix}.
```
The matrix ``E`` is implemented under [`StiefelProjection`](@ref) in `GeometricOptimizers`.
"""
mutable struct StiefelLieAlgHorMatrix{T, AT <: SkewSymMatrix{T}, ST <: AbstractMatrix{T}} <:
               AbstractLieAlgHorMatrix{T}
    A::AT
    B::ST
    N::Int
    n::Int

    # `N` and `n` follow from the blocks; the constructor takes them and asserts that they agree.
    function StiefelLieAlgHorMatrix(A::SkewSymMatrix{T}, B::AbstractMatrix{T}, N::Integer, n::Integer) where {T}
        @assert n == A.n == size(B, 2)
        @assert N == size(B, 1) + n

        new{T, typeof(A), typeof(B)}(A, B, N, n)
    end
end

@doc raw"""
    StiefelLieAlgHorMatrix(D::AbstractMatrix, n::Integer)

Take a big matrix as input and build an instance of `StiefelLieAlgHorMatrix`.

The integer ``N`` in ``St(n, N)`` is the number of rows of `D`.

# Extended help

If the constructor is called with a big ``N\times{}N`` matrix, then the projection is performed the following way:

```math
\begin{pmatrix}
A & B_1  \\
B_2 & D
\end{pmatrix} \mapsto
\begin{pmatrix}
\mathrm{skew}(A) & -B_2^T \\
B_2 & \mathbb{O}
\end{pmatrix}.
```

The operation ``\mathrm{skew}:\mathbb{R}^{n\times{}n}\to\mathcal{S}_\mathrm{skew}(n)`` is the skew-symmetrization operation. This is equivalent to calling of [`SkewSymMatrix`](@ref) with an ``n\times{}n`` matrix.

This can also be seen as the operation:
```math
D \mapsto \Omega(E, DE) = \mathrm{skew}\left(2 \left(\mathbb{I} - \frac{1}{2} E E^T \right) DE E^T\right).
```

Also see [`GeometricOptimizers.Ω`](@ref).
"""
function StiefelLieAlgHorMatrix(D::AbstractMatrix, n::Integer)
    N = size(D, 1)
    @assert N ≥ n

    @views A_small = SkewSymMatrix(D[1:n, 1:n])
    @views B = D[(n + 1):N, 1:n]
    StiefelLieAlgHorMatrix(A_small, B, N, n)
end

Base.parent(A::StiefelLieAlgHorMatrix) = (A.A, A.B)
Base.size(A::StiefelLieAlgHorMatrix) = (A.N, A.N)

manifold_type(::StiefelLieAlgHorMatrix) = StiefelManifold

# The bounds check comes first for the reason `getindex(::StrictlyLowerTriangular, …)` gives: the
# last line of the body answers `zero(T)` for an index outside the matrix as well as for one in the
# zero block, and the four branches above it index two different arrays, so an out-of-range pair can
# answer a legitimate-looking entry instead of raising.
function Base.getindex(A::StiefelLieAlgHorMatrix{T}, i, j) where {T}
    @boundscheck checkbounds(A, i, j)
    if i ≤ A.n
        if j ≤ A.n
            return A.A[i, j]
        end
        return -A.B[j - A.n, i]
    end
    if j ≤ A.n
        return A.B[i - A.n, j]
    end
    return zero(T)
end

# The sum kernel. `src/ambiguities.jl` has the `+` methods that reach it.
function _ladd(B::StiefelLieAlgHorMatrix, A::AbstractMatrix)
    @assert size(A) == size(B)

    # The destination is a fresh plain array rather than `copy(A)`. A structured `A` keeps its type
    # under `copy`, and its `setindex!` then constrains what the three blocks below can write: a
    # `SymmetricMatrix` symmetrizes each of them, and a triangular or a manifold point rejects the
    # half that falls outside its stored entries. The sum of a horizontal lift and an arbitrary
    # matrix carries none of those structures. The element type is promoted across both operands for
    # the same reason -- `copy(A)` gave the destination `A`'s element type alone.
    backend = KernelAbstractions.get_backend(A)
    C = KernelAbstractions.allocate(backend, promote_type(eltype(A), eltype(B)), size(A)...)
    copyto!(C, A)
    @views C[1:B.n, 1:B.n] .= B.A + A[1:B.n, 1:B.n]
    @views C[(B.n + 1):B.N, 1:B.n] .= B.B + A[(B.n + 1):B.N, 1:B.n]
    # `transpose` and not `adjoint`: `getindex` above builds this block as `-B.B[j, i]`, entrywise
    # and without conjugating, so the sum has to spell it the same way or the two disagree on a
    # complex element type. They agree on a real one.
    @views C[1:B.n, (B.n + 1):B.N] .= A[1:B.n, (B.n + 1):B.N] - transpose(B.B)

    C
end

# A sum of two owned matrices is dense everywhere else (`src/ambiguities.jl`); this pair is the
# exception, because a `StiefelLieAlgHorMatrix` is skew-symmetric by construction and so the sum of
# the two is as well. The result is built in the packed representation rather than dense and
# re-projected, which keeps an integer element type integer.
#
# Both blocks below land inside the destination's strict lower triangle, where the entry `(i, j)`
# with `i > j` sits at `S[(i - 2) * (i - 1) ÷ 2 + j]`. For a row `i ≤ n` that index is the one the
# inner `n × n` block uses for the same entry, so the first `n * (n - 1) ÷ 2` entries take `C.A.S`
# as one slice. Row `i > n` holds `C.B[i - n, :]` in its first `n` columns and zero in the rest,
# which is the loop.
function _owned_add(C::StiefelLieAlgHorMatrix, A::SkewSymMatrix)
    @assert size(A) == size(C)

    S = similar(A.S, promote_type(eltype(A), eltype(C)))
    copyto!(S, A.S)
    @views S[1:(C.n * (C.n - 1) ÷ 2)] .+= C.A.S
    for i in (C.n + 1):(C.N)
        offset = (i - 2) * (i - 1) ÷ 2
        @views S[(offset + 1):(offset + C.n)] .+= C.B[i - C.n, :]
    end

    SkewSymMatrix(S, C.N)
end

_owned_add(A::SkewSymMatrix, C::StiefelLieAlgHorMatrix) = _owned_add(C, A)

# `-` on the same mixed pair returns a dense matrix, although the difference of two skew-symmetric
# matrices is skew-symmetric as well. A structured `-` would be a behaviour change, so the asymmetry
# against `+` above is deliberate.

# The first `n` rows of `B * C`, which is the one part of the product that differs between the two
# lifts. This lift's top row of blocks is `[A  -Bᵀ]`, so both blocks contribute; the docstring on
# `*(::AbstractLieAlgHorMatrix, ::AbstractMatrix)` in `abstract_lie_algebra_horizontal.jl` says why
# the product needs a method at all.
#
# `transpose` and not `adjoint`, for the reason the comment on `+(::StiefelLieAlgHorMatrix,
# ::AbstractMatrix)` above gives: `getindex` builds that block as `-B.B[j - n, i]`, entrywise and
# without conjugating, so the product has to spell it the same way or the two disagree on a complex
# element type.
#
# The minus stays outside the product rather than moving onto the transpose, which is
# arithmetically the same and cheaper: `transpose(B.B)` is a lazy wrapper that feeds straight into
# the product, where `-transpose(B.B)` materializes a whole second `n × (N - n)` block first. The
# comment on `*(::AbstractMatrix, ::SkewSymMatrix)` in `special_matrices/skew_symmetric.jl`
# measures that difference for the case it is written about.
_hor_top_rows(B::StiefelLieAlgHorMatrix, C₁, C₂) = B.A * C₁ - transpose(B.B) * C₂

function Base.zeros(backend::KernelAbstractions.Backend,
        ::Type{StiefelLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    StiefelLieAlgHorMatrix(
        zeros(backend, SkewSymMatrix{T}, n), _zeros(backend, T, N - n, n), N, n)
end

# The `A` block is drawn before the `B` block, on every backend, so that a seeded draw names the same
# matrix whether or not the call names the backend.
function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{StiefelLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    A = rand(rng, backend, SkewSymMatrix{T}, n)
    StiefelLieAlgHorMatrix(A, _rand(rng, backend, T, N - n, n), N, n)
end

function StiefelLieAlgHorMatrix(V::AbstractVector, N::Int, n::Int)
    # length of skew-symmetric matrix
    skew_sym_size = n * (n - 1) ÷ 2
    # size of matrix component
    matrix_size = (N - n) * n
    @assert length(V) == skew_sym_size + matrix_size
    StiefelLieAlgHorMatrix(
        SkewSymMatrix(@view(V[1:skew_sym_size]), n),
        reshape(@view(V[(skew_sym_size + 1):(skew_sym_size + matrix_size)]), (N - n), n),
        N,
        n
    )
end

function KernelAbstractions.get_backend(B::StiefelLieAlgHorMatrix)
    KernelAbstractions.get_backend(B.B)
end

# `copyto!` accepts any destination at least as long as its source, so without this guard a
# mismatched pair partially overwrites `A` and leaves the rest stale. The structured `assign!`
# methods that reach this one through `foreach` check only their own `n`, not the block shapes.
function assign!(A::AbstractArray, B::AbstractArray)
    @assert size(A) == size(B)
    copyto!(A, B)

    nothing
end
