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

    #maybe modify this - you don't need N & n as inputs!
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

function Base.getindex(A::GrassmannLieAlgHorMatrix{T}, i::Integer, j::Integer) where {T}
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

# `src/ambiguities.jl` has the `+` and `-` methods that reach these two.
function _owned_add(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N
    @assert A.n == B.n
    GrassmannLieAlgHorMatrix(A.B + B.B,
        A.N,
        A.n)
end

function _owned_sub(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N
    @assert A.n == B.n
    GrassmannLieAlgHorMatrix(A.B - B.B,
        A.N,
        A.n)
end

function add!(C::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    @assert A.N == B.N == C.N
    @assert A.n == B.n == C.n
    add!(C.B, A.B, B.B)
end

function Base.:-(A::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(-A.B, A.N, A.n)
end

function Base.:*(A::GrassmannLieAlgHorMatrix, α::Real)
    GrassmannLieAlgHorMatrix(α*A.B, A.N, A.n)
end

Base.:*(α::Real, A::GrassmannLieAlgHorMatrix) = A*α

# The first `n` rows of `B * C` — see the comment on the `StiefelLieAlgHorMatrix` method of this
# function for the `transpose` and for where the minus sits. This lift's `(1, 1)` block is the zero
# matrix, so only the `-Bᵀ` block contributes.
#
# `C₁` goes unused here. It is taken so that both methods share one signature, which is what lets
# `*(::AbstractLieAlgHorMatrix, ::AbstractMatrix)` be written once for the two lifts.
_hor_top_rows(B::GrassmannLieAlgHorMatrix, C₁, C₂) = -(transpose(B.B) * C₂)

function Base.zeros(backend::KernelAbstractions.Backend,
        ::Type{<:GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    GrassmannLieAlgHorMatrix(_zeros(backend, T, N-n, n), N, n)
end

function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
        ::Type{<:GrassmannLieAlgHorMatrix{T}}, N::Integer, n::Integer) where {T}
    GrassmannLieAlgHorMatrix(_rand(rng, backend, T, N - n, n), N, n)
end

# `typeof(A)` is the two-parameter `GrassmannLieAlgHorMatrix{T, AT}`, which `zeros` has no
# method for; it has to be narrowed to the one-parameter form, as in the Stiefel case.
#
# The backend comes from `A`, for the reason given beside `similar(::StiefelLieAlgHorMatrix)`.
function Base.similar(A::GrassmannLieAlgHorMatrix, dims::Union{
        Integer, AbstractUnitRange}...)
    zeros(KernelAbstractions.get_backend(A), GrassmannLieAlgHorMatrix{eltype(A)}, dims...)
end
function Base.similar(A::GrassmannLieAlgHorMatrix)
    zeros(KernelAbstractions.get_backend(A), GrassmannLieAlgHorMatrix{eltype(A)}, A.N, A.n)
end

function scalar_add(A::GrassmannLieAlgHorMatrix, δ::Real)
    GrassmannLieAlgHorMatrix(A.B .+ δ, A.N, A.n)
end

#define these functions more generally! (maybe make a fallback script!!)
function ⊙²(A::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(A.B .^ 2, A.N, A.n)
end
function racᵉˡᵉ(A::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(sqrt.(A.B), A.N, A.n)
end
function /ᵉˡᵉ(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    GrassmannLieAlgHorMatrix(A.B ./ B.B, A.N, A.n)
end

function LinearAlgebra.mul!(C::GrassmannLieAlgHorMatrix, A::GrassmannLieAlgHorMatrix, α::Real)
    mul!(C.B, A.B, α)
    C
end
function LinearAlgebra.mul!(C::GrassmannLieAlgHorMatrix, α::Real, A::GrassmannLieAlgHorMatrix)
    mul!(C, A, α)
end
LinearAlgebra.rmul!(C::GrassmannLieAlgHorMatrix, α::Real) = mul!(C, C, α)

function _round(B::GrassmannLieAlgHorMatrix; kwargs...)
    GrassmannLieAlgHorMatrix(
        _round(B.B; kwargs...),
        B.N,
        B.n
    )
end

# The generic `AbstractArray` fallbacks for these route through `setindex!`, which this type
# does not define, so they have to be given explicitly — as they already are for
# `StiefelLieAlgHorMatrix`.
Base.zero(B::GrassmannLieAlgHorMatrix) = GrassmannLieAlgHorMatrix(zero(B.B), B.N, B.n)
Base.copy(B::GrassmannLieAlgHorMatrix) = GrassmannLieAlgHorMatrix(copy(B.B), B.N, B.n)

function Base.copyto!(A::GrassmannLieAlgHorMatrix, B::GrassmannLieAlgHorMatrix)
    copyto!(A.B, B.B)
    A
end

Base.fill!(A::GrassmannLieAlgHorMatrix, val) = (fill!(A.B, val); A)
