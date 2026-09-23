# The symplectic SR decomposition, the symplectic counterpart of the QR decomposition: it writes
# `A = S * R` with `S` symplectic (`SᵀJS = J`) and `R` of the block form below. The algorithm is
# the SROSH algorithm of [salam2008optimal](@cite), whose free parameters are taken at the optimal
# values that paper derives: `ρ = sign(a₁)‖a‖₂` minimizes the 2-norm condition number of the first
# reflector (its Lemma 4.3) and `μ = u₁ + ξ` that of the second (its Lemma 4.4). `symplectic_householder!`
# returns exactly the `(c₁, c₂, ρ, ν, μ)` of its Theorem 4.5, and `scripts/verify_salam2008.jl`
# checks the correspondence numerically. What is adjusted is the application rather than the
# mathematics: the reflectors are applied in place, the way this package's optimizers need them,
# rather than the way a general-purpose factorization would.
#
# [gao2024optimization](@cite) is what the decomposition is used for here.
#
# The one consumer here is `rand(SymplecticStiefelManifold, …)`, which takes `n` of `S`'s columns
# from each half. The orthonormalization that `rand(StiefelManifold, …)` uses will not do: its
# factor is orthogonal, and what a point of the symplectic Stiefel manifold has to satisfy is
# `UᵀJ_{2N}U = J_{2n}`.

@doc raw"""
    SymplecticHouseholderDecom(A, c₁, c₂, ρ, ν, μ)

The packed form of a symplectic SR decomposition, as [`sr!`](@ref) computes it.

`A` holds the reflector vectors below the diagonal of each block, in place, exactly as
`LinearAlgebra`'s `QRCompactWY` holds its own. The five vectors are the scalars the reflectors
were built with: `c₁` and `c₂` scale the two reflections of a step, and `ρ`, `ν` and `μ` are the
diagonal entries the step produced in `R`.

This is the storage, not the factors. [`Sfac`](@ref) and [`Rfac`](@ref) read `S` and `R` off it.
"""
struct SymplecticHouseholderDecom{T, AT <: AbstractMatrix{T}, VT <: AbstractVector{T}} <:
       LinearAlgebra.Factorization{T}
    A::AT
    c₁::VT
    c₂::VT
    ρ::VT
    ν::VT
    μ::VT
end

@doc raw"""
    Sfac(Λ::SymplecticHouseholderDecom, inverse::Bool = false)

The symplectic factor ``S`` of a symplectic SR decomposition, as an operator.

`S` is never formed: multiplying by it applies the reflectors `Λ` stores, in ``O(N^2)`` per vector.
`inv(S)` is the same object with the reflectors applied in the opposite order, which is why the
inverse costs what the forward application costs.

Indexing one entry builds the whole matrix and throws the rest away, because a reflector product
has no cheap entrywise form. Use `Matrix(S)` when more than a few entries are wanted; that is what
`rand(SymplecticStiefelManifold, …)` does before it slices.
"""
struct Sfac{inverse, T, ST <: SymplecticHouseholderDecom{T}} <: AbstractMatrix{T}
    Λ::ST
    function Sfac(Λ::SymplecticHouseholderDecom{T}, inverse::Bool = false) where {T}
        new{inverse, T, typeof(Λ)}(Λ)
    end
end

Base.size(S::Sfac) = (size(S.Λ.A, 1), size(S.Λ.A, 1))

function Base.Matrix(S::Sfac{false, T}) where {T}
    apply_S_left!(Matrix{T}(LinearAlgebra.I, size(S)...), S.Λ)
end

function Base.Matrix(S::Sfac{true, T}) where {T}
    apply_S_inverse_left!(Matrix{T}(LinearAlgebra.I, size(S)...), S.Λ)
end

Base.getindex(S::Sfac, i::Integer, j::Integer) = Matrix(S)[i, j]

Base.inv(S::Sfac{false}) = Sfac(S.Λ, true)
Base.inv(S::Sfac{true}) = Sfac(S.Λ, false)

@doc raw"""
    Rfac(Λ::SymplecticHouseholderDecom)

The ``R`` factor of a symplectic SR decomposition, as an operator.

``R`` is upper triangular in the symplectic sense: in the ``2N\times2M`` block form

```math
R = \begin{pmatrix} R_{11} & R_{12} \\ R_{21} & R_{22} \end{pmatrix},
```

``R_{11}`` and ``R_{22}`` are upper triangular and ``R_{21}`` is strictly upper triangular. The
entries come from the packed `Λ`; nothing is stored twice.
"""
struct Rfac{T, ST <: SymplecticHouseholderDecom{T}} <: AbstractMatrix{T}
    Λ::ST
    function Rfac(Λ::SymplecticHouseholderDecom{T}) where {T}
        new{T, typeof(Λ)}(Λ)
    end
end

Base.size(R::Rfac) = size(R.Λ.A)

function Base.getindex(R::Rfac{T}, i::Integer, j::Integer) where {T}
    # Every branch below either reads a packed array or returns a structural zero, so an index past
    # the end can reach a `zero(T)` and come back looking like a legitimate entry. The packed arrays
    # cannot be relied on to raise it: they are smaller than `R` and indexed by shifted indices.
    @boundscheck checkbounds(R, i, j)
    N, M = size(R.Λ.A) .÷ 2
    if j ≤ M
        i == j && return R.Λ.ρ[i]
        i < j && return R.Λ.A[i, j]
        i ≤ N && return zero(T)
        i < (N + j) && return R.Λ.A[i, j]
        return zero(T)
    end
    i == (j - M) && return R.Λ.μ[i]
    i == (j - M + N) && return R.Λ.ν[i - N]
    i < (j - M) && return R.Λ.A[i, j]
    i ≤ N && return zero(T)
    i < (j - M + N) && return R.Λ.A[i, j]
    return zero(T)
end

# The matrix and vector cases are separate methods on purpose. One method on `AbstractVecOrMat` is
# ambiguous against `LinearAlgebra`'s `*(::AbstractMatrix, ::AbstractVector)`, because `Sfac` is an
# `AbstractMatrix` itself: neither signature is more specific in both arguments, so `S * x` for a
# vector `x` is a `MethodError` at the call site.
Base.:*(S::Sfac{false}, B::AbstractMatrix) = apply_S_left(S.Λ, B)
Base.:*(S::Sfac{false}, b::AbstractVector) = apply_S_left(S.Λ, b)
Base.:*(S::Sfac{true}, B::AbstractMatrix) = apply_S_inverse_left(S.Λ, B)
Base.:*(S::Sfac{true}, b::AbstractVector) = apply_S_inverse_left(S.Λ, b)
Base.:*(B::AbstractMatrix, S::Sfac{false}) = apply_S_right(B, S.Λ)

Base.:*(B::AbstractMatrix, S::Sfac{true}) = apply_S_inverse_right(B, S.Λ)

# A row vector on the left is the one shape the two methods above leave unsettled: each stands off
# against `LinearAlgebra`'s own row-vector product, and neither wins. *A row vector meets an owned
# matrix* in `src/ambiguities.jl` gives the mechanism and lists every site. Each body is the one
# above it, so a row vector gets the answer that method gives every other matrix, and gets it the
# same cheap way: the reflectors are applied to the one row rather than assembled into a matrix.
#
# Four methods and not two, for the reason the `Sfac`-`Sfac` comment below gives: one method on
# `Sfac` is wider than `Sfac{false}` in that slot and so separates neither pair.
Base.:*(x::Adjoint{<:Any, <:AbstractVector}, S::Sfac{false}) = apply_S_right(x, S.Λ)
Base.:*(x::Transpose{<:Any, <:AbstractVector}, S::Sfac{false}) = apply_S_right(x, S.Λ)
Base.:*(x::Adjoint{<:Any, <:AbstractVector}, S::Sfac{true}) = apply_S_inverse_right(x, S.Λ)
function Base.:*(x::Transpose{<:Any, <:AbstractVector}, S::Sfac{true})
    apply_S_inverse_right(x, S.Λ)
end

# Without this, `S * inv(S)` — the most natural thing to write with two of these — is an ambiguous
# `MethodError` between the two methods above, because `Sfac` is itself an `AbstractMatrix` and
# neither signature is more specific in both arguments. Both factors are materialized rather than
# chained: it is the one resolution that is correct for every pair without a special case, and no
# caller in this package multiplies two of them, so the cost falls only on someone who asks for it.
# `S * inv(S)` is therefore the identity only up to the roundoff of the two kernels, which is the
# honest answer rather than an exact `I`: measured, `‖S·S⁻¹ - I‖` is 5.8e-16 at `2N = 4`, 4.4e-14
# at `2N = 10` and 1.7e-8 at `2N = 20`, growing with the size the way everything else here does.
#
# All four combinations are written out because one method on `(::Sfac, ::Sfac)` does not resolve
# it: against `(::Sfac{false}, ::AbstractMatrix)` it is narrower in the second argument and wider
# in the first, so neither dominates and the call stays ambiguous. Each pair below is narrower in
# both.
Base.:*(S₁::Sfac{false}, S₂::Sfac{false}) = Matrix(S₁) * Matrix(S₂)
Base.:*(S₁::Sfac{false}, S₂::Sfac{true}) = Matrix(S₁) * Matrix(S₂)
Base.:*(S₁::Sfac{true}, S₂::Sfac{false}) = Matrix(S₁) * Matrix(S₂)
Base.:*(S₁::Sfac{true}, S₂::Sfac{true}) = Matrix(S₁) * Matrix(S₂)

@doc raw"""
    SR(S::Sfac, R::Rfac)

The result of [`sr!`](@ref): the two factors of ``A = SR``, reached as `F.S` and `F.R`.
"""
struct SR{T, ST <: Sfac{<:Any, T}, RT <: Rfac{T}} <: LinearAlgebra.Factorization{T}
    S::ST
    R::RT
    function SR(S::Sfac{inverse, T}, R::Rfac{T}) where {inverse, T}
        new{T, typeof(S), typeof(R)}(S, R)
    end
end

@doc raw"""
    sr!(A)

Compute the symplectic SR decomposition of `A` in place, and return it as an [`SR`](@ref).

`A` is ``2N\times2M`` with ``N \geq M``, read as two stacked halves the way every symplectic
object in this package is. `A` is overwritten with the packed reflectors.

!!! warning "Not backward stable"
    ``S`` is symplectic and therefore not orthogonal, its condition number is unbounded, and this
    implementation has no re-orthogonalization step. The median residual grows with the size by two
    to three orders of magnitude per doubling, and `symplectic_householder!` can raise a
    `DomainError` from a cancelled square root at large sizes in `Float32`. `CHANGELOG.md` has the
    measured table; read its medians, because the maxima do not reproduce across draw orders. Use
    it in `Float64`, at small sizes, and check the result.

# Examples

```jldoctest
using GeometricOptimizers
using LinearAlgebra
import Random

Random.seed!(1234)

F = sr!(randn(6, 4))
S = Matrix(F.S)
J = [zeros(3, 3) I(3); -I(3) zeros(3, 3)]

norm(S' * J * S - J) < 1e-5

# output

true
```

See [`sr`](@ref) for the copying version.
"""
function sr!(A::AbstractMatrix{T}) where {T}
    N2, M2 = size(A)
    @assert iseven(N2)
    @assert iseven(M2)
    @assert N2 ≥ M2
    N = N2 ÷ 2
    M = M2 ÷ 2
    c₁ = zeros(T, M)
    c₂ = zeros(T, M)
    ρ = zeros(T, M)
    ν = zeros(T, M)
    μ = zeros(T, M)
    for i in 1:M
        row_ind = vcat(i:N, (N + i):N2)
        @views a = A[row_ind, i]
        @views b = A[row_ind, M + i]
        c₁[i], c₂[i], ρ[i], ν[i], μ[i] = symplectic_householder!(a, b)
        # apply the two reflections of this step to the columns that are left
        for j in (i + 1):M
            A[row_ind, j] .+= c₁[i] * symplectic_form(a, A[row_ind, j]) * a
            A[row_ind, j] .+= c₂[i] * symplectic_form(b, A[row_ind, j]) * b
            A[row_ind, j + M] .+= c₁[i] * symplectic_form(a, A[row_ind, j + M]) * a
            A[row_ind, j + M] .+= c₂[i] * symplectic_form(b, A[row_ind, j + M]) * b
        end
    end
    Λ = SymplecticHouseholderDecom(A, c₁, c₂, ρ, ν, μ)
    SR(Sfac(Λ), Rfac(Λ))
end

@doc raw"""
    sr(A)

Compute the symplectic SR decomposition of `A`, leaving `A` alone. See [`sr!`](@ref).
"""
sr(A::AbstractMatrix) = sr!(copy(A))

@doc raw"""
    apply_S_left!(B, Λ::SymplecticHouseholderDecom)

Overwrite `B` with ``SB``, applying the reflectors of `Λ` from the last step to the first.
"""
function apply_S_left!(b::AbstractVector, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A) .÷ 2
    @assert length(b) == 2 * N
    for j in M:-1:1
        row_ind = vcat(j:N, (N + j):(2 * N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j + M]
        b[row_ind] .-= Λ.c₂[j] * symplectic_form(v₂, b[row_ind]) * v₂
        b[row_ind] .-= Λ.c₁[j] * symplectic_form(v₁, b[row_ind]) * v₁
    end
    b
end

function apply_S_left!(B::AbstractMatrix, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A) .÷ 2
    @assert size(B, 1) == 2 * N
    for j in M:-1:1
        row_ind = vcat(j:N, (N + j):(2 * N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j + M]
        for i in axes(B, 2)
            B[row_ind, i] .-= Λ.c₂[j] * symplectic_form(v₂, B[row_ind, i]) * v₂
            B[row_ind, i] .-= Λ.c₁[j] * symplectic_form(v₁, B[row_ind, i]) * v₁
        end
    end
    B
end

@doc raw"""
    apply_S_inverse_left!(B, Λ::SymplecticHouseholderDecom)

Overwrite `B` with ``S^{-1}B``. Three things invert [`apply_S_left!`](@ref) together: the steps run
from the first to the last rather than the last to the first, the two reflectors within a step are
applied in the opposite order, and each update adds where the forward one subtracts.
"""
function apply_S_inverse_left!(b::AbstractVector, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A) .÷ 2
    @assert length(b) == 2 * N
    for j in 1:M
        row_ind = vcat(j:N, (N + j):(2 * N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j + M]
        b[row_ind] .+= Λ.c₁[j] * symplectic_form(v₁, b[row_ind]) * v₁
        b[row_ind] .+= Λ.c₂[j] * symplectic_form(v₂, b[row_ind]) * v₂
    end
    b
end

function apply_S_inverse_left!(B::AbstractMatrix, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A) .÷ 2
    @assert size(B, 1) == 2 * N
    for j in 1:M
        row_ind = vcat(j:N, (N + j):(2 * N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j + M]
        for i in axes(B, 2)
            B[row_ind, i] .+= Λ.c₁[j] * symplectic_form(v₁, B[row_ind, i]) * v₁
            B[row_ind, i] .+= Λ.c₂[j] * symplectic_form(v₂, B[row_ind, i]) * v₂
        end
    end
    B
end

@doc raw"""
    apply_S_right!(B, Λ::SymplecticHouseholderDecom)

Overwrite `B` with ``BS``. The reflections act on the rows of `B`, so this is written out rather
than expressed through [`apply_S_left!`](@ref).
"""
function apply_S_right!(B::AbstractMatrix, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A) .÷ 2
    @assert size(B, 2) == 2 * N
    for j in 1:M
        row_ind = vcat(j:N, (N + j):(2 * N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j + M]
        for i in axes(B, 1)
            @views b = B[i, row_ind]
            fac₁ = -Λ.c₁[j] * transpose(b) * v₁
            b[1:(N + 1 - j)] .-= fac₁ * v₁[(N + 2 - j):(2 * N + 2 - 2 * j)]
            b[(N + 2 - j):(2 * N + 2 - 2 * j)] .+= fac₁ * v₁[1:(N + 1 - j)]
            fac₂ = -Λ.c₂[j] * transpose(b) * v₂
            b[1:(N + 1 - j)] .-= fac₂ * v₂[(N + 2 - j):(2 * N + 2 - 2 * j)]
            b[(N + 2 - j):(2 * N + 2 - 2 * j)] .+= fac₂ * v₂[1:(N + 1 - j)]
        end
    end
    B
end

@doc raw"""
    apply_S_inverse_right!(B, Λ::SymplecticHouseholderDecom)

Overwrite `B` with ``BS^{-1}``. Three things invert [`apply_S_right!`](@ref) together, for the
reason given at [`apply_S_inverse_left!`](@ref): the steps run from the last to the first, the two
reflectors within a step are applied in the opposite order, and each factor is inverted, which for
``I + cvv^J`` means negating ``c`` because ``v^Jv = 0``.
"""
function apply_S_inverse_right!(B::AbstractMatrix, Λ::SymplecticHouseholderDecom)
    N, M = size(Λ.A) .÷ 2
    @assert size(B, 2) == 2 * N
    for j in M:-1:1
        row_ind = vcat(j:N, (N + j):(2 * N))
        @views v₁ = Λ.A[row_ind, j]
        @views v₂ = Λ.A[row_ind, j + M]
        for i in axes(B, 1)
            @views b = B[i, row_ind]
            fac₂ = Λ.c₂[j] * transpose(b) * v₂
            b[1:(N + 1 - j)] .-= fac₂ * v₂[(N + 2 - j):(2 * N + 2 - 2 * j)]
            b[(N + 2 - j):(2 * N + 2 - 2 * j)] .+= fac₂ * v₂[1:(N + 1 - j)]
            fac₁ = Λ.c₁[j] * transpose(b) * v₁
            b[1:(N + 1 - j)] .-= fac₁ * v₁[(N + 2 - j):(2 * N + 2 - 2 * j)]
            b[(N + 2 - j):(2 * N + 2 - 2 * j)] .+= fac₁ * v₁[1:(N + 1 - j)]
        end
    end
    B
end

apply_S_left(Λ::SymplecticHouseholderDecom, B::AbstractArray) = apply_S_left!(copy(B), Λ)

function apply_S_inverse_left(Λ::SymplecticHouseholderDecom, B::AbstractArray)
    apply_S_inverse_left!(copy(B), Λ)
end

apply_S_right(B::AbstractArray, Λ::SymplecticHouseholderDecom) = apply_S_right!(copy(B), Λ)

function apply_S_inverse_right(B::AbstractArray, Λ::SymplecticHouseholderDecom)
    apply_S_inverse_right!(copy(B), Λ)
end

@doc raw"""
    symplectic_form(a, b)

The canonical symplectic form ``a^TJb`` of two vectors of even length, evaluated without building
``J``.
"""
@views function symplectic_form(a::AbstractVector, b::AbstractVector)
    N2 = length(a)
    @assert iseven(N2)
    @assert length(b) == N2
    N = N2 ÷ 2
    -transpose(a[(N + 1):(2 * N)]) * b[1:N] + transpose(a[1:N]) * b[(N + 1):(2 * N)]
end

@doc raw"""
    symplectic_householder!(a, b)

Build one step of [`sr!`](@ref) in place, and return `(c₁, c₂, ρ, ν, μ)`.

The step is a pair of symplectic reflections that together map the two columns `a` and `b` onto
the shape ``R`` needs. `a` and `b` are overwritten with the reflector vectors.
"""
function symplectic_householder!(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
    N2 = length(a)
    @assert iseven(N2)
    @assert length(b) == N2
    N = N2 ÷ 2
    ρ = sign(a[1]) * LinearAlgebra.norm(a)
    c₁ = 1 / (ρ * a[N + 1])
    a[1] -= ρ
    if N == 1
        μν = b + c₁ * symplectic_form(a, b) * a
        b[1] = zero(T)
        b[2] = zero(T)
        return c₁, zero(T), ρ, μν[2], μν[1]
    end
    b .+= c₁ * symplectic_form(a, b) * a
    ν = b[N + 1]
    ξ = sqrt(LinearAlgebra.norm(b)^2 - b[1]^2 - ν^2)
    s = one(T)
    μ = b[1] + s * ξ
    c₂ = s / (ξ * ν)
    b[1] = -s * ξ
    b[N + 1] = zero(T)
    c₁, c₂, ρ, ν, μ
end

@doc raw"""
    symplectic_householder(a, b)

[`symplectic_householder!`](@ref) on copies, returning the five scalars followed by the two
reflector vectors.
"""
function symplectic_householder(a::AbstractVector, b::AbstractVector)
    a_copy, b_copy = copy(a), copy(b)
    symplectic_householder!(a_copy, b_copy)..., a_copy, b_copy
end
