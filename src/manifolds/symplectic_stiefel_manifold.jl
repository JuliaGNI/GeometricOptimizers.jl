@doc raw"""
    SymplecticStiefelManifold(A)

The symplectic Stiefel manifold, the set of ``2N\times2n`` matrices whose columns span a symplectic
subspace:

```math
Sp(2n, 2N) = \{U \in \mathbb{R}^{2N\times2n} : U^T\mathbb{J}_{2N}U = \mathbb{J}_{2n}\},
```

with ``\mathbb{J}`` the canonical Poisson tensor. Compare [`StiefelManifold`](@ref), whose
constraint is ``Y^TY = \mathbb{I}``: the two agree in form and differ in which bilinear form the
columns preserve.

The metric is the one of [gao2021riemannian](@cite), and `rand` builds a point through the
symplectic SR decomposition [`sr!`](@ref), which is the construction
[gao2024optimization](@cite) uses for its retraction. See also [bendokat2021real](@cite).

!!! warning "The accuracy degrades with the size, and `Float32` is out of reach"
    A point is only as good as the decomposition it came from, and that decomposition has no
    re-orthogonalization step. The residual `check` reports has a median of `1.3e-14` at `6x4` in
    `Float64` and `6.2e-8` at `40x20`; in `Float32` the median is already `0.014` at `20x10`, and
    at `40x20` some draws return a non-finite residual or throw. Only the medians reproduce: the
    maxima move by orders of magnitude with the draw order, so no particular worst case is quoted
    here. Nothing warns when a draw comes back far from the manifold. `CHANGELOG.md` carries the
    full table and what closing it would take.

# Examples

```jldoctest
using GeometricOptimizers
import Random

Random.seed!(1234)

check(rand(SymplecticStiefelManifold, 6, 4)) < 1e-5

# output

true
```
"""
mutable struct SymplecticStiefelManifold{T, AT <: AbstractMatrix{T}} <: Manifold{T}
    A::AT
    function SymplecticStiefelManifold(A::AbstractMatrix)
        @assert iseven(size(A, 1))
        @assert iseven(size(A, 2))
        @assert size(A, 1) ≥ size(A, 2)
        new{eltype(A), typeof(A)}(A)
    end
end

# The inner constructor above suppresses the default `SymplecticStiefelManifold{T, AT}(A)`, which
# the generic `Base.copy(::Manifold)` calls — and through it `_similar` and `GlobalSection`, which
# every manifold optimizer goes through. `StiefelManifold` keeps the default because it declares no
# inner constructor.
function SymplecticStiefelManifold{T, AT}(A::AT) where {T, AT <: AbstractMatrix{T}}
    SymplecticStiefelManifold(A)
end

Base.:*(U::SymplecticStiefelManifold, B::AbstractMatrix) = U.A * B
Base.:*(B::AbstractMatrix, U::SymplecticStiefelManifold) = B * U.A

# The canonical Poisson tensor, as a dense matrix and nothing more. `GeometricMachineLearning`
# exports a `PoissonTensor` type that carries phase-space `(q, p)` methods on top of this; that
# type belongs with the phase-space machinery it serves, so the plain form is rebuilt here rather
# than depended upon. Unifying the two is a separate change, and it is a breaking one downstream.
function _poisson_tensor(::Type{T}, n2::Integer) where {T}
    @assert iseven(n2)
    n = n2 ÷ 2
    J = zeros(T, n2, n2)
    for i in 1:n
        J[i, i + n] = one(T)
        J[i + n, i] = -one(T)
    end
    J
end

@doc raw"""
    rand(::Type{SymplecticStiefelManifold{T}}, N2, n2)

Draw a random point of the ``2N\times2n`` symplectic Stiefel manifold.

The draw is a Gaussian matrix put through the symplectic SR decomposition [`sr!`](@ref); the point
is `n` columns from each half of the symplectic factor. This is the symplectic counterpart of what
`rand(::Type{StiefelManifold}, …)` does with `qr`, and `qr` will not serve here: its `Q` preserves
the Euclidean form, not ``\mathbb{J}``.
"""
function Base.rand(rng::Random.AbstractRNG, ::Type{SymplecticStiefelManifold{T}},
        N2::Integer, n2::Integer) where {T}
    _rand_symplectic_stiefel(randn(rng, T, N2, n2), N2, n2)
end

function Base.rand(rng::Random.AbstractRNG, ::Type{SymplecticStiefelManifold},
        N2::Integer, n2::Integer)
    _rand_symplectic_stiefel(randn(rng, N2, n2), N2, n2)
end

function Base.rand(::Type{SymplecticStiefelManifold{T}}, N2::Integer, n2::Integer) where {T}
    rand(Random.default_rng(), SymplecticStiefelManifold{T}, N2, n2)
end

function Base.rand(::Type{SymplecticStiefelManifold}, N2::Integer, n2::Integer)
    rand(Random.default_rng(), SymplecticStiefelManifold, N2, n2)
end

# The backend-taking spellings belong to this type and not to the generic `Manifold{T}` draw in
# `abstract_manifold.jl`. That draw orthonormalises with `qr`, which preserves the Euclidean form
# and not ``\mathbb{J}``, and the inner constructor above asserts shape alone -- so the generic
# method wraps a matrix that is not on this manifold and nothing downstream says so. Both methods
# mirror the generic pair's signature, `MT <: SymplecticStiefelManifold{T}` rather than
# `MT <: Manifold{T}`, so a bare `SymplecticStiefelManifold` still picks up `default_eltype` first
# and arrives here parametrized.
function Base.rand(::CPU, rng::Random.AbstractRNG, ::Type{MT},
        N2::Integer, n2::Integer) where {T, MT <: SymplecticStiefelManifold{T}}
    rand(rng, SymplecticStiefelManifold{T}, N2, n2)
end

# `sr!` is a host factorization: `_rand_symplectic_stiefel` calls `Matrix` on its factor. A device
# draw is therefore a host draw and a transfer, which is a different operation from the device-native
# one the other two manifolds offer. Refusing says so; returning a host-drawn point from a call that
# named a device would not.
function Base.rand(backend::GPU, ::Random.AbstractRNG, ::Type{MT},
        N2::Integer, n2::Integer) where {T, MT <: SymplecticStiefelManifold{T}}
    throw(ArgumentError("$(backend) cannot draw a SymplecticStiefelManifold: the symplectic SR decomposition runs on the host. Draw on the host with rand(SymplecticStiefelManifold{$(T)}, $(N2), $(n2))."))
end

function _rand_symplectic_stiefel(A::AbstractMatrix, N2::Integer, n2::Integer)
    @assert N2 ≥ n2
    N, n = N2 ÷ 2, n2 ÷ 2
    # `Matrix` once rather than slicing the operator: an entry of `Sfac` costs a whole matrix.
    S = Matrix(sr!(A).S)
    SymplecticStiefelManifold(S[1:N2, vcat(1:n, (N + 1):(N + n))])
end

@doc raw"""
    rgrad(U::SymplecticStiefelManifold, ∇L::AbstractMatrix)

The Riemannian gradient of a point of the symplectic Stiefel manifold, for the metric of
[`metric(::SymplecticStiefelManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref).

``\nabla{}L`` is the Euclidean gradient, i.e. the derivative of the loss with respect to the
entries of `U` read as an unconstrained matrix.
"""
function rgrad(U::SymplecticStiefelManifold, ∇L::AbstractMatrix)
    J = _poisson_tensor(eltype(U), size(U, 1))
    ∇L * (U' * U) + J * U * (∇L' * J * U)
end

@doc raw"""
    metric(U::SymplecticStiefelManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)

The Riemannian metric of the symplectic Stiefel manifold, taken from
[gao2021riemannian](@cite).
"""
function metric(U::SymplecticStiefelManifold{T}, Δ₁::AbstractMatrix,
        Δ₂::AbstractMatrix) where {T}
    J = _poisson_tensor(T, size(U, 1))
    LinearAlgebra.tr(inv(U' * U) * Δ₁' *
                     (LinearAlgebra.I - (T(1) / 2) * J' * U * inv(U' * U) * U' * J) * Δ₂)
end

@doc raw"""
    check(U::SymplecticStiefelManifold)

How far `U` is from the manifold, as ``\|U^T\mathbb{J}_{2N}U - \mathbb{J}_{2n}\|``.

This replaces the generic [`check(::Manifold)`](@ref), whose residual is the orthonormality one.
"""
function check(U::SymplecticStiefelManifold)
    T = eltype(U)
    LinearAlgebra.norm(U' * _poisson_tensor(T, size(U, 1)) * U -
                       _poisson_tensor(T, size(U, 2)))
end

@doc raw"""
    global_section(U::SymplecticStiefelManifold)

A symplectic completion of `U`: the remaining ``2N - 2n`` directions, drawn at random and made
symplectic against `U`. The result is ``2N\times(2N - 2n)``, as
[`global_section(::StiefelManifold)`](@ref) is ``N\times(N - n)``, and it is what `GlobalSection`
expects.

The counterpart of that method, with the symplectic form in place of the Euclidean one.
"""
function global_section(U::SymplecticStiefelManifold)
    N2, n2 = size(U)
    N, n = N2 ÷ 2, n2 ÷ 2
    m = N - n
    A = randn(eltype(U), N2, N2 - n2)
    J₁ = _poisson_tensor(eltype(U), N2)
    J₂ = _poisson_tensor(eltype(U), n2)
    A -= U * J₂ * U' * J₁' * A
    # `sr!` returns the full `2N x 2N` symplectic factor; the completion is `m` of its columns from
    # each half, the same slice `_rand_symplectic_stiefel` takes. Returning the whole factor would
    # not be a completion at all -- `‖UᵀJΛ‖` over it is `O(100)` rather than zero.
    Matrix(sr!(A).S)[:, vcat(1:m, (N + 1):(N + m))]
end
