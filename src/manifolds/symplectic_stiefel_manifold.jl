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

# The product kernels. `src/ambiguities.jl` has the `*` methods that reach them.
function _lmul(U::SymplecticStiefelManifold, B::AbstractMatrix)
    _check_same_backend(U, B)
    U.A * B
end
function _rmul(B::AbstractMatrix, U::SymplecticStiefelManifold)
    _check_same_backend(U, B)
    B * U.A
end

# `U'` is where this type's own operations go: `rgrad` and `metric` form `U'U`, and `check` is
# `U'JU`. Without a kernel for the adjoint it keeps its wrapper into `LinearAlgebra`'s generic
# product, which reads the point one entry at a time — scalar indexing, which a device array does
# not serve. `LinearAlgebra` rewrites an `Adjoint` of a real matrix to a `Transpose` on its way into
# `mul!`, so that failure reports a `Transpose`.
function _lmul(U::Adjoint{T, SymplecticStiefelManifold{T, AT}},
        B::AbstractMatrix) where {T, AT <: AbstractMatrix{T}}
    _check_same_backend(parent(U), B)
    U.parent.A' * B
end

# The mirror, which [`metric`](@ref) needs: it forms `J'·U·inv(U'U)·U'·J`, so the adjoint appears on
# the right of a product as well as on the left.
function _rmul(B::AbstractMatrix,
        U::Adjoint{T, SymplecticStiefelManifold{T, AT}}) where {T, AT <: AbstractMatrix{T}}
    _check_same_backend(parent(U), B)
    B * U.parent.A'
end

# Writes the two off-diagonal blocks of the Poisson tensor. A kernel is what it takes to write them
# without scalar indexing, for the reason `write_ones_kernel!` and `unit_matrix` give one level up.
@kernel function write_poisson_blocks_kernel!(J::AbstractMatrix{T}, n) where {T}
    i = @index(Global)
    J[i, i + n] = one(T)
    J[i + n, i] = -one(T)
end

# The canonical Poisson tensor, as a dense matrix and nothing more. `GeometricMachineLearning`
# exports a `PoissonTensor` type that carries phase-space `(q, p)` methods on top of this; that
# type belongs with the phase-space machinery it serves, so the plain form is rebuilt here rather
# than depended upon. Unifying the two is a separate change, and it is a breaking one downstream.
#
# Every caller here passes the point it is building the tensor for, because the tensor has to land
# where the point already is: it is multiplied straight into `U`, and a host matrix against a device
# one reaches the generic product and scalar-indexes. The size is a separate argument because
# [`check`](@ref) needs both `2N` and `2n` for the same point.
function _poisson_tensor(U::AbstractMatrix{T}, n2::Integer) where {T}
    _poisson_tensor(KernelAbstractions.get_backend(U), T, n2)
end

# The host spelling is a loop over a `zeros(T, n2, n2)`, with no backend and no kernel launch.
# Routing it through `KernelAbstractions.zeros` and a kernel launch would make the common case pay
# for the device machinery, and the measurement behind that is the comment on
# `zeros(::Type{AT}, n)` in `special_matrices/triangular.jl`. `StiefelProjection` splits its
# constructor the same way, and this is the same split: the backendless form names the element type
# and places on the host, which is the third of the four call shapes in
# `docs/src/special_matrices.md`.
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

_poisson_tensor(::CPU, ::Type{T}, n2::Integer) where {T} = _poisson_tensor(T, n2)

function _poisson_tensor(
        backend::KernelAbstractions.Backend, ::Type{T}, n2::Integer) where {T}
    @assert iseven(n2)
    n = n2 ÷ 2
    _check_supported_eltype(backend, T)
    J = KernelAbstractions.zeros(backend, T, n2, n2)
    write_poisson_blocks! = write_poisson_blocks_kernel!(backend)
    write_poisson_blocks!(J, n; ndrange = n)

    J
end

@doc raw"""
    rand(::Type{SymplecticStiefelManifold{T}}, N2, n2)

Draw a random point of the ``2N\times2n`` symplectic Stiefel manifold.

The draw is a Gaussian matrix put through the symplectic SR decomposition [`sr!`](@ref); the point
is `n` columns from each half of the symplectic factor. This is the symplectic counterpart of the
orthonormalization `rand(::Type{StiefelManifold}, …)` does, and no orthonormalization will serve
here: an orthonormal factor preserves the Euclidean form, not ``\mathbb{J}``.
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
# `abstract_manifold.jl`. That draw orthonormalizes, which preserves the Euclidean form and not
# ``\mathbb{J}``, and the inner constructor above asserts shape alone -- so the generic
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
    J = _poisson_tensor(U, size(U, 1))
    ∇L * (U' * U) + J * U * (∇L' * J * U)
end

@doc raw"""
    metric(U::SymplecticStiefelManifold, Δ₁::AbstractMatrix, Δ₂::AbstractMatrix)

The Riemannian metric of the symplectic Stiefel manifold, taken from
[gao2021riemannian](@cite):

```math
g_U(\Delta_1, \Delta_2) = \mathrm{tr}\left( P \Delta_1^T \left(\mathbb{I}_{2N} - \frac{1}{2}\mathbb{J}_{2N}^TUPU^T\mathbb{J}_{2N}\right)\Delta_2 \right), \qquad P = (U^TU)^{-1}.
```

# Implementation

Every factor the trace needs is ``2n\times{}2n``, and the expression is evaluated that way:

```math
g_U(\Delta_1, \Delta_2) = \mathrm{tr}\left( P \left( \Delta_1^T\Delta_2 - \frac{1}{2} \left(\Delta_1^T\mathbb{J}_{2N}^TU\right) P \left(U^T\mathbb{J}_{2N}\Delta_2\right) \right) \right).
```

Written as the definition reads, the middle factor is a ``2N\times{}2N`` matrix, and forming it
costs ``O(N^3)`` — a matrix of the ambient dimension to reach a number, on every step the optimizer
takes. Grouped this way no product has two factors of the ambient dimension; the largest is
``\mathbb{J}_{2N}^TU``, and it is formed once, because ``U^T\mathbb{J}_{2N}`` is its adjoint. The
``2N\times{}2N`` identity is not built at all, and ``P`` is inverted once rather than twice.
``\mathbb{J}_{2N}`` is still assembled dense by `_poisson_tensor`, and it is what a call allocates
most of at the larger sizes.
`scripts/symplectic_metric_cost.jl` carries the measurement.
"""
function metric(U::SymplecticStiefelManifold{T}, Δ₁::AbstractMatrix,
        Δ₂::AbstractMatrix) where {T}
    J = _poisson_tensor(U, size(U, 1))
    P = inv(U' * U)

    # `X'` and not a second product: `U'J` is `(J'U)'` entry for entry, conjugation included, so
    # forming both spends one of the two `O(N^2n)` products this expression has left on a matrix
    # already in hand. Measured bitwise equal on `Float64` and on `ComplexF64`.
    X = J' * U
    LinearAlgebra.tr(P * (Δ₁' * Δ₂ - (T(1) / 2) * (Δ₁' * X) * P * (X' * Δ₂)))
end

@doc raw"""
    check(U::SymplecticStiefelManifold)

How far `U` is from the manifold, as ``\|U^T\mathbb{J}_{2N}U - \mathbb{J}_{2n}\|``.

This replaces the generic [`check(::Manifold)`](@ref), whose residual is the orthonormality one.
"""
function check(U::SymplecticStiefelManifold)
    LinearAlgebra.norm(U' * _poisson_tensor(U, size(U, 1)) * U -
                       _poisson_tensor(U, size(U, 2)))
end

@doc raw"""
    global_section(U::SymplecticStiefelManifold)

A symplectic completion of `U`: the remaining ``2N - 2n`` directions, drawn at random and made
symplectic against `U`. The result is ``2N\times(2N - 2n)``, as
[`global_section(::StiefelManifold)`](@ref) is ``N\times(N - n)``, and it is what `GlobalSection`
expects.

The counterpart of that method, with the symplectic form in place of the Euclidean one.

# This one is host-only, and the other three of this type's operations are not

[`rgrad`](@ref), [`metric`](@ref) and [`check`](@ref) run wherever the point is — [`metric`](@ref)
as far as the backend supplies an `lu`, since it forms ``\mathrm{inv}(U^TU)``. This does not, and
the reason is the same one that makes `rand(::GPU, ::Type{<:SymplecticStiefelManifold}, …)` refuse:
the completion is orthogonalized by the symplectic SR decomposition [`sr!`](@ref), which is a host
factorization — `_rand_symplectic_stiefel` calls `Matrix` on its factor. A device spelling would be
a host computation with two transfers around it, which is a different operation from the
device-native section [`global_section(::StiefelManifold)`](@ref) gives.

A device-backed point is therefore refused with an `ArgumentError` that says so, rather than left
to fail inside `sr!` with `Cannot access the contents of a private buffer`, which names neither the
manifold nor the call.
"""
function global_section(U::SymplecticStiefelManifold)
    backend = KernelAbstractions.get_backend(U)
    backend isa GPU &&
        throw(ArgumentError("global_section is host-only for a SymplecticStiefelManifold: the symplectic SR decomposition runs on the host, so a section of a $(backend) point would be a host computation with transfers around it. Move the point to the host first. rgrad and check do run on $(backend), and metric does wherever the backend supplies an lu."))

    N2, n2 = size(U)
    N, n = N2 ÷ 2, n2 ÷ 2
    m = N - n
    A = randn(eltype(U), N2, N2 - n2)
    J₁ = _poisson_tensor(U, N2)
    J₂ = _poisson_tensor(U, n2)
    A -= U * J₂ * U' * J₁' * A
    # `sr!` returns the full `2N x 2N` symplectic factor; the completion is `m` of its columns from
    # each half, the same slice `_rand_symplectic_stiefel` takes. Returning the whole factor would
    # not be a completion at all -- `‖UᵀJΛ‖` over it is `O(100)` rather than zero.
    Matrix(sr!(A).S)[:, vcat(1:m, (N + 1):(N + m))]
end
