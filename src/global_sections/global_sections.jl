@doc raw"""
    GlobalSection(Y)

Construct a global section for `Y`.

A global section ``\lambda`` is a mapping from a homogeneous space ``\mathcal{M}`` to the corresponding Lie group ``G`` such that

```math
\lambda(Y)E = Y,
```

Also see [`apply_section`](@ref) and [`global_rep`](@ref).

# Implementation

For an implementation of `GlobalSection` for a custom array (especially manifolds), the function [`global_section`](@ref) has to be generalized.
"""
struct GlobalSection{T, AT <: AbstractArray{T}, λT <: Union{AbstractArray{T}, Nothing}}
    Y::AT
    # the complement `Y⟂` of a manifold point, and `nothing` for a vector-space parameter
    λ::λT

    function GlobalSection(Y::AbstractVecOrMat)
        λ = global_section(Y)
        new{eltype(Y), typeof(Y), typeof(λ)}(copy(Y), λ)
    end
end

# `mapparameters` and not `map`: a container is a tree of layers, so its leaves are more than one
# level down and `map` would hand `GlobalSection` a whole layer. The plain `NamedTuple` method
# delegates here too, so both shapes take the same walk.
GlobalSection(ps::NamedTuple) = mapparameters(GlobalSection, ps)

# The container `NeuralNetworkParameters` holds a network's parameters in. `GlobalSection` is this
# package's own function, so a method on it is this package's to write -- and it has to be, because a
# package that merely uses both owns neither name and its method would be piracy.
#
# The result is deliberately the same `NamedTuple` tree the method above returns rather than a
# `NetworkParameters` of sections: a section is not a parameter, and everything downstream --
# `update_section!`, `apply_section`, `global_rep`, the four `copyto!` methods -- walks it as a plain
# container.
GlobalSection(ps::NetworkParameters) = GlobalSection(params(ps))

# The `NamedTuple` method above returns whatever shape `mapparameters` rebuilt, which for a section
# tree is a plain `NamedTuple` — so `apply_section`, `global_rep` and `update_section!` below take a
# `NamedTuple` on the section side and a [`NeuralNetworkParameters.NetworkParameters`](@extref) on the parameter side. Each is
# written with the section first, since `mapparameters` dispatches on its first argument and
# normalises the rest.

Base.size(λY::GlobalSection) = (size(λY.Y, 1), size(λY.Y, 2) + size(λY.λ, 2))

@doc raw"""
    Matrix(λY::GlobalSection)

Put `λY` into matrix form.

This is not recommended if speed is important!

Use [`apply_section`](@ref) and [`global_rep`](@ref) instead!
"""
function Base.Matrix(λY::GlobalSection)
    hcat(Matrix(λY.Y), Matrix(λY.λ))
end

@doc raw"""
    λY * Y

Apply the element `λY` onto `Y`.

Here `λY` is an element of a Lie group and `Y` is an element of a homogeneous space.
"""
Base.:*(λY::GlobalSection, Y::Manifold) = apply_section(λY, Y)

@doc raw"""
    apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT <: StiefelManifold{T}}

Apply `λY` to `Y₂`.

Mathematically this is the group action of the element ``\lambda{}Y\in{}G`` on the element ``Y_2`` of the homogeneous space ``\mathcal{M}``.

Internally it calls [`apply_section!`](@ref).
"""
function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT <: StiefelManifold{T}}
    Y = StiefelManifold(zero(Y₂.A))
    apply_section!(Y, λY, Y₂)

    Y
end

# The lift is bound to an array, as `global_section(::StiefelManifold)` returns one; this keeps the
# method disjoint from the `λ === nothing` one below. The same holds for the Grassmann method.
@doc raw"""
    apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:StiefelManifold{T}}

Apply `λY` to `Y₂` and store the result in `Y`.

This is the inplace version of [`apply_section`](@ref).
"""
function apply_section!(Y::AT, λY::GlobalSection{T, AT, <:AbstractArray},
        Y₂::MT) where {T, AT <: StiefelManifold{T}, MT <: StiefelManifold{T}}
    _apply_section!(Y.A, λY, Y₂.A)

    Y
end

# The product `λY.Y Y₂[1:n, :] + λY.λ Y₂[n+1:N, :]` into `A`, shared by the Stiefel and the
# Grassmann method.
#
# Two `mul!`s into `A` where it shares no memory with an input, which allocates nothing. Where it
# does, the two products are materialised before the broadcast writes `A`: `update_section!` without
# a workspace passes `Y === Y₂`, and an in-place product would read the rows of `Y₂` it has already
# overwritten. The second `mul!` adds its product into the first, so the two forms round differently
# and agree to `eps` rather than to the bit.
function _apply_section!(A::AbstractMatrix{T}, λY::GlobalSection, A₂::AbstractMatrix{T}) where {T}
    N, n = size(λY.Y)

    if Base.mightalias(A, A₂) || Base.mightalias(A, λY.Y.A) || Base.mightalias(A, λY.λ)
        @views A .= λY.Y * A₂[1:n, :] .+ λY.λ * A₂[(n + 1):N, :]
    else
        @views begin
            mul!(A, λY.Y.A, A₂[1:n, :])
            mul!(A, λY.λ, A₂[(n + 1):N, :], one(T), one(T))
        end
    end

    A
end

function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {
        T, AT <: GrassmannManifold{T}}
    Y = GrassmannManifold(zero(Y₂.A))
    apply_section!(Y, λY, Y₂)

    Y
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT, <:AbstractArray},
        Y₂::MT) where {T, AT <: GrassmannManifold{T}, MT <: GrassmannManifold{T}}
    _apply_section!(Y.A, λY, Y₂.A)

    Y
end

function apply_section(
        λY::GlobalSection{
            T, AT}, Y₂::AT) where {
        T, AT <: SymplecticStiefelManifold{T}}
    Y = SymplecticStiefelManifold(zero(Y₂.A))
    apply_section!(Y, λY, Y₂)

    Y
end

# The product is the Stiefel one: the section is stored as `[U Uˢ]`, the point first, so `λ(U)E` is
# its first `n` columns here as well. See `SymplecticLieAlgHorMatrix` for the basis this implies.
function apply_section!(Y::AT,
        λY::GlobalSection{T, AT, <:AbstractArray},
        Y₂::MT) where {
        T, AT <: SymplecticStiefelManifold{T}, MT <: SymplecticStiefelManifold{T}}
    _apply_section!(Y.A, λY, Y₂.A)

    Y
end

function apply_section(λY::GlobalSection{T}, Y₂::AbstractVecOrMat{T}) where {T}
    Y = copy(Y₂)
    apply_section!(Y, λY, Y₂)
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT, Nothing},
        Y₂::AbstractVecOrMat{T}) where {T, AT <: AbstractVecOrMat{T}}
    Y .= Y₂ .+ λY.Y
end

function apply_section(λY::NamedTuple, Y₂::NamedTuple)
    mapparameters(apply_section, λY, Y₂)
end

function apply_section!(Y::NetworkParameters, λY::NamedTuple, Y₂)
    mapparameters!(apply_section!, Y, λY, Y₂)
end

function global_rep(λY::NamedTuple, gx::NamedTuple)
    mapparameters(global_rep, λY, gx)
end

# The container versions of the two that *build* a tree, and the reason they are written with their
# arguments the other way round.
#
# `mapparameters` rebuilds in the shape of its **first** argument, so `mapparameters(f, λY, gx)` would
# hand back a plain `NamedTuple` — the section tree's shape — for a container `gx`. What these produce
# is a point and a gradient, i.e. *parameters*, and a parameter set has the parameters' shape. Getting
# this backwards is not a cosmetic matter: the gradient would come out a plain nested `NamedTuple`,
# which is not a container, so no alias in this package covers it and
# `_copyto!(gradient_array(cache), ·)` has no method for it.
#
# The section stays the plain tree `GlobalSection(::NetworkParameters)` returns; it is passed through
# as the second argument and `mapparameters` normalises it.
function apply_section(λY::NamedTuple, Y₂::NetworkParameters)
    mapparameters((y, λ) -> apply_section(λ, y), Y₂, λY)
end

function global_rep(λY::NamedTuple, gx::NetworkParameters)
    mapparameters((g, λ) -> global_rep(λ, g), gx, λY)
end

# The in-place form of the container versions, which writes the lift set into a destination that
# already has its shape. `mapparameters!` pairs the trees by key and reaches the leaves at any depth,
# where a manifold leaf takes the lift method of `global_rep!` and an ordinary array the copy below.
function global_rep!(D::Union{NamedTuple, NetworkParameters}, λY::NamedTuple,
        gx::Union{NamedTuple, NetworkParameters})
    mapparameters!(global_rep!, D, λY, gx)
    D
end

# The vector-space form: the section's lift is `nothing`, so the map is the identity and the write
# is a copy. `store_gradient!` reaches it for a plain-vector solution, and a parameter set reaches it
# for each of its vector-space leaves.
#
# It is disjoint from the lift methods below on the section, whose `λ` is `nothing` here and an
# array beside a manifold point.
function global_rep!(
        D::AbstractArray, λY::GlobalSection{T, <:AbstractVecOrMat{T}, Nothing},
        gx::AbstractVecOrMat{T}) where {T}
    (copyto!(D, gx); D)
end

function global_rep(::GlobalSection{T}, gx::AbstractVecOrMat{T}) where {T}
    gx
end

@doc raw"""
    global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:StiefelManifold{T}}

Express `Δ` (an the tangent space of `Y`) as an instance of `StiefelLieAlgHorMatrix`.

This maps an element from ``T_Y\mathcal{M}`` to an element of ``\mathfrak{g}^\mathrm{hor}``.

These two spaces are isomorphic where the isomorphism where the isomorphism is established through ``\lambda(Y)\in{}G`` via:

```math
T_Y\mathcal{M} \to \mathfrak{g}^{\mathrm{hor}}, \Delta \mapsto \lambda(Y)^{-1}\Omega(Y, \Delta)\lambda(Y).
```

Also see [`GeometricOptimizers.Ω`](@ref).

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: _round
import Random

Random.seed!(123)

Y = rand(StiefelManifold, 6, 3)
Δ = rgrad(Y, randn(6, 3))
λY = GlobalSection(Y)

_round(global_rep(λY, Δ); digits = 3)

# output

6×6 StiefelLieAlgHorMatrix{Float64, SkewSymMatrix{Float64, Vector{Float64}}, Matrix{Float64}}:
  0.0    -0.525  1.786   0.981   2.058  -0.4
  0.525   0.0    0.298  -0.424  -0.733   0.919
 -1.786  -0.298  0.0    -1.815  -1.409  -1.085
 -0.981   0.424  1.815   0.0     0.0     0.0
 -2.058   0.733  1.409   0.0     0.0     0.0
  0.4    -0.919  1.085   0.0     0.0     0.0
```

# Implementation

The function `global_rep` does in fact not perform the entire map ``\lambda(Y)^{-1}\Omega(Y, \Delta)\lambda(Y)`` but only

```math
\Delta \mapsto \mathrm{skew}(Y^T\Delta),
```

to get the small skew-symmetric matrix ``A\in\mathcal{S}_\mathrm{skew}(n)`` and

```math
\Delta \mapsto (\lambda(Y)_{[1:N, n:N]}^T \Delta)_{[1:(N-n), 1:n]},
```

to get the arbitrary matrix ``B\in\mathbb{R}^{(N-n)\times{}n}``.
"""
function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {
        T, AT <: StiefelManifold{T}}
    global_rep!(_zero_lift(StiefelLieAlgHorMatrix{T}, λY.Y), λY, Δ)
end

function _zero_lift(::Type{L}, Y::Manifold) where {L}
    zeros(KernelAbstractions.get_backend(parent(Y)), L, size(Y)...)
end

# `global_rep` written into a lift that already exists, which is the form a cache and the line
# search use, so a call allocates nothing beyond the `n × n` product the skew projection reads.
#
# `Δ` is the ambient gradient, which a gradient of this package produces as a manifold point over
# the same storage (`rgrad(Y, ·)` returns one), so it is read through its parent rather than through
# its `getindex` — a device array serves no scalar indexing, and the lifted product is a plain one.
#
# The `B` block is zeroed before the `mul!` because Metal's `mul!` scales the destination by `β = 0`
# rather than ignoring it, so a `NaN` the block holds, as a cache's lift does from its construction,
# would survive the product.
function global_rep!(D::AbstractLieAlgHorMatrix{T},
        λY::GlobalSection{T, <:Manifold, <:AbstractArray}, Δ::AbstractMatrix{T}) where {T}
    ∇L = _ambient(Δ)
    @assert (D.N, D.n) == size(λY.Y) == size(∇L)
    _lift_A!(D, λY.Y, ∇L)
    mul!(fill!(D.B, zero(T)), λY.λ', ∇L)
    D
end

_ambient(Δ::AbstractMatrix) = Δ
_ambient(Δ::Manifold) = parent(Δ)

# The `n × n` block: `skew(YᵀΔ)` for a Stiefel lift, and nothing for a Grassmann lift, whose `A`
# block is the zero matrix.
_lift_A!(D::StiefelLieAlgHorMatrix, Y, ∇L) = map_to_Skew!(D.A, Y.A' * ∇L)
_lift_A!(::GrassmannLieAlgHorMatrix, _, _) = nothing

@doc raw"""
    global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:GrassmannManifold{T}}

Express `Δ` (an element of the tangent space of `Y`) as an instance of [`GrassmannLieAlgHorMatrix`](@ref).

The method `global_rep` for [`GrassmannManifold`](@ref) is similar to that for [`StiefelManifold`](@ref).

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: _round
import Random

Random.seed!(123)

Y = rand(GrassmannManifold, 6, 3)
Δ = rgrad(Y, randn(6, 3))
λY = GlobalSection(Y)

_round(global_rep(λY, Δ); digits = 3)

# output

6×6 GrassmannLieAlgHorMatrix{Float64, Matrix{Float64}}:
  0.0     0.0    0.0     0.981   2.058  -0.4
  0.0     0.0    0.0    -0.424  -0.733   0.919
  0.0     0.0    0.0    -1.815  -1.409  -1.085
 -0.981   0.424  1.815   0.0     0.0     0.0
 -2.058   0.733  1.409   0.0     0.0     0.0
  0.4    -0.919  1.085   0.0     0.0     0.0
```
"""
function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {
        T, AT <: GrassmannManifold{T}}
    global_rep!(_zero_lift(GrassmannLieAlgHorMatrix{T}, λY.Y), λY, Δ)
end

@doc raw"""
    global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:SymplecticStiefelManifold{T}}

Express `Δ` (an element of the tangent space of `U`) as an instance of
[`SymplecticLieAlgHorMatrix`](@ref).

This is the map ``\Delta \mapsto \lambda(U)^{-1}\Omega(U, \Delta)\lambda(U)``, with ``\Omega``
the horizontal lift [`GeometricOptimizers.Ω`](@ref) of the symplectic Stiefel manifold, as for the
[`StiefelManifold`](@ref). A horizontal element is determined by its first ``2n`` columns, and those
are ``\lambda(U)^{-1}\Delta``, so the blocks are

```math
\mathcal{A} = U^+\Delta, \qquad \mathcal{B} = (U^s)^+\Delta,
```

with ``U^+ = \mathbb{J}_{2n}^TU^T\mathbb{J}_{2N}`` the symplectic inverse and ``\lambda(U) = [U\ \ U^s]``.
``\mathcal{A}`` is Hamiltonian for a tangent `Δ`; the Hamiltonian projection is taken all the
same, as [`global_rep`](@ref) takes the skew-symmetric one for the Stiefel manifold.

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: apply_section, StiefelProjection
import Random

Random.seed!(123)

U = rand(SymplecticStiefelManifold, 6, 4)
Δ = rgrad(U, randn(6, 4))
λU = GlobalSection(U)
B = global_rep(λU, Δ)

λU.Y * (B * StiefelProjection(6, 4))[1:4, :] + λU.λ * (B * StiefelProjection(6, 4))[5:6, :] ≈ Δ

# output

true
```
"""
function global_rep(λY::GlobalSection{T, AT},
        Δ::AbstractMatrix{T}) where {
        T, AT <: SymplecticStiefelManifold{T}}
    global_rep!(_zero_lift(SymplecticLieAlgHorMatrix{T}, λY.Y), λY, Δ)
end

# `𝕁X` and `𝕁ᵀX` for the Poisson tensor of the size of `X`'s rows, without building it: the two halves
# of the rows swap and one of them changes sign.
function _poisson_mul(X::AbstractMatrix)
    h = size(X, 1) ÷ 2
    @views vcat(X[(h + 1):(2h), :], .-X[1:h, :])
end

function _poisson_tmul(X::AbstractMatrix)
    h = size(X, 1) ÷ 2
    @views vcat(.-X[(h + 1):(2h), :], X[1:h, :])
end

function global_rep!(D::SymplecticLieAlgHorMatrix{T},
        λY::GlobalSection{T, <:SymplecticStiefelManifold, <:AbstractArray},
        Δ::AbstractMatrix{T}) where {T}
    ∇L = _ambient(Δ)
    @assert (D.N, D.n) == size(λY.Y) == size(∇L)
    JΔ = _poisson_mul(∇L)
    _hamiltonian_projection!(D, _poisson_tmul(λY.Y.A' * JΔ))
    D.B .= _poisson_tmul(λY.λ' * JΔ)
    D
end

@doc raw"""
    gradient_rep!(D, λY, Δ)
    gradient_rep!(D, λY, Δ, method)

Write into `D` the lift whose pairing with every lift ``B`` under [`_dot`](@ref) is the derivative
of the loss along ``\lambda(Y)BE``, i.e. the derivative of the loss with respect to the stored
entries of a lift at the section:

```math
\langle D, B\rangle = \langle\nabla{}L, \lambda(Y)BE\rangle_F \quad \forall{}B\in\mathfrak{g}^\mathrm{hor}.
```

`Δ` is the Riemannian gradient [`rgrad`](@ref), which is what a gradient of the optimizers returns.
This lift is what makes [`trial_slope`](@ref) the derivative of the line search's merit and a
descent check a statement about descent, and it is the gradient [`BFGS`](@ref) and [`DFP`](@ref)
hold: their secant pairs and their ``Q`` live in the same coordinates.

The four-argument form writes the gradient that `method` holds. For [`BFGS`](@ref), [`DFP`](@ref)
and every method without a `descent` it is the lift above. For [`GradientMethod`](@ref),
[`MomentumMethod`](@ref) and [`Adam`](@ref) it is the gradient of their `descent`:
[`global_rep`](@ref) of `Δ` for [`MetricDescent`](@ref), and for [`SectionDescent`](@ref) the
lift above with the [`SymmetricMatrix`](@ref) blocks of a symplectic lift rescaled to the inner
product ``\frac{1}{2}\mathrm{tr}(X^TY)`` of the dense lifts.

# The symplectic Stiefel manifold

For the [`StiefelManifold`](@ref) and the [`GrassmannManifold`](@ref) the lift above is
[`global_rep`](@ref)`(λY, Δ)` itself, because the inner product of [`_dot`](@ref) on a lift is the
canonical metric, and all of the forms above are the same. For the
[`SymplecticStiefelManifold`](@ref) they are not: the metric
[`metric(::SymplecticStiefelManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref) is not invariant
under the symplectic group, so no inner product on the lift reproduces it at every point.

The method recovers ``\langle\nabla{}L, \cdot\rangle_F = g_U(\Delta, \cdot)`` on the tangent space
from `Δ` and the metric, which is

```math
K = \Delta{}P - \frac{1}{2}\mathbb{J}_{2N}^TUP\,U^T\mathbb{J}_{2N}\Delta{}P, \qquad P = (U^TU)^{-1},
```

and writes the gradient of ``B \mapsto \langle{}K, U\mathcal{A} + U^s\mathcal{B}\rangle_F`` with
respect to the stored blocks of ``B``. With ``X = U^TK`` in four ``n\times{}n`` blocks, that is
``A_1 = X_{11} - X_{22}^T``, the storage gradients of ``X_{12}`` and ``X_{21}`` for the two
[`SymmetricMatrix`](@ref) blocks, and ``\mathcal{B} = (U^s)^TK = (U^s)^T\Delta{}P``. The metric enters
because the gradient of the optimizers is the Riemannian one; it is applied through a Cholesky
factor of ``U^TU`` and not through its inverse.

With [`MetricDescent`](@ref) the tangent vector ``\lambda(Y)DE`` of the held gradient is `Δ` for every
section, so at a fixed step size the path of [`GradientMethod`](@ref) and [`MomentumMethod`](@ref)
does not depend on the section ([`Adam`](@ref)'s entrywise second moments and coordinate step ceilings
still do). With
[`SectionDescent`](@ref) it depends on it: where the point is orthosymplectic and the section is
[`global_section`](@ref)'s, the inner product ``\frac{1}{2}\mathrm{tr}(X^TY)`` is the metric and the
two descents agree; elsewhere they differ, and a change of the section that leaves the point where it
is changes the step. `scripts/symplectic_gradient_options.jl` measures both statements.
"""
gradient_rep!(D, λY::GlobalSection, Δ) = global_rep!(D, λY, Δ)

function gradient_rep!(D::Union{NamedTuple, NetworkParameters}, λY::NamedTuple,
        gx::Union{NamedTuple, NetworkParameters})
    mapparameters!(gradient_rep!, D, λY, gx)
    D
end

function gradient_rep!(D::SymplecticLieAlgHorMatrix{T},
        λY::GlobalSection{T, <:SymplecticStiefelManifold, <:AbstractArray},
        Δ::AbstractMatrix{T}) where {T}
    G = _ambient(Δ)
    U = λY.Y.A
    @assert (D.N, D.n) == size(U) == size(G)
    k = D.n ÷ 2
    backend = KernelAbstractions.get_backend(G)

    # `P = (UᵀU)⁻¹` applied through its Cholesky factor
    C = cholesky(Symmetric(U' * U))
    GP = G / C
    # `UᵀK`, with `Uᵀ𝕁ᵀU = -𝕁ₙ`; the `Uˢ` block has no second term, because `(Uˢ)ᵀ𝕁ᵀU = 𝕆`
    X = U' * GP - _poisson_tmul(C \ (U' * _poisson_mul(GP))) ./ 2
    @views begin
        D.A₁ .= X[1:k, 1:k] .- transpose(X[(k + 1):(2k), (k + 1):(2k)])
        symmetric_storage_gradient_kernel!(backend)(D.B₁.S, X[1:k, (k + 1):(2k)]; ndrange = (
            k, k))
        symmetric_storage_gradient_kernel!(backend)(D.C₁.S, X[(k + 1):(2k), 1:k]; ndrange = (
            k, k))
    end
    mul!(fill!(D.B, zero(T)), λY.λ', GP)
    D
end

# The derivative with respect to the stored entries, `gradient_rep!`, turned into the representative
# in the inner product `½tr(XᵀY)` of the dense lifts. That inner product weighs every stored entry of a
# symplectic lift by one except the diagonals of the two `SymmetricMatrix` blocks, which it weighs by
# one half, so those are doubled. On every other lift `_dot` is that inner product already.
_section_sharp!(D) = D

function _section_sharp!(D::Union{NamedTuple, NetworkParameters})
    foreachparameters(_section_sharp!, D)
    D
end

function _section_sharp!(D::SymplecticLieAlgHorMatrix)
    backend = KernelAbstractions.get_backend(D)
    k = D.n ÷ 2
    double_packed_diagonal_kernel!(backend)(D.B₁.S; ndrange = k)
    double_packed_diagonal_kernel!(backend)(D.C₁.S; ndrange = k)
    D
end

# the diagonal entry `(i, i)` of the packed storage of a `SymmetricMatrix` is at `i(i + 1)/2`
@kernel function double_packed_diagonal_kernel!(S::AbstractVector{T}) where {T}
    i = @index(Global)
    j = i * (i + 1) ÷ 2
    S[j] += S[j]
end

@doc raw"""
    update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction, workspace = nothing)
    update_section!(Λ, B, retraction, workspace = nothing)

Transport the [`GlobalSection`](@ref) `Λ⁽ᵗ⁻¹⁾` along the step `B⁽ᵗ⁻¹⁾` and write the result into `Λᵗ`:

```math
\Lambda^{(t)} \leftarrow \Lambda^{(t-1)}\mathrm{Retraction}(B^{(t-1)}).
```

This is the fourth of the five steps an optimizer step consists of on a homogeneous space — see
[Optimization on Homogeneous Spaces](@ref) — and the one that makes the cache independent of the
iterate: the new point is read back out of the section afterwards with [`apply_section!`](@ref),
rather than the section being recomputed at the new point.

`B⁽ᵗ⁻¹⁾` is the final velocity the optimizer method produced, already scaled by the step length. It is
an [`AbstractLieAlgHorMatrix`](@ref) when the parameter is on a manifold, and of the parameter's own
type when it is not — a vector-space parameter carries no ``\lambda``, and the extended retraction on
a vector space is addition, so `retraction` is then ignored.

The three-argument form is the two-argument section written in place, `Λᵗ === Λ⁽ᵗ⁻¹⁾`. A `NamedTuple`
of parameters is walked leaf by leaf.

`workspace` is the [`RetractionWorkspace`](@ref) the retraction is taken in, or `nothing` to take it
in fresh arrays. [`Optimizer`](@ref) passes one at every call site on the step path; `nothing` is the
default so that a caller who has no optimizer, and every leaf of a parameter set that is not on a
manifold, need not build one.
"""
function update_section!(Λᵗ::GlobalSection{T, MT}, Λ⁽ᵗ⁻¹⁾::GlobalSection{T, MT},
        B⁽ᵗ⁻¹⁾::AbstractLieAlgHorMatrix{T}, retraction,
        workspace = nothing) where {T, MT <: Manifold{T}}
    _update_section!(workspace, Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction)
end

# The `RetractionWorkspace` counterpart of this is in `src/retractions/retraction_workspace.jl`,
# because that type is defined after this file is included and a signature cannot name it here.
function _update_section!(::Nothing, Λᵗ::GlobalSection, Λ⁽ᵗ⁻¹⁾::GlobalSection,
        B⁽ᵗ⁻¹⁾::AbstractLieAlgHorMatrix, retraction)
    N, n = B⁽ᵗ⁻¹⁾.N, B⁽ᵗ⁻¹⁾.n
    expB = retraction(B⁽ᵗ⁻¹⁾)
    apply_section!(expB, Λ⁽ᵗ⁻¹⁾, expB)
    Λᵗ.Y.A .= @view expB.A[:, 1:n]
    Λᵗ.λ .= @view expB.A[:, (n + 1):N]

    nothing
end

function update_section!(Λᵗ::GlobalSection{T, AT, Nothing}, Λ⁽ᵗ⁻¹⁾::GlobalSection{T, AT},
        B⁽ᵗ⁻¹⁾::AT, retraction, workspace = nothing) where {T, AT <: AbstractVecOrMat{T}}
    Λᵗ.Y .= Λ⁽ᵗ⁻¹⁾.Y .+ B⁽ᵗ⁻¹⁾

    Λᵗ
end

# Same update as the one above -- these are ordinary vector-space parameters and the extended
# retraction on a vector space is addition -- but written on the free parameters, which is where a
# [`VectorStorageMatrix`](@ref) keeps them. Three of the four have no `setindex!` for the broadcast
# above to write through, and for the fourth (`SymmetricMatrix`) it would visit each entry twice.
function update_section!(Λᵗ::GlobalSection{T, AT, Nothing}, Λ⁽ᵗ⁻¹⁾::GlobalSection{T, AT},
        B⁽ᵗ⁻¹⁾::AT, retraction, workspace = nothing) where {T, AT <: VectorStorageMatrix{T}}
    parent(Λᵗ.Y) .= parent(Λ⁽ᵗ⁻¹⁾.Y) .+ parent(B⁽ᵗ⁻¹⁾)

    Λᵗ
end

# A horizontal lift as a vector-space parameter: the same addition, block by block on the free
# parameters. `parent` gives the blocks, `(A, B)` or `(B,)`, and the `SkewSymMatrix` block `A` has no
# `setindex!`, so each block is written through its own `freeparameters`.
function update_section!(Λᵗ::GlobalSection{T, AT, Nothing}, Λ⁽ᵗ⁻¹⁾::GlobalSection{T, AT},
        B⁽ᵗ⁻¹⁾::AT, retraction, workspace = nothing) where {
        T, AT <: AbstractLieAlgHorMatrix{T}}
    foreach(parent(Λᵗ.Y), parent(Λ⁽ᵗ⁻¹⁾.Y), parent(B⁽ᵗ⁻¹⁾)) do y, y₀, b
        freeparameters(y) .= freeparameters(y₀) .+ freeparameters(b)
    end

    Λᵗ
end

# The direction `B⁽ᵗ⁻¹⁾` is of the parameters' shape and the two sections are plain `NamedTuple`s, so
# this is the mixed-shape walk: `mapparameters!` takes the section as its first argument and
# normalises the direction, whichever of the two shapes it arrived in.
function update_section!(Λᵗ::NamedTuple, Λ⁽ᵗ⁻¹⁾::NamedTuple, B⁽ᵗ⁻¹⁾::NetworkParameters, retraction)
    update_section_closure!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾) = update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction)
    mapparameters!(update_section_closure!, Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾)

    Λᵗ
end

# A parameter set's workspace is a tree of the section tree's shape, so it is walked with the rest
# rather than handed whole to each leaf: `retraction_workspace` puts a `RetractionWorkspace` at every
# leaf that is on a manifold and `nothing` at every leaf that is not. The `::Nothing` method below is
# for a caller who passes the default explicitly.
function update_section!(
        Λᵗ::NamedTuple, Λ⁽ᵗ⁻¹⁾::NamedTuple, B⁽ᵗ⁻¹⁾::NetworkParameters, retraction,
        workspace::NamedTuple)
    function update_section_closure!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, workspace)
        update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction, workspace)
    end
    mapparameters!(update_section_closure!, Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, workspace)

    Λᵗ
end

function update_section!(
        Λᵗ::NamedTuple, Λ⁽ᵗ⁻¹⁾::NamedTuple, B⁽ᵗ⁻¹⁾::NetworkParameters, retraction,
        ::Nothing)
    update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction)
end

function update_section!(Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction, workspace = nothing)
    update_section!(Λ⁽ᵗ⁻¹⁾, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction, workspace)
end

# The default for a `struct` is `===`, which is `false` for two sections that hold equal frames in
# different arrays -- and comparing the frames is exactly what `latest_gradient_is_current` needs in
# order to tell "the cache is at the iterate the state is at" from "it is at a line-search trial
# point". `λ === nothing` on Euclidean parameters, and `nothing == nothing` is `true`.
Base.:(==)(Λ₁::GlobalSection, Λ₂::GlobalSection) = Λ₁.Y == Λ₂.Y && Λ₁.λ == Λ₂.λ

# The storage type is left free in both arguments for the reason given on
# `copyto!(::Manifold, ::Manifold)`: a type parameter shared by the two sections binds the whole
# type, the array behind the frame included, so a host-backed and a device-backed section are
# already different concrete types and a shared parameter excludes exactly the transfer these
# methods exist for. `GlobalSection` is not an `AbstractArray` and so has no fallback either, which
# makes that a `MethodError` rather than a silent wrong answer. No species check is needed here:
# `copyto!(::Manifold, ::Manifold)` in `manifolds/abstract_manifold.jl` carries one, and the
# anchors are what this forwards to.
#
# The lift is bound to an array rather than left free, which is what keeps this method and the
# `λ === nothing` one below disjoint. A manifold whose `global_section` falls through to the
# `AbstractVecOrMat` default has no lift to copy, and so belongs to the method below.
function Base.copyto!(dest::GlobalSection{T, <:Manifold, <:AbstractArray},
        src::GlobalSection{T, <:Manifold, <:AbstractArray}) where {T}
    copyto!(dest.Y, src.Y)
    copyto!(dest.λ, src.λ)
    dest
end

function Base.copyto!(dest::GlobalSection{T, <:AbstractVecOrMat{T}, Nothing},
        src::GlobalSection{T, <:AbstractVecOrMat{T}, Nothing}) where {T}
    copyto!(dest.Y, src.Y)
    dest
end

# `Manifold <: AbstractMatrix`, so this also accepts a manifold point as the source and copies its
# entries into the section's plain array. That is the right answer for a section that has no lift to
# update, and it is the only direction in which the two species meet: the reverse, a plain array
# into a manifold-anchored section, goes to the method in `optimizers/named_tuple_wrapper.jl`, which
# takes a `Manifold` source and so rejects it.
function Base.copyto!(dest::GlobalSection{T, <:AbstractVecOrMat{T}, Nothing},
        src::AbstractVecOrMat{T}) where {T}
    copyto!(dest.Y, src)
    dest
end
