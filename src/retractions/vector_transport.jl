@doc raw"""
    AbstractVectorTransport

A vector transport on the Stiefel manifold: it moves a vector ``\Xi`` at ``Y`` to the tangent space
at ``Y_\mathrm{new}``, the retraction of a step ``\Delta`` at ``Y``. See [`transport!`](@ref),
[`ProjectionTransport`](@ref), which serves either retraction, [`DifferentiatedCayleyTransport`](@ref),
which serves [`cayley_direct!`](@ref), and [`ExponentialTransport`](@ref), which serves
[`geodesic_direct!`](@ref).
"""
abstract type AbstractVectorTransport end

@doc raw"""
    ProjectionTransport <: AbstractVectorTransport

The transport by orthogonal projection onto the tangent space at ``Y_\mathrm{new}``:

```math
\Xi_\mathrm{out} = \Xi - \frac{1}{2}Y_\mathrm{new}(Y_\mathrm{new}^T\Xi + \Xi^TY_\mathrm{new}).
```

It reads neither ``Y`` nor ``\Delta``.
"""
struct ProjectionTransport <: AbstractVectorTransport end

@doc raw"""
    DifferentiatedCayleyTransport <: AbstractVectorTransport

The differential of the Cayley retraction at ``\Delta`` in the direction ``\Xi``:

```math
\mathcal{T}_\Delta(\Xi) = (\mathbb{I}_M - \tfrac{1}{2}W(\Delta))^{-1} W(P_Y\Xi)
(\mathbb{I}_M - \tfrac{1}{2}W(\Delta))^{-1} Y,
\qquad P_Y\Xi = \Xi - \tfrac{1}{2}Y(Y^T\Xi + \Xi^TY),
```

with ``W`` the generator of [`cayley_direct!`](@ref). It is
``\frac{\mathrm{d}}{\mathrm{d}s}\big|_{s=0}\mathtt{cayley\_direct}(Y, \Delta + s\Xi)``. Any ``\Xi``
is accepted, and its part off the tangent space at ``Y`` is dropped; at ``\Delta = 0`` the result
is ``P_Y\Xi``.

It is computed in ``O(Mk^2)``, with the matrix ``M_{2k}`` of the retraction and the factors
``U_\Xi V_\Xi = W(P_Y\Xi)``, as

```math
\bigl(U_\Xi + \tfrac{1}{2}UM_{2k}^{-1}(VU_\Xi)\bigr)(V_\Xi W_+), \qquad
W_+ = (\mathbb{I}_M - \tfrac{1}{2}W(\Delta))^{-1}Y = Y + \tfrac{1}{2}UM_{2k}^{-1}VY.
```

``W_+`` comes from the factorization of the step, so the result does not read the entries of
``Y_\mathrm{new}``. Its solve is the `lu!` of [`cayley_direct!`](@ref), with the same backend limit.
"""
struct DifferentiatedCayleyTransport <: AbstractVectorTransport end

@doc raw"""
    ExponentialTransport(algorithm = ScaledSquaring()) <: AbstractVectorTransport

The transport by the group element of the geodesic:

```math
\mathcal{T}_\Delta(\Xi) = \exp(W(\Delta))P_Y\Xi,
\qquad P_Y\Xi = \Xi - \tfrac{1}{2}Y(Y^T\Xi + \Xi^TY),
```

with ``W`` the generator of [`geodesic_direct!`](@ref). ``\exp(W)`` is orthogonal and carries ``Y``
to ``Y_\mathrm{new} = \mathtt{geodesic\_direct}(Y, \Delta)``, so it carries the tangent space at
``Y`` onto the tangent space at ``Y_\mathrm{new}``. This is the transport along
``\exp(t\,W)Y`` of the canonical connection ``\nabla^{(0)}``, which the methods on a
[`GlobalSection`](@ref) apply when they keep a cache in ``\mathfrak{g}^\mathrm{hor}`` while the section
moves (see [Parallel Transport](@ref "Parallel Transport")). Any ``\Xi`` is accepted, and its part
off the tangent space at ``Y`` is dropped; at ``\Delta = 0`` the result is ``P_Y\Xi``.

It is computed in ``O(Mk^2)``, with the factors ``W = UV`` of [`geodesic_direct!`](@ref), as
``P_Y\Xi + U\,\mathfrak{A}(VU)\,VP_Y\Xi``, and ``\mathfrak{A}`` is evaluated with `algorithm`, as
in [`geodesic_direct!`](@ref). It does not read the entries of ``Y_\mathrm{new}``.
"""
struct ExponentialTransport{AT <: AbstractExponentialAlgorithm} <: AbstractVectorTransport
    algorithm::AT
end

ExponentialTransport() = ExponentialTransport(ScaledSquaring())

@doc raw"""
    transport!(Ξ_out, Y_new::StiefelManifold, Y::StiefelManifold, Δ, Ξ, tr::AbstractVectorTransport, ws::DirectCayleyWorkspace)

Write the transport `tr` of `Ξ` from `Y` to the retraction `Y_new` of `Δ` at `Y` into `Ξ_out`, and
return `Ξ_out`: `Y_new = cayley_direct(Y, Δ)` for [`DifferentiatedCayleyTransport`](@ref),
`Y_new = geodesic_direct(Y, Δ)` for [`ExponentialTransport`](@ref), and either for
[`ProjectionTransport`](@ref). `Ξ_out` may be `Ξ`, and must not be `Y_new.A`. A `Ξ_out`, `Y_new`,
`Δ`, `Ξ` or `ws` of another size than `Y` raises a `DimensionMismatch` before anything is written.
"""
function transport!(Ξ_out::AbstractMatrix, Y_new::StiefelManifold, Y::StiefelManifold,
        Δ::AbstractMatrix, Ξ::AbstractMatrix, tr::AbstractVectorTransport,
        ws::DirectCayleyWorkspace)
    _check_point_size(Y, Ξ_out)
    _check_point_size(Y, Y_new.A)
    _check_point_size(Y, Δ)
    _check_point_size(Y, Ξ)
    _check_workspace_size(ws, Y)
    _transport!(Ξ_out, Y_new, Y, Δ, Ξ, tr, ws)
end

function _transport!(Ξ_out, Y_new::StiefelManifold{T}, Y, Δ, Ξ, ::ProjectionTransport,
        ws::DirectCayleyWorkspace{T}) where {T}
    mul!(ws.YᵀX, Y_new.A', Ξ)
    ws.A .= (ws.YᵀX .+ ws.YᵀX') ./ 2
    copyto!(Ξ_out, Ξ)
    mul!(Ξ_out, Y_new.A, ws.A, -one(T), one(T))
    Ξ_out
end

function _transport!(Ξ_out, Y_new::StiefelManifold{T}, Y::StiefelManifold{T}, Δ, Ξ,
        ::DifferentiatedCayleyTransport, ws::DirectCayleyWorkspace{T}) where {T}
    F = _cayley_system!(ws, Y.A, Δ)
    _cayley_factors!(ws.U_Ξ, ws.Vᵗ_Ξ, ws.YᵀX, ws.A, Y.A, Ξ)
    # `M₂ₖ⁻¹(VU_Ξ)`
    mul!(ws.VU_Ξ, ws.Vᵗ', ws.U_Ξ)
    LinearAlgebra.ldiv!(F, ws.VU_Ξ)
    # `C = V_Ξ W₊`, with `W₊ = Y + UM₂ₖ⁻¹(VY)/2` from the factorization of the step
    mul!(ws.VY, ws.Vᵗ', Y.A)
    LinearAlgebra.ldiv!(F, ws.VY)
    copyto!(ws.Y_out, Y.A)
    mul!(ws.Y_out, ws.U, ws.VY, one(T) / 2, one(T))
    mul!(ws.C, ws.Vᵗ_Ξ', ws.Y_out)
    mul!(ws.D, ws.VU_Ξ, ws.C)
    mul!(Ξ_out, ws.U_Ξ, ws.C)
    mul!(Ξ_out, ws.U, ws.D, one(T) / 2, one(T))
    Ξ_out
end

# `P_YΞ` in the first `k` columns of `U_Ξ`, which this transport does not otherwise use. The
# factors of the step are formed first, because their scratch is the `YᵀX` and `A` of the
# projection.
function _transport!(Ξ_out, Y_new::StiefelManifold{T}, Y::StiefelManifold{T}, Δ, Ξ,
        tr::ExponentialTransport, ws::DirectCayleyWorkspace{T}) where {T}
    k = size(Y, 2)
    𝔄VU = _geodesic_system!(ws, Y.A, Δ, tr.algorithm)
    PΞ = view(ws.U_Ξ, :, 1:k)
    mul!(ws.YᵀX, Y.A', Ξ)
    ws.A .= (ws.YᵀX .+ ws.YᵀX') ./ 2
    copyto!(PΞ, Ξ)
    mul!(PΞ, Y.A, ws.A, -one(T), one(T))
    mul!(ws.C, ws.Vᵗ', PΞ)
    mul!(ws.D, 𝔄VU, ws.C)
    copyto!(Ξ_out, PΞ)
    mul!(Ξ_out, ws.U, ws.D, one(T), one(T))
    Ξ_out
end

@doc raw"""
    transport(Y_new::StiefelManifold, Y::StiefelManifold, Δ, Ξ, tr::AbstractVectorTransport)

The transport `tr` of `Ξ` from `Y` to the retraction `Y_new` of `Δ` at `Y`, in a new matrix of the
array type of `Y.A`. See [`transport!`](@ref).
"""
function transport(Y_new::StiefelManifold, Y::StiefelManifold, Δ::AbstractMatrix,
        Ξ::AbstractMatrix, tr::AbstractVectorTransport)
    _check_point_size(Y, Ξ)
    transport!(similar(Y.A), Y_new, Y, Δ, Ξ, tr, DirectCayleyWorkspace(Y))
end
