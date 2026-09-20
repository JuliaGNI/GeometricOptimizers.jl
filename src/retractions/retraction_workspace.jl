@doc raw"""
    RetractionWorkspace(x::Manifold)
    RetractionWorkspace(backend, T, N, n)

The buffers [`update_section!`](@ref) retracts a horizontal lift into, held for the life of an
[`Optimizer`](@ref) rather than rebuilt per call.

Every array a retraction of an ``N\times{}n`` point needs is a fixed shape once ``N`` and ``n`` are
known — the two factors of [`lift_factors`](@ref), the ``2n\times{}2n`` matrix the inverse is taken
of, the ``N\times{}N`` result, and the frame the result is transported into. An optimizer step
applies a retraction once per line-search trial and three times besides, so those shapes are
rebuilt six to fifteen times an iteration for a step that changes none of them.

# Implementation

The four blocks that do not depend on the lift are written once, here, and never again: the two
``n\times{}n`` identity blocks of ``B'`` and ``(B'')^T``, and the zero blocks beside them.
[`lift_factors!`](@ref) then writes only the four that do. `𝕀_small2` and `𝕀_big` are kept because
both sums are formed as a five-argument `mul!` into an identity, which needs the identity to copy
from.

The element type and the backend come from the point, so a workspace is on the same backend as the
solve it belongs to and no method here names either.

**What this does not remove**: the ``2n\times{}2n`` `inv` in [`cayley`](@ref), and whatever
[`GeometricOptimizers.𝔄`](@ref) allocates for [`geodesic`](@ref). Both are ``O(n^2)`` and neither
grows with ``N``. `inv` stays an `inv` because `lu!` and `rdiv!` would take it in place on the host
only, and a `KernelAbstractions` backend is under no obligation to supply either — `inv` is what
`cayley` already runs on Metal through.

A solve whose parameters are not on a manifold has no workspace: see
[`GeometricOptimizers.retraction_workspace`](@ref).
"""
struct RetractionWorkspace{T, AT <: AbstractMatrix{T}}
    N::Int
    n::Int
    unit::AT
    A_mat::AT
    B̂::AT
    B̄ᵗ::AT
    𝕀_small2::AT
    M::AT
    B̂C::AT
    𝕀_big::AT
    retracted::AT
    product::AT

    function RetractionWorkspace(
            backend::KernelAbstractions.Backend, ::Type{T}, N::Integer, n::Integer) where {T}
        unit = unit_matrix(backend, T, n)
        B̂ = KernelAbstractions.zeros(backend, T, N, 2n)
        B̄ᵗ = KernelAbstractions.zeros(backend, T, 2n, N)

        # The constant half of the factorisation, written once. `lift_factors!` writes the other
        # half and leaves these alone, which is why it may not be handed a workspace of another
        # shape.
        @views begin
            B̂[1:n, (n + 1):(2n)] .= unit
            B̄ᵗ[1:n, 1:n] .= unit
        end

        new{T, typeof(unit)}(N, n, unit,
            KernelAbstractions.zeros(backend, T, n, n), B̂, B̄ᵗ,
            unit_matrix(backend, T, 2n), KernelAbstractions.zeros(backend, T, 2n, 2n),
            KernelAbstractions.zeros(backend, T, N, 2n), unit_matrix(backend, T, N),
            KernelAbstractions.zeros(backend, T, N, N),
            KernelAbstractions.zeros(backend, T, N, N))
    end
end

function RetractionWorkspace(Y::Manifold{T}) where {T}
    N, n = size(Y)
    RetractionWorkspace(KernelAbstractions.get_backend(Y.A), T, N, n)
end

@doc raw"""
    NoWorkspace()

What [`GeometricOptimizers.retraction_workspace`](@ref) returns where there is no manifold to
retract on.

The vector-space methods of [`update_section!`](@ref) take a `workspace` argument and ignore it —
the extended retraction on a vector space is addition, which allocates nothing to begin with — so
this type is a *marker* and has no method of its own. It exists only because `nothing` cannot serve
as that marker.

# Implementation

`nothing` is the obvious spelling and is the one thing this may not be. A parameter set's workspace
is a tree walked in lockstep with its section tree, and `NeuralNetworkParameters.mapparameters!`
reads a `nothing` in that position as *skip this leaf*: the leaf function is then never called, so
the section is never transported and the iterate never moves. That is silent — the solve runs to its
iteration limit and reports the point it started from. A singleton says the same thing and is an
ordinary value to the walk.

(That name in plain code and not an `@extref`, for the reason `iterative_hessians.jl` gives about
the same choice: an `@extref` that the published inventory does not carry is a build error rather
than a dead link, and `mapparameters!` is not one of the names this package already links.)
"""
struct NoWorkspace end

@doc raw"""
    retraction_workspace(x)

The [`RetractionWorkspace`](@ref) a solve over `x` needs, or [`NoWorkspace`](@ref) where `x` carries
no manifold.

[`Optimizer`](@ref) builds one of these at construction and hands it to every
[`update_section!`](@ref) on the step path. A parameter set gets a tree of them in the shape its
section tree has, a `NoWorkspace` at every leaf that is an ordinary array — the extended retraction
on a vector space is addition, which allocates nothing to begin with.
"""
retraction_workspace(::AbstractVecOrMat) = NoWorkspace()
retraction_workspace(Y::Manifold) = RetractionWorkspace(Y)
function retraction_workspace(ps::NetworkParameters)
    mapparameters(retraction_workspace, params(ps))
end

@doc raw"""
    lift_factors!(ws::RetractionWorkspace, B::AbstractLieAlgHorMatrix)

[`lift_factors`](@ref) written into `ws`, returning nothing.

Only the lift-dependent blocks are written — four for a Stiefel lift and two for a Grassmann one,
whose ``A`` block is identically zero. The identity and zero blocks were written when the workspace
was built. `ws` has to have been built for `B`'s shape.
"""
function lift_factors!(ws::RetractionWorkspace{T}, B::StiefelLieAlgHorMatrix{T}) where {T}
    N, n = B.N, B.n
    @assert (ws.N, ws.n) == (N, n)
    mul!(ws.A_mat, B.A, ws.unit)

    # `transpose` and not `adjoint`, for the reason `lift_factors` gives at the same block.
    @views begin
        ws.B̂[1:n, 1:n] .= T(0.5) .* ws.A_mat
        ws.B̂[(n + 1):N, 1:n] .= B.B
        ws.B̄ᵗ[(n + 1):(2n), 1:n] .= T(0.5) .* ws.A_mat
        ws.B̄ᵗ[(n + 1):(2n), (n + 1):N] .= .-transpose(B.B)
    end

    nothing
end

function lift_factors!(ws::RetractionWorkspace{T}, B::GrassmannLieAlgHorMatrix{T}) where {T}
    N, n = B.N, B.n
    @assert (ws.N, ws.n) == (N, n)

    # `A ≡ 𝕆` for a Grassmann lift, so the two blocks the Stiefel method writes `A_mat` into stay
    # zero and are never written at all.
    @views begin
        ws.B̂[(n + 1):N, 1:n] .= B.B
        ws.B̄ᵗ[(n + 1):(2n), (n + 1):N] .= .-transpose(B.B)
    end

    nothing
end

@doc raw"""
    retraction_matrix!(ws::RetractionWorkspace, R::AbstractRetraction, B::AbstractLieAlgHorMatrix)

The ``N\times{}N`` matrix `retraction(R, B)` wraps in a manifold type, written into `ws.retracted`
and returned bare.

The matrix and not the manifold: [`update_section!`](@ref) reads the blocks of the result and
discards the wrapper, so wrapping it would be one allocation per line-search trial for a value
nothing keeps.

An [`AbstractRetraction`](@ref) this package does not ship has no in-place form here and falls
through to the allocating [`retraction`](@ref), with the answer copied in. A workspace never changes
what a retraction means. A bare callable is *not* covered — [`update_section!`](@ref) accepts one as
its `retraction`, and `src/utils.jl` passes one, but only ever together with no workspace, so there
is no retraction type left here to dispatch on and a caller who pairs the two gets a `MethodError`
rather than a silent fallback.
"""
function retraction_matrix!(ws::RetractionWorkspace{T}, R::AbstractRetraction,
        B::AbstractLieAlgHorMatrix{T}) where {T}
    copyto!(ws.retracted, retraction(R, B).A)
end

function retraction_matrix!(ws::RetractionWorkspace{T}, ::Cayley,
        B::AbstractLieAlgHorMatrix{T}) where {T}
    lift_factors!(ws, B)
    copyto!(ws.M, ws.𝕀_small2)
    mul!(ws.M, ws.B̄ᵗ, ws.B̂, -T(0.5), one(T))
    mul!(ws.B̂C, ws.B̂, inv(ws.M))
    copyto!(ws.retracted, ws.𝕀_big)
    mul!(ws.retracted, ws.B̂C, ws.B̄ᵗ, one(T), one(T))

    ws.retracted
end

function retraction_matrix!(ws::RetractionWorkspace{T}, R::Geodesic,
        B::AbstractLieAlgHorMatrix{T}) where {T}
    _geodesic_matrix!(ws, B, R.algorithm)
end

function _geodesic_matrix!(ws::RetractionWorkspace{T}, B::AbstractLieAlgHorMatrix{T},
        algorithm::AbstractExponentialAlgorithm) where {T}
    lift_factors!(ws, B)
    mul!(ws.B̂C, ws.B̂, 𝔄(ws.B̂, ws.B̄ᵗ', algorithm))
    copyto!(ws.retracted, ws.𝕀_big)
    mul!(ws.retracted, ws.B̂C, ws.B̄ᵗ, one(T), one(T))

    ws.retracted
end

# `ProjectedSkew` is the one algorithm that is not `𝔄` on the small product: it orthonormalises the
# range of the lift and exponentiates there, through a `qr` and an `eigen` that allocate whatever
# they allocate. There is nothing for a workspace to hold, so this falls back to the allocating
# method and copies the answer in — `update_section!` reads `ws.retracted` either way.
function _geodesic_matrix!(ws::RetractionWorkspace{T}, B::AbstractLieAlgHorMatrix{T},
        algorithm::ProjectedSkew) where {T}
    copyto!(ws.retracted, geodesic(B, algorithm).A)
end

# The workspace arm of `update_section!`; the `::Nothing` arm is in
# `src/global_sections/global_sections.jl`, next to the rest of that function, and this one is here
# because `RetractionWorkspace` does not exist yet when that file is included.
#
# `apply_section!` is not called. That function takes the transported frame in its own argument and
# would be handed `ws.retracted` as both source and destination -- which is why its two products are
# materialised rather than written in place, and so why it allocates two `N × N` matrices. With a
# second buffer the same expression is two `mul!`s, and the frame is then read straight out of it:
# nothing writes back into `ws.retracted` at all.
function _update_section!(ws::RetractionWorkspace{T}, Λᵗ::GlobalSection,
        Λ⁽ᵗ⁻¹⁾::GlobalSection, B⁽ᵗ⁻¹⁾::AbstractLieAlgHorMatrix{T},
        retraction::AbstractRetraction) where {T}
    N, n = B⁽ᵗ⁻¹⁾.N, B⁽ᵗ⁻¹⁾.n
    retracted = retraction_matrix!(ws, retraction, B⁽ᵗ⁻¹⁾)

    @views begin
        mul!(ws.product, Λ⁽ᵗ⁻¹⁾.Y.A, retracted[1:n, :])
        mul!(ws.product, Λ⁽ᵗ⁻¹⁾.λ, retracted[(n + 1):N, :], one(T), one(T))
        Λᵗ.Y.A .= ws.product[:, 1:n]
        Λᵗ.λ .= ws.product[:, (n + 1):N]
    end

    nothing
end
