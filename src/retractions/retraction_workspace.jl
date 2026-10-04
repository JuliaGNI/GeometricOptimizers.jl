@doc raw"""
    RetractionWorkspace(x::Manifold, retraction = nothing)
    RetractionWorkspace(backend, T, N, n, retraction = nothing)

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

The workspace also holds the scratch of [`GeometricOptimizers.𝔄!`](@ref), which a
[`geodesic`](@ref) step evaluates its ``\mathfrak{A}`` with, and of
[`retraction_differential!`](@ref), as much of it as `retraction` reads: the `𝔄!` scratch of its
algorithm for a [`Geodesic`](@ref), and that of the differential for [`Cayley`](@ref) on a host
`Matrix` of a LAPACK element type. Every other scratch array is `0 × 0`, so the type of a workspace
does not depend on its retraction. A workspace built for one retraction serves that retraction only;
one built with `retraction = nothing` holds all the scratch and serves every retraction.
[`Optimizer`](@ref) builds its workspace for its own retraction.

In a workspace, the ``\mathfrak{A}`` of [`ScaledSquaring`](@ref) and
[`NativePade`](@ref) allocates nothing, and that of [`AugmentedPade`](@ref) allocates what `Base.exp`
of its ``4n\times{}4n`` matrix allocates — on a ``20\times{}20`` argument 87 472 bytes in `Float64` and
44 224 in `Float32`. So in both precisions the two native algorithms are the lighter, and
`AugmentedPade` the heavier by exactly its `exp`. The figures are those of
`scripts/in_place_retraction_cost.jl`.

**What this does not remove**: the ``2n\times{}2n`` `inv` in [`cayley`](@ref) and the `exp` of
[`AugmentedPade`](@ref). Neither grows with ``N``. `inv` stays an `inv` because `lu!` and `rdiv!`
would take it in place on the host only, and a `KernelAbstractions` backend is under no obligation to
supply either — `inv` is what `cayley` already runs on Metal through. [`ProjectedSkew`](@ref) takes
its allocating [`geodesic`](@ref) and copies the answer in.

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
    # The scratch of `𝔄!`: its argument `X`, its result `𝔄X`, the column sums of its norm, up to eight
    # `2n × 2n` temporaries, and the `4n × 4n` matrix `AugmentedPade` exponentiates. Each of these and
    # of the four below is `0 × 0` where the retraction does not read it; see `_scratch_uses`.
    X::AT
    𝔄X::AT
    colsum::AT
    s₁::AT
    s₂::AT
    s₃::AT
    s₄::AT
    s₅::AT
    s₆::AT
    s₇::AT
    s₈::AT
    augmented::AT
    # The scratch of `retraction_differential!`: a `2n × n` right-hand side, two `N × n` columns,
    # and the pivots of the `2n × 2n` LU factorisation it takes in `s₁`.
    rhs::AT
    w₁::AT
    w₂::AT
    ipiv::Vector{LinearAlgebra.BlasInt}

    function RetractionWorkspace(
            backend::KernelAbstractions.Backend, ::Type{T}, N::Integer, n::Integer,
            retraction = nothing) where {T}
        unit = unit_matrix(backend, T, n)
        B̂ = KernelAbstractions.zeros(backend, T, N, 2n)
        B̄ᵗ = KernelAbstractions.zeros(backend, T, 2n, N)
        𝕀_small2 = unit_matrix(backend, T, 2n)

        uses = _scratch_uses(retraction, typeof(unit))
        function sized(use, rows, cols)
            KernelAbstractions.zeros(backend, T, use ? rows : 0, use ? cols : 0)
        end
        temporary(i) = sized(i ≤ uses.temporaries, 2n, 2n)
        augmented = sized(uses.augmented, 4n, 4n)

        # The constant half of the factorisation, written once. `lift_factors!` writes the other
        # half and leaves these alone, which is why it may not be handed a workspace of another
        # shape. `augmented` is `[X 𝕀; 𝕆 𝕆]`, and `𝔄!` writes only its `X` block.
        @views begin
            B̂[1:n, (n + 1):(2n)] .= unit
            B̄ᵗ[1:n, 1:n] .= unit
        end
        uses.augmented && @views augmented[1:(2n), (2n + 1):(4n)] .= 𝕀_small2

        small() = KernelAbstractions.zeros(backend, T, 2n, 2n)
        new{T, typeof(unit)}(N, n, unit,
            KernelAbstractions.zeros(backend, T, n, n), B̂, B̄ᵗ,
            𝕀_small2, small(),
            KernelAbstractions.zeros(backend, T, N, 2n), unit_matrix(backend, T, N),
            KernelAbstractions.zeros(backend, T, N, N),
            KernelAbstractions.zeros(backend, T, N, N),
            sized(uses.X, 2n, 2n), sized(uses.𝔄X, 2n, 2n), sized(uses.colsum, 1, 2n),
            temporary(1), temporary(2), temporary(3), temporary(4), temporary(5), temporary(6),
            temporary(7), temporary(8), augmented,
            sized(uses.differential, 2n, n),
            sized(uses.differential, N, n), sized(uses.differential, N, n),
            zeros(LinearAlgebra.BlasInt, uses.differential ? 2n : 0))
    end
end

function RetractionWorkspace(Y::Manifold{T}, retraction = nothing) where {T}
    N, n = size(Y)
    RetractionWorkspace(KernelAbstractions.get_backend(Y.A), T, N, n, retraction)
end

# The scratch fields of a `RetractionWorkspace` that `retraction` reads, given the type `AT` of the
# workspace's arrays: the `𝔄!` fields of `_𝔄_uses` for a `Geodesic`, and for `Cayley` the fields of
# the in-place differential, `X`, `s₁`, `rhs`, `w₁`, `w₂` and `ipiv`, which `retraction_differential!`
# reads on a host `Matrix` of a LAPACK element type only. A retraction this package does not ship reads
# none, and `nothing` stands for every retraction.
const _NO_SCRATCH = (
    X = false, 𝔄X = false, colsum = false, temporaries = 0, augmented = false,
    differential = false)

_scratch_uses(::AbstractRetraction, ::Type) = _NO_SCRATCH
_scratch_uses(R::Geodesic, ::Type) = merge(_NO_SCRATCH, _𝔄_uses(R.algorithm))

function _scratch_uses(::Cayley, AT::Type)
    in_place = AT <: Matrix{<:LinearAlgebra.BlasFloat}
    merge(_NO_SCRATCH, (X = in_place, temporaries = Int(in_place), differential = in_place))
end

function _scratch_uses(::Nothing, ::Type)
    (X = true, 𝔄X = true, colsum = true, temporaries = 8,
        augmented = true, differential = true)
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
    retraction_workspace(x, retraction = nothing)
    retraction_workspace(opt::Optimizer)

The [`RetractionWorkspace`](@ref) a solve over `x` with `retraction` needs, or [`NoWorkspace`](@ref)
where `x` carries no manifold. On an [`Optimizer`](@ref) it is the accessor instead, returning the
workspace that optimizer was built with.

[`Optimizer`](@ref) builds one of these for its retraction at construction and hands it to every
[`update_section!`](@ref) on the step path. A parameter set gets a tree of them in the shape its
section tree has, a `NoWorkspace` at every leaf that is an ordinary array — the extended retraction
on a vector space is addition, which allocates nothing to begin with.

The two meanings share a name and cannot collide: `Optimizer` is not a member of
`OptimizerSolution`, which is what the building methods dispatch on.
"""
retraction_workspace(::AbstractVecOrMat, retraction = nothing) = NoWorkspace()
retraction_workspace(Y::Manifold, retraction = nothing) = RetractionWorkspace(Y, retraction)
function retraction_workspace(ps::NetworkParameters, retraction = nothing)
    mapparameters(x -> retraction_workspace(x, retraction), params(ps))
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
    _dense_skew!(ws.A_mat, B.A, ws.unit)

    # `transpose` and not `adjoint`, for the reason `lift_factors` gives at the same block.
    @views begin
        ws.B̂[1:n, 1:n] .= T(0.5) .* ws.A_mat
        ws.B̂[(n + 1):N, 1:n] .= B.B
        ws.B̄ᵗ[(n + 1):(2n), 1:n] .= T(0.5) .* ws.A_mat
        ws.B̄ᵗ[(n + 1):(2n), (n + 1):N] .= .-transpose(B.B)
    end

    nothing
end

# The dense form of a skew block. On the host a broadcast reads it entry by entry and allocates
# nothing; anywhere else the product with the identity runs the backend's kernel, whose launch is
# the 128 bytes the host form saves. The two write the same matrix.
function _dense_skew!(C::Matrix{T}, A::SkewSymMatrix{T, Vector{T}}, ::AbstractMatrix) where {T}
    (C .= A)
end
_dense_skew!(C::AbstractMatrix, A::SkewSymMatrix, unit::AbstractMatrix) = mul!(C, A, unit)

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

`ws` has to have been built for `B`'s shape, as for [`lift_factors!`](@ref), and every arm asserts
it. The assertion is not decoration: the fallback arms reach `ws.retracted` through `copyto!`, and
`copyto!` into an oversized destination copies linearly rather than throwing, so a workspace of the
wrong shape would scramble the layout of the answer instead of rejecting it.
"""
function retraction_matrix!(ws::RetractionWorkspace{T}, R::AbstractRetraction,
        B::AbstractLieAlgHorMatrix{T}) where {T}
    @assert (ws.N, ws.n) == (B.N, B.n)
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
    # `B̄ᵗB̂` is the argument `𝔄(B̂, B̄, algorithm)` forms, and `𝔄!` evaluates it in the workspace.
    mul!(ws.X, ws.B̄ᵗ, ws.B̂)
    mul!(ws.B̂C, ws.B̂, 𝔄!(ws, ws.X, algorithm))
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
    @assert (ws.N, ws.n) == (B.N, B.n)
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

# The in-place `Cayley` arm of `retraction_differential!`, here and not in `retractions.jl` for the
# reason the arm above gives. The steps are `retraction_differential`'s, with `G = B̄ᵗB̂` in `ws.X`:
#
#     w₁ = E + B̂((𝕀 - aG) \ (aB̄ᵗE))     # M E
#     w₂ = B̂(B̄ᵗw₁)                      # B̄ M E
#     V  = w₂ - B̂((𝕀 + aG) \ (aB̄ᵗw₂))   # Mᵀ B̄ M E
#
# `B̄ᵗE` is the first `n` columns of `B̄ᵗ`, and adding `E` adds the identity to the top `n` rows. Each
# solve is factorised in `s₁` with the pivots in `ipiv`, and the right-hand side `rhs` is overwritten
# with the solution. `getrf!` checks its matrix for `Inf` and `NaN`, as the `lu` of `\` does, and a
# zero pivot in its `info` raises `SingularException`, as `\` does.
function retraction_differential!(
        D::AbstractLieAlgHorMatrix{T}, ws::RetractionWorkspace{T, Matrix{T}}, ::Cayley,
        B::AbstractLieAlgHorMatrix{T}, α) where {T <: LinearAlgebra.BlasFloat}
    if iszero(α)
        _copyto!(D, B)
        return D
    end

    n = B.n
    a = T(α) / 2
    G, M, rhs = ws.X, ws.s₁, ws.rhs
    lift_factors!(ws, B)
    mul!(G, ws.B̄ᵗ, ws.B̂)

    @views rhs .= a .* ws.B̄ᵗ[:, 1:n]
    M .= ws.𝕀_small2 .- a .* G
    _lu_solve!(M, ws.ipiv, rhs)
    mul!(ws.w₁, ws.B̂, rhs)
    @views ws.w₁[1:n, :] .+= ws.unit

    mul!(rhs, ws.B̄ᵗ, ws.w₁)
    mul!(ws.w₂, ws.B̂, rhs)

    mul!(rhs, ws.B̄ᵗ, ws.w₂)
    rhs .= a .* rhs
    M .= ws.𝕀_small2 .+ a .* G
    _lu_solve!(M, ws.ipiv, rhs)
    mul!(ws.w₁, ws.B̂, rhs)
    ws.w₂ .-= ws.w₁

    lift_from_columns!(D, ws.w₂)
end

function _lu_solve!(M::Matrix{T}, ipiv::Vector{LinearAlgebra.BlasInt},
        rhs::Matrix{T}) where {T <: LinearAlgebra.BlasFloat}
    _, _, info = LinearAlgebra.LAPACK.getrf!(M, ipiv)
    info > 0 && throw(LinearAlgebra.SingularException(info))
    LinearAlgebra.LAPACK.getrs!('N', M, ipiv, rhs)
end
