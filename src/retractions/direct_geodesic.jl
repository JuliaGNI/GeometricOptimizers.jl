# The factors of the step `Δ` at `Y` in `ws.U` and `ws.Vᵗ`, `VU` in `ws.M₂ₖ`, and `𝔄(VU)` in the
# scratch of the workspace, which this returns.
function _geodesic_system!(ws::DirectCayleyWorkspace{T}, Y::AbstractMatrix{T},
        Δ::AbstractMatrix{T}, algorithm::AbstractExponentialAlgorithm) where {T}
    _cayley_factors!(ws.U, ws.Vᵗ, ws.YᵀX, ws.A, Y, Δ)
    mul!(ws.M₂ₖ, ws.Vᵗ', ws.U)
    𝔄!(ws.𝔄_scratch, ws.M₂ₖ, algorithm)
end

function _geodesic_system!(::DirectCayleyWorkspace{T}, Y::AbstractMatrix{T},
        Δ::AbstractMatrix{T}, ::ProjectedSkew) where {T}
    throw(ArgumentError("ProjectedSkew exponentiates a skew-symmetric matrix, and the 2k × 2k " *
                        "matrix VU of a direct geodesic is not skew-symmetric; use another " *
                        "exponential algorithm"))
end

@doc raw"""
    geodesic_direct!(Y_new::StiefelManifold, Y::StiefelManifold, Δ::AbstractMatrix, ws::DirectCayleyWorkspace, algorithm = ScaledSquaring())

Write the geodesic retraction of the tangent vector `Δ` at `Y` into `Y_new`, and return `Y_new`.

The retraction is ``\exp(W)Y`` with the generator ``W = \Delta{}Y^T - Y\Delta^T + Y\Delta^TYY^T``
of [`cayley_direct!`](@ref), the point that [`geodesic`](@ref)`(Y, Δ)` reaches through a
[`GlobalSection`](@ref). It is computed without an ``M\times{}M`` matrix and without the orthogonal
complement of `Y`, through the factorization ``W = UV`` of [`cayley_direct!`](@ref) and

```math
Y_\mathrm{new} = Y + U\,\mathfrak{A}(VU)\,VY,
```

which is the formula of [`geodesic(::AbstractLieAlgHorMatrix)`](@ref) with ``U`` and ``V`` in place
of ``B'`` and ``(B'')^T``. [`GeometricOptimizers.𝔄!`](@ref) evaluates ``\mathfrak{A}`` on the
``2k\times{}2k`` matrix ``VU`` with `algorithm`, in the scratch of `ws`, so the cost is
``O(Mk^2)``. [`ProjectedSkew`](@ref) is an `ArgumentError`, because ``VU`` is not skew-symmetric.
For a `Δ` off the tangent space the result is the retraction of its tangent projection
``\Delta - \frac{1}{2}Y(Y^T\Delta + \Delta^TY)``.

`Y_new` may be `Y`: the result is written into `ws` first and copied into `Y_new.A` last. A `Δ`,
`Y_new` or `ws` of another size than `Y` raises a `DimensionMismatch` before anything is written.
"""
function geodesic_direct!(
        Y_new::StiefelManifold{T}, Y::StiefelManifold{T}, Δ::AbstractMatrix{T},
        ws::DirectCayleyWorkspace{T},
        algorithm::AbstractExponentialAlgorithm = ScaledSquaring()) where {T}
    _check_point_size(Y, Δ)
    _check_point_size(Y, Y_new.A)
    _check_workspace_size(ws, Y)
    𝔄VU = _geodesic_system!(ws, Y.A, Δ, algorithm)
    mul!(ws.C, ws.Vᵗ', Y.A)
    mul!(ws.VY, 𝔄VU, ws.C)
    copyto!(ws.Y_out, Y.A)
    mul!(ws.Y_out, ws.U, ws.VY, one(T), one(T))
    copyto!(Y_new.A, ws.Y_out)
    Y_new
end

@doc raw"""
    geodesic_direct(Y::StiefelManifold, Δ::AbstractMatrix, algorithm = ScaledSquaring())

The geodesic retraction of the tangent vector `Δ` at `Y`, as a new `StiefelManifold`. See
[`geodesic_direct!`](@ref).
"""
function geodesic_direct(Y::StiefelManifold{T}, Δ::AbstractMatrix{T},
        algorithm::AbstractExponentialAlgorithm = ScaledSquaring()) where {T}
    _check_point_size(Y, Δ)
    geodesic_direct!(StiefelManifold(similar(Y.A)), Y, Δ, DirectCayleyWorkspace(Y), algorithm)
end
