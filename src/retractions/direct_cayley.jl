@doc raw"""
    DirectCayleyWorkspace(Y::StiefelManifold)

The buffers of [`cayley_direct!`](@ref) and [`transport!`](@ref) at a point of the shape of `Y`.

For ``Y \in St(k, M)`` it holds four ``M\times{}2k`` matrices, one ``M\times{}k`` matrix, two
``k\times{}k`` matrices, two ``2k\times{}2k`` matrices and three ``2k\times{}k`` matrices, all on the
backend of `Y.A`: ``O(Mk + k^2)`` storage and no ``M\times{}M`` array. A workspace serves every
point of its shape and element type.
"""
struct DirectCayleyWorkspace{T, AT <: AbstractMatrix{T}}
    # the factors of the generator of the step: `U = [Y K]`, and `Vᵗ = Vᵀ = [YAᵀ - K  Y]`
    U::AT
    Vᵗ::AT
    # the same factors for the transported vector
    U_Ξ::AT
    Vᵗ_Ξ::AT
    # the result before it is copied into the output, so that the output may be the input
    Y_out::AT
    YᵀX::AT
    A::AT
    M₂ₖ::AT
    VU_Ξ::AT
    VY::AT
    C::AT
    D::AT
end

function DirectCayleyWorkspace(Y::StiefelManifold{T}) where {T}
    M, k = size(Y)
    backend = KernelAbstractions.get_backend(Y.A)
    buffer(m, n) = KernelAbstractions.allocate(backend, T, m, n)
    DirectCayleyWorkspace(buffer(M, 2k), buffer(M, 2k), buffer(M, 2k), buffer(M, 2k),
        buffer(M, k), buffer(k, k), buffer(k, k), buffer(2k, 2k), buffer(2k, 2k),
        buffer(2k, k), buffer(2k, k), buffer(2k, k))
end

function _check_workspace_size(ws::DirectCayleyWorkspace, Y::StiefelManifold)
    size(ws.Y_out) == size(Y.A) || throw(DimensionMismatch(
        "a workspace for points of size $(size(ws.Y_out)) at a point of size $(size(Y.A))"))
    nothing
end

# The factors `U = [Y K]` and `Vᵀ = [YAᵀ - K  Y]` of the generator `W(P_Y X) = UV` of the Cayley
# transform at `Y` along `X`, with `A = (YᵀX - XᵀY)/2` and `K = X - Y(YᵀX)`. For a tangent `X`,
# `P_Y X = X`. `YᵀX` and `A` are scratch.
function _cayley_factors!(
        U, Vᵗ, YᵀX, A, Y::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T}
    k = size(Y, 2)
    mul!(YᵀX, Y', X)
    A .= (YᵀX .- YᵀX') ./ 2
    @views begin
        K = U[:, (k + 1):(2k)]
        copyto!(U[:, 1:k], Y)
        copyto!(K, X)
        mul!(K, Y, YᵀX, -one(T), one(T))
        copyto!(Vᵗ[:, 1:k], K)
        mul!(Vᵗ[:, 1:k], Y, A', one(T), -one(T))
        copyto!(Vᵗ[:, (k + 1):(2k)], Y)
    end
    nothing
end

# The factors of the step `Δ` at `Y` in `ws.U` and `ws.Vᵗ`, and the LU factorization of
# `M₂ₖ = 𝕀 - VU/2` in `ws.M₂ₖ`.
function _cayley_system!(ws::DirectCayleyWorkspace{T}, Y::AbstractMatrix{T},
        Δ::AbstractMatrix{T}) where {T}
    _cayley_factors!(ws.U, ws.Vᵗ, ws.YᵀX, ws.A, Y, Δ)
    mul!(ws.M₂ₖ, ws.Vᵗ', ws.U, -one(T) / 2, zero(T))
    view(ws.M₂ₖ, LinearAlgebra.diagind(ws.M₂ₖ)) .+= one(T)
    LinearAlgebra.lu!(ws.M₂ₖ)
end

@doc raw"""
    cayley_direct!(Y_new::StiefelManifold, Y::StiefelManifold, Δ::AbstractMatrix, ws::DirectCayleyWorkspace)

Write the Cayley retraction of the tangent vector `Δ` at `Y` into `Y_new`, and return `Y_new`.

The retraction is ``(\mathbb{I}_M - \frac{1}{2}W)^{-1}(\mathbb{I}_M + \frac{1}{2}W)Y`` with the
generator ``W = \Delta{}Y^T - Y\Delta^T + Y\Delta^TYY^T``. It is computed without an ``M\times{}M``
matrix and without the orthogonal complement of `Y`, through the rank-``2k`` factorization
``W = UV`` with

```math
A = \frac{1}{2}(Y^T\Delta - \Delta^TY), \quad K = \Delta - Y(Y^T\Delta), \quad
U = [Y\;\; K], \quad V = \begin{bmatrix} AY^T - K^T \\ Y^T \end{bmatrix},
```

and the Sherman–Morrison–Woodbury identity

```math
Y_\mathrm{new} = Y + U M_{2k}^{-1} VY, \qquad M_{2k} = \mathbb{I}_{2k} - \frac{1}{2}VU.
```

The cost is ``O(Mk^2)``, and the only solve is the ``2k\times{}2k`` system ``M_{2k}``. For a `Δ`
off the tangent space the result is the retraction of its tangent projection
``\Delta - \frac{1}{2}Y(Y^T\Delta + \Delta^TY)``.

`Y_new` may be `Y`: the result is written into `ws` first and copied into `Y_new.A` last. A `Δ`,
`Y_new` or `ws` of another size than `Y` raises a `DimensionMismatch` before anything is written.
"""
function cayley_direct!(
        Y_new::StiefelManifold{T}, Y::StiefelManifold{T}, Δ::AbstractMatrix{T},
        ws::DirectCayleyWorkspace{T}) where {T}
    _check_point_size(Y, Δ)
    _check_point_size(Y, Y_new.A)
    _check_workspace_size(ws, Y)
    F = _cayley_system!(ws, Y.A, Δ)
    mul!(ws.VY, ws.Vᵗ', Y.A)
    LinearAlgebra.ldiv!(F, ws.VY)
    copyto!(ws.Y_out, Y.A)
    mul!(ws.Y_out, ws.U, ws.VY, one(T), one(T))
    copyto!(Y_new.A, ws.Y_out)
    Y_new
end

@doc raw"""
    cayley_direct(Y::StiefelManifold, Δ::AbstractMatrix)

The Cayley retraction of the tangent vector `Δ` at `Y`, as a new `StiefelManifold`. See
[`cayley_direct!`](@ref).
"""
function cayley_direct(Y::StiefelManifold{T}, Δ::AbstractMatrix{T}) where {T}
    _check_point_size(Y, Δ)
    cayley_direct!(StiefelManifold(similar(Y.A)), Y, Δ, DirectCayleyWorkspace(Y))
end
