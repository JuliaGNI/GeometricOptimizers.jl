# Global Tangent Spaces

In `GeometricOptimizers` standard neural network optimizers are generalized to [homogeneous spaces](@ref "Homogeneous Spaces") by leveraging the special structure of the tangent spaces of this class of manifolds. When we introduced homogeneous spaces we already talked about that every tangent space to a homogeneous space ``T_Y\mathcal{M}`` is of the form: 

```math
    T_Y\mathcal{M} = \mathfrak{g} \cdot Y := \{AY: A\in{}\mathfrak{g}\}.
```

We then have a decomposition of ``\mathfrak{g}`` into a vertical part ``\mathfrak{g}^{\mathrm{ver}, Y}`` and a horizontal part ``\mathfrak{g}^{\mathrm{hor}, Y}`` and the horizontal part is isomorphic to ``T_Y\mathcal{M}`` via:

```math
    \mathfrak{g}^{\mathrm{hor}, Y} = \{\Omega(\Delta): \Delta\in{}T_Y\mathcal{M} \}.
```

We now identify a special element ``E \in \mathcal{M}`` and designate the horizontal component ``\mathfrak{g}^{\mathrm{hor}, E}`` as our *global tangent space*. We will refer to this global tangent space by ``\mathfrak{g}^\mathrm{hor}``. We can now find a transformation from any ``\mathfrak{g}^{\mathrm{hor}, Y}`` to ``\mathfrak{g}^\mathrm{hor}`` and vice-versa (these spaces are isomorphic). If the metric on ``G`` is bi-invariant, as the canonical metric on ``SO(N)`` is, this transformation is a conjugation:

```@eval
Main.theorem(raw"Let the metric on ``G`` be bi-invariant, and let ``A\in{}G`` an element such that ``AE = Y``. Then we have
" * Main.indentation * raw"```math
" * Main.indentation * raw"A^{-1}\cdot\mathfrak{g}^{\mathrm{hor},Y}\cdot{}A = \mathfrak{g}^\mathrm{hor},
" * Main.indentation * raw"```
" * Main.indentation * raw"i.e. for every element ``B\in\mathfrak{g}^\mathrm{hor}`` we can find a ``B^Y \in \mathfrak{g}^{\mathrm{hor},Y}`` s.t. ``B = A^{-1}B^YA`` (and vice-versa). The map ``B^Y\mapsto{}A^{-1}B^YA`` is an isometry.")
```

```@eval
Main.proof(raw"For ``V\in\mathfrak{g}`` we have ``AVA^{-1}Y = AVE``, so ``V\mapsto{}AVA^{-1}`` maps ``\mathfrak{g}^{\mathrm{ver},E}`` onto ``\mathfrak{g}^{\mathrm{ver},Y}``. Because the metric is bi-invariant, its inner product on ``\mathfrak{g}`` is invariant under ``V\mapsto{}AVA^{-1}``, so this map also sends the orthogonal complement ``\mathfrak{g}^\mathrm{hor}`` of ``\mathfrak{g}^{\mathrm{ver},E}`` onto the orthogonal complement ``\mathfrak{g}^{\mathrm{hor},Y}`` of ``\mathfrak{g}^{\mathrm{ver},Y}``. Its inverse is the map of the theorem.")
```

Without bi-invariance the conjugation of an element of ``\mathfrak{g}^{\mathrm{hor},Y}`` in general has a vertical part, so the statement above fails (an example is the [symplectic Stiefel manifold](@ref "Global Tangent Space for the Symplectic Stiefel Manifold") with its Riemannian metric). The two spaces are still isomorphic for every complement of the vertical space, if the vertical part is removed:

```@eval
Main.theorem(raw"Let ``\mathfrak{g}^{\mathrm{hor},Y}`` be any complement of ``\mathfrak{g}^{\mathrm{ver},Y}``, ``\mathfrak{g}^\mathrm{hor}`` any complement of ``\mathfrak{g}^{\mathrm{ver},E}``, and ``A\in{}G`` an element such that ``AE = Y``. Then
" * Main.indentation * raw"```math
" * Main.indentation * raw"\mathfrak{g}^{\mathrm{hor},Y}\to\mathfrak{g}^\mathrm{hor}, \quad B^Y\mapsto\mathrm{pr}_{\mathfrak{g}^\mathrm{hor}}(A^{-1}B^YA),
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\mathrm{pr}_{\mathfrak{g}^\mathrm{hor}}`` is the projection along ``\mathfrak{g}^{\mathrm{ver},E}``, is an isomorphism. Its image ``B`` satisfies ``ABE = B^YY``, and if the metric is bi-invariant it is the conjugation of the theorem above.")
```

```@eval
Main.proof(raw"Write ``A^{-1}B^YA = V + B`` with ``V\in\mathfrak{g}^{\mathrm{ver},E}`` and ``B\in\mathfrak{g}^\mathrm{hor}``. Then ``ABE = A(V + B)E = B^YAE = B^YY``. If ``B = 0``, then ``B^YY = 0``, so ``B^Y`` is vertical and horizontal at ``Y``, and ``B^Y = 0``. The map is therefore injective, and the two spaces have the same dimension as ``T_Y\mathcal{M}``. In the bi-invariant case ``A^{-1}B^YA\in\mathfrak{g}^\mathrm{hor}`` by the theorem above, so ``V = 0``.")
```

The isomorphism of the second theorem is not an isometry in general: it carries the inner product of ``\mathfrak{g}^{\mathrm{hor},Y}`` to an inner product on ``\mathfrak{g}^\mathrm{hor}`` that can depend on ``A``.

We should note that we have written all Lie group and Lie algebra actions as simple matrix multiplications, like ``AE = Y``. For some Lie groups and Lie algebras, as the Lie group of isomorphisms on some domain ``\mathcal{D}``, this notation may not be appropriate [holm2009geometric](@cite). These Lie groups are however not relevant for what we use in `GeometricOptimizers` and we will stick to regular matrix notation.

## Global Sections

Note that the theorem above requires us to find an element ``A\in{}G`` such that ``AE = Y``. We will call such a mapping ``\lambda:\mathcal{M}\to{}G`` a *global section*[^1]. 

[^1]: Global sections are also crucial for [parallel transport](@ref "Parallel Transport") in `GeometricOptimizers`. A global section is first updated, i.e. ``\Lambda^{(t)} \gets \mathrm{update}(\Lambda^{(t-1)});`` and on the basis of this we then update the element of the manifold ``Y\in\mathcal{M}`` and the tangent vector ``\Delta\in{}T\mathcal{M}``.

```@eval
Main.definition(raw"We call a mapping ``\lambda`` from a homogeneous space ``\mathcal{M}`` to its associated Lie group ``G`` a **global section** if ``\forall{}Y\in\mathcal{M}`` it satisfies:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\lambda(Y)E = Y,
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``E`` is the distinct element of the homogeneous space.")
```

Note that in general global sections are not unique because the rank of ``G`` is in general greater than that of ``\mathcal{M}``. We give an example of how to construct such a global section for the Stiefel and the Grassmann manifolds below. 

## The Global Tangent Space for the Stiefel Manifold

We now discuss the specific form of the global tangent space for the [Stiefel manifold](@ref "The Stiefel Manifold"). We pick as distinct element ``E`` (which build by calling [`StiefelProjection`](@ref)):

```math
E = \begin{bmatrix}
\mathbb{I}_n \\ 
\mathbb{O}
\end{bmatrix}\in{}St(n, N).
```

Based on this, elements of the vector space ``\mathfrak{g}^{\mathrm{hor}, E} =: \mathfrak{g}^{\mathrm{hor}}`` are: 

```math
\bar{B} = \begin{pmatrix}
A & -B^T \\ B & \mathbb{O}
\end{pmatrix},
```

where ``A`` is a skew-symmetric matrix of size ``n\times{}n`` and ``B`` is an arbitrary matrix of size ``(N - n)\times{}n``. Arrays of type ``\mathfrak{g}^{\mathrm{hor}, E} \equiv \mathfrak{g}^\mathrm{hor}`` are implemented in `GeometricOptimizers` under the name [`StiefelLieAlgHorMatrix`](@ref).

We can call this with e.g. a skew-symmetric matrix ``A`` and an arbitrary matrix ``B``:

```@example call_stiefel_lie_alg_hor_matrix_1
using GeometricOptimizers # hide

N, n = 5, 2

A = rand(SkewSymMatrix, n)
```

```@example call_stiefel_lie_alg_hor_matrix_1
B = rand(N - n, n)
```

The constructor is then called as follows:

```@example call_stiefel_lie_alg_hor_matrix_1
B̄ = StiefelLieAlgHorMatrix(A, B, N, n)
```

We can also call it with a matrix of shape ``N\times{}N``:

```@example call_stiefel_lie_alg_hor_matrix_1
B̄₂ = Matrix(B̄) # note that this does not have any special structure

StiefelLieAlgHorMatrix(B̄₂, n)
```

Or we can call it on ``T_E\mathcal{M}\subset\mathbb{R}^{N\times{}n},`` i.e. a matrix of shape ``N\times{}n``:

```@example call_stiefel_lie_alg_hor_matrix_1
E = StiefelProjection(N, n)
```

```@example call_stiefel_lie_alg_hor_matrix_1
B̄₃ = B̄ * E

StiefelLieAlgHorMatrix(B̄₃, n)
```

We now demonstrate how to map from an element of ``\mathfrak{g}^{\mathrm{hor}, Y}`` to an element of ``\mathfrak{g}^\mathrm{hor}``:

```@example global_section
using GeometricOptimizers # hide
using GeometricOptimizers: Ω

N, n = 5, 2 # hide
Y = rand(StiefelManifold, N, n)
Δ = rgrad(Y, rand(N, n))
ΩΔ = Ω(Y, Δ)
λY = GlobalSection(Y) 

λY_mat = Matrix(λY)

round.(λY_mat' * ΩΔ * λY_mat; digits = 3)
```

Performing this computation directly is computationally very inefficient however and the user is strongly discouraged to call `Matrix` on an instance of `GlobalSection`. The better option is calling `global_rep`:

```@example global_section
using GeometricOptimizers: _round # hide

_round(global_rep(λY, Δ); digits = 3)
```

Internally `GlobalSection` calls the function [`GeometricOptimizers.global_section`](@ref) which does the following for the Stiefel manifold: 

```julia
A = randn(N, N - n) # or the gpu equivalent
A = A - Y * (Y' * A)
B = A / maximum(abs, A)                # so that the Gram matrix cannot overflow
Q = B / cholesky(Symmetric(B'B)).U     # first pass
λ = Q / cholesky(Symmetric(Q'Q)).U     # second pass; see below
Y⟂ = _cholesky_qr2(λ - Y * (Y' * λ))   # the same projection and orthonormalization once more
```

So we draw ``(N - n)`` new columns randomly, subtract the part that is spanned by the columns of ``Y`` and then orthonormalize the resulting matrix. The result is a matrix of ``(N - n)`` columns that is orthogonal to ``Y`` and is typically referred to as ``Y_\perp``  [absil2004riemannian, absil2008optimization, bendokat2020grassmann](@cite). We can easily check that this ``Y_\perp`` is indeed orthogonal to ``Y``.

The last line is there because of rounding. The first projection leaves a rounding error in the span of ``Y``, and the orthonormalization multiplies it by the condition number of the projected draw, which is large now and then; in `Float32` the worst of a few thousand draws has ``\|Y^T\lambda\|`` of order ``10^{-2}``. ``\lambda`` is orthonormal, so projecting and orthonormalizing it again amplifies nothing, and ``Y_\perp`` is orthogonal to ``Y`` to rounding. The proof below is in exact arithmetic, where one pass is enough.

The orthonormalization is **CholeskyQR2** — the Cholesky step shown twice above — and not `LinearAlgebra.qr`. `qr` is the textbook answer, but it is a host factorization for several of the backends supported here: `Metal` implements no `qr` for its array type at all, so a `qr` here would put [`GlobalSection`](@ref) and therefore [`Optimizer`](@ref) out of reach on such a device. CholeskyQR2 is expressible in matrix products, reductions and triangular solves alone, so it runs wherever the point already is, and its ``\|Q^TQ - \mathbb{I}\|`` is measured *smaller* than the host Householder QR's at every size tried. What the two give differs by the sign of each column, because CholeskyQR2's ``R`` has a positive diagonal and Householder's need not.

The price is that forming ``A^TA`` squares the condition number, and ``A`` above is square inside the complement of ``Y``. In `Float32` that breaks the factorization down on about one draw in a hundred and twenty. Such a draw is *replaced* rather than repaired — it is Gaussian noise and carries no information — which is what [`GeometricOptimizers.orthonormal_columns`](@ref) does. A replacement fails at a somewhat higher rate again, so the redraw is bounded and raises rather than returning a result it did not produce cleanly.

On a device, CholeskyQR2 keeps the point there; it is not the faster route. `scripts/orthonormalization_device_cost.jl` measures it on an M4 Max with Julia 1.13.1, Metal 1.11.1 and the development version of GeometricOptimizers after 0.8.0, in `Float32`, at the ``N\times(N-3)`` shape that [`global_section`](@ref) uses: a cold session, one warm-up, the median of five runs, `Metal.synchronize()` in the timed region, and the host on one BLAS thread.

| ``N`` | CholeskyQR2 on Metal | host `qr!` | download, host `qr!`, upload |
|---|---|---|---|
| 20 | 7.3 ms | 46 μs | 0.60 ms |
| 400 | 29 ms | 12 ms | 19 ms |

At these sizes the device time is mostly launch and synchronisation latency. The second pass is necessary: on Metal in `Float32` at the same shape, ``\|Q^TQ - \mathbb{I}\|`` after one pass is ``9.7\times10^{-6}`` at ``N = 20`` and ``1.9\times10^{-3}`` at ``N = 400``, against ``7.1\times10^{-7}`` and ``7.7\times10^{-6}`` after the second.

```@eval
Main.theorem(raw"The matrix ``Y_\perp`` constructed with the algorithm above satisfies
" * Main.indentation * raw"```math
" * Main.indentation * raw"Y^TY_\perp = \mathbb{O}_{n\times{}n},
" * Main.indentation * raw"```
" * Main.indentation * raw"and
" * Main.indentation * raw"```math
" * Main.indentation * raw"(Y_\perp)^TY_\perp = \mathbb{I}_n,
" * Main.indentation * raw"```
" * Main.indentation * raw"i.e. all the columns in the big matrix ``[Y, Y_\perp]\in\mathbb{R}^{N\times{}N}`` are mutually orthonormal and it therefore is an element of ``SO(N)``.")
```

```@eval
Main.proof(raw"The orthonormalization writes the drawn matrix as ``QR``, with ``Q`` orthonormal and ``R`` upper triangular — for CholeskyQR2 ``R`` is the product of the two Cholesky factors, and a product of upper triangular matrices is upper triangular. The second property is then trivially satisfied because ``Q`` is orthonormal. For the first property note that ``Y^TQR = \mathbb{O}`` is zero because we have subtracted the ``Y`` component from the matrix ``QR``. The matrix ``R\in\mathbb{R}^{(N-n)\times{}(N-n)}`` further has the property ``[R]_{ij} = 0`` for ``i > j`` and we have that 
" * Main.indentation * raw"```math
" * Main.indentation * raw"(Y^TQ)R = [r_{11}(Y^TQ)_{1\bullet}, r_{12}(Y^TQ)_{1\bullet} + r_{22}(Y^TQ)_{2\bullet}, \ldots, \sum_{i=1}^{N-n}r_{i(N-n)}(Y^TQ)_{i\bullet}].
" * Main.indentation * raw"```
" * Main.indentation * raw"Now all the coefficients ``r_{ii}`` are non-zero because the matrix we factorized has full rank — and for CholeskyQR2 they are in addition positive, which is what makes ``Q`` unique where a Householder ``QR`` leaves the sign of each column free. We can see that if ``(Y^TQ)R`` is zero ``Y^TQ`` also has to be zero.")
```

The function `global_rep` furthermore makes use of the following:

```math
    \mathtt{global\_rep}(Y) = \lambda(Y)^T\Omega(Y,\Delta)\lambda(Y) = EY^T\Delta{}E^T + \begin{bmatrix} \mathbb{O} \\ \bar{\lambda}^T\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O} & E\Delta^T\bar{\lambda} \end{bmatrix},
```
where ``\lambda(Y) = [Y, \bar{\lambda}].``

```@eval
Main.proof(raw"We derive the expression above: 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\begin{aligned}
" * Main.indentation * raw"\lambda(Y)^T\Omega(Y,\Delta)\lambda(Y)  & = \lambda(Y)^T[(\mathbb{I} - \frac{1}{2}YY^T)\Delta{}Y^T - Y\Delta^T(\mathbb{I} - \frac{1}{2}YY^T)]\lambda(Y) \\
" * Main.indentation * raw"                                        & = \lambda(Y)^T[(\mathbb{I} - \frac{1}{2}YY^T)\Delta{}E^T - Y\Delta^T(\lambda(Y) - \frac{1}{2}YE^T)] \\
" * Main.indentation * raw"                                        & = \lambda(Y)^T\Delta{}E^T - \frac{1}{2}EY^T\Delta{}E^T - E\Delta^T\lambda(Y) + \frac{1}{2}E\Delta^TYE^T \\ 
" * Main.indentation * raw"                                        & = \begin{bmatrix} Y^T\Delta{}E^T \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \frac{1}{2}EY^T\Delta{}E - \begin{bmatrix} E\Delta^TY & E\Delta^T\bar{\lambda} \end{bmatrix} + \frac{1}{2}E\Delta^TYE^T \\
" * Main.indentation * raw"                                        & = \begin{bmatrix} Y^T\Delta{}E^T \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} + E\Delta^TYE^T - \begin{bmatrix}E\Delta^TY & E\Delta^T\bar{\lambda} \end{bmatrix} \\
" * Main.indentation * raw"                                                & = EY^T\Delta{}E^T + E\Delta^TYE^T - E\Delta^TYE^T + \begin{bmatrix} \mathbb{O} \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O} & E\Delta^T\bar{\lambda} \end{bmatrix} \\
" * Main.indentation * raw"                                        & = EY^T\Delta{}E^T + \begin{bmatrix} \mathbb{O}_{n\times{}N} \\ \bar{\lambda}\Delta{}E^T \end{bmatrix} - \begin{bmatrix} \mathbb{O}_{N\times{}n} & E\Delta^T\bar{\lambda} \end{bmatrix},
" * Main.indentation * raw"\end{aligned}
" * Main.indentation * raw"```
" * Main.indentation * raw"which proofs our assertion.")
```

This expression of `global_rep` means we only need ``Y^T\Delta`` and ``\bar{\lambda}^T\Delta`` and this is what is used internally.

We now discuss the global tangent space for the Grassmann manifold. This is similar to the Stiefel case.

## Global Tangent Space for the Grassmann Manifold

In the case of the Grassmann manifold we construct the global tangent space with respect to the distinct element ``\mathcal{E}=\mathrm{span}(E)\in{}Gr(n,N)``, where ``E`` is again the same matrix.

The tangent tangent space ``T_\mathcal{E}Gr(n,N)`` can be represented through matrices: 

```math
\begin{pmatrix}
    0 & \cdots & 0 \\
    \cdots & \cdots & \cdots \\ 
    0 & \cdots & 0 \\
    b_{11} & \cdots & b_{1n} \\
    \cdots & \cdots & \cdots \\ 
    b_{(N-n)1} & \cdots & b_{(N-n)n}
\end{pmatrix}.
```

This representation is based on the identification ``T_\mathcal{E}Gr(n,N)\to{}T_E\mathcal{S}_E`` that was discussed in the section on the [Grassmann manifold](@ref "The Grassmann Manifold")[^2]. We use the following notation:

[^2]: We derived the following expression for the Riemannian gradient of the Grassmann manifold: ``\mathrm{grad}_\mathcal{Y}^{Gr}L = \nabla_Y{}L - YY^T\nabla_YL``. The tangent space to the element ``\mathcal{E}`` can thus be written as ``\bar{B} - EE^T\bar{B}`` where ``B\in\mathbb{R}^{N\times{}n}`` and the matrices in this tangent space have the desired form. 

```math
\mathfrak{g}^\mathrm{hor} = \mathfrak{g}^{\mathrm{hor},\mathcal{E}} = \left\{\begin{pmatrix} 0 & -B^T \\ B & 0 \end{pmatrix}: \text{$B\in\mathbb{R}^{(N-n)\times{}n}$ is arbitrary}\right\}.
```

This is equivalent to the horizontal component of ``\mathfrak{g}`` for the Stiefel manifold for the case when ``A`` is zero. This is a reflection of the rotational invariance of the Grassmann manifold: the skew-symmetric matrices ``A`` are connected to the group of rotations ``O(n)`` which is factored out in the Grassmann manifold ``Gr(n,N)\simeq{}St(n,N)/O(n)``. In `GeometricOptimizers` we thus treat the Grassmann manifold as being embedded in the Stiefel manifold. In [bendokat2020grassmann](@cite) viewing the Grassmann manifold as a quotient space of the Stiefel manifold is important for "feasibility" in "practical computations". 

## Global Tangent Space for the Symplectic Stiefel Manifold

The [`SymplecticStiefelManifold`](@ref) ``Sp(2n, 2N) = \{U\in\mathbb{R}^{2N\times2n} : U^T\mathbb{J}_{2N}U = \mathbb{J}_{2n}\}`` is a homogeneous space of the symplectic group ``Sp(2N)``, and [bendokat2021real](@cite) give it the same structure as the Stiefel manifold: a projection ``M\mapsto{}ME`` from the group, a vertical space, and, below their (3.4), a horizontal space that is the same subspace of the Lie algebra ``\mathfrak{sp}(2N)`` at every point. It is the complement of the vertical space with respect to the bi-invariant pseudo-Riemannian metric ``h`` of their (2.1). The horizontal space (3.13) of their Riemannian metric ``g``, the one [`metric`](@ref) implements, depends on the point, so only the first gives a global tangent space.

The section is stored as for the Stiefel manifold, ``\lambda(U) = [U\ \ U^s]`` with the point first, so ``E`` is the first ``2n`` columns. ``U^s`` is a symplectic completion: ``U^T\mathbb{J}_{2N}U^s = \mathbb{O}`` and ``(U^s)^T\mathbb{J}_{2N}U^s = \mathbb{J}_{2N-2n}``. In this basis the Poisson tensor is ``\bar{\mathbb{J}} = \mathrm{diag}(\mathbb{J}_{2n}, \mathbb{J}_{2N-2n})``, ``\lambda(U)^T\mathbb{J}_{2N}\lambda(U) = \bar{\mathbb{J}}``, and the horizontal space below (3.4) of [bendokat2021real](@cite) reads

```math
\mathfrak{g}^\mathrm{hor} = \left\{\begin{pmatrix} \mathcal{A} & -\mathcal{B}^+ \\ \mathcal{B} & \mathbb{O} \end{pmatrix}: \mathcal{A}\in\mathfrak{sp}(2n), \mathcal{B}\in\mathbb{R}^{(2N-2n)\times2n}\right\}, \qquad \mathcal{B}^+ = \mathbb{J}_{2n}^T\mathcal{B}^T\mathbb{J}_{2N-2n}.
```

This is the Stiefel case with the transpose replaced by the symplectic adjoint, and it is what [`SymplecticLieAlgHorMatrix`](@ref) stores. An element ``\Omega`` satisfies ``\Omega^T\bar{\mathbb{J}} + \bar{\mathbb{J}}\Omega = \mathbb{O}``, so its Cayley transform and its exponential preserve ``\bar{\mathbb{J}}``, ``\lambda(U)R`` is a section again for every retraction ``R`` of it, and ``\lambda(U)RE`` is a point of the manifold.

The tangent space at ``U`` is ``\{\Delta : U^+\Delta\in\mathfrak{sp}(2n)\}`` with the symplectic inverse ``U^+ = \mathbb{J}_{2n}^TU^T\mathbb{J}_{2N}``. A horizontal element is determined by its first ``2n`` columns, and those of ``\lambda(U)^{-1}\Omega(U,\Delta)\lambda(U)`` are ``\lambda(U)^{-1}\Delta``, with ``\Omega`` the horizontal lift [`GeometricOptimizers.Ω`](@ref) of (3.7) of [bendokat2021real](@cite). So `global_rep` needs only

```math
\mathcal{A} = U^+\Delta, \qquad \mathcal{B} = (U^s)^+\Delta.
```

This is the isomorphism of the second theorem at the top of this page, whatever horizontal space at ``U`` one starts from: the element ``B`` of ``\mathfrak{g}^\mathrm{hor}`` with ``\lambda(U)BE = \Delta`` is unique, so the vertical part of ``\lambda(U)^{-1}Z\lambda(U)`` is removed for the lift ``Z`` of (3.15) of [bendokat2021real](@cite), whose horizontal space is that of [`metric`](@ref), as it is for ``\Omega`` of (3.7), for which that part is zero. The first theorem, the conjugation, does not apply to the lift of (3.15): ``Sp(2N)`` has no bi-invariant Riemannian metric (see below), and at a generic point the conjugate of that lift has a vertical part.

Two things differ from the Stiefel case.

The first is the completion. The symplectic group is not compact, so a completion can be arbitrarily badly conditioned, and the coordinates of the lift, in which [`Adam`](@ref) and the quasi-Newton methods take their steps, are only as good as it. [`global_section`](@ref) therefore takes an orthonormal basis ``Q`` of the symplectic complement of ``U`` and brings the skew-symmetric ``Q^T\mathbb{J}_{2N}Q`` to its normal form with the eigenvectors of a Hermitian matrix, which rescales each column and keeps them orthogonal. Its condition number has not exceeded that of ``U`` on any point tried, and it is orthonormal where ``U`` is orthosymplectic.

!!! note "The symplectic group is not compact, and the sections can be ill-conditioned"
    On ``SO(N)`` every section is orthogonal, so its condition number is one. ``Sp(2N)`` is not compact, and neither is the stabilizer ``Sp(2N-2n)`` of a point. Two sections at one point differ by ``h = \mathrm{diag}(\mathbb{I}, S)`` with ``S\in{}Sp(2N-2n)``. Already ``S = \mathrm{diag}(t\mathbb{I}, t^{-1}\mathbb{I})`` has condition number ``t^2``, so the sections over one point have no bound on their conditioning. This enters in three places:

    - **The coordinates of the lift.** ``h`` sends ``(\mathcal{A}, \mathcal{B})`` to ``(\mathcal{A}, S^{-1}\mathcal{B})``. Everything held in these coordinates changes with the section: the moments of [`Adam`](@ref), the step of [`SectionDescent`](@ref), the secant pairs of [`BFGS`](@ref) and [`DFP`](@ref), and the cache of every momentum method.
    - **Rounding.** `apply_section`, `global_rep` and the section update multiply by ``\lambda(U)`` or its inverse ``\bar{\mathbb{J}}^T\lambda(U)^T\mathbb{J}_{2N}``, so their rounding error grows with the condition number of the section. This holds also for [`MetricDescent`](@ref), whose iterates do not depend on the section in exact arithmetic.
    - **The transported section.** The optimizers carry the section along their path and never reset it to [`global_section`](@ref), so its conditioning drifts from that of a fresh one.

    The manifold itself is unbounded too: an iterate can leave every bounded set, which a Stiefel or Grassmann iterate cannot. No choice of metric removes any of this, since no positive-definite inner product on ``\mathfrak{g}^\mathrm{hor}`` is invariant under the stabilizer (below).

    On the ``10\times4`` eigenvalue problem of the tests, the transported section goes from condition number 1.35 to 3.4–3.6 in the first 5000 steps of every method and then stays there. A fresh section of the final point has 2.83. With a noisy gradient ``2(M + \sigma(G + G^T)/2)U``, ``G`` Gaussian, over ``10^5`` steps of size ``10^{-2}``:

    - [`Adam`](@ref) ends at 4.31 against a fresh 2.78 (``\sigma = 1``), and at 6.15 against 5.28 (``\sigma = 10``).
    - [`GradientMethod`](@ref) diverges. Before its last step at ``\sigma = 1`` (step 7106), ``\|U\| = 1.3\cdot10^{25}``, ``\mathrm{cond}(U) = 2.1\cdot10^{10}``, and the section's condition number is ``3.7\cdot10^{63}``.

    A reset of the section to [`global_section`](@ref), with the caches carried over by ``\mathcal{B}\mapsto{}S\mathcal{B}``, would bound the drift, but the optimizers do not do it. Issue [#189](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/189) discusses this.

The second is the gradient. For the Stiefel manifold the inner product of two lifts, as [`GeometricOptimizers._dot`](@ref) takes it on their stored entries, is the canonical metric, so the lift of the Riemannian gradient pairs with the lift of a direction as the derivative of the loss along it. The metric of [`SymplecticStiefelManifold`](@ref) is only right-invariant, so no inner product on ``\mathfrak{g}^\mathrm{hor}`` reproduces it at every point, and this identity fails. Two lifts therefore take separate roles.

No other Riemannian metric can be reproduced that way either. Two sections at one point differ by ``h = \mathrm{diag}(\mathbb{I}, S)`` with ``S\in{}Sp(2N-2n)``, which sends ``(\mathcal{A}, \mathcal{B})`` to ``(\mathcal{A}, S^{-1}\mathcal{B})``. An inner product that gave the same value through every section would be invariant under this map. With ``S = \mathrm{diag}(t\mathbb{I}, t^{-1}\mathbb{I})`` and a ``\mathcal{B}`` that is zero below its first ``N - n`` rows, ``S^{-1}\mathcal{B} = \mathcal{B}/t``, so its quadratic form ``q`` satisfies ``q(\mathcal{B}) = q(\mathcal{B})/t^2`` for every ``t``, and ``q(\mathcal{B}) = 0``. The obstruction is the non-compact stabilizer of the point, not the metric chosen on ``Sp(2N)``. The same argument shows that ``Sp(2N)`` has no bi-invariant Riemannian metric, which would need an inner product on ``\mathfrak{sp}(2N)`` invariant under ``X\mapsto{}MXM^{-1}`` for every ``M\in{}Sp(2N)``. Take ``M = \exp(tD)`` with ``D = \mathrm{diag}(\mathbb{I}_N, -\mathbb{I}_N)\in\mathfrak{sp}(2N)``, so ``M = \mathrm{diag}(e^t\mathbb{I}_N, e^{-t}\mathbb{I}_N)``, and ``X = \begin{pmatrix}\mathbb{O} & K\\ \mathbb{O} & \mathbb{O}\end{pmatrix}`` with ``K = K^T\neq\mathbb{O}``, which is in ``\mathfrak{sp}(2N)``. Then ``MXM^{-1} = e^{2t}X``, so an invariant quadratic form ``q`` satisfies ``q(X) = e^{4t}q(X)`` for every ``t``, and ``q(X) = 0``. ``\mathfrak{sp}(2N)`` is simple, so the forms on it that are invariant under the whole group are the multiples of the trace form, and these are indefinite: the trace form gives the pseudo-Riemannian metric (3.8) of [bendokat2021real](@cite), whose horizontal space is the ``\mathfrak{g}^\mathrm{hor}`` above. `scripts/symplectic_metric_invariance.jl` checks these statements without the package.

The line search, [`BFGS`](@ref) and [`DFP`](@ref) pair a gradient with a direction, and they use the lift [`GeometricOptimizers.gradient_rep!`](@ref) gives: the derivative of the loss with respect to the stored entries of a lift at the section. With it the slope of a line search is the derivative of its merit, and the secant pairs of the quasi-Newton methods are in the coordinates their ``Q`` lives in.

[`GradientMethod`](@ref), [`MomentumMethod`](@ref) and [`Adam`](@ref) build their direction from the gradient itself, and their keyword `descent` selects which one. With [`MetricDescent`](@ref), the default, it is [`global_rep`](@ref) of the Riemannian gradient of [`metric`](@ref). Its tangent vector ``\lambda(U)BE`` does not depend on the section, and neither does the retraction ``\lambda(U)\exp(tB)E`` of a horizontal lift: a change ``\lambda\mapsto\lambda h`` with ``h = \mathrm{diag}(\mathbb{I}, S)`` sends ``B`` to ``h^{-1}Bh``, and ``\lambda h\exp(th^{-1}Bh)E = \lambda\exp(tB)E``. The path of [`GradientMethod`](@ref) is therefore a property of the point and the metric alone. That retraction is ``\exp(t\Omega(U,\Delta))U``, the geodesic of the pseudo-Riemannian metric (3.8) (Proposition 3.8 of [bendokat2021real](@cite)), and not the geodesic (3.18) of [`metric`](@ref): the two curves agree to first order in ``t`` and differ at second order. So it is a retraction for [`metric`](@ref), not its exponential map. It is, however, the geodesic of the canonical connection of the second kind [nomizu1954invariant, schlarb2024covariant](@cite), which needs no metric, only a horizontal space that the stabilizer maps to itself (Lemma 3.4 of [bendokat2021real](@cite)). Its parallel transport along ``\exp(t\Omega(U,\Delta))U`` is ``\Delta_2\mapsto\exp(t\Omega(U,\Delta))\Delta_2 = \lambda(U)\exp(tB)B_2E``, and this is exactly what the optimizers do when they keep a cache ``B_2\in\mathfrak{g}^\mathrm{hor}`` fixed while the section moves to ``\lambda(U)\exp(tB)``, as in [Parallel Transport](@ref "Parallel Transport") for the Stiefel manifold. So retraction and transport form one consistent pair, and the metric enters only through the gradient. The geodesic (3.18) of [`metric`](@ref) forms another pair, with the parallel transport of the Levi-Civita connection of [`metric`](@ref), and it costs more. With ``C = \lambda(U)^{-1}Z\lambda(U)``, where ``Z`` is the lift of (3.15), and ``P = \lambda(U)^T\lambda(U)``, (3.18) reads ``\lambda(U)\exp(t(C - P^{-1}C^TP))\exp(tP^{-1}C^TP)E``. ``C`` is not in ``\mathfrak{g}^\mathrm{hor}`` and the section enters through ``P``, so the retraction is no longer a map on ``\mathfrak{g}^\mathrm{hor}`` alone. The transport cannot keep the cache fixed either. Keeping it fixed while the section follows the horizontal lift of (3.18) gives a result that does not depend on the section, but it changes the length of the cache in [`metric`](@ref) and does not carry the initial velocity to the velocity of (3.18). The cache therefore has to be recomputed in every step. [#188](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/188) tracks this pair as an option. With [`SectionDescent`](@ref) the direction is the steepest descent of ``\frac{1}{2}\mathrm{tr}(X^TY)`` on the dense lifts at the section. By the statement above, that form defines no metric on the manifold, so `SectionDescent` is not a Riemannian method: its step is steepest descent at one section of the fibre over the point, not at the point. In the coordinates of [`global_section`](@ref) the metric is block-diagonal in ``(\mathcal{A}, \mathcal{B})`` at every point, and at an orthosymplectic point it is exactly ``\frac{1}{2}\mathrm{tr}(\mathcal{A}^T\mathcal{A}) + \mathrm{tr}(\mathcal{B}^T\mathcal{B})``, so there the two descents agree. Away from such points they differ, and the step of `SectionDescent` changes with a change of the section that leaves the point where it is, which the optimizers make by transporting the section along the path. `scripts/symplectic_gradient_options.jl` measures these statements.

## Library functions

[`AbstractLieAlgHorMatrix`](@ref), [`StiefelLieAlgHorMatrix`](@ref),
[`GrassmannLieAlgHorMatrix`](@ref), [`SymplecticLieAlgHorMatrix`](@ref), [`StiefelProjection`](@ref), [`GlobalSection`](@ref),
[`global_section`](@ref), [`global_rep`](@ref) and [`apply_section`](@ref). Their docstrings are on the [reference page](@ref GeometricOptimizers), where every docstring in the package is rendered once; the names above link to them.

[`GeometricMachineLearning`](@extref GeometricMachineLearning :doc:`index`) re-exports all of these,
so a neural network built there is optimized with exactly the machinery described above.


```@raw latex
\section*{Chapter Summary}

In this chapter we discussed mathematical core concepts in this dissertation: aspects from general topology and analysis. We furthermore discussed Riemannian geometry and parallel transport which are crucial concepts for manifold optimization. Homogeneous spaces were introduced as an important subcategory of manifolds and we showed how to find a global tangent space representation for them. This was done by identifying a distinct element $E\in\mathcal{M}$ and then considering the action of the Lie algebra $\mathfrak{g}$ on this element. We have presented our basic application user interface (API) for the \texttt{Julia} package \texttt{GeometricOptimizers}. This API will be extended in Part II with a general optimizer framework. 
```

```@raw latex
\begin{comment}
```

## References

```@bibliography
Pages = []
Canonical = false

absil2004riemannian
absil2008optimization
bendokat2020grassmann
bendokat2021real
Brantner:2023
frankel2011geometry
```

```@raw latex
\end{comment}
```

```@raw html
<!--
```

# References

```@bibliography
Pages = []
Canonical = false 

lipschutz1965general
lang2012real
lang2012fundamentals
bishop1980tensor
do1992riemannian
absil2004riemannian
bendokat2020grassmann
frankel2011geometry
o1983semi
bendokat2021real
```

```@raw html
-->
```