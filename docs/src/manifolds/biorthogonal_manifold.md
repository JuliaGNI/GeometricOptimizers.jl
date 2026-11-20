# The Biorthogonal Manifold

The *biorthogonal manifold* is the set of pairs of ``N\times{}n`` matrices whose columns are dual to each other:

```math
\mathcal{B}(n, N) = \{(\Phi, \Psi)\in\mathbb{R}^{N\times{}n}\times\mathbb{R}^{N\times{}n} : \Psi^T\Phi = \mathbb{I}_n\}.
```

[`BiorthogonalManifold`](@ref) stores a pair as the ``N\times2n`` matrix ``[\Phi\ \ \Psi]``. The pairs with ``\Phi = \Psi`` are the points of the [Stiefel manifold](@ref "The Stiefel Manifold"), so ``\mathcal{B}(n, N)`` contains ``St(n, N)``. The map ``(\Phi, \Psi)\mapsto\Phi\Psi^T`` sends ``\mathcal{B}(n, N)`` onto the set of projectors of rank ``n`` in ``\mathbb{R}^{N\times{}N}``, and ``St(n, N)`` onto the set of orthogonal ones; its fibers are the orbits ``\{(\Phi h, \Psi h^{-T}) : h\in{}GL(n)\}``.

## A symplectic lift

For ``\Phi, \Psi\in\mathbb{R}^{N\times{}n}`` let

```math
A = \begin{bmatrix} \Phi & \mathbb{O} \\ \mathbb{O} & \Psi \end{bmatrix}\in\mathbb{R}^{2N\times2n}, \qquad
\mathbb{J}_{2N} = \begin{bmatrix} \mathbb{O} & \mathbb{I}_N \\ -\mathbb{I}_N & \mathbb{O} \end{bmatrix}.
```

Then

```math
A^T\mathbb{J}_{2N}A = \begin{bmatrix} \mathbb{O} & \Phi^T\Psi \\ -\Psi^T\Phi & \mathbb{O} \end{bmatrix},
```

so ``A`` is a point of the [`SymplecticStiefelManifold`](@ref) ``Sp(2n, 2N)`` exactly when ``(\Phi, \Psi)\in\mathcal{B}(n, N)``. Its symplectic inverse is

```math
A^+ = \mathbb{J}_{2n}^TA^T\mathbb{J}_{2N} = \begin{bmatrix} \Psi^T & \mathbb{O} \\ \mathbb{O} & \Phi^T \end{bmatrix}, \qquad A^+A = \mathbb{I}_{2n}.
```

So ``(\Phi, \Psi)\mapsto\mathrm{diag}(\Phi, \Psi)`` embeds ``\mathcal{B}(n, N)`` in ``Sp(2n, 2N)``, and its image is the set of the symplectic lifts of ``\mathbb{R}^{2n}`` to ``\mathbb{R}^{2N}`` that map positions to positions and momenta to momenta. Biorthogonal weights of this kind are used in [otto2023learning, friedl2026learning](@cite). For ``\Phi = \Psi\in{}St(n, N)``, ``A^+ = A^T``, which is the lift of the proper symplectic decomposition, and it is the only symplectic lift of this form with two orthonormal frames: if ``\Phi, \Psi\in{}St(n, N)`` and ``\Psi^T\Phi = \mathbb{I}_n``, then ``\|\Phi - \Psi\|_F^2 = 2n - 2\,\mathrm{Tr}(\Psi^T\Phi) = 0``. A pair ``(\Phi h, \Psi h^{-T})``, ``h\in{}GL(n)``, of the fiber above has the lift ``A\,\mathrm{diag}(h, h^{-T})``, which differs from ``A`` by a symplectic change of the coordinates of ``\mathbb{R}^{2n}``.

## A homogeneous space of the general linear group

The general linear group ``GL(N)`` acts on ``\mathbb{R}^{N\times{}n}\times\mathbb{R}^{N\times{}n}`` by

```math
g\cdot(\Phi, \Psi) = (g\Phi, g^{-T}\Psi),
```

which keeps ``\Psi^T\Phi`` fixed, since ``(g^{-T}\Psi)^Tg\Phi = \Psi^Tg^{-1}g\Phi = \Psi^T\Phi``. Under the embedding above this is the action of the block-diagonal subgroup ``\{\mathrm{diag}(g, g^{-T}) : g\in{}GL(N)\}`` of the symplectic group ``Sp(2N)``.

The action is transitive on ``\mathcal{B}(n, N)``. Let ``E = \begin{bmatrix} \mathbb{I}_n \\ \mathbb{O} \end{bmatrix}\in\mathbb{R}^{N\times{}n}`` (the [`StiefelProjection`](@ref)), take a pair ``(\Phi, \Psi)``, and let the columns of ``K\in\mathbb{R}^{N\times(N-n)}`` be a basis of the kernel of ``\Psi^T``. The matrix ``G = [\Phi\ \ K]`` is invertible: if ``\Phi{}a + Kb = 0``, multiplying by ``\Psi^T`` gives ``a = 0``, and then ``b = 0`` because the columns of ``K`` are linearly independent. Since ``GE = \Phi`` and

```math
G^T\Psi = \begin{bmatrix} \Phi^T\Psi \\ K^T\Psi \end{bmatrix} = \begin{bmatrix} \mathbb{I}_n \\ \mathbb{O} \end{bmatrix} = E,
```

we have ``G\cdot(E, E) = (\Phi, \Psi)``.

The stabilizer of ``(E, E)`` consists of the ``g`` with ``gE = E`` and ``g^TE = E``: the first ``n`` columns and the first ``n`` rows of ``g`` are those of the identity,

```math
g = \begin{bmatrix} \mathbb{I}_n & \mathbb{O} \\ \mathbb{O} & B \end{bmatrix}, \qquad B\in{}GL(N-n).
```

Hence

```math
\mathcal{B}(n, N)\cong{}GL(N)/GL(N-n), \qquad \dim\mathcal{B}(n, N) = N^2 - (N-n)^2 = 2Nn - n^2,
```

``2Nn`` entries and ``n^2`` conditions ``\Psi^T\Phi = \mathbb{I}_n``. This is the counterpart of ``St(n, N)\cong{}O(N)/O(N-n)``: the involution ``g\mapsto{}g^{-T}`` of ``GL(N)`` fixes exactly ``O(N)``, and the involution ``(\Phi, \Psi)\mapsto(\Psi, \Phi)`` of ``\mathcal{B}(n, N)`` fixes exactly the pairs with ``\Phi = \Psi`` and ``\Phi^T\Phi = \mathbb{I}_n``, i.e. ``St(n, N)``. For ``n = N`` the manifold is ``GL(N)`` itself, as the pairs ``(g, g^{-T})``.

``GL(N)`` is not compact, nor, for ``n < N``, is the isotropy group ``GL(N - n)`` (for ``n = N`` it is trivial), and ``\mathcal{B}(n, N)`` is not compact either: for a pair ``(\Phi, \Psi)`` the curve ``t\mapsto(t\Phi, t^{-1}\Psi)``, ``t > 0``, lies in ``\mathcal{B}(n, N)`` and is unbounded.

## The bundle over the first factor

The two factors of a pair are not independent, but ``\mathcal{B}(n, N)`` is a bundle over the first one. The projection ``(\Phi, \Psi)\mapsto\Phi`` maps onto the matrices of full rank ``n``, and its fiber over ``\Phi`` is the affine space

```math
\{\Phi(\Phi^T\Phi)^{-1} + \Psi_\perp : \Psi_\perp\in\mathbb{R}^{N\times{}n},\ \Psi_\perp^T\Phi = \mathbb{O}\}
```

of dimension ``(N-n)n``. A matrix ``\Phi`` of full rank is in turn ``\Phi = UR`` with ``U\in{}St(n, N)`` spanning its columns and ``R\in{}GL(n)``, unique up to ``(U, R)\mapsto(UQ, Q^TR)`` with ``Q\in{}O(n)``. Together,

```math
\Phi = UR, \qquad \Psi = UR^{-T} + (\mathbb{I}_N - UU^T)W, \qquad U\in{}St(n, N),\ R\in{}GL(n),\ W\in\mathbb{R}^{N\times{}n},
```

is a parametrization of ``\mathcal{B}(n, N)`` by a Stiefel manifold and two vector spaces: ``\Psi^T\Phi = R^{-1}U^TUR + W^T(\mathbb{I}_N - UU^T)UR = \mathbb{I}_n`` for every ``U``, every invertible ``R`` and every ``W``, and every pair is reached, since ``UR^{-T} = \Phi(\Phi^T\Phi)^{-1}`` and ``(\mathbb{I}_N - UU^T)W`` runs through all ``\Psi_\perp``. It is redundant in the ``O(n)`` above and in the part ``UU^TW`` of ``W``, which does not enter ``\Psi``, and the count checks out: ``\dim{}St(n, N) + n^2 + (N-n)n - \dim{}O(n) = 2Nn - n^2``. It lets a [`StiefelManifold`](@ref) optimizer train ``U`` and a Euclidean one ``R`` and ``W``, at the price of keeping ``R`` invertible by hand. The rest of this page optimizes on ``\mathcal{B}(n, N)`` directly, which needs neither. [`rand(::Type{<:BiorthogonalManifold}, ::Integer, ::Integer)`](@ref) draws a point through the fiber.

## Tangent spaces and the metric

Differentiating ``\Psi^T\Phi = \mathbb{I}_n`` gives the tangent space at ``(\Phi, \Psi)``:

```math
T_{(\Phi, \Psi)}\mathcal{B}(n, N) = \{(\Delta_\Phi, \Delta_\Psi) : \Psi^T\Delta_\Phi + \Delta_\Psi^T\Phi = \mathbb{O}\}.
```

The manifold carries the metric of ``\mathbb{R}^{N\times2n}``, [`metric(::BiorthogonalManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref). For ``n < N`` no metric is invariant under the action, because the isotropy group ``GL(N - n)`` at ``(E, E)`` contains ``\mathrm{diag}(\mathbb{I}_n, t\mathbb{I}_{N-n})``, which multiplies the tangent vectors ``(\Delta_\Phi, \mathbb{O})`` with ``E^T\Delta_\Phi = \mathbb{O}`` by ``t``; for ``n = N`` the manifold is ``GL(N)``, and its left-invariant metrics are invariant. The Euclidean one is the simplest, and at the Stiefel manifold it is twice the Euclidean metric of ``St(n, N)``. Its normal space at ``(\Phi, \Psi)`` is ``\{(\Psi M, \Phi M^T) : M\in\mathbb{R}^{n\times{}n}\}``, and [`rgrad(::BiorthogonalManifold, ::AbstractMatrix)`](@ref) is the orthogonal projection onto the tangent space,

```math
\mathrm{grad}\,L = (\nabla_\Phi L - \Psi M,\ \nabla_\Psi L - \Phi M^T), \qquad \Psi^T\Psi M + M\Phi^T\Phi = \Psi^T\nabla_\Phi L + (\nabla_\Psi L)^T\Phi,
```

a Sylvester equation with two symmetric positive definite coefficients, which is solved in their eigenbases.

## The global tangent space

The global tangent space is introduced on the page [Global Tangent Spaces](@ref "The Global Tangent Space for the Biorthogonal Manifold") in the same five steps as for the Stiefel, the Grassmann and the symplectic Stiefel manifold: ``\mathfrak{g}^\mathrm{hor}`` is the complement ``\left\{\begin{pmatrix} A & C \\ B & \mathbb{O} \end{pmatrix}\right\}`` of the isotropy algebra ``\mathfrak{gl}(N-n)`` in ``\mathfrak{gl}(N)``, stored as a [`BiorthogonalLieAlgHorMatrix`](@ref), and the section stores the frame ``\lambda(Y) = [\Phi\ \ K]`` together with the dual frame ``\lambda(Y)^{-T} = [\Psi\ \ \tilde{K}]``. For the orthonormal ``K`` of [`global_section`](@ref), ``\tilde{K} = (\mathbb{I}_N - \Psi\Phi^T)K``; for any other ``K`` it is ``(\mathbb{I}_N - \Psi\Phi^T)K(K^TK)^{-1}``. The retractions below act on these two frames.

## Retractions on the Biorthogonal Manifold

A retraction of ``X\in\mathfrak{g}^\mathrm{hor}`` is a group element ``g``, and a step moves the frame to ``\lambda(Y)g`` and the dual frame to ``\lambda(Y)^{-T}g^{-T}``. Both retractions the package ships satisfy ``R(X)^{-1} = R(-X)``: ``\exp(X)^{-1} = \exp(-X)`` and ``\mathrm{Cayley}(X)^{-1} = \mathrm{Cayley}(-X)``. So the inverse is the same retraction of ``-X``, taken through the factorisation of [`GeometricOptimizers.lift_factors`](@ref), which here is

```math
\begin{pmatrix} A & C \\ B & \mathbb{O} \end{pmatrix} = \begin{bmatrix} \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix}\begin{bmatrix} \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & C \end{bmatrix},
```

the Stiefel factorisation with ``C`` in place of ``-B^T``, and the cost stays ``O(N^2n)``. [`cayley(::BiorthogonalLieAlgHorMatrix)`](@ref) and [`geodesic(::BiorthogonalLieAlgHorMatrix)`](@ref) return the pair ``[g\ \ g^{-T}]`` as a point of ``\mathcal{B}(N, N)\cong{}GL(N)``, as the retraction of a Stiefel lift returns an orthogonal matrix as a point of ``St(N, N)``.

Two things differ from the Stiefel case. The curve ``t\mapsto\lambda(Y)\exp(tX)\cdot(E, E)`` is the orbit of a one-parameter subgroup, but not a geodesic of the metric, which the group does not preserve. And ``\mathrm{Cayley}(X)`` is defined only where ``\mathbb{I} - \frac{1}{2}X`` and ``\mathbb{I} + \frac{1}{2}X`` are invertible, which for a lift that is not skew-symmetric is not every lift: a step whose lift has an eigenvalue near ``\pm2`` makes the retraction arbitrarily ill-conditioned. [`GeometricOptimizers.retraction_differential`](@ref) needs the first ``n`` rows of the differential as well as its first ``n`` columns, because the vertical space is the lower right block alone.

## The relation to the symplectic Stiefel manifold

Under ``X\mapsto\mathrm{diag}(X, -X^T)``, ``\mathfrak{gl}(N)`` is a subalgebra of ``\mathfrak{sp}(2N)``, and the section ``\mathrm{diag}(\lambda(Y), \lambda(Y)^{-T})`` of ``\mathrm{diag}(\Phi, \Psi)`` is symplectic. A biorthogonal lift is then a [`SymplecticLieAlgHorMatrix`](@ref), as the page [Global Tangent Spaces](@ref "The Global Tangent Space for the Biorthogonal Manifold") states block by block, and its retractions are the block-diagonal retractions of that lift. A symplectic optimizer started at ``\mathrm{diag}(\Phi, \Psi)`` would leave the block-diagonal points, because the gradient of a loss on ``Sp(2n, 2N)`` has components outside this subalgebra; optimizing on ``\mathcal{B}(n, N)`` keeps the structure and works with matrices of half the size.

## The optimizers

[`GradientMethod`](@ref), [`MomentumMethod`](@ref), [`Adam`](@ref), [`BFGS`](@ref) and [`DFP`](@ref) take a [`BiorthogonalManifold`](@ref) and a parameter set with such a leaf, under [`Optimizer`](@ref) and, for the first three, under [`TrainingOptimizer`](@ref) and in a [`CompositeMethod`](@ref). The line search, [`BFGS`](@ref) and [`DFP`](@ref) pair [`GeometricOptimizers.gradient_rep!`](@ref), the derivative of the loss with respect to the stored blocks, with a direction, so the slope of a line search is the derivative of its merit.

The gradient [`GradientMethod`](@ref) and [`MomentumMethod`](@ref) hold is [`global_rep`](@ref) of the Riemannian gradient, and the path of [`GradientMethod`](@ref) does not depend on the section. The derivative with respect to the stored blocks is that lift times the Gram matrix ``M`` of the metric in the stored blocks, and ``M`` is not the identity here, not even at a Stiefel point. [`BFGS`](@ref) and [`DFP`](@ref) start their inverse Hessian at ``M^{-1}`` ([`GeometricOptimizers._metric_inverse!`](@ref)), so their first step, and their step after a restart, is the one of `GradientMethod`. The line search caps the step at ``c/\rho(X)`` ([`GeometricOptimizers.step_αmax`](@ref)). A real eigenvalue ``\mu\neq0`` of ``X`` puts a pole of ``\mathrm{Cayley}(\alpha{}X)`` or of its inverse at ``\alpha = 2/|\mu|``, and a complex one puts none on the real axis, so for the default ``c = 1`` the cap is half the distance to the nearest pole when an eigenvalue of largest modulus is real, and less than half otherwise. A change of the section keeps the eigenvalues of the lift and transforms ``M`` with its coordinates, so under a line search the paths of `GradientMethod`, `BFGS` and `DFP` do not depend on the section.

The path of [`Adam`](@ref) does, through its entrywise moments, and [`Adam`](@ref) holds [`GeometricOptimizers.gradient_rep!`](@ref) instead of `global_rep`, for the reason the next section gives.

## Adam on the biorthogonal manifold

Fix a section ``\lambda(Y) = [\Phi\ \ K]`` and a metric ``g``, and let ``M`` be the Gram matrix of ``g`` in the stored blocks ``(A, B, C)`` of the lift ``\begin{pmatrix} A & C \\ B & \mathbb{O}\end{pmatrix}\in\mathfrak{g}^\mathrm{hor}``. If ``b`` is the vector of these blocks for `global_rep` of the Riemannian gradient of ``g``, a step along the lift with blocks ``w`` changes the loss at the rate ``b^TMw``. The first direction of [`Adam`](@ref) is ``w = -Db`` with the positive diagonal ``D = \mathrm{diag}\big(1/(|b_i| + \delta)\big)``, so its rate is ``-b^TMDb``. [`GradientMethod`](@ref) and [`MomentumMethod`](@ref) have ``D = \mathbb{I}`` and the rate ``-b^TMb < 0`` for every ``M``. For a positive diagonal ``D`` the rate is negative for every ``b`` and every ``D`` if and only if ``M`` is diagonal: for ``m_{ij}\neq0``, ``b = e_i + se_j`` with ``s = -m_{ij}/(2m_{jj})`` and ``D = \mathrm{diag}(\varepsilon, 1)`` on ``(i, j)`` give a positive rate for small ``\varepsilon``. The ``D`` of Adam depends on ``b``, and for ``\delta\to0`` its direction is ``-s`` with ``s = \mathrm{sign}(b)``. The rate ``-b^TMs`` is negative for every ``b\neq0`` if ``M`` is strictly diagonally dominant by rows, since ``b^TMs\geq\sum_i|b_i|\big(m_{ii} - \sum_{j\neq{}i}|m_{ij}|\big)``. If ``m_{ii} < \sum_{j\neq{}i}|m_{ij}|`` for some ``i``, it is positive for ``b = e_i + \varepsilon{}s`` with ``s_i = 1``, ``s_j = -1`` where ``m_{ij} > 0``, ``s_j = 1`` elsewhere, and small ``\varepsilon > 0``, since then ``\mathrm{sign}(b) = s`` and ``b^TMs = m_{ii} - \sum_{j\neq{}i}|m_{ij}| + O(\varepsilon)``. A sign-like rescaling turns the gradient by more than ``90^\circ`` in the metric ``M``, although by less than ``90^\circ`` in the coordinates.

On the [`StiefelManifold`](@ref) with the canonical metric, ``M = \mathbb{I}`` for every section, because a change of section multiplies the ``B`` block by an orthogonal matrix. On ``\mathcal{B}(n, N)``, a change of section ``\lambda\mapsto\lambda\,\mathrm{diag}(\mathbb{I}_n, S)``, ``S\in{}GL(N-n)``, maps the blocks of a fixed tangent vector to ``(A, S^{-1}B, CS)``, and ``S = t\mathbb{I}_{N-n}`` scales ``B`` by ``t^{-1}`` and ``C`` by ``t``. No inner product is invariant under this, so for ``n < N`` the Gram matrix of any metric depends on the section, and for ``N - n\geq2`` every point has sections for which it is not diagonal. For ``N - n = 1``, ``S`` is a scalar, and a metric whose ``M`` is diagonal for one section has a diagonal ``M`` for every section. This is a property of the non-compact isotropy group ``GL(N-n)``, not of the choice of ``\mathfrak{g}^\mathrm{hor}``: the isotropy algebra acts trivially at ``(E, E)``, so every complement of it gives the same blocks and the same change of section. For the metric of [`metric(::BiorthogonalManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref), the inherited one, ``M`` couples ``A`` and ``B`` through ``2\,\mathrm{Tr}(A^T\Phi^TKB)``, and ``\Phi^TK\neq\mathbb{O}`` exactly where ``\mathrm{span}\,\Phi\neq\mathrm{span}\,\Psi``. So at every such point ``M`` is not diagonal for any section. The change of section ``S = t\mathbb{I}_{N-n}`` multiplies this coupling by ``t`` and keeps the ``A`` block of ``M``, so for large ``t`` a row of ``A`` is not diagonally dominant: every such point has sections, and gradients, for which the first step of Adam on `global_rep` raises the loss. At a Stiefel point with an orthonormal ``K``, ``M`` is diagonal, which is where a lift initialized as a cotangent lift starts; it leaves these points with its first step.

[`GeometricOptimizers.gradient_rep!`](@ref) holds the derivative of the loss with respect to the stored blocks, the pulled-back differential ``b^\natural = Mb``, which needs neither the metric nor the Sylvester equation of [`rgrad`](@ref). The rate of a step ``w`` is ``(b^\natural)^Tw``, so the first direction of Adam on it lowers the loss at the rate ``-\sum_i(b^\natural_i)^2/(|b^\natural_i| + \delta) < 0``, for every section and every metric, unless ``b^\natural = 0``. This is the gradient [`Adam`](@ref) holds on a [`BiorthogonalManifold`](@ref). It is `global_rep` of the gradient of the metric that makes the blocks of the current section orthonormal; on the sections with an orthonormal ``K``, which differ by ``S\in{}O(N-n)``, that metric is one and the same,

```math
g^\natural_{(\Phi, \Psi)}(\Delta, \Delta) = \big\|[\Phi\ \ K]^{-1}\Delta_\Phi\big\|_F^2 + \big\|(\mathbb{I}_N - \Psi(\Psi^T\Psi)^{-1}\Psi^T)\Delta_\Psi\big\|_F^2.
```

The section transported by the retraction leaves this set after the first step. The guarantee covers the first step only: later steps rescale the first moment, a combination of past gradients, which need not be a descent direction of the current loss, on ``\mathbb{R}^d`` as here. The over-parametrization of [otto2023learning](@cite), Euclidean Adam on unconstrained pairs mapped to ``\mathcal{B}(n, N)``, also takes the derivative of the loss pulled back to the coordinates it rescales, and its first step descends for the same reason; its step depends on the representative in the non-compact fibre ``GL(n)``, as the step here depends on the section.

That Adam depends on the coordinate system is known, in ``\mathbb{R}^d`` as on manifolds. In [xie2025adam; Theorem 2.2](@cite) the authors show that "`SGD` and `AdaSGD` are rotation-equivariant. `Adam` and `SignGD` are only permutation-equivariant", and in [zhang2025understanding; Section 2.1](@cite) that "Adam is not rotation equivariant, due to its element-wise division". In [becigneul2019riemannian; Section 1](@cite) the authors write that "on a Riemannian manifold, one is generally not given an intrinsic coordinate system, rendering meaningless the notions sparsity or coordinate-wise update", in [kasai2019riemannian; Section 1](@cite) they name "the absence of a canonical coordinate system on manifolds" as a key difficulty, and in [Brantner:2023; Section 3.1](@cite) the Hadamard product and the elementwise division "are coordinate-dependent and are therefore not easily generalizable to arbitrary manifolds". These works treat the dependence as a loss of equivariance or of meaning; the loss of descent above is not specific to manifolds either, and arises on ``\mathbb{R}^d`` for an entrywise rescaling of any gradient whose Gram matrix is not diagonal.

`scripts/adam_pullback_severity.jl` measures what holding `global_rep` costs, at the defaults of [`Adam`](@ref), over 50 seeds each:

| problem | `global_rep` | `gradient_rep!` |
|:--|:--|:--|
| the test problem of `test/integration/biorthogonal_optimizer_tests.jl`, [`Optimizer`](@ref), 500 steps at ``10^{-3}`` | final 44.9 to 8982 from 1405, 4 above the start | 38.7 to 45.2 |
| the same, [`TrainingOptimizer`](@ref), 200 steps at ``10^{-3}`` | 132.6 to 1520.5, 36 above a fifth of the start | 48.7 to 69.3 |
| ``\|Y - Y_0\|_F^2``, ``(N, n) = (2, 1)``, 1000 steps at ``10^{-2}`` | 2 first steps ascend, 1 run ends above its start | 0, 0 |
| the same, ``(N, n) = (6, 2)`` | 4 first steps ascend, 19 runs end above their start | 0, 0 |
| the same, ``(N, n) = (20, 3)`` | 5 first steps ascend, 48 runs end above their start, 17 off the manifold | 0, 0, 0 |

The [`SymplecticStiefelManifold`](@ref), whose isotropy group is not compact either, is in the same position (K57), and the script measures it too; there [`Adam`](@ref) still holds `global_rep`.

The section is transported along the path by elements of ``GL(N)``, so, unlike a Stiefel section, it does not stay orthogonal, and the coordinates of the lift are only as well conditioned as it is. [`AdamWithEuclideanDecay`](@ref) refuses the manifold with an `ArgumentError`: it is not compact, so ``\frac{\lambda}{2}\|Y\|_F^2`` is not constant on it, and a horizontal lift cannot subtract ``\lambda{}Y`` entrywise.

```@raw latex
\begin{comment}
```

## References

```@bibliography
Pages = []
Canonical = false

otto2023learning
friedl2026learning
bendokat2021real
Brantner:2023
becigneul2019riemannian
kasai2019riemannian
xie2025adam
zhang2025understanding
```

```@raw latex
\end{comment}
```
