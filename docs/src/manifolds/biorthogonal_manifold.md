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

So ``(\Phi, \Psi)\mapsto\mathrm{diag}(\Phi, \Psi)`` embeds ``\mathcal{B}(n, N)`` in ``Sp(2n, 2N)``, and its image is the set of the symplectic lifts of ``\mathbb{R}^{2n}`` to ``\mathbb{R}^{2N}`` that map positions to positions and momenta to momenta. Biorthogonal weights of this kind are used in [otto2023learning, friedl2026learning](@cite). For ``\Phi = \Psi\in{}St(n, N)``, ``A^+ = A^T``, which is the lift of the proper symplectic decomposition.

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

Neither ``GL(N)`` nor the isotropy group ``GL(N - n)`` is compact, and ``\mathcal{B}(n, N)`` is not compact either: for a pair ``(\Phi, \Psi)`` the curve ``t\mapsto(t\Phi, t^{-1}\Psi)``, ``t > 0``, lies in ``\mathcal{B}(n, N)`` and is unbounded.

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

The manifold carries the metric of ``\mathbb{R}^{N\times2n}``, [`metric(::BiorthogonalManifold, ::AbstractMatrix, ::AbstractMatrix)`](@ref). No metric is invariant under the action, because the isotropy group is not compact; the Euclidean one is the simplest, and at the Stiefel manifold it is twice the Euclidean metric of ``St(n, N)``. Its normal space at ``(\Phi, \Psi)`` is ``\{(\Psi M, \Phi M^T) : M\in\mathbb{R}^{n\times{}n}\}``, and [`rgrad(::BiorthogonalManifold, ::AbstractMatrix)`](@ref) is the orthogonal projection onto the tangent space,

```math
\mathrm{grad}\,L = (\nabla_\Phi L - \Psi M,\ \nabla_\Psi L - \Phi M^T), \qquad \Psi^T\Psi M + M\Phi^T\Phi = \Psi^T\nabla_\Phi L + (\nabla_\Psi L)^T\Phi,
```

a Sylvester equation with two symmetric positive definite coefficients, which is solved in their eigenbases.

## The global tangent space

The global tangent space is introduced on the page [Global Tangent Spaces](@ref "The Global Tangent Space for the Biorthogonal Manifold") in the same five steps as for the Stiefel, the Grassmann and the symplectic Stiefel manifold: ``\mathfrak{g}^\mathrm{hor}`` is the complement ``\left\{\begin{pmatrix} A & C \\ B & \mathbb{O} \end{pmatrix}\right\}`` of the isotropy algebra ``\mathfrak{gl}(N-n)`` in ``\mathfrak{gl}(N)``, stored as a [`BiorthogonalLieAlgHorMatrix`](@ref), and the section stores the frame ``\lambda(Y) = [\Phi\ \ K]`` together with the dual frame ``\lambda(Y)^{-T} = [\Psi\ \ \tilde{K}]``, ``\tilde{K} = (\mathbb{I}_N - \Psi\Phi^T)K``. The retractions below act on these two frames.

## Retractions

A retraction of ``X\in\mathfrak{g}^\mathrm{hor}`` is a group element ``g``, and a step moves the frame to ``\lambda(Y)g`` and the dual frame to ``\lambda(Y)^{-T}g^{-T}``. Both retractions the package ships satisfy ``R(X)^{-1} = R(-X)``: ``\exp(X)^{-1} = \exp(-X)`` and ``\mathrm{Cayley}(X)^{-1} = \mathrm{Cayley}(-X)``. So the inverse is the same retraction of ``-X``, taken through the factorisation of [`lift_factors`](@ref), which here is

```math
\begin{pmatrix} A & C \\ B & \mathbb{O} \end{pmatrix} = \begin{bmatrix} \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix}\begin{bmatrix} \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & C \end{bmatrix},
```

the Stiefel factorisation with ``C`` in place of ``-B^T``, and the cost stays ``O(N^2n)``. [`cayley(::BiorthogonalLieAlgHorMatrix)`](@ref) and [`geodesic(::BiorthogonalLieAlgHorMatrix)`](@ref) return the pair ``[g\ \ g^{-T}]`` as a point of ``\mathcal{B}(N, N)\cong{}GL(N)``, as the retraction of a Stiefel lift returns an orthogonal matrix as a point of ``St(N, N)``.

Two things differ from the Stiefel case. The curve ``t\mapsto\lambda(Y)\exp(tX)\cdot(E, E)`` is the orbit of a one-parameter subgroup, but not a geodesic of the metric, which the group does not preserve. And ``\mathrm{Cayley}(X)`` is defined only where ``\mathbb{I} - \frac{1}{2}X`` and ``\mathbb{I} + \frac{1}{2}X`` are invertible, which for a lift that is not skew-symmetric is not every lift: a step whose lift has an eigenvalue near ``\pm2`` makes the retraction arbitrarily ill-conditioned. [`retraction_differential`](@ref) needs the first ``n`` rows of the differential as well as its first ``n`` columns, because the vertical space is the lower right block alone.

## The relation to the symplectic Stiefel manifold

Under ``X\mapsto\mathrm{diag}(X, -X^T)``, ``\mathfrak{gl}(N)`` is a subalgebra of ``\mathfrak{sp}(2N)``, and the section ``\mathrm{diag}(\lambda(Y), \lambda(Y)^{-T})`` of ``\mathrm{diag}(\Phi, \Psi)`` is symplectic. A biorthogonal lift is then a [`SymplecticLieAlgHorMatrix`](@ref), as the page [Global Tangent Spaces](@ref "The Global Tangent Space for the Biorthogonal Manifold") states block by block, and its retractions are the block-diagonal retractions of that lift. A symplectic optimizer started at ``\mathrm{diag}(\Phi, \Psi)`` would leave the block-diagonal points, because the gradient of a loss on ``Sp(2n, 2N)`` has components outside this subalgebra; optimizing on ``\mathcal{B}(n, N)`` keeps the structure and works with matrices of half the size.

## The optimizers

[`GradientMethod`](@ref), [`MomentumMethod`](@ref), [`Adam`](@ref), [`BFGS`](@ref) and [`DFP`](@ref) take a [`BiorthogonalManifold`](@ref) and a parameter set with such a leaf, under [`Optimizer`](@ref) and, for the first three, under [`TrainingOptimizer`](@ref) and in a [`CompositeMethod`](@ref). The line search, [`BFGS`](@ref) and [`DFP`](@ref) pair [`GeometricOptimizers.gradient_rep!`](@ref), the derivative of the loss with respect to the stored blocks, with a direction, so the slope of a line search is the derivative of its merit.

With [`MetricDescent`](@ref), the default, the gradient [`GradientMethod`](@ref), [`MomentumMethod`](@ref) and [`Adam`](@ref) hold is [`global_rep`](@ref) of the Riemannian gradient, and the path of [`GradientMethod`](@ref) does not depend on the section. With [`SectionDescent`](@ref) it is that derivative, i.e. the steepest descent of ``\mathrm{tr}(X^TY)`` on the lifts at the section. For [`Adam`](@ref) the difference matters: its direction is close to the sign of the gradient it holds, entry by entry, and that is a descent direction in the coordinates the derivative pairs with, which are those of `SectionDescent`. With `MetricDescent` the coordinates are the dual frame's, and on the test problem of `test/integration/biorthogonal_optimizer_tests.jl` the result of a fixed number of steps then depends on the completion ``K`` drawn at the start, while with `SectionDescent` it does not. `Adam(; descent = SectionDescent())` is therefore the one to use here; on a [`StiefelManifold`](@ref) or a [`GrassmannManifold`](@ref) the two are the same, so a parameter set with leaves of several kinds loses nothing by it.

The section is transported along the path by elements of ``GL(N)``, so, unlike a Stiefel section, it does not stay orthogonal, and the coordinates of the lift are only as well conditioned as it is. [`AdamWithEuclideanDecay`](@ref) refuses the manifold: it is not compact, and what Euclidean weight decay should do on it is not decided.

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
```

```@raw latex
\end{comment}
```
