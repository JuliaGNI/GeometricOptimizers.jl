```@meta
CurrentModule = GeometricOptimizers
```

# Optimization on Homogeneous Spaces

The manifold optimizers in this package implement [brantner2023generalizing](@cite). This page summarizes that paper: what problem it solves, how it solves it, and which function in `GeometricOptimizers` corresponds to which operation in the algorithm.

## The problem: Adam has no coordinate-free formulation

Standard optimizers such as gradient descent generalize to Riemannian manifolds without much trouble: they only need a gradient and a retraction, both of which are defined intrinsically. Adam does not. Adam assigns *one learning rate per coordinate*, because its second moment ``\mathcal{B}^\mathtt{cache}_2`` is accumulated and divided entry by entry. On a manifold there is no intrinsic coordinate system, so "coordinate-wise update" is not a well-defined operation, and the cache cannot simply be carried from one tangent space ``T_{Y^{(t)}}\mathcal{M}`` to the next.

Existing generalizations resolve this either by restricting to Lie groups — where the Lie algebra ``\mathfrak{g}`` is a global tangent space and the cache can live there — or by keeping the cache in an ambient vector space and projecting back onto the manifold at every step, which either loses the vector-valued second moment or introduces a projection.

## The idea: a global tangent space for homogeneous spaces

A **homogeneous space** is a manifold ``\mathcal{M}`` on which a Lie group ``G`` acts transitively. Fixing a distinct element ``E \in \mathcal{M}``, every ``Y \in \mathcal{M}`` is ``Y = \Lambda{}E`` for some ``\Lambda \in G``, and a **section** is a map ``\lambda: \mathcal{M} \to G`` with ``\lambda(Y)E = Y``. This is [`GlobalSection`](@ref).

For the [`StiefelManifold`](@ref) ``St(n, N) = \{Y \in \mathbb{R}^{N \times n} : Y^TY = \mathbb{I}_n\}`` the group is ``G = SO(N)``, the distinct element is ``E = [\mathbb{I}_n; \mathbb{O}]``, and the section is computed by completing ``Y`` to an orthonormal basis with a ``QR`` decomposition: ``\lambda(Y) = [Y, Y_\perp]``.

The kernel of ``\mathfrak{g} \to T_Y\mathcal{M}`` is the *vertical* component ``\mathfrak{g}^{\mathrm{ver},Y}``; its orthogonal complement in ``\mathfrak{g}`` is the *horizontal* component ``\mathfrak{g}^{\mathrm{hor},Y} \simeq T_Y\mathcal{M}``. Evaluating this at the distinct element gives the **global tangent space representation**

```math
\mathfrak{g}^\mathrm{hor} \equiv \mathfrak{g}^{\mathrm{hor},E},
```

which is the same space for every ``Y``. For the Stiefel manifold it has the sparse form

```math
\mathfrak{g}^\mathrm{hor} = \left\{
    \begin{bmatrix} A & -B^T \\ B & \mathbb{O} \end{bmatrix} :
    A \in \mathbb{R}^{n \times n} \text{ skew-symmetric}, \ B \in \mathbb{R}^{(N-n) \times n} \text{ arbitrary}
\right\},
```

which is what [`StiefelLieAlgHorMatrix`](@ref) stores — only ``A`` and ``B``, never the full ``N \times N`` matrix. The analogue for the Grassmann manifold is [`GrassmannLieAlgHorMatrix`](@ref).

**This is the key point of the paper.** The Adam cache is kept in ``\mathfrak{g}^\mathrm{hor}``, a fixed vector space that does not depend on the current iterate. Coordinate-wise operations are meaningful there, so the Adam update can be written down unchanged. And because

```math
\dim{}St(n, N) = \dim\mathfrak{g}^{\mathrm{hor},Y} = \dim\mathfrak{g}^\mathrm{hor} = n(N - n) + \tfrac{1}{2}n(n-1),
```

no dimensions are added anywhere and **no projection is needed** — unlike approaches that carry the cache in ``\mathbb{R}^{N \times n} \times \mathfrak{so}(n)``.

## The algorithm

One step, for weights ``Y^{(t)}``, a cache, the Euclidean gradient ``\nabla{}L`` from automatic differentiation, optimizer parameters ``\Xi`` and a section ``\Lambda^{(t)}``:

```math
\begin{aligned}
\Delta^{(t)}     &\gets \mathtt{rgrad}(Y^{(t)}, \nabla{}L)                    && \text{Riemannian gradient, an element of } T_{Y^{(t)}}\mathcal{M} \\
\mathcal{B}^{(t)} &\gets \mathtt{global\_rep}(\Lambda^{(t)}, \Delta^{(t)})     && \text{lift it to } \mathfrak{g}^\mathrm{hor} \\
\mathtt{cache}   &\gets \mathtt{update}(\mathtt{cache}, \mathcal{B}^{(t)}, t, \Xi) && \text{the ordinary optimizer update, in } \mathfrak{g}^\mathrm{hor} \\
W^{(t)}          &\gets \mathtt{velocity}(\mathtt{cache}, \Xi)                 && \\
\Lambda^{(t+1)}  &\gets \mathtt{update\_section}(\Lambda^{(t)}, W^{(t)}) = \Lambda^{(t)}\,\mathrm{retraction}(W^{(t)}) && \\
Y^{(t+1)}        &\gets \Lambda^{(t+1)}E &&
\end{aligned}
```

Only the first, second and last two lines are new relative to a Euclidean optimizer; `update` and `velocity` — that is, the *definition* of Adam, momentum or plain gradient descent — are untouched.

### The Riemannian gradient

Converts the Euclidean gradient into a Riemannian one, i.e. into an element of ``T_Y\mathcal{M}``, via the metric:

```math
\mathrm{Tr}\!\left((\nabla_YL)^TV\right) = g_Y(\mathtt{rgrad}(Y, \nabla_YL), V) \quad \forall V \in T_Y\mathcal{M}.
```

For the Stiefel manifold with the canonical metric ``g_Y(V_1, V_2) = \mathrm{Tr}(V_1^T(\mathbb{I} - \tfrac{1}{2}YY^T)V_2)`` this is ``\mathrm{grad}_YL = \nabla{}L - Y\nabla{}L^TY``. In the package: [`rgrad`](@ref rgrad(::StiefelManifold, ::AbstractMatrix)).

### The lift to the global tangent space

Maps ``T_Y\mathcal{M} \to \mathfrak{g}^\mathrm{hor}``. It is the composition of two isomorphisms — ``\Omega: T_Y\mathcal{M} \to \mathfrak{g}^{\mathrm{hor},Y}``,

```math
\Omega(V_Y) = \left(\mathbb{I} - \tfrac{1}{2}YY^T\right)V_YY^T - YV_Y^T\left(\mathbb{I} - \tfrac{1}{2}YY^T\right),
```

and the conjugation ``Z \mapsto \Lambda^{-1}Z\Lambda`` that moves ``\mathfrak{g}^{\mathrm{hor},Y}`` to ``\mathfrak{g}^{\mathrm{hor},E}``. In practice the two are done at once: with ``\Lambda = [Y, Y_\perp]`` the result is just ``A = Y^T\Delta`` and ``B = Y_\perp^T\Delta``. In the package: [`global_rep`](@ref global_rep(::GlobalSection{T,AT}, ::AbstractMatrix{T}) where {T, AT<:StiefelManifold{T}}).

### The extended retraction

A classical retraction maps ``T_Y\mathcal{M} \to \mathcal{M}``. Here the argument lives in ``\mathfrak{g}^\mathrm{hor}`` instead, so the paper defines an **extended retraction** ``\overline{\mathrm{retraction}}: \mathfrak{g}^\mathrm{hor} \to \mathcal{M}``, characterized by ``\overline{\mathrm{retraction}} \circ \Omega`` being a classical retraction. Computationally it splits into `update_section` (right-multiply the section by ``\mathrm{retraction}(W^{(t)})``) and `apply_section` (right-multiply by ``E``).

Two choices ship with the package: `Geodesic`, the closed-form geodesic of the Stiefel manifold, and `Cayley`, the Cayley transform. Both exploit the sparsity of ``\mathfrak{g}^\mathrm{hor}``, so their cost is set by ``n``, not by ``N``.

Note that the section is *not* recomputed at every step. It is built once, when the optimizer is initialized (the ``QR`` decomposition above), and thereafter parallel-transported along the optimization trajectory by `update_section`.

## The Euclidean case falls out

If ``\mathcal{M}`` is a vector space ``\mathcal{V}`` then ``\mathcal{M} \equiv T_Y\mathcal{M} \equiv \mathfrak{g}^\mathrm{hor} = \mathcal{V}``, the distinct element is ``E = \mathbb{O}``, `rgrad` and `global_rep` are the identity, and the extended retraction is addition. The algorithm above collapses to textbook Adam. This is why the same [`Adam`](@ref), [`MomentumMethod`](@ref) and `GradientMethod` in this package work on a plain `Array` and on a [`StiefelManifold`](@ref) without any change on the caller's side: the method supplies `update` and `velocity`, the parameter type supplies everything else.

The construction is not specific to Adam either. Any first-order method whose cache is a vector space element — RMSProp, AdaGrad, BFGS — generalizes the same way, and it applies to any homogeneous space, including the [`GrassmannManifold`](@ref) and the symplectic Stiefel and Grassmann manifolds.

## The numerical experiment

The paper trains a vision transformer on MNIST and Fashion-MNIST: 16 transformer blocks, 7 attention heads, ``49 \times 16`` inputs (``28 \times 28`` images cut into 16 patches of ``7 \times 7``), batch size 2048, 500 epochs, `Float32`, no hyperparameter tuning. The projection matrices ``W^Q_i, W^K_i, W^V_i`` of the multi-head attention layers are the weights that are put on ``St(7, 49)``; the feedforward and classification weights stay unconstrained. Four configurations are compared:

| weights | optimizer | outcome |
|---|---|---|
| unconstrained | Adam | does not learn — loss stuck at ``\approx 1.34`` |
| Stiefel | gradient descent | learns |
| Stiefel | momentum | learns, second best |
| Stiefel | Adam | learns, **best** |

Constraining the projections to the Stiefel manifold is what makes the network trainable at all. The unconstrained baseline suffers from **vanishing gradients**: with 16 blocks and none of the usual remedies — layer normalization, dropout, regularization, pre-training — the gradient that reaches the early blocks dies, the network collapses onto a *trivial prediction* ``e_i`` and stays there. The plateau is exactly ``\sqrt{\tfrac{9}{10} \cdot 2} \approx 1.34``, the ``L_2`` distance between a one-hot guess and a one-hot target that disagree on 9 of 10 digits. An orthonormal ``Y`` neither amplifies nor damps what passes through a block, which is what removes the problem.

Two conclusions follow. First, hard geometric constraints can replace the heuristics a transformer normally needs, and they do so without adding a hyperparameter — unlike a soft orthogonality penalty ``\tilde{L} = L + \mu\Sigma``, which adds ``\mu`` and guarantees nothing. Second, among the Stiefel optimizers, Adam gives speed-ups over gradient descent and momentum of the same order as it does in the vector space case, which is the point of generalizing it rather than settling for a first-order method.

The experiment is reproduced by the scripts in `scripts/` of this repository.

## Reference

```@bibliography
Pages = ["manifold_optimizers.md"]
Canonical = false
```
