```@meta
CurrentModule = GeometricOptimizers
```

# Weight decay on a manifold

[`AdamWithEuclideanDecay`](@ref) is [`Adam`](@ref) with the *decoupled weight decay* of
[loshchilov2019decoupled](@cite), the method usually called AdamW. The name is deliberately
different here: on a compact Stiefel or Grassmann manifold, standard Euclidean weight decay has
zero Riemannian gradient.

## Decoupled weight decay

Ordinary ``L^2`` regularization adds ``\lambda{}x`` to the gradient. In [`Adam`](@ref), that term
then enters the moments and is rescaled with the gradient. Decoupled weight decay instead adds it
to the direction after the moments have been formed:

```math
x \gets x - \eta\frac{m_1}{\sqrt{m_2} + \delta} - \eta\lambda{}x.
```

Thus ``\lambda{}x`` does not enter ``m_1`` or ``m_2``. The decay is still scaled by the learning
rate ``\eta``; in this package, ``\eta`` is the ``\alpha`` supplied by the line search.

## Why the decay vanishes

The decay term is the Euclidean gradient of ``\frac{\lambda}{2}||Y||_F^2``. On the Stiefel
manifold,

```math
||Y||_F^2 = \mathrm{tr}(Y^TY) = \mathrm{tr}(\mathbb{I}_n) = n,
```

and the same norm is constant on the Grassmann representation used by this package. The penalty is
therefore constant, and its Riemannian gradient [`rgrad(::StiefelManifold, ::AbstractMatrix)`](@ref) makes this explicit:

```math
\mathtt{rgrad}(Y, \lambda{}Y) = \lambda{}Y - Y(\lambda{}Y)^TY
  = \lambda{}Y - \lambda{}Y = \mathbb{O}.
```

The constraint already prevents the norm from growing, so the Euclidean penalty has nothing left to do. Consequently, [`AdamWithEuclideanDecay`](@ref)

- decays ordinary array parameters,
- leaves [`Manifold`](@ref) parameters unchanged, and
- coincides with [`Adam`](@ref) on a bare manifold for every ``\lambda``.

!!! info
    The `StiefelManifold` is compact, which is why this argument applies. For other, noncompact spaces,
    Euclidean decay may have a nontrivial effect.

For a mixed parameter container, such as a model with Stiefel attention weights and ordinary
weights and biases, only the ordinary arrays are decayed:

```julia
ps = (w = rand(StiefelManifold, 5, 3), b = randn(3))
algorithm = AdamWithEuclideanDecay(; λ = 1e-2)
optimizer = Optimizer(ps, loss; algorithm = algorithm, linesearch = Static(1e-3))
```

If every parameter is a manifold, a nonzero ``\lambda`` cannot affect the run. The optimizer warns
at construction instead of silently behaving exactly like [`Adam`](@ref). The method reuses Adam's
cache and state; `OptimizerState(AdamWithEuclideanDecay(), ps)` returns an [`AdamState`](@ref).

## Iterate versus direction

The conclusion above applies when the iterate is constrained: each update is retracted back onto
the manifold. It does not say that every method using a tangent projection makes weight decay
irrelevant.

A seemingly related optimizer to the generalized manifold Adam is Mano (see [gu2026mano](@cite)). It first updates in ``\mathbb{R}^{m\times{}n}`` and then projects to the more trivial *oblique manifold*. Its weights are therefore free to change norm in the first step. The Mano paper also reports that a retracting Riemannian-SGD baseline failed to converge at one tested large language-model scale. That result is relevant empirical context, not a contradiction of the geometric argument above. The package's premise, following [brantner2023generalizing](@cite), is that the constraint is part of the model rather than a projection fitted to an unconstrained AdamW solution. Furthermore, the implementation's [`cayley`](@ref) retraction uses a ``2n\times{}2n`` solve, not an SVD or QR of the full weight, and hence reduces cost.

## Why the name is not `AdamW`

On a bare manifold, `AdamWithEuclideanDecay` is [`Adam`](@ref) for every ``\lambda``. Calling it `AdamW` would make the familiar name silently denote a no-op. The package therefore reserves `AdamW` and defines it only to raise an explanatory error pointing to [`AdamWithEuclideanDecay`](@ref) and [issue #28](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28).

The qualifier in `AdamWithEuclideanDecay` describes the decay, ``\lambda{}x``; the method itself is the manifold optimizer. A reference-point regularizer or decay of lifted coordinates would be a different method, not standard AdamW.

## Related questions

The settled no-op does not resolve questions about [`Adam`](@ref) itself. Its moments are accumulated element-wise in the horizontal Lie-algebra representation while [`GlobalSection`](@ref) re-bases that representation after each step. Questions regarding parallel transport, the related covariant derivative, and how much the random section affects optimization through holonomy remain open.
