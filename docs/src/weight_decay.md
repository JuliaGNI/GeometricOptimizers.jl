```@meta
CurrentModule = GeometricOptimizers
```

# Weight Decay on Manifolds

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

Both manifolds, and to machine precision rather than by assertion:

```jldoctest
julia> using GeometricOptimizers, LinearAlgebra, Random

julia> Random.seed!(1234);

julia> λ = 0.5;

julia> [norm(rgrad(Y, λ * Y.A)) < 1e-14
        for Y in (rand(StiefelManifold, 6, 3), rand(GrassmannManifold, 6, 3))]
2-element Vector{Bool}:
 1
 1
```

The constraint already prevents the norm from growing, so the Euclidean penalty has nothing left to do. Consequently, [`AdamWithEuclideanDecay`](@ref)

- decays ordinary array parameters,
- leaves [`StiefelManifold`](@ref) and [`GrassmannManifold`](@ref) parameters unchanged, and
- coincides with [`Adam`](@ref) on a bare manifold for every ``\lambda``.

!!! info "Compactness, not manifoldness"
    Both manifolds of this package are compact, which is why this argument applies. For a
    noncompact space Euclidean decay may well have a nontrivial effect, so the no-op is declared
    on the two concrete manifolds rather than on [`Manifold`](@ref): a manifold added later gets an
    error from [`_is_decayable`](@ref) until it has decided the question for itself.

For a mixed parameter container, such as a model with Stiefel attention weights and ordinary
weights and biases, only the ordinary arrays are decayed:

```jldoctest mixed
julia> using GeometricOptimizers, LinearAlgebra, Random

julia> Random.seed!(1234);

julia> A = randn(5, 3);

julia> ps = (w = rand(StiefelManifold, 5, 3), b = randn(3));

julia> loss(ps::NamedTuple) = norm(A - ps.w * ps.w' * A) + norm(ps.b);

julia> algorithm = AdamWithEuclideanDecay(; λ = 1e-2);

julia> optimizer = Optimizer(ps, loss; algorithm = algorithm, linesearch = Static(1e-3));

julia> OptimizerState(algorithm, ps) isa AdamState   # Adam's state, reused verbatim
true
```

Only `b` is decayed; `w` is left to [`Adam`](@ref) alone. The
[test file](https://github.com/JuliaGNI/GeometricOptimizers.jl/blob/main/test/adam_with_euclidean_decay.jl)
pins both halves of that, including that `b` moves by exactly ``-\eta\lambda{}b`` more than it does
under [`Adam`](@ref) on the first step.

If every parameter is a manifold, a nonzero ``\lambda`` cannot affect the run. The optimizer warns
at construction instead of silently behaving exactly like [`Adam`](@ref). The method reuses Adam's
cache and state; `OptimizerState(AdamWithEuclideanDecay(), ps)` returns an [`AdamState`](@ref).

## Which line search

The decay is applied to the *direction*, and the line search then scales that direction by its
``\alpha``, so the decay per step is ``\alpha\lambda``. That is deliberate — it is how
[loshchilov2019decoupled](@cite) couples the decay to the learning-rate schedule and nothing else —
but it means the choice of line search is part of the method's semantics rather than a performance
knob. See [`default_linesearch`](@ref):

- `Static(η)`, the default, gives the decay of the paper with ``\eta`` fixed.
- [`DecayingStatic`](@ref) is the schedule case, and equally well defined: ``\lambda`` keeps its
  meaning relative to ``\eta`` while both go to zero.
- A *searching* line search is not recommended, and not only because [`Adam`](@ref)'s direction is
  a poor thing to search along. The merit it minimizes is the bare objective ``f``, not
  ``f + \frac{\lambda}{2}||x||^2`` — the penalty is never assembled anywhere, which is what
  *decoupled* means — so the search spends its ``\alpha`` undoing the decay's contribution to
  ``f``, and the effective regularization strength becomes whatever ``\alpha`` it settled on.

## What the argument does and does not depend on

The argument depends on the *iterate* being constrained: every step ends with a retraction back
onto the manifold, so ``||Y||_F^2 = n`` holds at every iterate and not merely in the limit. What it
does **not** say is that any method with a geometric flavour makes Euclidean weight decay
irrelevant. Projecting a gradient, a momentum or a direction onto a tangent space constrains the
*step*; unless the point is then returned to the constraint set, ``||Y||_F`` is free to drift and a
Euclidean penalty has something to act on again. Whether the decay vanishes is therefore a question
about the iterates, and it has to be asked of each method separately.

That distinction matters because manifold-flavoured optimizers for large language models are an
active area — Mano ([gu2026mano](@cite)) projects the momentum onto the tangent space of the
parameters and constrains it on a rotational oblique manifold, and reports outperforming AdamW and
Muon at lower memory and compute cost. This package takes the other route, following
[brantner2023generalizing](@cite): the constraint is part of the model, maintained at every
iterate, rather than a projection wrapped around an essentially unconstrained solution. Its cost
is kept down in the retraction instead — [`cayley`](@ref) needs only a ``2n\times{}2n`` solve, not
an SVD or a QR decomposition of the full weight.

## Why the name is not `AdamW`

On a bare manifold, `AdamWithEuclideanDecay` is [`Adam`](@ref) for every ``\lambda``. Calling it `AdamW` would make the familiar name silently denote a no-op. The package therefore reserves `AdamW` and defines it only to raise an explanatory error pointing to [`AdamWithEuclideanDecay`](@ref) and [issue #28](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28).

The qualifier in `AdamWithEuclideanDecay` describes the decay, ``\lambda{}x``; the method itself is the manifold optimizer. A reference-point regularizer or decay of lifted coordinates would be a different method, not standard AdamW.

## Two unrelated decays

Three names in this package contain the word *decay*, and only one of them decays weights. The
collision is worth spelling out, because the two that are easiest to confuse —
[`AdamWithEuclideanDecay`](@ref) and [`AdamOptimizerWithDecay`](@ref) — differ in what they act on,
in what parameterizes them, and in which half of the optimizer they live in.

| | acts on | parameterized by | lives in | effect |
|---|---|---|---|---|
| [`AdamWithEuclideanDecay`](@ref) | the **weights** | ``\lambda`` | the [`OptimizerMethod`](@ref), i.e. the direction | adds ``-\lambda{}x`` to the direction |
| [`DecayingStatic`](@ref), and hence [`AdamOptimizerWithDecay`](@ref) | the **learning rate** | ``\eta_1``, ``\eta_2``, ``n`` | the `SimpleSolvers.LinesearchMethod`, i.e. the step size | ``\alpha(t) = \gamma^t\eta_1`` |

Neither implies the other and they compose freely: a run may decay its weights, its learning rate,
both, or neither. `AdamWithEuclideanDecay` has a *fixed* learning rate by default
([`default_linesearch`](@ref) gives it `Static`), and `AdamOptimizerWithDecay` never touches a
weight — it is [`Adam`](@ref), which has no ``\lambda``.

### Where `GeometricMachineLearning`'s method belongs

`GeometricMachineLearning` defines an `AdamOptimizerWithDecay` that stores ``\eta_1``, ``\eta_2``,
``n_\mathrm{epochs}`` alongside Adam's ``\rho_1``, ``\rho_2``, ``\delta``, and steps by
``\eta_1\gamma^t`` with ``\gamma = \exp(\log(\eta_2/\eta_1)/n_\mathrm{epochs})``. Despite the name it
is the **second** row of the table: an exponential decay of the learning rate, with no weight decay
anywhere in it. Its schedule is the one [`DecayingStatic`](@ref) already implements, factor for
factor:

| | GML's `AdamOptimizerWithDecay` | [`DecayingStatic`](@ref) |
|---|---|---|
| decay factor | ``\gamma = \exp(\log(\eta_2/\eta_1)/n_\mathrm{epochs})`` | ``\gamma = \exp(\log(\eta_2/\eta_1)/n)`` |
| step at ``t`` | ``\eta_1\gamma^t`` | ``\gamma^t\eta_1`` |
| first step taken | ``\eta_1\gamma`` (`o.step` is incremented before `update!`) | ``\gamma\eta_1`` (`increase_iteration_number!` runs before `solver_step!`) |

The left column is GML's *code*; its docstring states the reciprocal,
``\gamma = \exp(\log(\eta_1/\eta_2)/n_\mathrm{epochs})``, which would grow the step rather than decay
it. The third row matters as much as the second: an equality of formulas is not an equality of
schedules unless the two also count ``t`` alike, and both start at ``t = 1``.

So migrating it here is a matter of splitting it across the two halves it was bundling —
``\rho_1``, ``\rho_2``, ``\delta`` to [`Adam`](@ref) and ``\eta_1``, ``\eta_2``,
``n_\mathrm{epochs}`` to the line search — which is what [`AdamOptimizerWithDecay`](@ref) does under
the original name, so that GML can delete its copy rather than port it. The name is all that carries
over unchanged: GML derives the element type from ``\eta_1``, whose default is a `Float32` literal, and
takes the step sizes and moment coefficients positionally, so migrated call sites still need editing
(see the note on [`AdamOptimizerWithDecay`](@ref)).

This is also what v0.1.0's `AdamWithDecay` did, with the same three fields on the method itself. It
was removed when the step size moved out of the [`OptimizerMethod`](@ref)s; the schedule survived it,
the bundling did not.

## Related questions

The settled no-op does not resolve questions about [`Adam`](@ref) itself. Its moments are accumulated element-wise in the horizontal Lie-algebra representation while [`GlobalSection`](@ref) re-bases that representation after each step. Questions regarding parallel transport, the related covariant derivative, and how much the random section affects optimization through holonomy remain open.
