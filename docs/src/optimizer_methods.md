```@raw latex
In the previous chapter we introduced a general optimizer framework without giving explicit examples of neural network optimizers; this is done here. This chapter discusses standard neural network optimizers, including gradient descent, momentum, and Adam. In the implementation of all these optimizers the \textit{optimizer cache} will play an important role.
```

# Standard Neural Network Optimizers

In this section we discuss optimization methods that are often used in training neural networks. From a perspective of manifolds the *optimizer methods* outlined here operate on ``\mathfrak{g}^\mathrm{hor}`` only. Each of them has a cache associated with it[^1] and this cache is updated by `GeometricOptimizers.update!`. The precise role of this function is described below.

[^1]: In the case of the [gradient optimizer](@ref "The Gradient Optimizer") this cache is trivial.

## The Gradient Optimizer

The gradient optimizer is the simplest optimization algorithm used to train neural networks. It was already briefly discussed when we introduced [Riemannian manifolds](@ref "Gradient Flows and Riemannian Optimization").

It simply does: 

```math
\mathrm{weight} \leftarrow \mathrm{weight} + (-\eta\cdot\mathrm{gradient}),
```

where addition has to be replaced with appropriate operations in the manifold case[^2].

[^2]: In the manifold case the expression ``-\eta\cdot\mathrm{gradient}`` is an element of the [global tangent space](@ref "Global Tangent Spaces") ``\mathfrak{g}^\mathrm{hor}`` and a retraction maps from ``\mathfrak{g}^\mathrm{hor}``. We then still have to compose it with the [updated global section](@ref "Parallel Transport") ``\Lambda^{(t)}``.

A method carries no learning rate: it produces a *direction*, and how far to go along it is a
[`SimpleSolvers.LinesearchMethod`](@extref) — [`Static`](@extref SimpleSolvers.Static) for a fixed
step, [`DecayingStatic`](@ref) for one that decays.

```@example optimizer_methods
using GeometricOptimizers  # hide
import GeometricOptimizers  # hide
const η = 0.01
method = GradientMethod()
```

What a method *does* hold between steps is its [`OptimizerState`](@ref), built from the method and
the parameters it will be applied to. The parameters may be a vector, a `NamedTuple` of arrays, or
one of the manifolds:

```@example optimizer_methods
weight = (A = zeros(4, 4), )
state = OptimizerState(method, weight)

typeof(state)
```

For gradient descent that state is trivial — the direction is the gradient itself, scaled by ``-\eta``
— so there is nothing in it to look at. The two methods below are where it starts to matter.

## The Momentum Optimizer

The momentum optimizer is similar to the gradient optimizer but further stores past information as *first moments*. We let these first moments *decay* with a *decay parameter* ``\alpha``:

```math
\mathrm{weights} \leftarrow \mathrm{weights} + (\alpha\cdot\mathrm{moment} - \eta\cdot\mathrm{gradient}),
```

where addition has to be replaced with appropriate operations in the manifold case.

In the case of the momentum optimizer the cache is non-trivial:

```@example optimizer_methods
const α = 0.5
method = MomentumMethod(α)
weight = (A = zeros(4, 4), )
state = OptimizerState(method, weight)

# the moment is stored for each array in `weight` (which is a `NamedTuple`)
GeometricOptimizers.momentum(state).A
```

It is initialized with zeros, so the first step of a solve is the same one plain gradient descent
would take; from the second step on the moment carries the history.

If the weight is on a manifold the moment is allocated on ``\mathfrak{g}^\mathrm{hor}`` instead, and
[`OptimizerState`](@ref) does that without being told:

```@example optimizer_methods
manifold_weight = (Y = rand(StiefelManifold, 5, 3), )

typeof(GeometricOptimizers.momentum(OptimizerState(method, manifold_weight)).Y)
```

So if the weight is ``Y\in{}St(n,N)`` the corresponding cache is initialized as the zero element on ``\mathfrak{g}^\mathrm{hor}\subset\mathbb{R}^{N\times{}N}`` as this is the global tangent space representation corresponding to the StiefelManifold.

## The Adam Optimizer

The Adam Optimizer is one of the most widely used neural network optimizers. The cache of the Adam optimizer consists of *first and second moments*. The *first moments* ``B_1``, similar to the momentum optimizer, store linear information about the current and previous gradients, and the *second moments* ``B_2`` store quadratic information about current and previous gradients. These second moments can be interpreted as approximating the curvature of the optimization landscape.  

If all the weights are on a vector space, then we directly compute updates for ``B_1`` and ``B_2``:
1. ``B_1 \gets ((\rho_1 - \rho_1^t)/(1 - \rho_1^t))\cdot{}B_1 + (1 - \rho_1)/(1 - \rho_1^t)\cdot{}\nabla{}L,``
2. ``B_2 \gets ((\rho_2 - \rho_1^t)/(1 - \rho_2^t))\cdot{}B_2 + (1 - \rho_2)/(1 - \rho_2^t)\cdot\nabla{}L\odot\nabla{}L,``

where ``\odot:\mathbb{R}^n\times\mathbb{R}^n\to\mathbb{R}^n`` is the *Hadamard product*: ``[a\odot{}b]_i = a_ib_i.`` ``\rho_1`` and ``\rho_2`` are hyperparameters. Their defaults, ``\rho_1=0.9`` and ``\rho_2=0.99``, are taken from [goodfellow2016deep; page 301](@cite). After having updated the `cache` (i.e. ``B_1`` and ``B_2``) we compute a *velocity* with which the parameters of the network are then updated:
* ``W_t\gets -\eta{}B_1/\sqrt{B_2 + \delta},``
* ``Y^{(t+1)} \gets Y^{(t)} + W^{(t)},``

where the last addition has to be replaced with appropriate operations when dealing with manifolds. Further ``\eta`` is the *learning rate* and ``\delta`` is a small constant that is added for stability. The division, square root and addition in the computation of ``W_t`` are performed element-wise.

In the following we show a schematic update that Adam performs for the case when no elements are on manifolds (also compare this figure with the [general optimization framework](@ref "Generalization to Homogeneous Spaces")):

![Schematic representation of the Adam optimizer. The first Adam step updates the first and second moments, and the second Adam step outputs the final velocity.](tikz/adam_optimizer_light.png)
![Schematic representation of the Adam optimizer. The first Adam step updates the first and second moments, and the second Adam step outputs the final velocity.](tikz/adam_optimizer_dark.png)

We demonstrate the Adam state on the same example from before. Note the spelling: ``\rho_1`` and
``\rho_2`` above are `β₁` and `β₂` here, as everywhere else in this package.

```@example optimizer_methods
const ρ₁ = 0.9
const ρ₂ = 0.99
const δ = 1e-8

method = Adam(Float64; β₁ = ρ₁, β₂ = ρ₂, δ)
state = OptimizerState(method, weight)

GeometricOptimizers.first_moment(state).A
```

and the second moment alongside it:

```@example optimizer_methods
GeometricOptimizers.second_moment(state).A
```

### Weights on Manifolds

The problem with generalizing Adam to manifolds is that the Hadamard product ``\odot`` as well as the other element-wise operations (``/``, ``\sqrt{}`` and ``+``) lack a clear geometric interpretation. In `GeometricOptimizers` we get around this issue by utilizing the [global tangent space representation](@ref "Global Tangent Spaces"). A similar approach is shown in [kong2023momentum](@cite).

### Cayley ADAM: a scalar second moment

[`ScalarMomentAdam`](@ref) is the other published answer to that problem, available here as a
baseline to compare against: it is *Cayley ADAM*, [li2020efficient; Algorithm 2](@cite), which avoids
the Hadamard product by **collapsing the second moment to a scalar**. Where `Adam` accumulates
``\bar{G}\odot\bar{G}\in\mathfrak{g}^\mathrm{hor}``, it accumulates ``\lVert\bar{G}\rVert^2``, one
number, and divides the first moment by ``\sqrt{m_2 + \delta}``:

```math
m_1 \gets \frac{\beta_1 - \beta_1^t}{1 - \beta_1^t}m_1 + \frac{1 - \beta_1}{1 - \beta_1^t}\bar{G},
\qquad
m_2 \gets \frac{\beta_2 - \beta_2^t}{1 - \beta_2^t}m_2 + \frac{1 - \beta_2}{1 - \beta_2^t}\lVert\bar{G}\rVert^2,
\qquad
W_t \gets -\frac{m_1}{\sqrt{m_2 + \delta}}.
```

So the whole matrix gets one adaptive learning rate instead of one per coordinate, and the second
moment carries no direction — which is what the method is named for, and the reason this is a
baseline rather than the package's recommended method. It is nonetheless a reproduction of a
published algorithm, not a straw man: on a given objective it may well beat [`Adam`](@ref).

Everything else is shared with [`Adam`](@ref) — the gradient, its
[global tangent space representation](@ref "Global Tangent Spaces"), the bias-correction convention,
the section/retraction path and the line search — which is what makes the two comparable on the same
problem. One thing about that comparison is worth knowing before running it: `Adam` normalizes
*componentwise*, so ``\lVert{}W_t\rVert \approx \sqrt{\dim}``, while the divisor above normalizes the
whole lift at once, so ``\lVert{}W_t\rVert \approx 1``. Both methods take the same default
``\eta``, so the same number is a step ``\sqrt{\dim}`` shorter here.

Two things in the source are deliberately *not* reproduced: its two-step approximation of the
Cayley transform, because this package's [`Cayley`](@ref) retraction is exact and any other
[`AbstractRetraction`](@ref) may be used instead, and the step-length cap that approximation needs,
because bounding the step is [`GeometricOptimizers.step_αmax`](@ref)'s job here. A third difference —
which ``\lVert\cdot\rVert^2`` the second moment takes — is a keyword,
`ScalarMomentAdam(; ambient_norm = true)` being the source's ambient Euclidean choice. The method's
docstring maps every symbol of Algorithm 2 onto the code and records all three.

The source's algorithm is Stiefel-only and so is this: the method accepts exactly one
`StiefelManifold`, and ordinary arrays, `NamedTuple`s, Grassmann solutions and mixed trees throw an
`ArgumentError`.

#### The source's projection *is* `global_rep`

Algorithm 2 spends three of its lines building the tangent-space projection it needs — an auxiliary
matrix ``\hat{W} = ZY^T - \frac{1}{2}Y(Y^TZY^T)``, its skew-symmetrization ``W = \hat{W} - \hat{W}^T``
and the projection ``\pi_{T_Y}(Z) = WY`` (its equation (2)). None of it has to be implemented here,
because that map is [`GeometricOptimizers.Ω`](@ref) and its conjugate is
[`global_rep`](@ref). Writing ``\hat{W} = (\mathbb{I} - \frac{1}{2}YY^T)ZY^T``,

```math
W = \hat{W} - \hat{W}^T
  = \left(\mathbb{I} - \tfrac{1}{2}YY^T\right)ZY^T - YZ^T\left(\mathbb{I} - \tfrac{1}{2}YY^T\right)
  = \Omega(Y, Z),
```

and conjugating by the global section ``\lambda(Y) = [\,Y \mid \lambda\,]`` gives, using
``Y^T(\mathbb{I} - \frac{1}{2}YY^T) = \frac{1}{2}Y^T`` and ``\lambda^TY = \mathbb{O}``, block by block

| block of ``\lambda(Y)^TW\lambda(Y)`` | value | field of [`StiefelLieAlgHorMatrix`](@ref) |
| --- | --- | --- |
| ``Y^TWY`` | ``\frac{1}{2}(Y^TZ - Z^TY) = \mathrm{skew}(Y^TZ)`` | ``A`` |
| ``\lambda^TWY`` | ``\lambda^TZ`` | ``B`` |
| ``Y^TW\lambda`` | ``-(\lambda^TZ)^T`` | ``-B^T`` |
| ``\lambda^TW\lambda`` | ``\mathbb{O}`` | the zero block |

which is exactly `global_rep(GlobalSection(Y), Z)` — for *any* ``Z``, tangent or not. So
``W = \lambda(Y)\bar{G}\lambda(Y)^T`` for ``\bar{G} = `` `global_rep(λY, Z)`, the source's ``W_k`` *is*
the horizontal lift the first-order caches already receive their gradient in, and its ``\pi_{T_Y}(Z)``
is that lift read back at ``Y``. `test/scalar_moment_adam.jl` pins the identity numerically against a
literal transcription of the source's formula, so it cannot silently stop holding.

What the source needs those lines for a *second* time — re-projecting the momentum onto the tangent
space at the new iterate, its equation (6) — is [`update_section!`](@ref)'s job here, and that one is
a genuine departure: transport by the global section and transport by re-projection agree for a Lie
group and differ on a proper homogeneous space.

## The Adam Optimizer with Decay
The Adam optimizer with decay is similar to the standard Adam optimizer with the difference that the learning rate ``\eta`` decays exponentially. We start with a relatively high learning rate ``\eta_1`` (e.g. ``10^{-2}``) and end with a low learning rate ``\eta_2`` (e.g. ``10^{-8}``). If we want to use this optimizer we have to tell it beforehand how many epochs we train for such that it can adjust the learning rate decay accordingly:

There is no new method here and no schedule of its own: [`AdamOptimizerWithDecay`](@ref) is
[`Adam`](@ref) paired with a [`DecayingStatic`](@ref) line search, returned as a `NamedTuple` to
splat into an [`Optimizer`](@ref).

```@example optimizer_methods
const η₁ = 1e-2
const η₂ = 1e-6
const n_epochs = 1000

pairing = AdamOptimizerWithDecay(n_epochs, Float64; η₁ = η₁, η₂ = η₂)
pairing.linesearch
```

The state is therefore exactly Adam's, because the method is:

```@example optimizer_methods
state = OptimizerState(pairing.algorithm, weight)

typeof(GeometricOptimizers.first_moment(state).A)
```

## Library functions

[`GradientMethod`](@ref), [`MomentumMethod`](@ref), [`Adam`](@ref), [`ScalarMomentAdam`](@ref),
[`AdamOptimizerWithDecay`](@ref), [`DecayingStatic`](@ref) and [`OptimizerState`](@ref). Their
docstrings are on the [reference page](@ref GeometricOptimizers), where every docstring in the
package is rendered once; the names above link to them.

Taking a step with one of these over the parameter tree of a *neural network* is
[`GeometricMachineLearning`](@extref GeometricMachineLearning :doc:`index`)'s job: it wraps an
[`OptimizerState`](@ref) and a cache in an optimizer of its own and walks the tree with them. The
framework it does that within is described [here](@ref "The optimizer framework, step by step").

```@raw latex
\begin{comment}
```

## References

```@bibliography 
Pages = []
Canonical = false

goodfellow2016deep
```

```@raw latex
\end{comment}
```
