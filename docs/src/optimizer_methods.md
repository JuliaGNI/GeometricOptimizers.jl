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

The problem with generalizing Adam to manifolds is that the Hadamard product ``\odot`` and the
other element-wise operations (``/``, ``\sqrt{}`` and ``+``) are coordinate-dependent: changing the
basis changes the update, so these operations do not define intrinsic operations on tangent vectors.
`GeometricOptimizers` resolves this for [`Adam`](@ref) by applying them in the
[global tangent space representation](@ref "Global Tangent Spaces"). A similar approach is shown in
[kong2023momentum](@cite).

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

The source does not require a global tangent space representation: its scalar second moment is
already independent of the choice of coordinates. This implementation nevertheless uses the same
horizontal lift, bias-correction convention, section update and line search as [`Adam`](@ref), so the
two methods can be compared inside the same optimization framework.

By default, ``\lVert\bar{G}\rVert^2`` is the norm of that horizontal lift. This is the quotient-space
quantity naturally available to the optimizer and avoids another gradient evaluation.
`ScalarMomentAdam(; ambient_norm = true)` instead uses the squared Frobenius norm of the ambient
Euclidean gradient ``\nabla{}L``, as the source does. These norms come from different views of the
Stiefel manifold — as a homogeneous quotient and as an embedded matrix manifold — and need not agree.

!!! info "Why the source is Cayley-specific"
    A retraction is a two-argument map: its update depends on both the current point and a tangent
    vector at that point. Algorithm 2 specializes this map to the Cayley retraction and evaluates the
    resulting implicit equation with two fixed-point iterations. Its step-length cap guarantees that
    this iteration is a contraction. `GeometricOptimizers` separates the optimizer from the
    retraction, evaluates [`Cayley`](@ref) directly, and enforces admissible step lengths through
    [`GeometricOptimizers.step_αmax`](@ref); consequently any [`AbstractRetraction`](@ref) may be used.

The source's algorithm is Stiefel-only and so is this: the method accepts exactly one
`StiefelManifold`, and ordinary arrays, `NamedTuple`s, Grassmann solutions and mixed trees throw an
`ArgumentError`.

#### The source's Algorithm 2, symbol by symbol

In its notation — ``\mathcal{G}`` the stochastic Euclidean gradient, ``X`` the iterate, ``l`` the
learning rate, ``q = 0.5``, ``s = 2``:

```
 2  X₁ orthonormal, M₁ = 0, v₁ = 1
 4  M_{k+1} ← β₁ M_k + (1 - β₁) 𝒢(X_k)
 5  v_{k+1} ← β₂ v_k + (1 - β₂) ‖𝒢(X_k)‖²
 6  v̂_{k+1} ← v_{k+1} / (1 - β₂ᵏ)
 7  r       ← (1 - β₁ᵏ) √(v̂_{k+1} + ε)
 8  Ŵ_k     ← M_{k+1} X_kᵀ - ½ X_k (X_kᵀ M_{k+1} X_kᵀ)
 9  W_k     ← (Ŵ_k - Ŵ_kᵀ) / r
10  M_{k+1} ← r W_k X_k                       # project the momentum onto the tangent space
11  α       ← min{ l, 2q / (‖W_k‖ + ε) }
12  Y⁰      ← X_k - α M_{k+1}
13  for i = 1 to s
14      Yⁱ  ← X_k - (α/2) W_k (X_k + Y^{i-1})
15  X_{k+1} ← Y^s
```

Lines 8–10 are its equation (2), lines 12–15 are ``s`` fixed-point iterations of its equation (5) —
its closed-form Cayley transform (3) written implicitly — and line 11 is the contraction condition of
its Theorem 1, ``\alpha \in (0, \min\{1, 2/\lVert{}W\rVert\})``. The pseudocode is cross-checked
against the authors' implementation (`stiefel_optimizer.py`, class `AdamG`, in
`JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform`); where the two disagree, the
implementation is followed, and [`ScalarMomentAdam`](@ref)'s docstring records both disagreements
along with the three places this port departs from the source deliberately.

##### Where each symbol lives

For a single `Y::StiefelManifold{T}`, ``Y \in \mathbb{R}^{N\times{}n}``, at iteration
`t = state.iterations ≥ 1`:

| Source | Representation | Implementation |
| --- | --- | --- |
| ``X_k`` | ambient Stiefel matrix | `state.x`, `cache.x` and `state.section` |
| ``\mathcal{G}(X_k)`` | horizontal lift in ``\mathfrak{g}^\mathrm{hor}`` | `gradient_array(cache)`, i.e. `global_rep(section(state), ∇L)` |
| ``M_{k+1}`` | horizontal lift | `cache.m₁`, `state.m₁`, bias-corrected |
| ``v_{k+1}`` | scalar | `cache.m₂`, `state.m₂`, bias-corrected |
| ``\hat{m}``, ``\hat{v}`` | — | absorbed into the bias-corrected storage |
| ``r`` | scalar | `cache.m̃₂ = √(m₂ + δ)`; the ``(1-\beta_1^k)`` half is absorbed |
| ``W_k`` | horizontal lift | `-direction(cache)` |
| ``\alpha`` | scalar | the line search's step length, capped by [`GeometricOptimizers.step_αmax`](@ref) |
| lines 12–15 | — | [`update_section!`](@ref)`(section, α⋅direction, retraction)` |
| ``\varepsilon`` | scalar | `method.δ` |
| ``k`` | iteration counter | `state.iterations` |

Bias-corrected storage is [`Adam`](@ref)'s convention here and not a departure from the source:
substituting ``m = (1 - \beta_1^t)\hat{m}`` into line 4 gives the ``m_1`` recursion above, and
likewise for ``v`` with line 5, after which line 7's ``r`` is ``\sqrt{\hat{v} + \varepsilon}`` alone
and line 9 is ``W = \hat{m}/\sqrt{\hat{v}+\varepsilon}``. Lines 9 and 10 divide and re-multiply by
``r`` so that the *stored* momentum is un-normalized; storing `m₁` un-normalized is the same thing.

**Lines 8–10 cost nothing to port**, which is what the next section is about.

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
