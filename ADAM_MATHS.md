# Cayley ADAM in the global tangent space representation: reproduction record

## Source and scope

The implemented algorithm is *Cayley ADAM*, Algorithm 2 of Li, Li, and Todorovic, *Efficient
Riemannian Optimization on the Stiefel Manifold via the Cayley Transform*, arXiv:2002.01113 (arXiv v1,
February 4, 2020; ICLR 2020), cross-checked against the authors' implementation
(`stiefel_optimizer.py`, class `AdamG`, in `JunLi-Galios/Optimization-on-Stiefel-Manifold-via-Cayley-Transform`).
Where the pseudocode and that implementation disagree, the disagreements are listed below and the
implementation is followed.

The source's Algorithm 2, in its notation (`𝒢` is the stochastic Euclidean gradient, `X` the iterate,
`l` the learning rate, `q = 0.5`, `s = 2`):

```text
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

Lines 8–10 are the source's equation (2), lines 12–15 are `s` fixed-point iterations of its equation
(5), which is its closed-form Cayley transform (3) written implicitly, and line 11 is the contraction
condition of its Theorem 1, `α ∈ (0, min{1, 2/‖W‖})`.

## The identity the port rests on

For `Y ∈ St(N, n)`, the package's `Ω(Y, ·)` and the source's lines 8–9 are the same map. With
`Ŵ = ZYᵀ - ½Y(YᵀZYᵀ) = (𝕀 - ½YYᵀ)ZYᵀ`,

```math
W = Ŵ - Ŵᵀ = (𝕀 - ½YYᵀ)ZYᵀ - YZᵀ(𝕀 - ½YYᵀ) = Ω(Y, Z),
```

and conjugating by the global section `λ(Y) = [Y | λ]` gives, block by block,

| block of `λ(Y)ᵀ W λ(Y)` | value | `StiefelLieAlgHorMatrix` field |
| --- | --- | --- |
| `YᵀWY` | `½(YᵀZ - ZᵀY) = skew(YᵀZ)` | `A` |
| `λᵀWY` | `λᵀZ` | `B` |
| `YᵀWλ` | `-(λᵀZ)ᵀ` | `-Bᵀ` |
| `λᵀWλ` | `0` | the zero block |

which is exactly `global_rep(λY, Z)`. So `W = λ(Y) Ḡ λ(Y)ᵀ` for `Ḡ = global_rep(λ(Y), Z)`, and the
source's projection `π_{T_X}(Z) = WX` is `Ḡ` read back at `Y`. **Lines 8–10 are the representation
this package's first-order caches receive their gradient in already**; there is nothing to implement
for them, and `test/non_geometric_adam.jl` asserts the identity numerically.

## Symbol table

For a single `Y::StiefelManifold{T}`, `Y ∈ R^(N×n)`, at iteration `t = state.iterations ≥ 1`:

| Source | Representation | Implementation |
| --- | --- | --- |
| `X_k` | ambient Stiefel matrix | `state.x`, `cache.x`, and `state.section` |
| `𝒢(X_k)` | horizontal lift in `𝔤ʰᵒʳ` | `gradient_array(cache)`, i.e. `global_rep(section(state), ∇L)` |
| `M_{k+1}` | horizontal lift in `𝔤ʰᵒʳ` | `cache.m₁`, `state.m₁`, bias-corrected |
| `v_{k+1}` | scalar | `cache.m₂`, `state.m₂`, bias-corrected |
| `m̂`, `v̂` | — | absorbed into the bias-corrected storage, see below |
| `r` | scalar | `cache.m̃₂ = √(m₂ + δ)`; the `(1 - β₁ᵏ)` half is absorbed |
| `W_k` | horizontal lift in `𝔤ʰᵒʳ` | `-direction(cache)` |
| `α` | scalar | the line search's step length, capped by `step_αmax` |
| lines 12–15 | — | `update_section!(section, α·direction, retraction)` |
| `ε` | scalar | `method.δ` |
| `k` | iteration counter | `state.iterations` |

Bias-corrected storage is the package's [`Adam`](@ref) convention and is not a departure from the
source: substituting `m = (1 - β₁ᵗ)m̂` into line 4 gives

```math
m̂ ← \frac{β₁ - β₁ᵗ}{1 - β₁ᵗ} m̂ + \frac{1 - β₁}{1 - β₁ᵗ} Ḡ,
```

and likewise for `v` with line 5, after which line 7's `r` is `√(v̂ + ε)` alone and line 9 is
`W = m̂/√(v̂ + ε)`. Lines 9 and 10 divide and re-multiply by `r` so that the *stored* momentum is
un-normalized; storing `m₁` un-normalized is the same thing.

## The three deliberate departures

1. **The retraction, and with it the step cap.** Lines 12–15 are an approximation of the Cayley
   transform, and line 11 exists to keep it contractive (Theorem 1). This package retracts exactly —
   `Cayley()` evaluates the transform through the Sherman-Morrison-Woodbury formula, and `Geodesic()`
   and any other `AbstractRetraction` are equally admissible — so neither the two-step iteration nor
   its contraction bound is ported. The step is still bounded: `step_αmax(c, δ) = 2πc/‖δ‖`, which has
   the shape of line 11, and `step_ceiling = 1/2π` reproduces its `2q/‖W‖` with `q = 0.5` up to the
   source's use of the induced 1-norm in place of this package's Euclidean one. This is the departure
   the method's name no longer advertises and the docstring states.

2. **What `‖·‖²` is taken of.** The source accumulates `‖𝒢(X_k)‖²`, the squared Frobenius norm of the
   *ambient Euclidean* gradient (`torch.norm(g)**2` on `p.grad` in the authors' implementation). Here
   it is `l2norm(gradient_array(cache))^2`, the squared norm of the horizontal lift's free
   parameters, `½‖A‖²_F + ‖B‖²_F`. Two reasons: the ambient Euclidean gradient is not part of this
   package's optimizer protocol — `Gradient` applied to a `Manifold` returns `rgrad(Y, ∇L)`, and
   recovering `∇L` would cost a second gradient evaluation per step, defeating the reuse
   `store_gradient!` documents — and this is the norm the rest of the package measures gradients and
   steps with (`rg` in `OptimizerStatus`, `step_αmax`). Both are ambient-frame quantities and neither
   is intrinsic; the choice changes the scalar by a bounded factor, not the character of the method.

3. **Transport of the momentum.** The source transports `M_k` to the new tangent space by
   re-projecting it there (its equation (6), and the reason lines 8–10 are applied to the momentum
   rather than only to the gradient). Here the momentum is a horizontal lift and the section carries
   it: `update_section!` moves `λ(Y)` along the accepted step, and the same element of `𝔤ʰᵒʳ` is the
   transported momentum at the new iterate. The two agree for a Lie group and differ on a proper
   homogeneous space. This is the one departure that changes the iterates against the authors'
   implementation on the same objective and seed.

Two further discrepancies are *between* the source and its own implementation, not departures of this
port:

- **`v₁`.** Line 2 initializes `v₁ = 1`, the implementation with `0`; and line 6's `1 - β₂ᵏ` is `0` at
  the `k = 0` its loop starts from. `0` is followed, which with the bias-corrected storage makes the
  first direction `-Ḡ/√(‖Ḡ‖² + δ)`, a normalized gradient step, and which is also `Adam`'s
  convention here.
- **`ε` inside or outside the root.** Line 7 and the implementation both put it inside, `√(v̂ + ε)`.
  The package's `Adam` puts it outside, `√m₂ + δ`. The source's placement is used.

## Difference from `Adam`, in one line

`Adam` accumulates `Ḡ ⊙ Ḡ`, an element of `𝔤ʰᵒʳ`, and divides componentwise; `NonGeometricAdam`
accumulates `‖Ḡ‖²`, a number, and divides by a number. Everything else — the gradient, its
representation, the bias-correction convention, the direction's sign, the absence of a learning-rate
factor, the section/retraction path, the line search — is shared, which is what makes the two
comparable on the same problem.

## Supported API

Exactly one `StiefelManifold{T}` solution, `T <: AbstractFloat`. Ordinary arrays, `NamedTuple`s,
Grassmann solutions and mixed trees are rejected with an `ArgumentError` before cache/state
initialization. The constructor is `NonGeometricAdam(T=Float64; β₁=0.9, β₂=0.99, δ=1e-8)`, all
hyperparameters converted to `T`, with `0 ≤ β₁, β₂ < 1` and `δ ≥ 0` validated. `NonGeometricAdamState`
is public for manual solver steps; the cache is private. Nothing in the algorithm is Stiefel-specific
once the second moment is a scalar — the Grassmann and `NamedTuple` cases would need no new
mathematics — but the source is a Stiefel algorithm and widening the scope is left as a separate
decision.

## Package pseudocode

```text
Ḡ       ← global_rep(section(state), ∇L(x))        # = store_gradient!, and the source's lines 8-10
m₁      ← fac₁₁ · m₁_state + fac₁₂ · Ḡ            # 𝔤ʰᵒʳ,  source line 4 bias-corrected
m₂      ← fac₂₁ · m₂_state + fac₂₂ · ‖Ḡ‖²         # scalar, source lines 5-6 bias-corrected
m̃₂      ← √(m₂ + δ)                                # source line 7's r
cache.δ ← -m₁ / m̃₂                                 # 𝔤ʰᵒʳ,  minus the source's W of line 9
```

with `fac₁₁ = (β₁ - β₁ᵗ)/(1 - β₁ᵗ)`, `fac₁₂ = (1 - β₁)/(1 - β₁ᵗ)` and the same for `β₂`. The step
`α · cache.δ` and its retraction are the optimizer's, not the method's.
