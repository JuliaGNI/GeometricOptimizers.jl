# Non-geometric Adam: decision record

## Source and scope

Version one is based on the arXiv v1 source (February 4, 2020) of Li, Li, and Todorovic,
*Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform*,
arXiv:2002.01113. The source defines Cayley SGD with momentum and Cayley ADAM in the
optimization-algorithm section (the ADAM moment equations are the equations immediately before
its Cayley-ADAM update algorithm). Its ADAM-like rule is the standard two-moment recursion applied
to the projected Stiefel gradient: maintain first and second moments, bias-correct them, form
`-m̂/(sqrt(v̂)+δ)`, and use that direction in the Cayley update. The source's Cayley transform is
an efficient implementation of its retraction, not a requirement of the moment recursion.

This package ports that moment rule only. It is an experimental comparison method, not a
replacement for `Adam` and not a claim that the source's complete Cayley ADAM implementation has
been reproduced.

## Representation contract

For a single `Y::StiefelManifold{T}` with `Y ∈ R^(N×n)`:

| Source/package symbol | Representation | Implementation |
| --- | --- | --- |
| `Y` | ambient Stiefel matrix | `state.x`, `cache.x` |
| `G_t` | ambient tangent matrix at `Y` | local `G` reconstructed from the horizontal gradient |
| `m_t` | ambient tangent matrix | `state.m₁`, `cache.m₁` |
| `v_t` | ambient tangent matrix | `state.m₂`, `cache.m₂` |
| `m̂_t`, `v̂_t` | ambient tangent matrices | local moment-correction operations |
| `d_t` | ambient tangent matrix before conversion | local `D` |
| `B_t` | horizontal Lie-algebra representation | `cache.δ`, then the existing section/retraction path |
| `t` | Euclidean iteration counter | `state.iterations` |

The squared quantity is the **ambient tangent matrix** `G_t`, element by element. The square is
not taken on `StiefelLieAlgHorMatrix.A` and `.B`; those blocks are only the package transport/update
representation. The explicit conversion is `G = Matrix(section) * Matrix(global_gradient)[:, 1:n]`.

## Package differences and adaptations

Package `Adam` stores bias-corrected moments directly in the horizontal representation and squares
the horizontal blocks. It uses the same first-update indexing (`t = 1`), direction sign, `δ`
placement, and no embedded learning-rate factor. `NonGeometricAdam` instead stores uncorrected
ambient moments, applies the standard corrections as `m̂ = m/(1-β₁^t)` and `v̂ = v/(1-β₂^t)`,
squares the ambient tangent gradient, and converts the normalized ambient direction into a
horizontal representation only at the update boundary. This makes the representation difference
observable while preserving the package's optimizer protocol.

The source's Cayley transform is adapted here: the package's existing `retraction` argument and
`update_section!` path remain authoritative. No source-specific iterative Cayley implementation or
source-specific vector transport is added. Accordingly, this method is a paper-derived moment-rule
baseline with the package's retraction semantics, not a full Cayley ADAM reproduction.

The direction is negative because `update_section!` interprets it as a descent velocity. The line
search supplies the step length; no learning-rate factor is present in `cache.δ`. `δ` is converted
to `T` and added after the elementwise square root, exactly as `sqrt(v̂) + δ`.

## Meaning of “non-geometric”

The name describes this package's selected representation, not a label claimed by the source. The
second moment is a coordinatewise ambient-array accumulator. It is neither an intrinsic tensor on
the Stiefel manifold nor transported between tangent spaces; its entries depend on the chosen
ambient frame and section. The accepted step is nevertheless manifold-valid because the normalized
ambient tangent direction is explicitly mapped to the package's horizontal representation and the
existing retraction/section path is used.

## Supported API and pseudocode

Version one accepts exactly one `StiefelManifold{T}` solution, where `T <: AbstractFloat`. Ordinary
arrays, `NamedTuple`s, Grassmann solutions, and mixed trees are rejected before cache/state
initialization with an `ArgumentError`. The public constructor is
`NonGeometricAdam(T=Float64; β₁=0.9, β₂=0.99, δ=1e-8)`; all hyperparameters are converted to `T`,
with `0 ≤ β₁, β₂ < 1` and `δ ≥ 0` validated. `NonGeometricAdamState` is public for manual solver
steps; the cache remains private implementation scratch.

For each accepted update, with `t = state.iterations ≥ 1`:

```text
G_global ← global_rep(section(state), gradient(x))
G        ← Matrix(section(state)) * Matrix(G_global)[:, 1:n]
m        ← β₁ m_previous + (1 - β₁) G
v        ← β₂ v_previous + (1 - β₂) (G ⊙ G)
m̂        ← m / (1 - β₁^t)
v̂        ← v / (1 - β₂^t)
D        ← -m̂ / (sqrt(v̂) + δ)
B        ← global_rep(section(state), D)
cache.δ  ← B
```

State/cache mapping is one-to-one: `m_previous` and `v_previous` are `state.m₁` and `state.m₂`;
`m`, `v`, `m̂`, `v̂`, and `D` are cache/local scratch; `B` is `cache.δ`; `G_global` is
`cache.g`; and `t` is `state.iterations`. There are no unresolved representation ambiguities in
this narrowed scope. The remaining deliberate adaptation is the package retraction documented
above.
