# GeometricOptimizers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaGNI.github.io/GeometricOptimizers.jl/stable/)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaGNI.github.io/GeometricOptimizers.jl/latest/)
[![Build Status](https://github.com/JuliaGNI/GeometricOptimizers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGNI/GeometricOptimizers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaGNI/GeometricOptimizers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaGNI/GeometricOptimizers.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/G/GeometricOptimizers.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/G/GeometricOptimizers.html)

Optimizers on vector spaces and on matrix manifolds: the Stiefel and the Grassmann manifold.

## Example

A solve on a vector, and a solve on the Stiefel manifold: the minimum of `-tr(Y' * A * Y)` over the
`5 × 2` matrices `Y` with orthonormal columns is minus the sum of the two largest eigenvalues of `A`.

```julia
julia> using GeometricOptimizers, LinearAlgebra

julia> F(x) = sum(abs2, x .- 1);

julia> x = zeros(3);

julia> solve!(x, OptimizerState(BFGS(), x), Optimizer(x, F));

julia> x ≈ ones(3)
true

julia> A = Symmetric(diagm(0 => [5.0, 4.0, 3.0, 2.0, 1.0]));

julia> G(Y) = -tr(Y' * A * Y);

julia> Y = rand(StiefelManifold, 5, 2);

julia> solve!(Y, OptimizerState(BFGS(), Y), Optimizer(Y, G));

julia> G(Y) ≈ -9
true
```

## References
- Brantner B. Generalizing Adam To Manifolds For Efficiently Training Transformers. arXiv preprint arXiv:2305.16901, 2023.

## Development

### Git hooks

Two hooks live in `.githooks`. They are **not active in a fresh clone** — `core.hooksPath` is local
configuration and does not travel with a push — so enable them once per clone:

```sh
git config core.hooksPath .githooks
```

**`pre-commit`** acts on **staged `.jl` files only**, and exits immediately when a commit stages
none, so a documentation- or workflow-only commit is not slowed down by it:

- **JuliaFormatter `--check`**, honouring this repository's own `.JuliaFormatter.toml` — **blocks**
  the commit. Formatting is mechanical and always fixable.
- **`fatou lint`**, when `fatou` is installed — **advisory only**, and deliberately so: its
  `unused-import` rule does not follow `include`, so it flags the load-bearing imports of every
  module file.
- **`using <Package>`**, which catches a syntax error or a broken `include` — **blocks**.

**`pre-push`** runs the full test suite with `--check-bounds=auto`, but **only when pushing to
`main` or `master`**; a topic branch is left to CI. It prints nothing for **10–30 minutes**, which
looks exactly like a network hang and is not one. If you do interrupt it, check for an orphaned
Julia process that the killed hook left behind.

Either hook can be bypassed for a single command with `--no-verify`, for a change you know it does
not apply to:

```sh
git commit --no-verify
git push --no-verify
```

The hooks are generated from one shared copy and are byte-identical across the related
repositories, so edit them there rather than here — a local edit is silently undone by the next
install.
