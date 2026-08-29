# Changelog

All notable changes to GeometricOptimizers.jl are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (pre-1.0, so a minor bump is a
breaking release).

## [0.7.0]

**A whole set of parameters is a `NetworkParameters` and never a bare `NamedTuple`.** This is a
breaking change to what the package *accepts*, and everything else here follows from it.

`ParameterContainer{T}` was `Union{ArrayNamedTuple{T}, NetworkParameters{T}}`, and its first member
was `ArrayNamedTuple{T,S} = NamedTuple{S,<:Tuple{Vararg{AbstractArray{T}}}}` — **an alias for
`Base.NamedTuple`**, not a type of its own. That is what made it a problem rather than merely a wide
type, and it caused three separate things:

- **A method on it was a method on `Base.NamedTuple`.** Five of the eight sites of issue
  [#16](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/16) had to move upstream to the
  packages that own their generics for exactly this reason, rather than be narrowed in place. 0.6.1
  closed that as a property; this release removes the shape that caused it.
- **It collided with `GlobalSectionNamedTuple`**, which is an alias for `NamedTuple` too, and with the
  bare `NamedTuple` a *nested* section tree has to be written on. Every section-copy and
  parameter-copy pair in `optimizers/named_tuple_wrapper.jl` overlapped on a shape neither method
  could claim. 0.6.1 patched that with four disambiguations; they are **deleted** here, because the
  overlap no longer exists.
- **Its `T` meant two different things.** For the `NamedTuple` half it guaranteed the set was flat
  *and* that every leaf was an `AbstractArray{T}`; for the container it is a promotion over leaves at
  any depth. One signature said one thing about one argument shape and something else about the other.

### Measured

`Test.detect_ambiguities(GeometricOptimizers; recursive = true)`, on Julia 1.13 against
`GeometricBase` 0.14.9, `SimpleSolvers` 0.13.2 and `NeuralNetworkParameters` 0.3.0:

| | 0.6.1 | 0.7.0 |
|---|---|---|
| total reported | 138 | 129 |
| **both methods owned by this package** | **35** | **26** |
| the `copyto!`/`_copyto!` family | 10 | **1** |

The one that is left is documented in place and has no witness: a `GlobalSection` anchored on a
`Manifold` whose lift is `nothing`, which `GlobalSection(::Manifold)` never builds. The two that used
to be listed beside it — the empty `NamedTuple`, vacuously a `NamedTuple` of arrays *and* of
`GlobalSection`s at once, and the mixed-element-type parameter pair — are gone rather than documented,
because both needed a parameter set to be a `NamedTuple`.

The remaining 26 are between this package's structured matrices (`SymmetricMatrix`, `SkewSymMatrix`,
the triangulars, `StiefelProjection`) and `LinearAlgebra`/`ArrayLayouts` methods on `AbstractMatrix`
— `*`, `+`, `mul!`, `vcat`, `hcat`. None was introduced by this work or by the de-piracy wave.

### Migrating

Wrap. `NetworkParameters(ps)` **shares the leaf arrays** rather than copying them, so an in-place
solve still writes through to the arrays the caller holds and nothing has to be copied back:

```julia
ps = (Y = rand(StiefelManifold, 6, 3), W = randn(3, 4), b = zeros(6))
opt = Optimizer(NetworkParameters(ps), F; algorithm = Adam(Float64))
```

`NetworkParameters` is **re-exported** for this, so `using GeometricOptimizers` is enough. That is the
one exception to the rule that this package does not re-export `NeuralNetworkParameters`' names, and
it is there because the wrap is now this package's own entry condition rather than a foreign concern.
The name collides with nothing: `AbstractNeuralNetworks`, `SymbolicNeuralNetworks` and
`GeometricMachineLearning` all take it from `NeuralNetworkParameters` too.

The container forwards `keys`, `values`, `ps[k]`, `ps.field` and `length`, so an objective, a
hand-written `∇F!` and any regrouping written against the bare `NamedTuple` need no change. That is
how `GMLDatasets`' five MNIST scripts converted — the wrap and the type annotations, nothing else.

`GeometricMachineLearning` hands this package one **layer** at a time and cannot convert by wrapping
at the root. It wraps at the boundary instead (`GML/src/optimizers/optimizer.jl`, `_as_go_solution`),
and takes the flatness test that `ArrayNamedTuple` used to make by dispatch — a layer is a flat
`NamedTuple` of arrays, a subtree is not — as a rule written out in its own source.

### Changed

- **`GeometricBase = "0.14.9"`**, i.e. unchanged from 0.6.0 — an earlier revision of this release
  raised it to `"0.14.10"` and that bump is withdrawn, so no bound moves. It existed only so that
  `L2norm` over a parameter set would be available, and the method now lives in
  `NeuralNetworkParameters`' `src/norms.jl` — the package that owns the type and the walk,
  and the only one of the two that can test it, since `GeometricBase` supports Julia 1.10 and cannot
  resolve `NeuralNetworkParameters` at all. `test/aqua_tests.jl` asserts the new location.
  `GeometricBase` consequently drops out of the registration chain: 0.14.9 is registered and
  unchanged, so only `SimpleSolvers` 0.13.2 has to precede this release.
- `ArrayTuple`, `ArrayNamedTuple` and `ParameterContainer` are **removed**. Signatures written in
  `ParameterContainer{T}` now take `NetworkParameters{T}` directly, which is the same type with one
  fewer name in front of it.
- `OptimizerSolution{T}` is `Union{AbstractVector{T}, Manifold{T}, NetworkParameters{T}}`.
- `GradientArrayOrNamedTuple{T}` is renamed **`GradientStorage{T}`** and `LiftOrNamedTuple{T}` to
  **`LiftOrParameters{T}`**: neither has a `NamedTuple` in it any more, and a name that says
  otherwise is worse than no name.
- `DottableSet` narrows from `Union{AbstractLieAlgHorMatrix, ParameterSet}` to
  `Union{AbstractLieAlgHorMatrix, NetworkParameters}`.
- The four `_copyto!` disambiguations added in 0.6.1 are deleted, and the comment block above them now
  records why they are not needed rather than why they were.
- `test/named_tuple_parameters.jl` is renamed **`test/flat_parameters.jl`**. It used to pin the
  opposite commitment — that a bare `NamedTuple` is a parameter set — and now drives the *flat*
  container, which is the shape `GMLDatasets` keeps a transformer's parameters in.
  `test/network_parameters_optimizer.jl` is its nested counterpart, and the difference is not
  cosmetic: a nested container's leaves sit one level below what `Base.map` reaches.
- A new testset in `test/flat_parameters.jl` pins the new commitment directly: a bare `NamedTuple` is
  turned away at `OptimizerCache`, `OptimizerState` and `Optimizer`, and the wrap shares its leaves.

### Fixed

- **The documentation build was failing before this release, and the failure was invisible.**
  `docs/make.jl` listed `SimpleSolvers` and `GeometricMachineLearning` as `InterLinks` inventories but
  not `NeuralNetworkParameters`, while `RiemannianGradient`'s docstring
  (`src/utils.jl`) has `[`NeuralNetworkParameters.NetworkParameters`](@extref)` in it. Documenter
  stopped at `ExtCrossReferences` with `:external_cross_references` and never reached
  `RenderDocument`. `NeuralNetworkParameters` is now an inventory entry, so the reference resolves and
  the build renders.

  Worth recording how it hid: `make.jl` runs doctests *before* cross-references, so a log filtered for
  "error" and "doctest" shows `[ Info: Doctest: running doctests.` and nothing alarming, and a build
  piped into `grep` reports the exit status of `grep`. The check that the build succeeded is that
  `[ Info: RenderDocument: rendering document.` appears.

- **The signatures that took `NeuralNetworkParameters.ParameterSet` now take `NetworkParameters`.**
  That alias is removed upstream in 0.3.0, because a method on it was a method on `Base.NamedTuple` and
  because it named two questions at once — a whole set, and a *branch* of one. `geodesic`, `cayley`,
  `retraction_differential`, `solution_scale`, `_manifold_αmax`, `apply_section!` and `update_section!`
  take the container; `_is_decayable` has two methods, because it recurses through `values` itself and
  so meets a container's layers on the way down.

  `l2norm` of a whole set comes from `NeuralNetworkParameters`' `src/norms.jl` and is
  likewise container-only. Every fixture in `test/` that handed one of these a bare `NamedTuple` now
  wraps.

### Not changed, and deliberately

The signatures still written on `NeuralNetworkParameters.ParameterSet` — `geodesic`, `cayley`,
`retraction_differential`, `solution_scale`, `_manifold_αmax`, `_is_decayable`, `apply_section!`,
`update_section!` — stay wide. Each is this package's *own* function, so none is piracy, and each
either recurses through `values` itself or is applied to a **direction**, which `mapparameters`
rebuilds in the shape of whatever it walked — including the plain `NamedTuple` of a nested section
tree. Narrowing them is a separate question from this one and depends on
`NeuralNetworkParameters.ParameterSet`, which cannot narrow while `SymbolicNeuralNetworks` uses it for
*equation* sets that are bare `NamedTuple`s by construction.

The section aliases (`GlobalSectionTuple`, `GlobalSectionNamedTuple`,
`GlobalSectionSingleOrNamedTuple`) also stay. Collapsing them means making
`GlobalSection(::NetworkParameters)` return a *container* of sections rather than a plain
`NamedTuple`, which needs a `parameter_eltype` for a `GlobalSection` upstream and changes what
`GeometricMachineLearning`'s `_tree_optim_step!` descends into. It is the next simplification of this
kind and it is not this one.

## [0.6.1]

**The last type piracy in this package is gone, and a test keeps it that way.** Goal 2 of the
ecosystem plan — "no type piracy in the GML ecosystem" — had this package as its only remaining
holder. `Aqua.Piracy.hunt` found **eight** methods, where the 0.6.0 changelog's table counted one and
issue [#16](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/16) counted four. That
discrepancy is the point of the change: a list of sites goes stale, and this one had gone stale in
both directions — it recorded two entries as open after they had been fixed, and missed five that
were not. `test/aqua_tests.jl` runs `Aqua.test_piracies` and does not go stale.

Type piracy here means a method whose function *and* all of whose dispatched argument types belong to
other packages. It matters because it is global: any package that loads this one, even indirectly,
silently gets the new behaviour for the rest of its session.

### Changed

- **Five methods moved to the packages that own their generics**, as extensions on
  `NeuralNetworkParameters`. There is no dependency cycle to worry about: that package's only
  dependency is `ChainRulesCore`, so neither owner depends on this one.

  | was here | is now |
  |---|---|
  | `l2norm(::ParameterSet)` (`optimizers/optimizer_status.jl`) | `NeuralNetworkParameters` 0.3.0, `src/norms.jl`, as `L2norm(::NetworkParameters)` |
  | `GradientAutodiff(F, ::ParameterSet)` (`optimizers/named_tuple_wrapper.jl`) | `SimpleSolvers` 0.13.2, `ext/SimpleSolversNeuralNetworkParametersExt.jl` |
  | `GradientFunction(F, ∇F!, ::ParameterSet)` (same file) | the same extension |
  | `alloc_h`, the `ParameterContainer` arm (`iterative_hessians/bfgs/bfgs_state.jl`) | the same extension |
  | `GradientAutodiff(F, ::Matrix)` (`utils.jl`) | `SimpleSolvers` 0.13.2 proper, widened to `AbstractMatrix` |

  `l2norm` was the dangerous one. `ParameterSet` admits **any** keyed `NamedTuple`, so the meaning of
  `l2norm` of any `NamedTuple` changed for the whole session of anything that loaded this package.
  Upstream it is written as `L2norm`, from which the generic `l2norm(x) = sqrt(L2norm(x))` follows,
  and it still calls `l2norm` on the leaves — so `l2norm(::AbstractLieAlgHorMatrix)` and
  `l2norm(::VectorStorageMatrix)`, which are this package's and stay here, keep deciding what a lift
  and a `VectorStorageMatrix` contribute. Recursing through `L2norm` instead would have read a skew
  block's dense `n × n` interface and counted its entries twice.

  The `alloc_h` arm for a `Manifold` stays, because `Manifold` is this package's type; only the
  parameter-set arm moved. `GradientAutodiff(F, ::Manifold)` stays for the same reason, and takes
  precedence over the new `AbstractMatrix` method upstream, which is what it has to do: it rebuilds
  the manifold with `manifold_constructor` where a `reshape` would only give back a matrix.

  **This raises the compat bound to `SimpleSolvers = "0.13.2"`**, because four of the five are only
  there from that release. The fifth went to `NeuralNetworkParameters` rather than to `GeometricBase`,
  so `GeometricBase = "0.14.9"` stands and that package leaves the registration chain entirely. See
  *Known issues* below for what the `SimpleSolvers` bound costs downstream.

- **`(::Gradient)(::Matrix)` and `(::Gradient)(::ParameterContainer)` are de-pirated in place, by a
  new `RiemannianGradient <: SimpleSolvers.Gradient`.** These two could not follow the five above:
  their bodies are `rgrad`, this package's Riemannian projection, which `SimpleSolvers` neither has
  nor should. Wrapping is the other way to de-pirate a method, and it is the shape
  `GeometricMachineLearning` already uses for its `_GMLGradient`.

  `RiemannianGradient` holds the Euclidean gradient, forwards the coordinate interface to it
  unchanged, and applies `rgrad` — leaf by leaf for a parameter set, whole for a matrix — when called
  on an iterate. `Optimizer` wraps for you at all three of its entry points, on a caller's own
  gradient as well as on `default_gradient`'s, and only where the wrapper changes something: a plain
  vector has no projection to apply, and a `Manifold` reaches `(::Gradient)(::Manifold)`, which is
  owned already because `Manifold` is this package's type. Wrapping is idempotent, so passing a
  gradient that is already wrapped is not a second wrapper.

  `gradient(opt)` therefore returns a `RiemannianGradient` for a parameter set where it returned a
  `GradientAutodiff` before. The name is exported for that reason. A bare `SimpleSolvers` gradient
  called on a parameter set is now a `MethodError` rather than a silently projected answer — which is
  what "this method belonged to nobody" meant in practice.

- **`(::Hessian)(::AbstractMatrix, ::OptimizerSolution)`**, the "this has to be called together with
  a cache" error, is now two methods on this package's own Hessian types (`IterativeHessian` and
  `NoHessian`) rather than one on `SimpleSolvers.Hessian`. `OptimizerSolution` is an alias for a union
  of types this package does not own, so the method owned neither side. The error is about the
  hessians *here* — they are built from a cache, and there is nothing to say about anyone else's — so
  naming them is the more accurate signature as well as the fix.

### Added

- **`test/aqua_tests.jl`**, on a new test dependency `Aqua = "0.8"`. `Aqua.test_piracies` is the whole
  of the ownership property; the rest of the file asserts, with `which`, that each of the five moved
  methods now resolves to the extension that owns it, and that the two which stayed resolve here.
  Without that second half a move upstream is indistinguishable from a deletion, which is how a
  capability gets lost in a de-piracy pass.

  The `GeometricBase` half is asserted here and not in `GeometricBase`, because that package's test
  environment would have to resolve `NeuralNetworkParameters`, whose floor is Julia 1.11, while
  `GeometricBase` still supports 1.10. Both are hard dependencies here.

  Only the piracy check. `Aqua.test_all` would also run `test_ambiguities`, which reports **139** —
  almost all between this package's structured matrices and `LinearAlgebra`/`ArrayLayouts` methods on
  `AbstractMatrix`, none of them touched by this work. That is its own piece of work; turning it on
  before then would only mean a test expected to fail.

- **`RiemannianGradient`** is exported, since it is what `gradient(opt)` returns for a parameter set
  and therefore what a caller dispatches on.

- **`test/optimizer_status_tests.jl` asserts the shape of the quadrature fold**, not only its value.
  The fold moved to `GeometricBase` with the rest of the group, and what could have been lost in the
  move is that it calls `l2norm` on each leaf and not `L2norm` — so a leaf that keeps its numbers
  behind another interface still decides what it contributes. A `StiefelLieAlgHorMatrix` leaf is the
  case where the two answers differ: it presents a dense `N × N` skew-symmetric matrix over its free
  parameters, so the generic `L2norm(::AbstractArray)` counts the off-diagonal blocks twice. The
  testset pins `l2norm(ps) ≈ √(l2norm(B)^2 + …)` and `l2norm(B) ≉ norm(B)`, so a future rewrite
  upstream that folds through `L2norm` fails here rather than quietly changing every stopping
  criterion on a manifold network.

### Fixed

- **Four reachable method ambiguities in the `_copyto!` family.** All four raised
  `MethodError: … is ambiguous` at run time, and one of them is reached by an ordinary `solver_step!`:
  a **flat** `NetworkParameters` with a manifold leaf died on its first step, inside
  `_copyto!(solution(cache(opt)), section(cache(opt)))`.

  | call | ambiguous between |
  |---|---|
  | `NetworkParameters ← GlobalSectionNamedTuple` | `(::ParameterContainer, ::GlobalSectionNamedTuple)` and `(::NetworkParameters, ::NamedTuple)` |
  | `GlobalSectionNamedTuple ← NetworkParameters` | `(::GlobalSectionNamedTuple, ::ParameterContainer)` and `(::NamedTuple, ::NetworkParameters)` |
  | `ArrayNamedTuple ← NetworkParameters` | `(::ParameterContainer{T}, ::ParameterContainer{T})` and `(::NamedTuple, ::NetworkParameters)` |
  | `NetworkParameters ← ArrayNamedTuple` | `(::ParameterContainer{T}, ::ParameterContainer{T})` and `(::NetworkParameters, ::NamedTuple)` |

  The cause is the one that made half of issue #16 piracy, showing up as ambiguity instead:
  `ArrayNamedTuple` and `GlobalSectionNamedTuple` are **aliases for `NamedTuple`**. Each pair of
  methods is written flat-then-nested — `GlobalSectionNamedTuple` for a flat section tree, a bare
  `NamedTuple` for a nested one, because "a `NamedTuple` of `GlobalSection`s to any depth" is a
  recursive type Julia cannot express — and a bare `NamedTuple` also matches a flat
  `GlobalSectionNamedTuple` while `NetworkParameters` is also a `ParameterContainer`. On the overlap
  neither method wins.

  Four disambiguating methods close them, each taking the body the semantics pick rather than an
  arbitrary tie-break; on the overlap the candidates agree anyway, so no shape that reached one of
  them by accident changes its answer.

  **The suite could not have caught this**, which is the part worth keeping:
  `test/network_parameters_optimizer.jl` drives a *nested* container, whose section tree is a
  `NamedTuple` of `NamedTuple`s and therefore not a `GlobalSectionNamedTuple`, and
  `test/named_tuple_parameters.jl` drives a *bare flat* `NamedTuple`, which is not a
  `NetworkParameters`. Flat-and-wrapped is the corner between the two, and it is what a caller gets by
  wrapping a flat parameter `NamedTuple`. `test/container_section_copy.jl` covers it, including an
  end-to-end `solver_step!` on a flat container with a `StiefelManifold` leaf.

  Three further pairs in this family stay reported by `Test.detect_ambiguities` and are documented in
  place rather than closed, because each intersects only at a shape this package's API cannot build:
  the empty `NamedTuple` (`Optimizer` on an empty set raises before any copy happens), the
  mixed-element-type section (a section is built from the iterate it is a frame for, so its element
  type is the iterate's by construction), and a `Manifold`-anchored `GlobalSection` with no lift
  (`GlobalSection(::Manifold)` always builds one). A method that exists only to satisfy a static
  checker is a method somebody later has to reason about.

- **The three measurement harnesses timed on the wall clock.** `first_call` in
  `scripts/walk_compile_cost.jl` and `scripts/nested_kwargs_cost.jl`, and `fc` in
  `scripts/adam_cache_attribution.jl`, used `time()`, which steps when the system adjusts it. The
  sweep behind `NeuralNetworkParameters`' 0.2.5 verification produced a reported **-1.4 s** for one
  row of its `leaf_layout_cost.jl` — the sibling of `walk_compile_cost.jl` — where two re-runs read
  0.53. All three now use `time_ns()`, which is monotonic.

  A negative first-call time is at least obvious. A step of the same size in the other direction is
  not, and would be indistinguishable from a genuine 1.4 s regression on a harness whose whole
  purpose is to make a compile-time cliff visible — these three are what found the last two defects
  upstream, so a figure they get wrong is a figure that gets committed.

### Known issues

- **This release cannot be resolved beside `GeometricIntegratorsBase` 0.6.3 or
  `GeometricIntegrators` 0.17.** Both pin `SimpleSolvers = "0.12.1 - 0.12"`, and this release requires
  0.13.2. That bound already held `NonlinearIntegrators` at `GeometricOptimizers` 0.5.0 and
  `GeometricMachineLearning`'s *test* environment at `SimpleSolvers` 0.12.2 before this change — it is
  not new — but 0.6.0 could still resolve against 0.12 and 0.6.1 cannot. `GeometricMachineLearning`'s
  own package dependencies are unaffected; it is its test extra `GeometricIntegrators = "0.16 - 0.17"`
  that blocks. The fix is upstream of all of this: `GeometricIntegratorsBase` has to widen to
  `SimpleSolvers = "0.13"`.

  Verified against this release with that one test file set aside: `GeometricMachineLearning`'s suite
  passes in full otherwise, including every optimizer and structured-parameter test.

## [0.6.0]

**A `NetworkParameters` runs through the optimizer.** 0.5.0 made it a member of `OptimizerSolution{T}`,
which bound `T` at the eleven sites that take the element type from the *type* of the solution and left
everything else dispatching on `ArrayNamedTuple` — so a container got several frames into a solve
before failing. This finishes it.

**Three of this entry's own claims were re-measured against `NeuralNetworkParameters` 0.2.3 and did not
survive**, and they failed the same way rather than three different ways: each was a sound measurement
attributed to the wrong cause. The layout cliff was one unused type parameter upstream, not nesting and
not `parameterlayout`; most of the allocation saving was upstream's D13 and D17, not the flat buffers;
and the nested cache figure is `AdamCache`'s constructor on the compat floor, not the layout. No code
from #68 or #69 is reverted — the designs hold and 0.2.3 improves both — and the harness now sweeps the
axis whose absence made the first of those possible. Two new *Known issues* came out of it.

### Added

- **The elementwise primitives, and the ~40 sites around them, take a container.** A new alias,
  `ParameterContainer{T} = Union{ArrayNamedTuple{T}, NetworkParameters{T}}`, is what they dispatch on;
  `GradientArrayOrNamedTuple` and `OptimizerSolution` are written in terms of it.

  It is used **only where `T` genuinely has to bind** — beside a `Matrix{T}`, an `f̄::T`, a `b::T` —
  because `T` means two different things across the union: for `ArrayNamedTuple{T}` it guarantees that
  every leaf is an `AbstractArray{T}` *and* that the set is flat, while for `NetworkParameters{T}` it is
  a promotion over leaves that may nest to any depth. So a method written on it accepts a nested
  container while rejecting the nested plain `NamedTuple` describing the same network.

  Everywhere else the signature is **`NeuralNetworkParameters.ParameterSet`**, the same union without
  either bound, added upstream in 0.2.2 at this package's request. Sixteen sites here had been spelling
  `Union{NamedTuple,NetworkParameters}` out inline, each with a comment explaining why the narrower alias
  would not do; `AbstractNeuralNetworks` had eight of the same, `GeometricMachineLearning` fifteen, and
  `SymbolicNeuralNetworks` had named it `EquationSet`. One name now, in the package that owns the type.
  This is what raises the compat bound to `NeuralNetworkParameters = "0.2.2"`.

  `test/network_parameters_optimizer.jl` drives a **nested** container — a real network's shape, which
  the flat `NamedTuple` of `test/named_tuple_parameters.jl` does not cover — through every algorithm,
  both retractions and both element types, plus `solve!`, `BFGS` and `DFP`. One of its testsets is the
  statement that the change is behaviour-preserving: a nested container and the flat `NamedTuple`
  describing the same problem reach the same iterates.

  `test/named_tuple_parameters.jl` passes **unchanged**, which is the additive claim: a bare
  `NamedTuple` is still a solution and still takes the same steps.

- **The walks those bodies are written in are `NeuralNetworkParameters`'.** `mapparameters` out of
  place, `mapparameters!` in place. Both recurse on the branches and reach leaves at any depth, so a
  container comes back a container and a flat `NamedTuple` a flat `NamedTuple`, and both check that the
  keys agree at every level — which is what `Base.foreach` would not do, going through `zip` and
  iterating values.

  This branch carried a local copy for most of its life, `_mapleaves`/`_mapleaves!` in
  `src/parameter_walks.jl`, written here because `mapparameters` could not be compiled on the shape
  this package has a named consumer for: its across-children walk was an `@inline`d `Base.tail` chain,
  one specialisation per child over argument types each as long as the branch, and inference on that
  grew as the cube of the width.

  **Retiring it took two upstream releases, not one, and the second only happened because the first
  was measured wrong.** `NeuralNetworkParameters` 0.2.2 wrote those walks out as `@generated` bodies,
  and the figure that said the job was done — `mapparameters(zero, ps)` at "0.00 s" on the 369-entry
  flat set — came from a harness that swept its shapes in one process, so the flat table was reading
  what the nested one had already compiled. Cold, the same call cost **1.51 s** against `Base.map`'s
  0.01 s, and deleting this file on the strength of that figure took a cold `OptimizerCache` on that
  set from **1.57 s to 71.16 s**. A 45× regression read as an improvement for exactly as long as
  nobody ran the tables in the other order.

  What fixed it upstream is a *split* rather than more of the same: `_map_zip`, which is all
  `mapparameters` and `mapstorage` are, is written out to 32 children and hands wider branches back to
  `Base.map`, while the walks that run in an inner loop — `flatten!`, `unflatten!`,
  `foreachparameters`, the out-of-place `unflatten` — stay written out at every width. The rule is
  "in an inner loop or not" and not "wide or narrow": `map` past 32 fields drops to `Base`'s `Any32`
  loop and gives up inference, which for a once-per-cache walk is one dynamic dispatch at the call
  site and for a per-iteration walk would be one per call.

  Measured cold, one shape per process, Julia 1.11.9, on the 369-entry flat set:

  | | 0.5.0 (`map`) | `_mapleaves` retired, before the split | after |
  |---|---|---|---|
  | `_zero` | 0.01 s | 1.42 s | **0.09 s** |
  | `_copy` | 0.01 s | 0.79 s | **0.09 s** |
  | `_similar` | 0.01 s | 1.18 s | **0.10 s** |
  | `GlobalSection` | 0.01 s | 0.90 s | **0.20 s** |
  | `OptimizerCache(Adam)` | 1.57 s | 71.16 s | **2.42 s** |

  The last row is the one to read: it is the entry point `GeometricMachineLearning` reaches without
  going through `flatten`, and `scripts/walk_compile_cost.jl --caches` is where it comes from. It
  **reproduces** — re-measured against `NeuralNetworkParameters` 0.2.3 it is 2.45 s bare and 2.38 s
  wrapped at flat 369 on 1.11.9, and 2.34 s on 1.13 — so the case for retiring the local walk stands
  where it was made. The 0.85 s this release is dearer than 0.5.0 is `mapparameters` against `map` and
  nothing else: an earlier draft of this line called it "the honest price of the container — a
  `ParameterLayout` is built and stored where `map` over a `NamedTuple` needed none", and `AdamCache`
  builds no layout. It reaches `_zero`, `_similar`, `_fill!` and `GlobalSection` and stops. The same
  mistake, attributing a walk's cost to the layout, is what the *Known issues* entry below is about.

  **The four rows above it are not reproducible from the committed harness as it stood**, which is worth
  saying rather than quietly fixing: `scripts/walk_compile_cost.jl` did not print `_zero`, `_copy`,
  `_similar` or `GlobalSection` at all, so where those figures came from cannot be recovered. It prints
  them now, and on 0.2.3 at flat 369 they read 0.04 / 0.01 / 0.01 / 0.01 s — below the column above,
  which is a row order the old script did not have rather than a change in the walk. Treat the first
  three columns as the record of a decision and the harness as the thing that can be re-run.

  The **nested** container is a different matter: `OptimizerCache` on a 16 × 24 network of 384 leaves
  costs 87.56 s, which 0.5.0 cannot be compared against because it raises a `MethodError` on `_copy`
  there — the whole of what this release is about. **It reproduces at 88.59 s against
  `NeuralNetworkParameters` 0.2.3**, whose `parameterlayout` on that shape is 0.87 s, so it was never
  what this file used to say it was. See *Known issues*.

  Deleting the file fixes a latent bug of its own, and that is the part worth keeping. `_as_walkable`,
  the normaliser it put its trailing arguments through, had a **catch-all** where
  `NeuralNetworkParameters`' `_as_namedtuple` has three exhaustive methods — so a tree zipped against a
  *leaf* raised upstream and did not raise here. What happened here instead was worse than raising:
  `map` over a `NamedTuple` and a bare `Vector` matches neither of `Base`'s specific methods, falls
  through to the generic iterator `map`, and `zip`s the branch's *entries* against the leaf's
  *elements*, handing back an `Array`. Measured: `map(f, (a = [1.0], b = [2.0]), [9.0, 9.0])` gives
  `[([1.0], 9.0), ([2.0], 9.0)]`, and a leaf shorter than the branch is silently truncated. Nothing
  here paired a leaf with a branch, so nothing depended on either behaviour.

  Its three uses that were not the walk — `_dot_leaves`, `_block_αmax` and `_linesearch_parameters`,
  each `values(_as_walkable(x))` — need no normaliser at all: `Base.values(::NetworkParameters)` is
  defined upstream, so one `values(x)` serves both members of the union. Two of those three no longer
  take a `values` either, and the first no longer exists: the fold entry below replaces `_dot_leaves`
  and `_block_αmax`'s branch descent with upstream's zipped walk, which normalises the shapes itself.

  `scripts/walk_compile_cost.jl` stays, and **runs each shape in a process of its own now**, which is
  the change that matters more than any row in it. There is no `--cold-mapparameters` flag any more,
  because a number you have to ask for in a special way is a number the default run is getting wrong.
  It is also this package's half of the answer to open issue C9, and what would catch the cliff
  returning.

### Changed

- **Julia 1.11 is the minimum**, up from the 1.10 LTS, in step with the rest of this family of
  packages. Nothing here was accommodating 1.10 directly; what the floor buys is that
  `NeuralNetworkParameters` 0.2.2 can be depended on, since its `@generated` walks need 1.11 to be
  inlined and 1.10 will not inline one. The CI matrix is `1.11 / 1.12 / 1.13 / nightly`, so open issue
  **D1** — a 140× compile-time cliff on 1.12 that 1.13 does not have — stays observed rather than
  becoming the floor.

- **`GeometricBase = "0.14.9"`**, which is the release that takes `L2norm` over any array. See the
  `l2norm` entry under *Removed* below.

- **`NeuralNetworkParameters = "0.2.3"`**, up from `"0.2.2"`. 0.2.2 is what the code needs — it is the
  release whose `@generated` walk split let `src/parameter_walks.jl` be retired, and 0.2.3 changes
  `src/layout.jl` and nothing else, so nothing here would fail against 0.2.2. The bound moves because
  the figures this release quotes are 0.2.3's, and admitting 0.2.2 would admit the elevenfold
  `parameterlayout` cliff the *Known issues* entry below says is gone. Source-compatible either way:
  `LeafLayout` lost a field and a type parameter in 0.2.3, and nothing in this package constructs a
  layout by hand or names its parameters.

  Two things it buys that are this package's rather than upstream's, both on the quasi-Newton caches,
  and both measured by the new `scripts/cache_type_cost.jl` — which is a separate file because neither
  is a time, and `walk_compile_cost.jl` forks a process per row in order to measure times honestly.

  `_flat_scratch` gives `BFGSCache`/`DFPCache` four `FlatParameters` buffers, and
  `FlatParameters{T,DT,LT}` carries its `ParameterLayout` as `LT`, so the layout type is a type
  parameter of the cache through `flat::FT` and whatever the layout holds the cache holds:

  | `_flat_scratch` on | type nodes | bytes over the buffers' floor |
  |---|---|---|
  | 369 leaves, bare | 7415 → **2987** | 79 960 → **47 488** |
  | 369 leaves, in a `NetworkParameters` | 7419 → **2991** | 79 960 → **47 488** |
  | 128 leaves, alternating `Matrix`/`Vector` | 2595 → **1059** | 25 856 → **14 592** |
  | one 400 × 400 leaf and one 400 vector | 75 → **51** | 1 283 832 → **480** |

  - **The cache's own type shrinks by 2.5×.** Under 0.2.2's `LeafLayout{N,P}` every leaf's *concrete
    array type* was in the cache's signature.
  - **The cache stops retaining the parameter arrays**, which is the last row. 1 283 832 bytes over the
    floor is the 400 × 400 leaf (1 280 000) plus the 400 vector (3 200) plus 632 of layout: a 0.2.2
    layout held a live reference to each leaf, so the four buffers pinned a second copy of the whole
    parameter set for the lifetime of the cache. On 0.2.3 they retain **480 bytes**, whatever the leaves
    weigh.

  Neither is a run-time cost and `scripts/optimizer_allocations.jl` confirms it: a 20-iteration `solve!`
  allocates within 0.04 % of the same bytes on 0.2.2 and 0.2.3, on all four shapes and both methods.

  What *is* a cost, and the one figure in this release that a quasi-Newton user feels directly:
  `OptimizerCache(BFGS)` on the flat 369-entry set goes **27.88 s → 12.24 s** bare and 17.01 s → 11.70 s
  wrapped, on Julia 1.11.9. That is the cache's `flatten`, which the `Adam` cache does not do — and
  `OptimizerCache(Adam)` on the same set is 2.42 s on 0.2.2 and 2.45 s on 0.2.3, unmoved, as a walk
  should be.

- **`global_rep` and `apply_section` rebuild in the shape of the *parameters*, not of the section.**
  These walks are written with their arguments the other way round from the rest, and it is not
  cosmetic. `GlobalSection(::NetworkParameters)` deliberately returns a plain tree — a section is not a
  parameter — so a walk driven by the section hands back a plain *nested* `NamedTuple` for a container.
  That shape is neither an `ArrayNamedTuple` (its values are branches, not arrays) nor a container, so
  no alias here covers it and `_copyto!(gradient_array(cache), ·)` has no method for it. What these two
  produce is a point and a gradient, and a parameter set has the parameters' shape.

- **The section-side `copyto!` methods come in pairs.** `GlobalSectionNamedTuple` is flat by
  construction, and a container's section tree is nested; "a `NamedTuple` of `GlobalSection`s to any
  depth" is a recursive type Julia cannot express, so there is no widening of that alias that covers
  both. The *other* side carries the dispatch instead, which it can because a container is a type with
  a name rather than an alias for `NamedTuple`. That is the one place where taking the container bought
  this file something beyond a wider signature.

- **`l2norm` and `solution_scale` on a parameter set are recursive** rather than `map` + `sum` over one
  level. Over a container the quantities to combine in quadrature are at the *leaves*; combining its
  layers instead would have handed `l2norm` a whole layer, and every stopping criterion of `solve!` is
  computed from these.

  The recursion itself allocates nothing, and as of this release neither does `l2norm` of a parameter
  set. The 32 bytes per matrix leaf it used to cost were the `vec` in `l2norm(a::AbstractMatrix)`, one
  of the two pirated methods of **#16** group 1; the entry under *Removed* below is what took them
  upstream, and `test/flat_buffer_allocations.jl` asserts the exact zero rather than "one wrapper per
  matrix leaf". An earlier draft of this entry said the fix "belongs with the upstreaming those two
  methods are waiting for" — it did, and the upstreaming is in this wave.

- **`_manifold_αmax` descends a branch** instead of writing it off as "not a `Manifold`". Every
  top-level value of a container is a *layer*, so without this a container would have got `Inf` — no
  step ceiling at all — for exactly the parameter shape a network has, and issue A1b would be back for
  it. A flat `ArrayNamedTuple` never reaches the new method, its values being arrays by construction.

- **`NeuralNetworkParameters` moves to `"0.2.4"`**, from `"0.2.3"`, for the zipped `foldparameters` and
  `foldstorage` the *Fixed* entry on the four folds is about. A floor and not a widening: the four folds
  and `_dot`'s element type call into 0.2.4 unconditionally, so 0.2.3 is an `UndefVarError` rather than a
  slower path. Nothing else in the resolution moves with it.

  Also imported for the first time: `parameter_eltype`, which is how `_dot` gets an element type now that
  it no longer takes one off its signature. Neither the folds nor it are re-exported, for the reason the
  `using` block in `src/GeometricOptimizers.jl` gives about `flatten`.

### Fixed

- **`_dot` of two horizontal lifts whose element types differ returned the *ambient* product, at twice
  the right value.** Silently: it did not raise, and no test could see it, because every assertion on
  that path checked a type or compared two spellings of the same element type.

  `AbstractLieAlgHorMatrix{T} <: AbstractMatrix{T}`, so a `Float32` lift beside a `Float64` one missed
  the alias that binds a `T` — `LiftOrNamedTuple{T}` needs one `T` for both — and fell through to
  `_dot(::AbstractVecOrMat, ::AbstractVecOrMat)`, which is `LinearAlgebra.dot`. That is the ambient
  Frobenius product, and it counts each off-diagonal block of a lift twice. `St(6, 3)`:

  | | before | after | `dot(flatten(a), flatten(b))` |
  |---|---|---|---|
  | `_dot(lift{Float32}, lift{Float64})` | 5.504356027190567 | **2.7521780135952834** | 2.7521780135952834 |

  Exactly the factor of two `docs/src/linesearch_on_manifolds.md` gives a section to — "paired
  ambiently, the slope came out as ``2\varphi'(\alpha)`` where ``\varphi'(\alpha)`` was wanted" — and it
  reaches the three quantities that chapter names: `trial_slope`, the quasi-Newton denominator
  ``\delta^\mathsf{T}\gamma``, and the predicted decrease. A mixed-precision solve would have searched
  along a curve whose derivative was doubled.

  The `_dot` widening recorded under *Removed (breaking)* is what fixes it: `DottableSet` binds no
  element type, so the pair reaches `foldstorage` over the free parameters like every other one does.
  It works because `parameter_eltype` recurses — its `AbstractArray` method asks `freeparameters`
  before falling back to `eltype`, so a lift answers with the promotion over its blocks rather than
  with the `Union{}` catch-all.

  **Found by review of that widening rather than by the widening itself**, which is why it is recorded
  separately: that entry claimed the differing-eltype pair "used to be a `MethodError`", and that was
  true of two `NamedTuple`s and two containers and not of two lifts.
  `test/flat_buffer_allocations.jl` now asserts the *value* against the flattening for that pair, and
  asserts that the ambient product is the other number, so the assertion cannot be satisfied by both.
  A same-eltype pair is the control; it was never affected, which is the whole reason this survived.

- **The flat buffers are no longer allocated per call.** Every quantity a quasi-Newton method forms
  lives in the *flattened* coordinates — `Q` is sized by the length of the flattening, `outer!` forms
  its outer products there, `_dot` pairs there — while the parameters are a `NamedTuple`, a container,
  or a horizontal lift of the ambient shape. Each crossing built a fresh flat vector: two per `_dot`,
  two per `outer!`, one plus a `ParameterLayout` per `_mul!`, one more for the `γ` of each `update!`.

  Two different fixes, because two different things were wrong.

  - **`_dot` needed no buffer at all**, which is the better half: `dot` of two flattenings is the sum
    of the per-leaf `dot`s, so the sum can be taken without the vectors. It is also the half that
    matters most — `_dot` is the hottest of the sites, once per line-search trial slope and once per
    `OptimizerStatus` — and it reaches *every* cache, including the first-order ones, where a buffer on
    the quasi-Newton caches would not have.

    The summation order changes with it, per leaf and then across rather than once over the
    concatenation. Both are ``\sum_i a_ib_i`` and they differ at round-off;
    `test/flat_buffer_allocations.jl` pins the two against each other, and
    `test/optimizer_convergence/svd_optim.jl` — where `_dot` feeds `curvature_is_usable`'s relative
    test and the line search's ``\varphi'`` — is unmoved.

  - **`outer!` and `_mul!` genuinely need the flat form**, so `BFGSCache` and `DFPCache` carry buffers
    to write into: one field and one unbounded type parameter each, a `NamedTuple` of
    `NeuralNetworkParameters.FlatParameters` built from `_zero(x)` and not `x`, for the reason the
    `flatlength(_zero(x))` beside it gives. `FlatParameters` rather than a bare `Vector` because it
    carries its own layout and keeps it through `similar`, which is what retired the `parameterlayout(c)`
    call the deleted `_mul!` methods made. See `_flat_scratch`.

    `nothing` for an `AbstractVector` solution, where the parameters *are* the flat coordinates and
    buffering would add a copy per iteration and buy nothing.

  `γᵀQγ` goes with them: `dot(γ, Q, γ)`, the three-argument `dot`, where `Δg2' * Q * Δg2` materialised
  `Q * γ`.

  `test/flat_buffer_allocations.jl` pins each site at exactly zero, for all four shapes of solution and
  both methods. What that is worth end to end, from `scripts/optimizer_allocations.jl` — a 20-iteration
  `solve!`, bytes allocated, Julia 1.11.9, Apple M4 Max.

  **Three columns, and the middle one is why.** An earlier version of this table had two, and said "both
  columns re-measured […] so the two are comparable rather than merely recorded". They were not
  comparable: the 0.5.0 column resolved `NeuralNetworkParameters` **0.2.1** and this release's resolved
  0.2.2, and 0.2.2 is the release that fixed *upstream's* D13 and D17 — `flatten!`/`unflatten!` and
  `mapparameters!` allocating past about forty children. So the table charged this release for an
  improvement that was mostly upstream's. Holding the dependency fixed at 0.2.3 separates them:

  | | 0.5.0, NNP 0.2.1 | 0.5.0, NNP 0.2.3 | this release, NNP 0.2.3 |
  |---|---|---|---|
  | `BFGS`, `Vector` | 36 232 | 28 616 | **22 536** |
  | `BFGS`, bare `Manifold` | 1 474 408 | 1 103 000 | **1 056 760** |
  | `BFGS`, flat `NamedTuple` | 2 431 912 | 1 694 072 | **1 559 832** |
  | `BFGS`, container | `MethodError` | `MethodError` | **1 566 072** |
  | `DFP`, `Vector` | 35 432 | 27 640 | **21 240** |
  | `DFP`, bare `Manifold` | 1 474 888 | 1 102 024 | **1 055 464** |
  | `DFP`, flat `NamedTuple` | 2 425 608 | 1 684 936 | **1 550 216** |
  | `DFP`, container | `MethodError` | `MethodError` | **1 556 456** |

  The first column is the old one reproduced **byte for byte** by pinning 0.2.1 against the same 0.5.0
  worktree, which is what identifies the confound rather than merely suspecting it. Read across the flat
  `NamedTuple` row: 2 431 912 → 1 559 832 is 872 080 bytes, still about a third — but **737 840 of them
  are upstream's** and 134 240 are this release's, so roughly a sixth of the saving is ours. The `Vector`
  row is where the split is closest to even, 7 616 against 6 080, and it is also the row with no
  parameter tree at all: `flatten(T, ::Vector)` copies, and `Δg2' * Q * Δg2` allocated there too.

  **This does not change what the code does or whether it should stay.** The flat sites are at exactly
  zero and `test/flat_buffer_allocations.jl` asserts it site by site; what changes is the size of the
  claim, and 6 712 bytes an iteration on the shape `GMLDatasets` uses is what removing four flat vectors
  per `update!` is actually worth. The remaining 1.56 MB is the gradient evaluation and the parameter
  trees, which is what *Known issues* has always said.

  The figures moved twice while this branch was in review, and both moves are the walk rather than the
  weather. First, `_mapleaves!` became `foreachparameters`: the discarded result tree cost one
  `NamedTuple` per branch per call, worth about 17 000 bytes on the parameter-set rows and 194 000 on
  the container ones. Then `_mapleaves` itself went, and `l2norm` stopped taking a `vec` per matrix
  leaf, worth another 16 000 and 32 000. The `Vector` and `Manifold` rows have no branches at all and
  moved by neither — which is what says the difference is where it is claimed to be.

  The 0.5.0 column is within 0.06 % of the figures recorded before it was re-measured (36 312 against
  36 232, 2 430 488 against 2 431 912, and so on), the residue being the Julia version the first
  measurement was taken on. That is a *reproduction*, and it is recorded because six measurements in
  this file's own catalogue have failed to be.

  And it reproduced a third time, which is how the middle column above got its meaning: pinning
  `NeuralNetworkParameters` 0.2.1 against that same 0.5.0 worktree returns **36 232 / 1 474 408 /
  2 431 912 / 35 432 / 1 474 888 / 2 425 608**, byte for byte. A figure that reproduces exactly under a
  dependency you did not think you were varying is the strongest evidence available that the dependency
  is what the comparison was measuring. Neither reproduction was wrong; between them they say the
  measurement was sound and the *attribution* was not, which is the same distinction the layout entry
  under *Known issues* turns on.

- **`test/flat_buffer_allocations.jl` was measuring its own harness.** Every `@allocated` in it sat in
  the body of a `@testset`, with only the *call* inside a function. A `@testset` body is a closure and a
  `for` inside one captures its loop variables, so what those twelve assertions measured was the boxing
  of the arguments on the way in. On Julia 1.13 the boxes are elided and all twelve passed; on 1.11 they
  are not, and all twelve failed — 16 bytes for `_dot`, 32 for the secant pair. Nothing in `src/` was
  wrong.

  It matters because 1.11 is the floor now, so the file would have failed outright, and because the
  file's own header states the rule it was breaking. The `@allocated` is *inside* each `_measured_*`
  function now, each of which warms the call before measuring it, and the assertions read
  `@test _measured_dot(a, b) == 0`.

  This is the second time this exact mistake has been caught in this ecosystem by writing the rule down
  and then not following it — `NonlinearIntegrators`' `test/quality/inference_and_allocations.jl`
  records the same one, where it inflated a figure from 11 424 bytes to 31 411.

- **The four folds over a parameter set are upstream's zipped fold, and the 26–71 s compile cliff on
  Julia 1.12 and 1.13 is gone.** This was a *Known issue* of this release until
  `NeuralNetworkParameters` 0.2.4, and the entry that stood here is replaced by this one. Issue [#70],
  upstream [NeuralNetworkParameters#19][nnp19].

  `l2norm`, `solution_scale` and `_dot` were `Base.tail` recursions written here, for one reason: there
  was no *zipped* `foldparameters`, so a consumer wanting `Σ aᵢbᵢ` or `Σ f(aᵢ)²` over a parameter set had
  to write the recursion itself. A branch of `k` children yields `k` specialisations whose argument types
  are each `O(k)` long — twice over for a zipped fold, which is why `_dot` was about 2.2× the other two
  on every version — and at `k = 369` that is 26 to 71 s of inference. 369 entries in one flat
  `NamedTuple` is not a synthetic width: it is the MNIST transformer of `GMLDatasets`, the consumer this
  file has been quoting all release.

  **There were four such folds, not three.** `_manifold_αmax` is the fourth, and #70's own count missed
  it: a zipped `Base.tail` chain, positional over `values`, and the only one of the four on the
  *per-iteration* path, through `linesearch_parameters`. It was also not a row in
  `scripts/walk_compile_cost.jl`, so replacing only the three the issue named would have made the sweep
  report an all-clear that a manifold or `NamedTuple` solve on a 369-wide set did not have. It has a row
  now.

  `scripts/walk_compile_cost.jl`, cold, one process per cell, flat 369, seconds. The `0.2.3` columns are
  the *Known issue* being closed and are kept as the "before", since a future reader comparing against
  them is how the cliff coming back would be noticed:

  | | 1.11.9 | 1.12.7 | 1.13.0-rc3 |
  |---|---|---|---|
  | `l2norm`, was | 0.69 | **26.44** | **35.50** |
  | `l2norm`, now | **0.87** | **1.27** | **0.95** |
  | `solution_scale`, was | 0.63 | **26.07** | **35.14** |
  | `solution_scale`, now | **0.50** | **0.48** | **0.52** |
  | `_dot`, was | 1.47 | **57.34** | **70.98** |
  | `_dot`, now | **1.71** | **2.04** | **1.59** |
  | `αmax`, now | **0.03** | **0.03** | **0.03** |
  | **sum, was** (three folds) | **2.79** | **109.85** | **141.62** |
  | **sum, now** (four folds) | **3.11** | **3.82** | **3.09** |

  So 29× on 1.12 and 46× on 1.13, and **the three columns now agree**, which is the pass condition the
  sweep's header states. The nested 16 × 24 shape, which never had the cliff, is unmoved: 0.25–1.07 s
  across all three.

  **And on 1.11 it is slightly dearer**, which is the honest half of the table: 0.87 s against 0.69 for
  `l2norm` and 1.71 against 1.47 for `_dot`, about a third of a second over the four. A generated body at
  literal indices is not free, it is merely not quadratic. That is the trade, and 1.11 was the version
  that had nothing to gain.

  Two things fall out of the replacement rather than being aimed at, and both are guarantees the old code
  said it lacked:

  - **The pairing is by key, and the widths are checked** — in upstream's *generator*, so both cost
    nothing at run time and a mismatch raises before the fold is specialised. `_dot` and
    `_manifold_αmax` paired positionally over `values` and checked neither; `_dot`'s comment said this
    was where such a check would go if one were ever wanted. Key *order* is the case that matters, since
    that one produced a number rather than an error. Pinned in
    `test/neural_network_parameters_protocol.jl`.

    **`_manifold_αmax` bounds its first argument to a `ParameterSet` so that the guarantee cannot be
    spelled around**, and that bound was added on review rather than with the fold. Upstream pairs a
    keyed branch by key and a `Tuple` branch *positionally* — a `Tuple`'s blocks have no keys to agree
    on — so an unconstrained signature still accepted `_manifold_αmax(values(sol), values(δ), c)`, which
    is how every call site was spelled until this release, and folded it positionally. On a two-block
    set with the direction's keys crossed: `3.12` by key, `ArgumentError` by key with the sets given
    whole, and `6.40` through `values`. The four-method recursion this replaced made that call
    unspellable by running out of methods, so the bound is not new strictness — it is the strictness the
    deletion dropped. `_dot` needs no such bound; its narrowest signature is already `DottableSet`.
  - **The grouping of the leaves no longer shows in the sum.** Upstream threads its accumulator through
    the nested branches, so a left fold over a tree is the left fold over the flat leaf list whatever the
    tree's shape — where the recursion this replaced was a right fold that happened to align. The same
    numbers written flat, written nested and wrapped in a container pair to the same `Float64` exactly,
    and `test/flat_buffer_allocations.jl` asserts `==` for the three.

  Allocations are unchanged at zero, and that is now pinned at the width where it could fail: a
  de-specialised `op` costs 3 088 bytes at arity one and 6 144 at arity two on a 369-leaf set, where the
  three small shapes the file used to test would have shown 16 to 48 and could have passed while boxing.
  Seven shapes for `_dot`, and `solution_scale` and `αmax` are pinned for the first time.

[nnp19]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/19
[#70]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/70

- **`AdamCache`'s four-argument constructor cost 81 s of a nested container's 88 on Julia 1.11.9, and
  now costs 3.** This was a *Known issue* of this release, which named splitting the constructor as "the
  obvious thing to try" and did not try it. It has been tried. Splitting alone did nothing; splitting
  **and** forbidding inference to cross the boundary is the fix.

  `OptimizerCache(Adam, ps)` on the 16 × 24 nested container in a `NetworkParameters`, cold, one process
  per cell, `scripts/walk_compile_cost.jl --caches`. Both columns measured on *this* branch, so the only
  variable is the constructor:

  | | before | after |
  |---|---|---|
  | 1.11.9 | 88.23 s | **6.87 s** |
  | 1.12.7 | 12.65 s | 17.35 s |
  | 1.13.0-rc3 | 15.14 s | 20.65 s |

  Attributed by `scripts/adam_cache_attribution.jl` on 1.11.9, the constructor's own body goes from
  **81.49 s to 2.91 s**, and everything around it is unmoved.

  **The fix is two things and neither works alone**, which is the part worth recording. The `new` — ten
  fields, four large type parameters — moves into `_adam_cache`, reached with every member already
  computed and passed in, so that frame infers from its own signature rather than through the tree that
  built them. That change *by itself* measured **84.37 s against 81.49**, i.e. nothing, or slightly
  worse. Marking `_adam_cache` `@noinline` on top of it takes it to 2.91 s. So it is not the split and it
  is not the annotation; it is the two together, the annotation being what stops inference simply
  re-crossing the boundary the split created.

  That is worth setting beside open issue **D1**, whose shape this is and whose `@noinline` control
  *failed* (925 s against 940 s). The difference is where the barrier goes: D1 put it around the
  construction, leaving the expensive composition intact on the far side, while this puts it between the
  composition and the `new`. D1's own successful control was flattening the nesting so each frame infers
  from its signature — which is what the split does — so both halves here have a precedent in D1, and
  neither precedent worked on its own.

  **And it costs 5 s on 1.12 and 1.13**, which is the honest half. `@noinline` is a real instruction and
  the versions that inferred the composition perfectly well now pay for a separate specialisation. The
  trade is −81.4 s on the compat floor against +4.7 and +5.5 above it: the worst cell across the three
  supported versions goes from 88.23 s to 20.65, and the spread collapses from 88/13/15 to 7/17/21. A
  version-dependent `@noinline` was not attempted — it would be a `@static if VERSION` in the hot
  constructor of the default optimizer, and this file has enough version cliffs recorded in it to be
  wary of encoding one in dispatch.

  `OptimizerState(Adam)`, `OptimizerCache(BFGS)` and `OptimizerState(BFGS)` are unmoved on every version.
  `MomentumCache` carries the same shape per `gradient_optimizer.jl:30` and is **not** changed here: it
  is not on the sweep and was not measured, so it is the next thing to look at rather than something to
  assume.

### Removed (breaking)

- **Two pirated `Base.copyto!` methods are gone, which is two of the three surviving sites of [#16]
  group 3.** `copyto!(::GlobalSectionNamedTuple, ::ParameterContainer)` and
  `copyto!(::NamedTuple, ::NetworkParameters)` are `_copyto!` methods now, at the same signatures, so
  dispatch resolves exactly as it did.

  These needed no coordination to remove, which is what distinguishes them from the third site. Every
  caller already went through `_copyto!`. The call sites that reach *these two signatures* — not the
  package's twenty-odd `_copyto!` calls in total — are three here
  (`src/manifold_optimizers/gradient_optimizer.jl:151`,
  `src/manifold_optimizers/momentum_optimizer.jl:141`,
  `src/manifold_optimizers/scalar_moment_adam_optimizer.jl:234`) and six in
  `GeometricMachineLearning` (`src/optimizers/optimizer.jl:233-245`) — checked against the working copy
  rather than assumed. The two `Base.copyto!` methods existed only to be forwarded to by the two
  `_copyto!` one-liners that are also gone. `copyto!` is still what is handed to `mapparameters!` as the
  *leaf* operation, and that is not piracy: at the bottom of the walk a pair is two arrays, which is
  `Base`'s own method, or a `GlobalSection` and its anchor, which dispatches on a type of this package's.

  Breaking in the sense that `copyto!(a_section_tree, some_parameters)` spelled with `Base.copyto!` no
  longer resolves. Nothing in this package, `GeometricMachineLearning` or `GMLDatasets` spelled it that
  way, and it was neither exported nor documented.

- **Four `outer!`/`_mul!` methods are gone, and one of them was a type-piracy site of [#16].** With both
  always reached through the flat buffers, nothing in `src/`, `test/`, `docs/` or `scripts/` called
  `outer!` on a [`ParameterContainer`](@ref) or on an `AbstractLieAlgHorMatrix`, or `_mul!` on either —
  and nothing in `GeometricMachineLearning` or `GMLDatasets` ever did. The elementwise
  `_mul!(c::ParameterContainer, a::ParameterContainer, b::ParameterContainer)` goes with them, having
  had no caller before this release either.

  Deleting the container `outer!` **retires one of the five sites #16 group 3 tracks** — `outer!` is
  `SimpleSolvers`' generic and this package owns neither arm of the union. That is possible for this one
  and not for the group, and the reason is worth stating: `outer!` is internal to the quasi-Newton
  caches, so no consumer calls it and no consumer has to change. The other four are on generics a
  consumer does reach.

  The ambient-versus-intrinsic reasoning the lift method documented has moved to `_flat_scratch`, where
  the flat form now lives; `docs/src/linesearch_on_manifolds.md` points there.

- **`update!(::BFGSState, ::Gradient, x, retraction)` is gone.** It had no caller in `src/`, `test/`,
  `docs/` or `scripts/`, and none in `GeometricMachineLearning` or `GMLDatasets` either: for a
  `BFGSState` the live path is `update!(state, opt, x)`, which reaches the six-argument method and
  hands it `problem(opt).F(x)` directly.

  It is worth naming for what it did rather than because it is missed. It set `f̄` with
  `gradient.F(flatten(T, x)[1])`, and `gradient.F` is the closure `_x -> F(unflatten(layout, _x))` — so
  that line flattened `x` and unflattened it again to compute `F(x)`, allocating twice for a value the
  caller already had. 0.5.0's *Known issues* counted it among the sites that wanted a scratch buffer on
  the state; taking `f` as an argument is the fix, and a buffer would only have made the round trip
  cheaper. **`BFGSState` gains no field and no type parameter**, which is a departure from what that
  entry said the work would be.

- **`l2norm` and `solution_scale` take any parameter shape, not just a solution's.** They dispatched on
  `ParameterContainer`, which leaves out the nested plain `NamedTuple`; `_sumsq_leaves` recurses, so
  covering it costs nothing, and leaving it out cost something. `GeometricBase.l2norm` is variadic, so an
  uncovered `NamedTuple` did not raise a `MethodError` naming `l2norm` — it was splatted into
  `L2norm(::NamedTuple, ::Matrix, ::Matrix, …)` and reported against a function the caller never named.

  **`_dot` is widened with them after all**, and the paragraph that stood here saying it was
  "deliberately not" is retracted. The reason given was sound and is simply no longer binding: `_dot`
  returned `T(...)`, so `T` had to bind, and `ParameterContainer{T}` was the alias that bound it.
  Accumulating in `zero(T)` rather than converting after the fact removes the need for the conversion,
  and `NeuralNetworkParameters.parameter_eltype` supplies the element type where no `T` binds — so the
  nested bare `NamedTuple` is covered, and the sweep's `MethodError` row is a figure now. See the fold
  entry below. The pair whose element types *differ* is covered with it, by promotion over both sets,
  where for two `NamedTuple`s or two containers it used to be a `MethodError` — and, for two *lifts*,
  something worse than a `MethodError`. That last case has its own *Fixed* entry above, and it is the
  reason this widening is not only a widening.

  Two methods and not one widened signature, which is measured rather than stylistic:
  `parameter_eltype` is upstream's `@generated` `promote_type` chain, and on a **bare** `NamedTuple` of
  369 entries it costs 6 144 bytes a call even though it infers to `Type{Float32}`. That is the flat
  MNIST shape, and `trial_slope` is the hottest caller in the package. So the method that binds `T` on
  its signature keeps it — free, and it covers every shape that worked before plus the wide flat one —
  and `parameter_eltype` is confined to the method for the shapes that have no such parameter, which are
  the nested ones, whose branches are narrow enough that it folds to nothing.
  `test/flat_buffer_allocations.jl` pins both paths at zero on seven shapes.

- **`scripts/walk_compile_cost.jl` ran, and then swept the axis it was comparing across.** As first
  committed it died on its first table, before printing anything: `map(zero, ·)` on a *nested* set hands
  `zero` a whole layer, which is the very claim the script exists to demonstrate, so every row that
  raises is caught and reported as a row now rather than raising. That is how the `l2norm` gap above
  surfaced, and it is why the exception rows are kept: the elementwise primitives turning a nested bare
  `NamedTuple` away is a statement about `ArrayNamedTuple` being flat by construction, not a hole in the
  harness.

  It now sweeps **shape × wrapping**, four cells and a process each, where it swept flat-bare against
  nested-wrapped and called the difference nesting. `--gradient` is a third mode for the one row that
  cannot share a process with anything, and `--caches` gained the `BFGS` rows, which are the ones that
  flatten. See *Known issues* for what the one-axis sweep cost, which is the reason for all of it.

  The nested set is also built with **distinct keys per block** now — sixteen branches of sixteen types,
  which is what a network has, against sixteen of one type compiled once. That is a change to what the
  figures are *of*, so: it moves the two rows that can be compared by less than a cold measurement's
  spread, `parameterlayout` at 16 × 24 wrapped on 0.2.2 reading 14.28 s against the 13.91 s recorded
  before and `OptimizerCache(Adam)` 85.70 s against 87.56 s.

- **`add!` on a parameter set is deleted rather than narrowed.** It was `::NamedTuple` ×3 and had become
  `::ParameterContainer` ×3, which drops a nested plain `NamedTuple` and any layer whose weights do not
  share one element type — the only signature in this release that got *narrower*. It had no caller in
  `src/`, `test/`, `docs/` or `scripts/`, and `GeometricMachineLearning` deleted its own `NamedTuple` arm
  of `AbstractNeuralNetworks.add!` in 0.7 for the same reason.

- **One quadrature fold, not two.** `l2norm` and `solution_scale` each carried a four-method copy of the
  same recursion, differing only in the leaf function; both call `_sumsq_leaves(f, x)` now, which is one
  method over `NeuralNetworkParameters.foldparameters`. See the *Fixed* entry on the four folds for the
  compile figures and for why the `Base.tail` recursion this began as did not survive the release.

  `foldparameters` and emphatically not `foldstorage`, which is the walk `_dot` wants. `foldstorage`
  descends past a leaf into the numbers it stores, and `solution_scale(Y::Manifold)` is `√size(Y, 2)` —
  a *nominal* value that has nothing to do with those numbers. It is the one leaf method whose answer
  would change; `l2norm` of a lift or of a `VectorStorageMatrix` is over the free parameters either way.

  Its `f` is still annotated `::F where {F}`, and **that annotation is no longer load-bearing** — the
  claim that stood here is corrected. It was load-bearing for the four-method version, where two methods
  only *passed `f` along* and Julia does not specialise on a function argument it never sees called, so
  `f` arrived boxed and each leaf cost a dynamic dispatch: 128 bytes against 64 on the mixed set, which
  #69's `test/flat_buffer_allocations.jl` caught. The single method closes over `f` instead, and a
  closure is a `new`, which counts as a use — measured, the annotation now changes nothing on any shape.
  It is kept because the obligation is real for whoever hands an `op` on: upstream's `foldparameters`
  docstring says so, its own `op` slot does de-specialise, and that costs nothing only because it is
  `@inline` and folds into the caller. Put a `@noinline` between the two and it is 3 088 bytes at arity
  one and 6 144 at arity two on a 369-leaf set.

  (An earlier draft of this entry reported 0.01 s and 0.00 s for the compile cost, which was the warm
  figure the old harness produced, and then 0.57 s against 0.00 s, which had the nested figure from a
  run that had already compiled the flat one. The figure that replaced those — "about 0.65 s, which is
  the fold's own `k³`, at a width where a *fold* still gets away with it" — was measured correctly and
  was still wrong about the thing that mattered, being true only on 1.11. That is what the *Fixed* entry
  below is about.)

- **`l2norm(::AbstractMatrix)` and `l2norm(::AbstractFloat)` are gone, upstream.** They were
  **issue [#16] group 1**, and the plainest piracy in the package: `l2norm` is `GeometricBase.Utils`'
  (`SimpleSolvers` only re-exports it) and both argument types are `Base`'s, so every package that
  loaded this one inherited them. Their own comment said where they belonged.

  `GeometricBase` 0.14.9 takes `L2norm(x::AbstractArray)` where it had `AbstractVector`, which is the
  matrix method with the `vec` removed. The `AbstractFloat` one turned out to be **redundant as well as
  pirated**: `L2norm(x::Real) = x^2` and `l2norm(x) = √L2norm(x)` were already `abs` for any `Real`, so
  nothing ever needed it.

  Removing the `vec` is not only ownership. `vec` of a `Matrix` allocates a 32-byte reshape wrapper, so
  `l2norm` of a parameter set cost 32 bytes per matrix leaf on every stopping criterion of every
  iteration of `solve!`; it is zero now, for a lift, a flat `NamedTuple`, a container and a plain set of
  vectors alike, and `test/flat_buffer_allocations.jl` says so as an exact equality.

  This is the whole of group 1. Groups 2 and 3 are unmoved — see *Known issues*.

- **One shape normaliser, not one and a half.** `_as_blocks` (`linesearch_problem.jl`) and
  `_as_walkable` (`parameter_walks.jl`) were the same function under two names in two files; the first
  went earlier in this release and the second went with the file. Neither is replaced, because
  `Base.values(::NetworkParameters)` is defined upstream and one `values(x)` serves both members of the
  union.

- `apply_section!(Y, λY::NamedTuple, Y₂)` widened two of three slots to `Any`, where it accepts a
  `ParameterSet` and fails inside the walk for anything else. Named now.

### Known issues

- **A solve is not allocation-free, and this release does not make it one.** The flat vectors are gone;
  what remains is the gradient evaluation and the parameter trees `global_rep` and
  `retraction_differential` build per line-search evaluation in `trial_slope`. A 20-iteration `solve!` on
  the flat `NamedTuple` problem still allocates 1.56 MB, which is essentially all of it: this release
  removed 6 712 bytes an iteration. See the three-column table under *Fixed*, and note that most of what
  its two-column predecessor claimed was upstream's rather than ours.

  The third item that stood here — "the `map` inside the elementwise walks returning a tree of results
  that is then discarded", 992 bytes per `update!` — is **fixed** in this release: `_mapleaves!` is
  `NeuralNetworkParameters.foreachparameters`. See the review findings above.

- **The walk is upstream's now, and the file that held the local copy is gone.** The entry that stood
  here in an earlier draft of this release — "`src/parameter_walks.jl` should be
  retired in favour of `NeuralNetworkParameters.mapparameters`" — is **done**, and the file is deleted.
  See the walks entry under *Added* above, and open issue **D9**, which is closed with it.

- **The cost this entry used to describe was one unused type parameter upstream, and the diagnosis
  here was wrong in every part.** What stood here said that building a layout for a *nested* container
  cost about 88 s, that the cause was `NeuralNetworkParameters.parameterlayout`, that it was "the price
  of the property that package exists to provide", and — in a list headed *recorded so they are not
  tried again* — that erasing `LeafLayout`'s prototype type parameter "would not help here". Erasing it
  is what fixed it, in `NeuralNetworkParameters` 0.2.3, and the compat bound moves to `"0.2.3"` for it.

  `LeafLayout` carried a `prototype` field that nothing read — a `LeafLayout` is by construction the
  case where `freeparameters(x) === x`, so `rebuild` on it is the identity — and its type parameter put
  each leaf's *concrete array type* into the layout type of every branch above it. Upstream counts 1849
  nodes in the type tree of a 369-leaf wrapped layout against 742 without it.
  `_layout(::NetworkParameters, ::Int)` is where a caller paid, inferring through the child walk's whole
  return type to wrap its result, and inference on such a type grows faster than the type does.

  **Only the wrapped shape has such a caller, which is why this entry's table could not see it.** That
  table compared a bare flat `NamedTuple` against a nested `NetworkParameters` and read the difference
  as a cost of *nesting*. Both axes moved at once. Swept properly — `scripts/walk_compile_cost.jl` runs
  shape × wrapping now, four cells, one process each — the wrapper is the whole of it and nesting is not
  a cost at all. Julia 1.11.9, seconds, `0.2.2 → 0.2.3`:

  | | `parameterlayout` bare | `parameterlayout` wrapped | `flatten` bare | `flatten` wrapped | **total, wrapped** |
  |---|---|---|---|---|---|
  | flat, 369 entries | 1.34 → 1.03 | **14.08 → 1.04** | 15.39 → 3.98 | 4.83 → 3.78 | 18.91 → **4.82** |
  | 16 × 24, 384 leaves | 1.24 → 0.88 | **14.28 → 0.87** | 14.05 → 1.86 | 1.94 → 1.59 | 16.22 → **2.46** |

  Read the bare `parameterlayout` column on 0.2.2: **1.24 s nested against 1.34 s flat**, at the same
  leaf count, where wrapping either of them costs a factor of eleven. Nesting was never the variable.
  On 0.2.3 all four cells agree, and the nested set comes out slightly *cheaper* than the flat one.

  The `flatten` columns are the other half, and they are why the old entry could be written at all: on
  0.2.2 the two wrappings split nearly the same total between different entry points — 1.34 + 15.39 =
  16.73 s bare against 18.91 s wrapped at flat 369 — so whichever one you measured, the other had
  absorbed the difference, and a one-axis sweep could produce a coherent wrong answer. Taking the type
  parameter out reduces the *total*, by 3.9× at flat 369 and 6.6× at 16 × 24.

  Of the three explanations that entry listed as tested-and-wrong, the first two stand and were
  answering the wrong question anyway, having compared a nested-wrapped type against a flat-bare one.
  The third was the answer. Its evidence is still sound — alternating `Matrix` and `Vector` leaves cost
  11.32 s against 13.9 s homogeneous, so leaf **diversity** is not the driver — but it never compared
  the layout type with and without the parameter, which is where the elevenfold difference was. A
  negative result about one thing was written down as a negative result about a neighbouring one, in the
  one kind of note whose whole purpose is to stop the work being redone.

  **The lesson is not about the clock, and that is what makes it new.** D16 and D19 upstream, and this
  file's own harness fix in 0.6.0, were all about measuring cold and measuring the right process. This
  one was measured cold and in the right process and was still wrong, because the confound was in *which
  two things the table's two columns were*. `scripts/walk_compile_cost.jl` sweeps both axes now, and its
  header records the inversion: agreeing columns are the pass condition, and a gap reappearing between
  bare and wrapped is what the sweep exists to catch.

  **The one piece of advice that entry gave survives with its reason replaced.** "Use `flatlength`
  wherever only the size is wanted" was justified there by the clock, 1.19 s against 13.22 s. Upstream's
  own `flatlength` docstring now measures 1.03 s against 1.05 s. The reason to prefer it is the `Int` it
  returns, so that nothing downstream of the call has to infer through a layout type — which is a claim
  about the caller's code, not about any release of upstream. `alloc_h` (`bfgs_state.jl:45`) does;
  `_flat_scratch` needs the layout itself and cannot, and now says so.

  Tracked upstream as [NeuralNetworkParameters#15][nnp15], fixed in [#18][nnp18].
  [NeuralNetworkParameters#16][nnp16], which this entry used to point at, is closed as **not a defect**:
  `parameterlayout` never was superlinear in the leaf count of a nested set.

[nnp15]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/15
[nnp18]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/pull/18
[nnp16]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl/issues/16

- **Taking the container did not close issue [#16], and the *Known issues* of 0.5.0 said it would.**
  That entry read "what remains is the swap itself, i.e. the `ArrayNamedTuple` half of **#16**". The
  swap is here and #16's group 3 is still open, because closing it needs the `NamedTuple` methods
  *removed* rather than joined by container ones — and they cannot be. `GeometricMachineLearning` hands
  this package one bare layer `NamedTuple` per layer (`src/optimizers/optimizer.jl:65-87`, through
  `_use_go_cache`'s `x isa OptimizerSolution` test) and dispatches `_GMLGradient`'s functor on
  `GeometricOptimizers.ArrayNamedTuple` (`:16`); GMLDatasets' three MNIST scripts pass a flat
  `NamedTuple` directly. Removing the methods is a coordinated breaking change across three
  repositories, and it is not this one.

  A method on `ParameterContainer` is therefore piracy for *both* members: `ArrayNamedTuple` is an
  alias for `Base.NamedTuple`, which is the original complaint, and `NetworkParameters` belongs to
  `NeuralNetworkParameters`, so a method pairing it with `l2norm`, `outer!` or `copyto!` owns neither
  side either. The alias' docstring says so.

  **The audit itself was stale, and re-running it leaves one site rather than four.** #16 lists group 3
  as four methods; measured against this release:

  | site as listed | now |
  |---|---|
  | `outer!(::AbstractMatrix, ::ArrayNamedTuple, ::ArrayNamedTuple)` | **gone** — deleted in this release; see *Removed (breaking)*, and only calls remain |
  | `(::Gradient)(::ArrayNamedTuple)` | **not piracy** — it dispatches on `Gradient`, which is this package's type. This is #16's own group 4 rule, "one owned argument type is enough", applied to an entry group 3 got wrong |
  | `Base.copyto!(::GlobalSectionNamedTuple, ::ArrayNamedTuple)` | **gone** — `_copyto!` now, with the second of the pair; see *Removed (breaking)* |
  | `l2norm(::ArrayNamedTuple)`, now `l2norm(::ParameterSet)` | **real, and the one that is blocked** |

  So what is left of group 3 is `l2norm`, and `solution_scale` is *not* beside it — that one is this
  package's own function, not `GeometricBase`'s, so widening it cost nothing in ownership. Nor is `_dot`,
  which this release widened to `ParameterSet`: it is local.

  `l2norm` is the blocked one for the reason above, and the block is specific rather than general:
  `GeometricMachineLearning` dispatches `_GMLGradient`'s functor on `GeometricOptimizers.ArrayNamedTuple`
  (`src/optimizers/optimizer.jl:16`), so the alias cannot be retired from under it, and removing the
  `NamedTuple` method needs the three repositories moved together.

  **Group 2 is closed, and by dissolution rather than by a fix.** It was six
  `ParameterHandling.flatten` methods dispatching purely on `Base` types — "the widest of the four", and
  it was. `ParameterHandling` is not a dependency of this package any more, is not in the resolved
  manifest, and `src/` defines no `flatten` method at all: `flatten` is `NeuralNetworkParameters`',
  imported and only called. The migration to that package took the group with it as a side effect, which
  is worth writing down precisely because nobody set out to do it.

[#16]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/16

## [0.5.0]

**`ParameterHandling` is gone, and the package that owns a network's parameters walks them instead.**
0.4.x already supplied the `freeparameters`/`rebuild` protocol for these types through an extension;
this release makes [NeuralNetworkParameters.jl][nnp] a hard dependency and lets it do the flattening,
which retires the whole `ParameterHandling` shim -- including five methods on Base types that were type
piracy and said so, and a one-argument method that silently promoted a `Float32` network to `Float64`.

**It also takes `changebackend` for these types**, which was living in
`GeometricMachineLearning`'s HDF5 extension, a package that owns neither the function nor the types.

The `NNP-D` labels below are `NeuralNetworkParameters`' `PLAN.md` numbering of the parameter-container
consolidation this release finishes, and are written with that prefix because the *Open Issues*
catalogue at the foot of this file numbers its own **D1**--**D8** independently: bare `D5` and `D8`
here would name a Documenter gap and a `SimpleSolvers` documentation link, which is not what is meant.
`A21` and `#16` are this file's own.

### Removed (breaking) — dependencies

- **`ParameterHandling` is no longer a dependency.** The shim it existed for is deleted in full:

  - the one-argument `flatten(::ArrayNamedTuple{T})` that forwarded to `T` because
    `ParameterHandling.flatten` otherwise defaults to `Float64`. `NeuralNetworkParameters.flatten`
    takes the element type from the parameters, so there is nothing to work around. **NNP-D6**.
  - the five methods on `NamedTuple`, `Tuple`, `Tuple{}`, `Vector` and
    `AbstractMatrix`/`AbstractArray{,3}`. These were piracy -- the generic is `ParameterHandling`'s and
    the types are Base's -- and the comment above them said so, citing issue **#16**. They also shadowed
    methods `ParameterHandling` already defines, because `T<:AbstractFloat` is narrower than its
    `T<:Real`. The `Vector` one was concrete in `Vector`, which is the hole a `CuVector` fell through
    (**A21**).
  - the four methods on this package's own types, which `src/parameter_protocol.jl` supersedes.

  `manifold_constructor` loses its flattening caller with them, and with it the bug class its
  docstring describes: reconstructing a manifold from a type name is how a `GrassmannManifold` used to
  come back a `StiefelManifold` on every round trip. `NeuralNetworkParameters.rebuild` takes a
  *prototype*, so that is now structurally impossible rather than guarded against. **NNP-D5** is
  closed for the flat path.

  Three test files reached `ParameterHandling` through this package and now use
  `NeuralNetworkParameters` directly. Their assertions are unchanged -- only
  `flatten(ps)` becomes `flatten`/`unflatten(layout, v)`, because a `ParameterLayout` is a value rather
  than a closure.

- **`ForwardDiff` is no longer a dependency either**, and it goes for the same reason rather than as a
  separate cleanup: its only use in `src/` was the `unflatten_to_Vector(v::Vector{<:ForwardDiff.Dual})`
  clause of the pirated `flatten(::Type{T}, ::Vector{R})` above, which existed so that a round trip
  through the flat form would not convert `Dual`s back to the leaf's element type.
  `NeuralNetworkParameters.unflatten` keeps `eltype(v)` by construction, so the clause has nothing to
  do. Nothing else in `src/` or `test/` named it; `docs/Project.toml` has its own entry, which is
  unaffected. `ForwardDiff` still arrives transitively through `SimpleSolvers`, which is what actually
  differentiates a `GradientAutodiff`, so no caller loses anything.

- **`NeuralNetworkParameters` is a hard dependency, and `NeuralNetworkParametersExt` is gone.** The
  extension's contents moved to `src/parameter_protocol.jl` unchanged and its `__init__` became the
  module's. Nothing about the arrangement changed; the protocol is simply always present, which is what
  the flattening needs. A caller who was checking for the extension with `Base.get_extension` has to
  stop.

### Added

- **`NetworkParameters{T}` is a member of `OptimizerSolution{T}`.** `NeuralNetworkParameters` 0.2
  carries the element type of the leaves as the container's first type parameter, so a parameter set
  binds `T` in the `where {T, VT<:OptimizerSolution{T}}` clauses the caches, states and `Optimizer`
  constructors are written with, exactly as an `AbstractVector` or a `Manifold` does. It is the
  cheapest member of the union to intersect: `T` comes from a direct type parameter rather than through
  a `Vararg` bound on value types.

  One caveat that the other members do not have. `NetworkParameters{T}`'s `T` is a *promotion* over the
  leaves and not a guarantee that every leaf is a `T`, where `ArrayNamedTuple{T}` guarantees exactly
  that. A mixed-precision set is a `NetworkParameters{Float64}` with `Float32` leaves still in it.

  Like every alias in `src/optimizer_solution.jl`, this one stays out of `struct` type-parameter
  bounds; the note at the head of that file says why.

- **`changebackend` for `Manifold`, `VectorStorageMatrix` and `AbstractLieAlgHorMatrix`**, in a new
  `AbstractNeuralNetworks` weak-dependency extension. **NNP-D3** and the second half of **NNP-D8**.

  The generic is `AbstractNeuralNetworks`' and the types are this package's, so a method on the pair has
  one owned argument on each side and belongs in one of those two packages. It was in a third:
  `GeometricMachineLearning`'s HDF5 extension, whose own comment says so and asks for this move. Two
  consequences beyond the ownership -- `changebackend(GPU(), nn)` on a network with a manifold weight
  was a `MethodError` unless HDF5 happened to be loaded, because the methods sat inside an HDF5
  extension and had nothing to do with HDF5; and the horizontal lifts never had a method at all.

  One method covers every family where `GeometricMachineLearning` had five and was missing three:
  `mapstorage` hands the function the storage of a leaf and rebuilds the leaf around it, so a type added
  to `src/parameter_protocol.jl` later is covered without a change here. `test/changebackend.jl` pins
  the type, the metadata that is not in the storage, the element type, that a lift's structured block
  stays structured rather than being densified, and that a transfer copies rather than aliases.

- **`GlobalSection(::NetworkParameters)`.** **NNP-D11**. `GlobalSection` is this package's function and the
  container is `NeuralNetworkParameters`', so this method is this package's to write --
  `GeometricMachineLearning` carried it until now, owning neither name. The result is the same plain
  tree the `NamedTuple` method returns rather than a `NetworkParameters` of sections: a section is not a
  parameter, and everything downstream walks it as an ordinary container.

### Changed

- **`apply_toNT` is deleted. It was `Base.map`.** All 30 call sites call `map` instead.

  ```julia
  apply_toNT(f, a, b)   →   map(f, a, b)
  ```

  `map` over `NamedTuple`s takes any number of arguments and already throws
  `ArgumentError: Named tuple names do not match.` on mismatched *or reordered* keys, which is what the
  hand-rolled `@assert keys(ps[1]) == keys(p)` was approximating -- except that Base's check is
  unconditional where an `@assert` can be compiled out. Heterogeneous values map fine, which is the case
  `ArrayTuple` exists to permit. Verified on 1.10, the compat floor, as well as 1.13. It was unexported,
  and `GeometricMachineLearning` reached it by qualified call; that call is gone in its 0.7.0.

- **The two `Gradient` constructors close over a `ParameterLayout`** instead of the chain of nested
  closures `ParameterHandling.flatten` returned, one per level of the tree, which was not type stable.

- **`_mul!` flattens once instead of twice, and writes back in place.** The destination supplies its
  layout, not its numbers, so only one argument needs flattening; and `unflatten!` writes through
  `copyto!` rather than building a fresh parameter set for `_copyto!` to copy out of. Both the
  `NamedTuple` and the horizontal-lift method.

- **The `Gradient` functor uses `mapparameters`, not `mapstorage`.** `rgrad` is the Riemannian
  projection and needs the point it projects at, not the point's storage. Same result, one walk instead
  of a `for` loop over `keys`.

### Fixed

- **Three sites allocated a zeroed copy of the parameters and a vector the size of it in order to call
  `length`.** `BFGSCache`, `DFPCache` and `alloc_h` sized `Q` with
  `ParameterHandling.flatten(_zero(x))` and used only `length(v)`. They use `flatlength` now, which
  counts without building the flat vector.

  The `_zero(x)` stays, and a comment says why, because dropping it is a real bug that
  `test/optimizer_convergence/svd_optim.jl` catches: `zero` of a manifold element is its *horizontal
  lift*, whose free-parameter count is the intrinsic dimension and not the size of the dense storage --
  12 against 18 for a `StiefelManifold(6, 3)` -- and `Q` multiplies gradients, which are lifts.

### Known issues

- **The container swap is not in this release, but it is no longer blocked.** The blocker was upstream
  and is gone: `NeuralNetworkParameters` 0.2 carries the element type as `NetworkParameters{T}`'s first
  type parameter, so `NetworkParameters{T}` is a member of `OptimizerSolution{T}` and the eleven sites
  that derive `T` from the *type* of the solution -- every cache and state inner constructor, both
  `Optimizer` constructors and the `BFGSState` `update!` methods -- bind it for a parameter set with no
  edit of their own. This release makes it a member.

  What remains is the swap itself, i.e. the `ArrayNamedTuple` half of **#16**. Every elementwise
  primitive in `src/optimizers/named_tuple_wrapper.jl` -- `_zero`, `_copy`, `_similar`, `_copyto!`,
  `_mul!`, `_rac!`, `_div!`, `_square!`, `rgrad` -- dispatches on `ArrayNamedTuple` and none accepts a
  container. So widening the union widens signatures without making an optimizer run on one: a
  container handed to `Optimizer` now gets further in before failing rather than being turned away at
  the door. Those methods, and `GlobalSection`'s deliberate unwrapping at
  `src/global_sections/global_sections.jl:42`, are the work.

- **The flat buffers are still allocated per call.** The three sites above were the ones that
  allocated a flat vector in order to *not* use it; the ones that use it still build a fresh one every
  time. `_dot` flattens both arguments per inner product -- once per line-search trial slope, the
  hottest of them -- `outer!` both per update, `_mul!` one plus a `ParameterLayout`, and the `Δg2` of
  each quasi-Newton `update!` one more. `NeuralNetworkParameters` has the allocation-free counterparts
  (`flatten!`, `unflatten!`) and a `FlatParameters` that carries its own layout and preserves it
  through `similar`, so what is missing is only the scratch to write into.

  Left out because of where that scratch has to live rather than because of the rewrite. It belongs on
  the cache and the state, not on the `Gradient`: one `Gradient` is shared between `solver_step!` and
  the line search's closure, which is why the cache already keeps `gradient` and `latest_gradient`
  apart -- `src/optimizers/linesearch_problem.jl:211-224` records the momentum corruption that sharing
  caused. So it is a new field and a new type parameter on `BFGSCache`, `DFPCache` and `BFGSState`,
  which `test/named_tuple_parameters.jl` requires to be unbounded, plus the `@allocated == 0` tests
  that are the only thing that would make "preallocated" more than a claim. That is its own release.

- **`l2norm(::AbstractMatrix)` and `l2norm(::AbstractFloat)` are still piracy on Base types**
  (`src/optimizers/optimizer_status.jl`), and no parameter container fixes them; the comment there
  already says they belong upstream in `GeometricBase`.

- **The elementwise primitives still hand-write their `parent` unwrapping.** `_add!`, `_rac!`,
  `_square!`, `_div!` and `_rmul!` each carry a `VectorStorageMatrix` method that is a pure `parent`
  call and, for some, a lift method that is a `foreach` over `parent` -- which is exactly
  `mapstorage!`. Collapsing them was left out of this release on purpose: several of these families
  have a *deliberate* manifold exception that reaching for the storage would silently bypass
  (`_fill!(::Manifold, ::T)` returns the point untouched; `_similar(::Manifold)` draws a fresh random
  point because `similar` is an error on purpose), and `_difference!` has no manifold method at all, so
  a manifold leaf currently errors where storage would succeed. Each conversion needs deciding on its
  own evidence rather than in a sweep.

[nnp]: https://github.com/JuliaGNI/NeuralNetworkParameters.jl

## [0.4.2]

**The geodesic gets a second portable exponential, and the portability the first one was chosen for
becomes true rather than merely stated.** `ScaledSquaring` is the default because it runs on a backend
that forbids scalar indexing; it did not, and the reason was one line. `NativePade` is the
independent implementation that had been missing there.

### Added

- **`NativePade`, a fifth `AbstractExponentialAlgorithm` and the second portable one.** It evaluates
  ``\mathfrak{A} = \varphi_1`` directly on the existing ``2n\times{}2n`` factor with the degree-6
  diagonal Padé approximant ``q_6^{-1}p_6``, whose coefficients are the ``[7/6]`` Padé approximant of
  ``\exp`` rearranged — ``\varphi_1 \approx (N-D)/(xD)``, and ``N-D`` divides by ``x`` because both
  have constant term one. The dense solve a rational approximant normally needs is replaced by five
  fixed Newton--Schulz steps, and the scale is undone by the same low-rank squaring recursion
  `ScaledSquaring` uses, so nothing larger than the input is ever formed and the whole path is
  reductions and matrix products.

  It exists as the *independent* implementation, which is what the entry below is about: until now
  the only cross-check on `ScaledSquaring` was `AugmentedPade`, and that one calls the dense-LAPACK
  `Base.exp` on a matrix twice the dimension. `test/retractions/exponential_accuracy.jl` now runs
  both portable algorithms on a `JLArray` — a backend that *errors* on scalar indexing — and against
  `AugmentedPade` over both manifolds, both element types and the whole norm sweep.

  `ScaledSquaring` stays the default, and the measurements say why rather than the other way round.
  On the isolated ``\mathfrak{A}`` call at ``N = 200``, ``n = 10``: `0.021 ms` and `201 KiB` against
  `0.037 ms` and `330 KiB`, with `AugmentedPade` at `0.053 ms` and `114 KiB`. `Float64` accuracy is
  indistinguishable — `check` `3.6e-14` against `3.9e-14` at ``\|\bar{B}\| = 361``, forward error
  `2.1e-14` for both — but `Float32` is not: at the top of the norm sweep `NativePade`'s `check` is
  `1.0e-4` where `ScaledSquaring`'s is `4.0e-5` and `AugmentedPade`'s `3.4e-5`, the worst of the
  three, while its *forward* error there is no outlier at all. So what `Float32` costs it is the
  orthogonality of the retracted point, not the agreement with `exp`. It also inherits the whole of
  open issue **A6**: `s` comes from ``\|X\|_1`` the same way, so it too takes about twice the
  squarings it needs.

  The allocation figures are new to this repository and do not rank the algorithms the way runtime
  does: `AugmentedPade`, which builds a ``4n\times{}4n`` matrix and throws three quarters of the
  result away, is the *lightest* allocator of the three, because `Base.exp` reuses buffers where both
  native algorithms produce a fresh ``2n\times{}2n`` temporary per operation. On a CPU that is a
  detail; on a backend where an allocation costs a synchronisation it may not be, which is worth
  knowing about the algorithm whose whole purpose is portability. `scripts/retraction_accuracy.jl`
  reports allocations for the isolated call and for one whole retraction, and
  `docs/src/retractions.md` carries both tables.

  The coefficients' provenance is written down rather than left to be re-derived: ``q_6`` is the
  denominator of the ``[7/6]`` Padé approximant of ``\exp``, and ``p_6`` is ``(N - D)/x`` for its
  numerator, which divides exactly and inherits the ``O(x^{13})`` order. Both that and the
  Newton--Schulz iteration are cited to Higham. What is *not* settled is a backward-error criterion —
  see **C16** below.

  Its threshold is bounded where `ScaledSquaring`'s is not, which is the one place the two are not
  interchangeable. `ScaledSquaring(θ)` sums its series until the terms vanish and is accurate over the
  32-fold range its own docstring sweeps; `NativePade` does a *fixed* five Newton--Schulz steps, whose
  residual ``(\mathbb{I}-q_6)^{32}`` stops being small once ``\theta`` grows. Worst relative error
  over 400 random ``6\times6`` arguments of one-norm exactly ``\theta`` is `6e-16` at
  ``\theta = 1``, `1.2e-10` at ``\theta = 3/2``, `1.1e-5` at ``\theta = 2`` and `169` at
  ``\theta = 3``, and nothing about the result says so. `NativePade(θ)` therefore rejects
  ``\theta > 1/2``, which is also the default.

- **`ScalarMomentAdam`, a Stiefel-only baseline to compare [`Adam`](@ref) against.** It is *Cayley
  ADAM*, Algorithm 2 of Li, Li and Todorovic, [*Efficient Riemannian Optimization on the Stiefel
  Manifold via the Cayley Transform*](https://arxiv.org/abs/2002.01113) (ICLR 2020), the
  `li2020efficient` entry added to the bibliography by this PR's first round. It is reproduced with
  this package's infrastructure, and departs from the source in three places, all deliberate and all
  recorded in the method's docstring: any of the package's retractions may be used in place of the
  source's two-step approximation of the Cayley transform (and with it the source's step cap goes),
  which ``\lVert\cdot\rVert^2`` the second moment squares is a keyword, and the momentum is
  transported by the global section rather than re-projected. Its second moment is a
  **scalar**,
  ``v \gets \beta_2v + (1-\beta_2)\lVert\mathcal{G}(X)\rVert^2`` — one adaptive learning rate for the
  whole matrix instead of one per coordinate. That is the source's entire departure from ordinary Adam
  and it is what the name says: reducing the second moments to a scalar drops their structure, where
  this package's `Adam` keeps it by accumulating ``\bar{G}\odot\bar{G}`` in
  ``\mathfrak{g}^\mathrm{hor}``. Everything else is shared with `Adam` — the gradient, its global
  tangent space representation, the bias-correction convention, the section/retraction path and the
  line search — which is what makes the two comparable on the same problem. One thing about that
  comparison is not shared and is worth knowing before running it: `Adam` normalizes componentwise, so
  its direction has norm ``\approx\sqrt{\dim}``, while this one normalizes the whole lift at once and
  has norm ``\approx1``. Both take the same `Static(DEFAULT_LEARNING_RATE)`, so the same ``\eta`` is a
  step ``\sqrt{\dim}`` shorter here.

  The port needs less machinery than it looks like it should, because the source's lines 8-10 — its
  auxiliary matrix ``\hat{W} = ZY^T - \frac{1}{2}Y(Y^TZY^T)``, the skew-symmetrization
  ``W = \hat{W} - \hat{W}^T`` and the projection ``\pi(Z) = WY`` — are this package's
  ``\Omega(Y, \cdot)``, and `global_rep` is that map conjugated into
  ``\mathfrak{g}^\mathrm{hor}``. The source's ``W_k`` *is* the horizontal lift the first-order caches
  already receive their gradient in. The *Optimizer Methods* manual page derives that block by
  block; the method's docstring maps every symbol of Algorithm 2 onto the code and
  records the three places where the reproduction deliberately departs from the source: the exact
  retraction and with it the source's step cap, which norm is squared, and how the momentum is
  transported.

  The second of those three is a keyword. `ambient_norm = false`, the default, squares the horizontal
  lift, which is free and keeps the gradient identical to `Adam`'s; `ambient_norm = true` squares the
  source's ambient Euclidean ``\lVert\nabla{}L\rVert_F`` and costs one extra gradient evaluation per
  step. The two are not interchangeable up to a constant — `rgrad` annihilates the normal component,
  so for ``\nabla{}L = YS`` with ``S`` symmetric the lift norm is zero where the ambient one is not.

  `ScalarMomentAdamState` is public; the cache is not. Ordinary arrays, `NamedTuple`s, Grassmann
  solutions, mixed parameter trees and a `StiefelManifold` whose element type does not match the
  method's are rejected with an `ArgumentError`, on the `Optimizer` path as well as the
  `OptimizerState` one. ([#51])

### Fixed

- **`ScaledSquaring` did not in fact run on a backend that forbids scalar indexing, which is the
  property it is the default *for*.** `𝔄(A)` — the series it sums, and the whole of `TaylorSeries` —
  opened with `Aⁿ = one(A)` and `𝔄A = one(A)`, and `Base.one(::AbstractMatrix)` is `Base._one`:
  `similar`, `fill!`, then a scalar-indexed loop over the diagonal. So the documented claim that the
  default algorithm uses "nothing but matrix products and norms" was false one level down, and
  `GeometricOptimizers.opnorm₁` — which exists solely to keep that claim — was being undone by the
  next line.

  This is the first half of open issue **A19**, and the fix is the one that entry asked for: the
  identity is built with `KernelAbstractions.zeros` plus the `write_ones_kernel!` that
  `src/utils.jl` already had, now shared as `unit_matrix` by all five places that need one — the
  `Base.one` methods of `SymmetricMatrix`, `SkewSymMatrix`, the two `AbstractTriangular`s and
  `AbstractLieAlgHorMatrix`, which each carried their own copy of it, and `𝔄`, which had none.

  A19 is **not** closed. Its second half is that no run in this repository exercises `geodesic` on a
  *GPU*, and a `JLArray` is not a GPU — it is an array type that errors on scalar indexing, which is
  the specific failure mode CUDA.jl and Metal.jl exhibit and the specific thing that was wrong here.
  The entry is narrowed to what is left, and the transcript of an actual CUDA or Metal run is still
  what would close it.

### Documentation

- **`docs/src/retractions.md` gains an allocation account and a `Float32` forward-error table**, and
  both cover all five algorithms rather than only the new one. The allocation figures had never been
  reported anywhere in this repository, and the `Float32` half of the accuracy picture had `check`
  without the forward error beside it — which turns out to matter, because the two rank the algorithms
  differently: `ProjectedSkew` is the best of the four on `Float32` `check` by an order of magnitude
  and the *worst* on `Float32` forward error at every norm — `1.3×` to `2.9×` over most of the sweep
  and `14×` at the smallest lift — which is the trade it makes stated in both directions for the first
  time. `scripts/retraction_accuracy.jl` prints both, so every figure in that section still comes from
  one run of it. [#54]

- **`docs/make.jl`'s `size_threshold` goes from 400 KiB to 450 KiB**, because `ScalarMomentAdam`'s
  docstrings take `api.md` — the one catch-all `@autodocs` over the whole package — to 411.84 KiB,
  and `size_threshold` is an error rather than a warning. This is the smaller half of the fix and
  not the fix: the page grows with every documented addition, and what bounds it is continuing the
  migration [#59] started, which moved the manifold, matrix, global-section and retraction
  docstrings onto the chapters that explain them. `size_threshold_warn` stays at its default so the
  build keeps saying so. [#51]

[#54] and [#51] are the whole release: the second portable exponential with the scalar-indexing fix
that made the first one's portability true, and the Stiefel-only Adam baseline. A patch and not a
minor bump — nothing existing changed name or behaviour. `NativePade`, `ScalarMomentAdam` and
`ScalarMomentAdamState` are new public names, `ScaledSquaring` stays the default algorithm and
`Adam` is untouched, so a package that calls neither of the new types sees exactly the library
[0.4.1](#041) shipped — with `𝔄` now doing on a backend that forbids scalar indexing what it had
always claimed to.

Two entries in the Open Issues catalogue move with it. **A19** loses its first half — `𝔄` no longer
builds its identity through `Base.one` — and is narrowed to what is left, which is that no run here
exercises `geodesic` on an actual GPU. **C16** is new, from #54's review: `NativePade`'s
``\theta \le 1/2`` is measured rather than derived from a backward-error criterion, which is the
fourth of the four things [#52] asked for and the reason that issue stays open.

## [0.4.1]

**The structured matrices get a `NeuralNetworkParameters` protocol.** Nothing existing changes
behaviour; what is new is that a package serialising or differentiating these types no longer has to
know them one by one.

### Added

- **A `NeuralNetworkParameters` extension**, giving the manifolds, the `VectorStorageMatrix`es and the
  horizontal lifts that package's leaf protocol. `NeuralNetworkParameters` walks a parameter set over
  two methods per leaf type — `freeparameters`, saying where the differentiable numbers live, and
  `rebuild`, putting a leaf back together around them — so everything written against it (flattening,
  the elementwise optimizer primitives, the HDF5 traversal) now works for these types without a
  hand-written case per type in whichever package happens to need one.

  `freeparameters` is a single delegating method, because the relation already exists here as
  `Base.parent`: `A.S` for a storage matrix, `A.A` for a manifold element, the tuple of blocks
  `(A, B)` / `(B,)` for a lift. What has to be spelled out is `rebuild` per family — which is what
  lets the storage come back with a different element type, as it does when a flattened set is
  differentiated — and the `parameter_metadata` a file needs when it has no prototype to rebuild
  against: the `n` of a storage matrix, the `N` and `n` of a lift. A manifold element needs none, its
  storage being the dense matrix. `__init__` registers each of the eight types, so
  `load(NetworkParameters, h5)` rebuilds them with no prototype supplied.

  The methods belong here and not in a package that *uses* both. `freeparameters(::SymmetricMatrix)`
  written anywhere else is piracy twice over — on `NeuralNetworkParameters`' generic and on this
  package's type — and two such packages can silently disagree.
  [`GeometricMachineLearning`](https://github.com/JuliaGNI/GeometricMachineLearning.jl) carried
  exactly that: five `h5save` methods tagging a `gml_type` attribute and a recursive reader to match,
  which [GeometricMachineLearning#246](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/246)
  deletes. Files already written in that layout keep loading — the reconstructors registered here
  normalise it, which is something only this package can do, since this is where the types are.

  It matters beyond tidiness: `SymmetricMatrix <: AbstractMatrix`, so without these methods
  `NeuralNetworkParameters`' `freeparameters(x::AbstractArray) = x` fallback treats one as terminal,
  flattening it as ``n^2`` numbers rather than ``n(n+1)/2`` and writing the dense form to file. For
  the skew-symmetric and triangular types there is no `setindex!` to broadcast through at all.
  `GeometricMachineLearning`'s more specific `h5save` methods currently win on the write path, so
  nothing is silently broken today, but that specificity is the only thing standing in the way.

- **`test/neural_network_parameters_protocol.jl`** — one leaf of each family plus a plain array,
  checked for `freeparameters === parent` and non-terminality, `rebuild` inversion, `rebuild` across an
  element-type change for a storage matrix, a manifold element and a lift, a flat length equal to the
  sum of the *storage* lengths, `parameter_metadata`, HDF5 round trips both with and without a
  prototype, a `Float32` set that must not widen, and a file hand-written in
  `GeometricMachineLearning`'s old layout, so dropping its duplicated reader does not quietly make its
  existing files unreadable.

  The `(D, n)` constructors slice with `@views`, so a lift built that way holds `SubArray`s and can
  never compare type-equal to one rebuilt from a flat vector. The tests build the blocks outright, so
  the type assertions say something about the protocol rather than about slicing.

All of the above is [#60]. Its review added the erroring `rebuild` on the three abstract families:
`freeparameters` is defined on those, `rebuild` only on the concrete types, and since all of them are
`AbstractMatrix`es, `NeuralNetworkParameters`' `rebuild(::AbstractArray, data) = data` would otherwise
catch a subtype added later and hand back a densified parameter with no error anywhere.

### Documentation

- **The docstrings move to a page of their own.** `index.md` ended in a catch-all `@autodocs` over
  everything the topic pages did not document by hand, which made it by far the largest page in the
  build and left the introduction sitting on top of the whole library. `api.md` now holds that block
  and the `@index` beside it, and `index.md` is the landing page it reads as. The chapters link into
  `api.md` rather than repeat a docstring, since Documenter renders each one only once.

  `index.md`'s closing sentence said that `GeometricMachineLearning` "re-exports most of what
  follows". What followed was the `@autodocs` block, and it no longer does; the sentence names the
  page instead.

- **The theme stylesheet is wired back in.** [0.4.0](#040) brought
  `docs/src/assets/extra_styles.css` over with the chapters whose figures need it — each of those
  figures exists twice, once per Documenter theme, and both are included, the stylesheet hiding
  whichever does not belong to the active one — and the `assets` entry that loads it was dropped
  again by the commit that split `api.md` off. In that window all seven of those pages would have
  shown both variants of every figure, stacked. No release fell in the window, so nothing published
  was ever affected.

  The entry now says in a comment what depends on it. It has gone missing twice: once by never
  arriving with the chapters, and once by this. [#59]

[#60] and [#59] are the whole release: the protocol with its tests, and the two documentation
entries above. A patch and not a minor bump — nothing in `src/` changed and no exported name moved.
What is new lives in `ext/`, which Julia loads only when `NeuralNetworkParameters` is present, so a
package that does not depend on it sees exactly the library [0.4.0](#040) shipped.

## [0.4.0]

**The manifold geometry becomes public API, and its documentation moves here.** The types were always
this package's; what changes is that a downstream package can now reach them without qualifying, and
that the chapters explaining them live next to them. Driven by
[GeometricMachineLearning#239](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/239),
which deletes GML's own copies of eleven of these types.

### Added

- **Optimizer primitives for `SymmetricMatrix` and the triangular types.** `similar`, `fill!`, the
  elementwise `_add!`, `_rac!`, `_square!`, `_div!`, `_rmul!`, `_difference!`, then `l2norm`,
  `ParameterHandling.flatten` and `update_section!` on a `GlobalSection` over one of them. Without
  these, a `SymmetricMatrix` or a `LowerTriangular` could not be an optimizer parameter at all: the
  generic methods either broadcast — and three of these four types have no `setindex!` to broadcast
  through — or reshape, and ``n(n\pm1)/2`` numbers do not reshape back to ``n \times n``.

  They are written once, over the new `VectorStorageMatrix` alias for the four types that keep their
  free parameters in one vector (`SkewSymMatrix`, `SymmetricMatrix` and the two `AbstractTriangular`s),
  which is also what the `SkewSymMatrix`-specific methods that existed before collapse into.
  `GeometricMachineLearning` carried the missing ones itself, in a `go_bridges.jl` that is deleted
  with this.

  This is what its SympNet, symplectic-attention and volume-preserving layers are parametrized by, so
  it is the difference between those networks being trainable here and not.

- **`test/special_matrices/triangular.jl`** — there were no tests for the triangular types at all.
  Storage layout, the two-triangles-plus-diagonal partition, multiplication against a dense matrix,
  linearity, and the `vec`/constructor round trip. Adapted from GML's `test/arrays/triangular.jl`,
  whose remaining half tests GML's own tensor kernels.

- **`test/special_matrices/optimizer_primitives.jl`** for the methods above, including that
  `_rmul!` scales the *matrix* and not merely its storage — which differ by a sign in the upper
  triangle of a skew-symmetric matrix — and that an optimizer *runs* over one of these types, for
  each of the four types and each of the three first-order methods. The end-to-end case is the one
  that covers `_difference!` and `flatten`: neither is called from `update!`, so testing the
  primitives one at a time leaves both of them out.

### Changed (breaking)

- **The manifold, section and retraction interface is exported.** Previously internal, so downstream
  code named them `GeometricOptimizers.`-qualified:

  | | |
  |---|---|
  | geometry | `metric`, `check`, `Ω` (`rgrad` was already exported) |
  | matrices | `AbstractTriangular`, `StiefelProjection`, `AbstractLieAlgHorMatrix` |
  | sections | `global_section`, `apply_section`, `apply_section!`, `update_section!` |
  | retractions | `AbstractRetraction`, `Geodesic`, `Cayley`, `geodesic`, `cayley`, `retraction` |
  | optimizer | `OptimizerMethod`, `OptimizerSolution` |

  Breaking because a name that is now exported can collide with a downstream one. `Optimizer`,
  `Manifold`, `SkewSymMatrix` and eight others *did* collide with `GeometricMachineLearning` until
  #239 aliased them; that is the whole point of exporting them.

  The **caches stay internal**, for every method alike — `GradientCache`, `MomentumCache`,
  `AdamCache`, `BFGSCache`, `DFPCache`, `NewtonOptimizerCache` and the `OptimizerCache` supertype.
  They are `solver_step!` scratch and nothing outside a step should read one. `test/exports.jl`
  pins both halves of this.

### Fixed

- **Three tests in `test/special_matrices/skew_symmetric.jl` were defined but never called**, so
  `scalar_multiplication` carried a `SkeySymMatrix` typo that would have failed immediately had it
  run. `check_map_to_Skew`, `scalar_multiplication` and `test_random_array_generation` are now
  invoked, the typo is fixed, and the linearity of `+` is covered too. Their properties were tested
  only by GML's copies of these tests, which is where the omission surfaced.

- **`test/global_sections/global_sections.jl` never tested the Stiefel global section.**
  `stiefel_global_section` built a `GrassmannManifold`, making it a second copy of
  `grassmann_global_section`. It now asserts the defining property, `λ(Y)E == Y`, and a new
  `global_tangent_space_rep` asserts that `global_rep` and `apply_section` are inverse — which
  nothing covered, the `Ω` tests reaching only the first of the two isomorphisms `global_rep`
  composes.

- **`stiefel_proj`'s element type defaulted to `Flaot32`.** Harmless, since every call passes `T`,
  but a default nothing reaches is a default nothing checks; it is gone rather than spelled
  correctly. The same file now also asserts that `E` *is* `[I; O]`, which orthonormality does not
  imply.

- **`scripts/retraction_accuracy.jl` loads again.** [0.3.0](#030)'s rename of `_BFGS`/`_DFP` to
  `BFGS`/`DFP` reached `src/`, `test/` and `docs/` and missed the one script the repository kept,
  which still opened with `using GeometricOptimizers: _BFGS, _DFP` and named the old spelling in ten
  `COMBINATIONS` entries and two comments. `UndefVarError` on the fourth line of the file, since
  0.3.0.

  Nothing in `src/` changes and no figure moves. The point of saying so is that this script is where
  every iteration and evaluation count the package quotes comes from — the tables in
  [`default_linesearch`](@ref)'s docstring and in `test/optimizer_convergence/svd_optim.jl` — so for
  two releases the only way to check any of them was to reconstruct the harness by hand. Confirmed
  against the docstring after the fix: `BFGS` with `Backtracking(expand = true)` under `Geodesic`
  gives 95 iterations and 2 441 objective evaluations, which is the cell that table prints.

  This is the kind of gap open issue **C9** is about — the entry counts how few of this package's
  quoted figures have a named harness behind them, and here the one that does had been broken since
  the release that renamed the methods it drives.

- **Downstream: `GeometricMachineLearning` is on 0.3.** Its bound was `"0.2.1"`. Nothing it calls was
  touched by 0.3.0 — it reaches this package through a qualified `import` plus a named `using` list
  that holds neither `BFGS` nor `DFP`, and the three exports 0.3.0 removed never existed — so the
  adoption is the bound and a re-resolve. Recorded here because 0.3.0's entry named GML as the place
  the dead-export bug had already been fixed, and this closes that thread.

### Documentation

- **The manifold, special-matrix and optimizer chapters move here from
  `GeometricMachineLearning`**, thirteen pages in all: the seven-page `Manifolds` chapter (general
  topology through homogeneous spaces), `special_matrices.md` and `global_tangent_spaces.md`, and the
  four-page `Optimizer` chapter, whose framework half merges into `manifold_optimizers.md` and whose
  retraction theory merges into `retractions.md`. They document types that live here.

  The `Optimizer Methods` examples are rewritten against this package's own API — `OptimizerState`
  and the moment accessors — where they used to construct GML's `Optimizer` and call its
  `optimization_step!`.

- **The TikZ figures come with them**, as sources: `docs/src/tikz` holds the six `.tex` files and a
  Makefile, and `Documenter.yml` compiles them before `make.jl` — as does `docs/Makefile`, for a local
  build. That needs `texlive-xetex`, `texlive-science` and `poppler-utils` in the docs job. Committing
  the PNGs instead would have let them drift from the sources.

- **`docs/src/assets/extra_styles.css`** comes with them too. Each figure on the moved pages exists
  twice, once per Documenter theme, and both are included; this is what hides the one that does not
  belong to the active theme, keyed on Documenter's own theme class. Without it every one of those
  pages shows both, stacked. The `_light`/`_dark` suffix is therefore load bearing: a figure with only
  one variant must not carry it, which is why the two on `linesearch.md` no longer do.

- **`DocumenterInterLinks` gains a `GeometricMachineLearning` entry.** Nine references on the moved
  pages point at chapters that stayed there — the pullback machinery, the SympNet and transformer
  architectures — and they stay references. This is a documentation-only edge: nothing under `src/`
  or `test/` depends on GML, and `InterLinks` reads a published `objects.inv` rather than a local
  build, so linking both ways creates no build-order cycle.

- Twelve bibliography entries arrive with the moved chapters, and `docs/make.jl` gains the
  `Main.definition`/`Main.theorem`/`Main.proof` helpers they are written against.

Almost all of the above is [#50]; the two `scripts/retraction_accuracy.jl` and
`GeometricMachineLearning`-bound bullets under *Fixed* are [#49], which merged first and is
unreleased, so it ships here. A minor bump and therefore breaking, for the reason the *Changed
(breaking)* section gives: the newly exported names can collide downstream.

[#50]'s review is where the two missing primitives came from. `ParameterHandling.flatten` and
`_difference!` had not moved onto `VectorStorageMatrix` with the rest of the family, and both are
reached from outside `update!` — `flatten` from the `Optimizer` constructor, `_difference!` from
`gradient_difference!` on every `OptimizerStatus` — so every direct test of `_add!` and friends
passed while none of the three newly supported types could actually be optimized. The end-to-end
testset in `test/special_matrices/optimizer_primitives.jl` is what closes that gap, and it is the
reason the *Added* entry above lists nine methods rather than seven. The review also found the
missing `docs/src/assets/extra_styles.css`, without which every figure on the moved pages rendered
twice.

## [0.3.1]

### Removed

- **The MNIST material has moved to [GMLDatasets.jl](https://github.com/JuliaGNI/GMLDatasets.jl).**
  Nothing in `src/` changes and nothing a user calls goes away; what leaves the repository is the
  experiment that was run *against* the package. That is the nine files of `scripts/` that carry it
  — `mnist.jl`, `mnist_cuda.jl`, `mnist_cuda_repetitions.jl`, `mnist_metal.jl`,
  `mnist_metal_short.jl`, `metal_memory_probe.jl`, `distill_mnist_results.jl` and the two `screen`
  wrappers — together with `scripts/Project.toml`, the three checked-in result series under
  `docs/src/data/`, and the "Reproducing the experiment" half of `docs/src/manifold_optimizers.md`,
  which plotted them.

  All of it arrived with [#14], and it earned its place: it found three limitations of the optimizer
  interface and two bugs in `Adam` and `MomentumMethod`, every one of them fixed in `src/` and
  recorded under [0.2.0](#020) below. But this is a library for scientific machine learning, and an
  image data set is not part of its subject — nor is a documentation build that ships 540 rows of
  somebody else's training run.

  `docs/src/manifold_optimizers.md` keeps its derivation, which documents this package's own
  construction rather than a run of it, and its experiment section becomes a paragraph pointing at
  GMLDatasets.jl. `scripts/retraction_accuracy.jl` **stays**: it regenerates this package's own
  retraction tables, from the root project rather than from `scripts/Project.toml`, and C9 below is
  about how few harnesses of that kind there are.

  `MNIST_PORT.md` is deleted with them. Its account of what was done duplicates the changelog; its
  two findings about `src/` that were *not* in the changelog now are, under 0.2.2's known issues and
  as A21 in [Open Issues](#open-issues) below; and its operational half went to the GMLDatasets.jl
  documentation, which those known issues link.

  All of the above is [#48]. A21 and C9's sixth bullet are what its review turned up — A21 is the
  only catalogue entry this release adds, and the finding in it dates from [#14] rather than from
  this work. Nothing in `src/` changed, which is why this is a patch and not a minor bump.

## [0.3.0]

### Changed (breaking)

- **`_BFGS` and `_DFP` are now `BFGS` and `DFP`, and they are exported.** They were the only optimizer
  methods carrying a leading underscore, they are what `Optimizer` takes as its default `algorithm`,
  and they are what the tests construct — an unexported, underscore-prefixed name for the primary
  entry point of a package. There is no compatibility alias: the underscore spelling is gone.

### Added

- **`BFGSState` and `DFPState` are exported**, which they had to be for the entry above to be true
  of the whole family. Every other state already was — `GradientState`, `MomentumState` and
  `AdamState` alongside their methods, `NewtonOptimizerState` above them — so `BFGSState` was the
  one state type a user had to reach into the module for, and `test/named_tuple_parameters.jl`
  printed the asymmetry in a single expression: three bare names followed by
  `GeometricOptimizers.BFGSState`. `DFPState` is an alias for `BFGSState`. The *caches* stay
  internal, for every method alike, and that is the line the export list now draws.
- **Docstrings for `BFGS` and `DFP`.** Both said only "Algorithm taken from `nocedal2006numerical`"
  — the same sentence twice, with no signature line, so the two were indistinguishable in the
  rendered reference. That was tolerable while they were unexported names; `docs/src/index.md` is a
  whole-module `@autodocs`, so exporting them promotes two near-empty entries into the public
  reference next to `Newton`'s. Each now states its update, and `BFGS` says that it is
  `Optimizer`'s default `algorithm` — the reason it is exported at all.
- **A test that every exported name is defined**, `test/exports.jl`. The check below is what
  surfaced the dead exports, and calling it "worth keeping in mind" is exactly the guarantee that
  had already failed once: a dangling export is silent *because* nothing resolves the name, so
  nothing but the check itself will ever notice. It runs on every `Pkg.test()` now, together with a
  spot check that the methods and their states are exported and the caches are not.

### Removed (breaking)

- **The exports `NewtonOptimizer`, `BFGSOptimizer` and `DFPOptimizer`, none of which existed.**
  `names(GeometricOptimizers)` listed all three and `isdefined` was `false` for all three, so
  `using GeometricOptimizers; BFGSOptimizer()` raised `UndefVarError`. The block was transcribed from
  SimpleSolvers' export list in `0eab6b1`, and SimpleSolvers 0.7.x had the identical bug — its 0.3.8
  shims (`BFGSOptimizer(args...) = NewtonOptimizerState(args...; hessian = HessianBFGS)`) had been
  dropped while the exports stayed. SimpleSolvers removed the names in 0.8; this package was the last
  place carrying them. `Newton` was already exported separately, so `NewtonOptimizer` was surplus
  even had it existed.

  GeometricMachineLearning hit the same bug and fixed it in its `d986e61a` by deleting the dead
  exports; the fix was never mirrored here. The consequence was not cosmetic: GML's changelog points
  users at `_BFGS()` as the replacement for its own removed `BFGSOptimizer`, and a user following
  that to this package found neither name.

  `filter(n -> !isdefined(GeometricOptimizers, n), names(GeometricOptimizers))` is now empty, and
  `test/exports.jl` asserts that it stays empty.

  All of the above is [#47]; the three `### Added` entries are what its review turned up, and none
  of them left an entry open in [Open Issues](#open-issues).

## [0.2.2]

### Fixed

- **A `GrassmannManifold` can be optimized over.** It could not, at all — as a bare point or inside a
  `NamedTuple` — which was catalogue entry A11 and is the concrete content of [#27]. Every
  `GrassmannManifold` test in the suite exercised the manifold, its lift, its retraction and its
  `check`; none exercised a solve, because none could. The three failures, reproduced on `main`
  before anything was changed:

  ```
  Optimizer(rand(GrassmannManifold, 5, 3), F)
  # MethodError: no method matching GradientAutodiff(::typeof(F), ::GrassmannManifold{Float64, Matrix{Float64}})

  solve!(ps, OptimizerState(_BFGS(), ps), Optimizer(ps, F; algorithm = _BFGS()))
  # CanonicalIndexError: setindex! not defined for GrassmannManifold{Float64, Matrix{Float64}}

  solve!(ps, OptimizerState(Adam(Float64), ps), Optimizer(ps, F; algorithm = Adam(Float64)))
  # The function `similar` does not make sense in this context. Consider using rand.
  ```

  The retraction layer was already generic — `lift_factors`, `geodesic`, `cayley`,
  `lift_from_columns` and `retraction_differential` all had Grassmann methods — so nothing about the
  geometry was missing. What was missing was **a set of helpers in the optimizer plumbing written
  against `StiefelManifold` / `StiefelLieAlgHorMatrix` instead of against the abstract types**, and
  A11's estimate that the fix was "small and in two places" was the count of the ones that raise on
  the first step rather than of the ones a solve needs:

  | | site | what it broke |
  |---|---|---|
  | `GradientAutodiff` | `utils.jl` | `Optimizer` construction, for a bare point |
  | `_similar` | `named_tuple_wrapper.jl` | the `Adam` / `Momentum` / `Gradient` state constructors |
  | `copyto!` | `stiefel_manifold.jl` | `update!(::BFGSCache, …)` |
  | `_copyto!` on a `GlobalSection` | `named_tuple_wrapper.jl` | the frame comparison in `store_gradient!` |
  | `_difference!` | `named_tuple_wrapper.jl` | the quasi-Newton secant pair |
  | `_add!`, `_rac!`, `_div!`, `_square!` | `named_tuple_wrapper.jl`, `stiefel_lie_algebra_horizontal.jl` | the momentum recursion and the `Adam` moments |
  | `assign!`, `vec` | `stiefel_lie_algebra_horizontal.jl` | — |
  | `l2norm` | `optimizer_status.jl` | **see below** |
  | `one` | `stiefel_lie_algebra_horizontal.jl` | **see below** |

  Every one of them is now a single method over an abstract type, and for those that act on a
  lift, over `Base.parent` — the tuple of blocks a lift stores its free parameters in, `(A, B)` for
  the Stiefel lift and `(B,)` for the Grassmann one. The Stiefel bodies *were* that `foreach`, so
  the generic form is bit-identical for them and the next lift type inherits it rather than
  rediscovering the gap. `apply_section!` on a `GrassmannManifold` also assigned `Y.A = …` where the
  Stiefel method writes `@views Y.A .= …`, i.e. it replaced the point's array on every solver step
  and returned that array rather than the point.

  **Two of the nine would not have raised.** `l2norm` on a horizontal lift is the norm of its
  *flattening* — the coordinates `Q` is sized by, `outer!` forms its outer product in and a line
  search's `α` parameterizes. Without its own method the Grassmann lift fell through to
  `l2norm(::AbstractMatrix)`, the *ambient* Frobenius norm, which counts each off-diagonal block
  twice and so is `√2` too large; it feeds `step_αmax` (the step ceiling of A1b),
  `curvature_is_usable`, `rxₐ` and `rg`. And `Base.one` — which `geodesic` calls on every retraction
  — existed as a `KernelAbstractions` kernel for the Stiefel lift and fell through to `Base._one`,
  whose diagonal write is the scalar indexing that kernel exists to avoid, for the Grassmann one.
  That second is a piece of issue A19 rather than a fix for it: A19 is about `𝔄`, whose argument is
  a bare matrix and which is still on the scalar-indexed path.

  **Two corrections to A11's own text**, both found by reproducing it. It says the `NamedTuple` case
  fails because `GrassmannManifold` does not define `setindex!` while `StiefelManifold` does; neither
  defines it, and what `StiefelManifold` had is the `copyto!` above, which is why the generic
  `AbstractArray` path was never reached for it. And A11 names `_copyto!` as the `NamedTuple`
  failure where `test/adam_with_euclidean_decay.jl` named `_similar` — both are right, they are the
  `_BFGS` path and the `Adam` path, which is why they are two rows of the table and not one.

  **No Stiefel figure moves**, which is the bar rather than "the tests pass": every change is either
  a widened signature that resolves to the same body for a `StiefelManifold` or a method only the
  Grassmann types reach. Verified with `svd_tables()` from `scripts/retraction_accuracy.jl` — all
  twenty (method, line search) combinations over eight starting points, iterations, evaluations, `rg`
  and `check` — which reproduces `main` **to the digit in every cell**.

  New coverage in `test/grassmann_optimizer_tests.jl`: `Gr(1, 3)` and `Gr(2, 5)`, all five methods,
  both retractions, `Float32` and `Float64`, plus a `NamedTuple` holding a `GrassmannManifold`
  beside an ordinary `Matrix` and one holding both kinds of manifold at once. That last is what
  would have caught the 0.2.0 defect in `ParameterHandling.flatten(T, ::Manifold)` — that it rebuilt
  a `StiefelManifold` for every kind of manifold, so a `GrassmannManifold` came back Stiefel with a
  different `rgrad` and a different retraction and no error anywhere — which until now had no
  end-to-end test, because a solve could not reach it.

  A11 asked for "a decision about what the Grassmann *tests* should then assert, since the quotient
  means two representatives of the same point are equally correct answers". The decision is that the
  objective is the Rayleigh quotient ``-\mathrm{tr}(Y^TMY)``, which is constant on the class
  ``Y \sim YO`` by construction, and that every assertion is made about the projector ``YY^T`` and
  never about ``Y``. `test/manifold_optimizers_with_new_interface.jl`'s problem — the distance to a
  target point — could not be reused for exactly that reason: ``Y`` and ``-Y`` are the same point of
  ``Gr(1, 3)`` and are at different distances from it.

  The Grassmann half of [#27] closes with this. **Two** things remain under that issue, not one —
  this paragraph said only the first until the review of [#46] checked the second against merged
  `main` rather than taking the issue's own wording for it:

  - `mode = :finitediff` has no `GradientFiniteDifferences` method for a bare `Manifold` of *either*
    kind, which is the same gap [#24] records for a `NamedTuple`.
  - `default_gradient` for a bare `Manifold` still falls through to its `AbstractArray` method, which
    is [#27]'s *second* bullet and is untouched by the change above. It is A20 below.

### Known issues

One entry is new in [Open Issues](#open-issues) below, from the review of [#46]. It is not fixed
here, and it says what closing it would take:

- **A20** — `default_gradient(problem, x)` has methods for `AbstractArray` and for
  `ArrayNamedTuple`, and `Manifold <: AbstractMatrix`, so a bare manifold takes the first: a gradient
  sized by `length(x)` that composes `problem.F` with a flat vector instead of rebuilding the point.
  It does not raise where it is built — it raises at the first gradient evaluation, and only on an
  objective that names its argument type, which is why nothing in the suite shows it. The fix is one
  method delegating to the `GradientAutodiff(F, ::Manifold)` this release added.

Two more come from the MNIST port of [#14], which found them but did not fix them. They were
recorded in a `MNIST_PORT.md` at the repository root; that file went with the scripts it described
(see [0.3.1](#031) above), and this is what it said that is still true of `src/`. The
first is an entry in the catalogue below as well; the second is a note about code that is no longer
here to fix:

- **A21** — **the optimizer interface cannot hold GPU arrays**, so a GPU run keeps its parameters on
  the host. This is a regression against `GeometricMachineLearning`, whose optimizers did run on
  `CUDABackend()`. Two things block it: `ParameterHandling.flatten` has no method that a GPU array
  reaches, and the optimizer flattens its parameter `NamedTuple` on every step; and
  `_similar(::Manifold)` always allocates on the host, so the state would mix host and device arrays
  even if the flattening did not. It cost the port little at the size of network it ran — ≈1.2 MB up
  and down per step against ≈3 GB of device-side activations — which is why it was left. It would
  matter for a parameter set large enough to be worth keeping resident. Both causes, the
  measurement, and what closing it would take are in the entry below.

- **Two `Zygote` pullback workarounds lived in the scripts rather than in `src/`**: `mat_tensor_mul`'s
  pullback indexes scalars, which a GPU array cannot serve, and it produces lazy `Transpose`s that
  have to be materialized. The scripts moved to
  [GMLDatasets.jl](https://github.com/JuliaGNI/GMLDatasets.jl) and carried the workarounds with them, so
  this is recorded rather than tracked: it is not an open issue in this package.

The operational half of `MNIST_PORT.md` — what the four training configurations are for, why the
unconstrained baseline is *expected* to plateau at ``\sqrt{2 \cdot 0.9} \approx 1.342``, and the
`Metal` unified-memory handling the GPU scripts needed — is now
[Running the Experiments](https://juliagni.github.io/GMLDatasets.jl/latest/running_the_experiments/)
in the GMLDatasets.jl documentation.

## [0.2.1]

### Added

- **`𝔄exp(B̂, B̄, algorithm = ScaledSquaring())`**, computing
  ``\exp(B'(B'')^T)`` as ``\mathbb{I} + B'\mathfrak{A}(B', B'')(B'')^T`` — the identity
  [`𝔄`](@ref) exists for, packaged as the exponential it computes. [`geodesic`](@ref) already
  computed this product inline; this is for callers that want the exponential of a low-rank product
  on its own, at a cost set by ``n`` rather than by ``N``.

  It comes from `GeometricMachineLearning`, which carried it as a one-line wrapper over this
  package's `𝔄` and is dropping it as replicated functionality
  (GeometricMachineLearning#230). Not exported, as `𝔄` is not.

  **The default is `ScaledSquaring`, as [`geodesic`](@ref)'s is**, and deliberately not the unscaled
  series that `𝔄(B̂, B̄)` is. The reason is the one under geodesic's "The default changed in 0.2.0":
  the series cancels catastrophically once ``\|\bar{B}\| \gtrsim 50``, which is not a regime a
  function presenting itself as *the exponential* may quietly get wrong. Relative error against
  `exp(Matrix(B))` for `B = scale * rand(StiefelLieAlgHorMatrix, 10, 2)`:

  | `scale` | 1 | 10 | 50 | 100 |
  | --- | --- | --- | --- | --- |
  | ``\|\bar{B}\|`` | 3.8 | 36.3 | 145.8 | 324.9 |
  | `TaylorSeries` | 5.3e-16 | 1.1e-7 | 1.8e24 | 1.7e79 |
  | `ScaledSquaring` | 4.0e-16 | 1.9e-15 | 7.8e-15 | 1.9e-14 |

  `algorithm` is defined exactly where `𝔄(X, algorithm)` is — `TaylorSeries`, `ScaledSquaring`,
  `AugmentedPade` — and so, like `𝔄(B̂, B̄, algorithm)`, its `AbstractExponentialAlgorithm` signature
  also admits `ProjectedSkew`, for which there is no `𝔄` method. That hole is `𝔄`'s and is left as
  it is rather than papered over here; the docstring says which three are supported.

  GML's test for it came over too, into `test/retractions/exponential_accuracy.jl`: the identity
  swept over `Float32`/`Float64` and every shape with `N = 1:10`, `n = 1:N`, plus the `algorithm`
  form. The docstring jldoctests assert it for one 10×2 `Float64` lift, which cannot pin the element
  type of the result or reach rectangular arguments, so this is new coverage of `𝔄` as much as of
  `𝔄exp`. That sweep scales its arguments by `0.1`, so it is silent about the default; a second
  testset asserts the identity at lift norms up to ~600 and against `geodesic`, and fails on three
  of its four scales if the default moves back to the unscaled series.

### Fixed

- **The optimizer caches and states no longer take an hour to compile through a function.** The
  cache and state structs bounded their type parameters by `OptimizerSolution`,
  `GradientArrayOrNamedTuple` and `GlobalSectionSingleOrNamedTuple`. Those bounds are now gone, from
  every optimizer: `GradientCache`/`GradientState`, `MomentumCache`/`MomentumState`,
  `AdamCache`/`AdamState`, `BFGSCache`/`BFGSState` (which is also `DFPState`), `DFPCache` and the
  `VT` of `OptimizerResult`, which is on the return path of every `solve!`.
  `NewtonOptimizerCache`/`NewtonOptimizerState` follow suit although their bounds
  (`AbstractArray{T}`, `GlobalSection{T}`) were never the expensive kind — Newton is
  `AbstractArray`-only, so nothing expands behind them — so that the family stays uniform and nobody
  has to work out per struct whether a given bound happens to be one that costs.

  The symptom was not an error but a hang, and only in callers that had to *infer* the type of an
  optimizer rather than take it from a concrete argument — so it did not show up at the REPL, where
  every intermediate is concrete, but did show up for anyone who wrapped training in a function.
  `GeometricMachineLearning`'s symplectic-autoencoder test is one: nine layers, mixing
  `StiefelManifold` weights with ordinary matrices and vectors. Compiled statement by statement it
  takes ~14 s; compiled as one method body it did not finish in over an hour, with the time going to
  `subtype_unionall` under `ml_matches`. It is ~14 s either way now.

  Worth being precise about the cause, because the obvious reading is wrong. The bounds did not cost
  concrete inference: `OptimizerCache(Adam(Float64), ps)` on a `NamedTuple` of parameters infers to a
  `UnionAll` with or without them, since the outer constructors are written in the same aliases.
  What the bounds added was *coupling* — one `T` tying all four parameters together underneath three
  nested `Vararg` unions, the last over a three-parameter `UnionAll` — so every method-table
  intersection involving an inferred cache had to re-solve that constraint system. Unbounded, the
  inferred type has the same shape but independent parameters and a bare `<:Tuple`, which costs
  nothing to intersect.

  No behaviour changes and nothing is unchecked that was checked before: the invariant lives in the
  constructors, whose signatures take `x::OptimizerSolution{T}` and
  `g::AT where AT<:GradientArrayOrNamedTuple{T}` and build the `GlobalSection` themselves. That holds
  for the *inner* constructors of `BFGSCache`, `BFGSState`, `DFPCache` and the two Newton structs as
  readily as for the outer ones elsewhere — what must not carry the aliases is the `struct` parameter
  list, not the methods. A note above the aliases in `optimizer_solution.jl` now says not to use them
  as struct bounds and why, and `test/named_tuple_parameters.jl` pins every cache and state as
  unbounded so reinstating any of them fails a test rather than silently costing an hour. See
  GeometricMachineLearning#230.

### Bookkeeping

- **The 0.2.0 release notes are closed out** (C4). That version is tagged and registered, so its
  section is `## [0.2.0]` rather than `## [Unreleased] — targeting 0.2.0`, it has the
  `[0.2.0]: …/releases/tag/v0.2.0` definition that `[0.1.0]` always had, and `[Unreleased]` compares
  against the latest tag instead of `v0.1.0`. The two links from the Open Issues catalogue into that
  section pointed at the old heading's anchor and are repaired with it.

- **`Random` and `LinearAlgebra` have `[compat]` entries** (C3), matching the `Printf = "1"` that was
  already there. All three are stdlibs in `[deps]` and used the same way; two of them had no bound.

### Known issues

Four entries are new in [Open Issues](#open-issues) below, all from the review of this release. None
is fixed here, and each says what closing it would take:

- **A18** — `𝔄(B̂, B̄, algorithm)` and `𝔄exp(B̂, B̄, algorithm)` accept an
  `AbstractExponentialAlgorithm`, which admits `ProjectedSkew`: a `geodesic`-level algorithm with no
  `𝔄` method. The call dispatches and then fails one frame in, naming `𝔄` rather than what was
  called. `𝔄exp` inherits the hole rather than adding one, and narrowing only `𝔄exp` would put the
  two out of step.
- **A19** — nothing in this repository exercises `geodesic` on a GPU backend, and `𝔄` reaches
  `Base.one`, whose diagonal write is scalar indexing. The documented claim that `ScaledSquaring` is
  "the only usable algorithm that runs unchanged on a `KernelAbstractions` GPU backend" is therefore
  unverified here, and there is a concrete reason to check it.
- **C14** — `geodesic` and the new `𝔄exp` assemble the same ``\mathbb{I} + B'\mathfrak{A}B''^T``
  independently, and now default the same way in two places rather than one.
- **C15** — the compile-time figures quoted above were measured against the six first-order structs.
  The quasi-Newton and Newton paths were widened on the strength of their *inferred types*, which is
  a sound argument and not a measurement.

## [0.2.0]

v0.1.0 carried **two parallel optimizer hierarchies**: `src/optimizers/`, ported from
GeometricMachineLearning, which knew about manifolds and was driven by `optimization_step!`, and
`src/euclidean_optimizers/`, which had the newer `OptimizerState` / `solver_step!` / `solve!`
interface but only worked on `AbstractVector`s. This release collapses them into one. `Optimizer` is
now the single entry point and runs over anything the new `OptimizerSolution` union covers — an
`AbstractVector`, a `Manifold`, or a `NamedTuple` of arrays — with the same driver, the same states
and the same caches.

Unifying them meant reading both implementations against each other, which turned up a number of
defects that had been invisible: two of the three first-order methods computed a step that did not
match their own documented formula, two optimizer states were reading uninitialized memory, and
`mul!` on the special matrix types returned the wrong object. Those are fixed here, so **the iterates
this package produces change**, in most cases substantially for the better.

### Removed (breaking)

- **`EuclideanOptimizer` is gone.** Replace `EuclideanOptimizer(x, F; …)` with `Optimizer(x, F; …)`.
- **The old manifold entry points are gone**: `optimization_step!`, `init_optimizer_cache`,
  `setup_gradient_cache` / `setup_momentum_cache` / `setup_adam_cache` / `setup_bfgs_cache`,
  `perform_optimization!`, and the manifold `BFGS` method with its `_BFGSCache`. The
  `Optimizer(method, ps; retraction = cayley)` + `optimization_step!(o, λY, ps, dx)` sequence is
  replaced by `Optimizer` / `OptimizerState` / `solve!`:

  ```julia
  # before
  o = Optimizer(Momentum(), ps; retraction = cayley)
  λY = GlobalSection(ps)
  optimization_step!(o, λY, ps, dx)

  # now
  opt   = Optimizer(ps, L; algorithm = MomentumMethod(), retraction = Cayley())
  state = OptimizerState(MomentumMethod(), ps)
  solve!(ps, state, opt)
  ```

- **`AdamWithDecay` is removed.** It differed from `Adam` only in an exponentially decaying learning
  rate, and the learning rate is now the line search's business (see below); a decaying schedule
  belongs in a `LinesearchMethod`, which is what `DecayingStatic` is (see *Added*). Note that the
  decay it named was of the *learning rate*, not of the weights — `AdamWithEuclideanDecay` (see
  *Added*) is a different method and not a rename of this one.
- **`Adam`'s `η` field is removed.** It was never applied to the direction, so `Adam(1e-3)` and
  `Adam(1e2)` produced identical results.
- Two stranded test files (`test/optimizer_status_tests.jl`, `test/special_matrices/poisson_tensor.jl`)
  were deleted. Neither was included by `runtests.jl` and both referenced symbols that no longer exist
  (`QuasiNewtonOptimizer`, `PoissonTensor`).

### Changed (breaking)

- **Method types renamed.** `EuclideanOptimizerMethod` → `OptimizerMethod` (there is one abstract type
  now); `Gradient` / `_Gradient` → `GradientMethod`; `Momentum` → `MomentumMethod`. `_BFGS` and `_DFP`
  are unchanged. `GradientMethod`, `MomentumMethod`, `Adam` and their states are now exported.
- **`Adam`'s constructor changed shape.** Because `η` used to be the *first positional* argument, `β₁`,
  `β₂` and `δ` are keyword arguments now, so an old `Adam(1e-3)` call fails loudly instead of silently
  setting `β₁ = 1e-3`. The element type is what `Adam` takes positionally: `Adam(Float32)`. Unlike
  `MomentumMethod`, `Adam` is not converted by `Optimizer`, so the type has to match the parameters.
- **The learning rate moved to the line search.** Every `OptimizerMethod` now produces only a
  *direction*; how far the optimizer goes along it is the line search's `α`. A fixed learning rate `η`
  is `linesearch = Static(η)`, and `Static` is exported for that reason. `default_linesearch` makes
  `Static(1e-3)` the default for `Adam`, whose direction is a moving average that is deliberately not
  required to descend, and `Backtracking` the default for everything else: the (quasi-)Newton methods
  because they come with a scale of their own, `GradientMethod` and `MomentumMethod` because a
  searching line search is what makes them converge rather than merely descend.

  On the SVD problem, letting the line search pick the step takes the relative error after 1000
  iterations from 1.4e-2 to 2.3e-3 (`GradientMethod`) and from 1.4e-2 to 2.2e-3 (`MomentumMethod`).
  `Bisection` reaches 6.2e-7 in the same number of *iterations*, but it costs ≈583 objective
  evaluations per iteration against `Backtracking`'s ≈25, so on any fixed budget of work
  `Backtracking` is ahead; see the table in `default_linesearch`. Pass a searching method explicitly
  when iterations rather than evaluations are what you are paying for.

  The `Backtracking` is `Backtracking(T; expand = true)`, which is *not* SimpleSolvers' own default —
  see the next entry.
- **The line search may now lengthen a step**, which is what SimpleSolvers 0.11 was needed for at the
  time this entry was written; the floor is `0.12` now, see below. A shrink-only
  backtracking search starts its trial step at `α = 1` and can never exceed it. That is right for a
  direction already scaled like a Newton step — `_BFGS` accepts `α = 1` on 74% of its iterations — but
  wrong for one that is systematically under-scaled. `_DFP`'s wants a median `α` of 8, so on the SVD
  problem it accepted the ceiling on *every* iteration and needed 47 115 of them; handing the same
  search a trial step of 3 instead of 1 was worth a factor of 217, which is what identified the ceiling
  rather than the method as the cause.

  That measurement became JuliaGNI/SimpleSolvers.jl#174 and, in SimpleSolvers 0.11, the `expand` key:
  an accepted *first* trial step is lengthened while each longer trial still satisfies sufficient
  decrease and strictly improves the merit. `default_linesearch` switches it on. Iterations, then total
  objective evaluations, Geodesic / Cayley:

  | | `expand = false` | `expand = true` |
  |---|---|---|
  | `_BFGS` | 114 / 136 iters, 2 915 / 3 446 evals | **93 / 118**, **2 389 / 3 021** |
  | `_DFP` | 47 115 / 26 479, 1 177 919 / 662 019 | **702 / 1 366**, **18 258 / 35 329** |

  It costs under 4% per iteration and, on a well-scaled problem, exactly nothing: the extrapolation
  reuses `φ(0)`, `φ'(0)` and `φ(α)`, all known once the trial step is accepted, so declining to expand
  costs no evaluation. On the sphere problem the evaluation counts are identical with and without it.

  The `_DFP` figures are for one starting point, but they are now roughly representative: across eight,
  `_DFP` under the default ranges over 387–845 iterations (`Geodesic`) and 466–1 366 (`Cayley`), and
  `_BFGS` over 93–154 / 114–156. Those `_DFP` ranges read 512–77 890 and 465–3 834 when this entry was
  first written, because `Q` became badly conditioned and how fast the expansion dug it out was close to
  arbitrary; that was the missing curvature condition, fixed below. For a DFP-heavy workload
  `StrongWolfe(T; c₂ = 0.1)` is still somewhat faster and steadier — see the next entry.

  0.11 also removes `Backtracking.α₀`, the field that appeared to configure the trial step and was never
  read — unused here, so nothing in this package changes for it. The compat bound went to `"0.11"` and
  not `"0.10, 0.11"` because `expand` does not exist in 0.10; it is `"0.12"` as released, for the step
  ceiling of issue A1b.
- **`StrongWolfe` is re-exported.** It was the only one of SimpleSolvers' six line searches this
  package did not pass through. It is not a default, but it is the better explicit choice for a
  DFP-heavy workload: `StrongWolfe(T; c₂ = 0.1)` takes `_DFP` to 207 / 279 iterations and
  16 873 / 23 818 evaluations, 1.4× to 2.0× faster in wall clock than the expanding `Backtracking`
  default, and stays inside 207–755 / 215–447 across eight starting points. The "part that matters
  more" this entry originally claimed — that the default ranged over two orders of magnitude where
  `StrongWolfe` did not — no longer holds: the curvature-condition fix below brings the default to
  387–845 / 466–1 366, so the remaining case for `StrongWolfe` here is cost rather than reliability.
  `c₂ = 0.1` and not its own default of `0.9`, at which the Wolfe conditions already hold at `α = 1` on
  99.4% of iterations, so its bracketing phase never fires and it crawls too — 26 978 iterations
  against 207.
- **Retractions are passed as instances, not functions**: `retraction = Cayley()` / `Geodesic()`, not
  `retraction = cayley`. The default is `Cayley()`.
- **`MomentumMethod`'s recursion is fixed.** It accumulated `p ← p + α∇L`, which is not momentum but an
  undamped sum: for a constant gradient it grew linearly instead of saturating at `∇L/(1 - α)`, and it
  kept pushing after `∇L → 0`. It is `p ← αp + ∇L` now, with direction `-p`. On the SVD test problem
  the relative error drops from 1.9e-2 to 9.7e-3 — from *worse* than plain gradient descent to
  slightly better, as momentum should be. ([#18])
- **`Adam`'s update formulas are fixed.** Three defects, each of which changes the iterates:
  - the bias correction was evaluated at `t + 1` rather than `t`, making the very first direction
    `0.7425⋅sign(∇L)` instead of the `sign(∇L)` the correction is supposed to give;
  - the recursion factors were `β/(1 - βᵗ)` rather than `(β - βᵗ)/(1 - βᵗ)`, which is what the
    bias-corrected form requires — the former amplifies the moments by up to `1/(1 - β)` in the first
    iterations;
  - the square root was applied to `m₂` instead of to `m̃₂ = m₂ + δ`, i.e. the direction was
    `-m₁/(m₂ + δ)` instead of `-m₁/(√m₂ + δ)`, and it corrupted the stored second moment afterwards.

  Relative error on the SVD problem improves from 1.1e-3 to 1.7e-5 (`Geodesic`) and from 1.7e-3 to
  6.0e-5 (`Cayley`).
- **`_DFP` accepts a `Manifold` and a `NamedTuple`, like `_BFGS`.** Its cache was `AbstractVector`-only,
  so anything else fell through to a `NewtonOptimizerCache` and a `MethodError`. It is lifted to
  `OptimizerSolution` the way `BFGSCache` already was: the solution and the gradient get separate type
  parameters (on a manifold they are a point and a horizontal lift, of different shapes), the section
  may be a `NamedTuple` of sections, `Q` is sized by the length of the flattening rather than by
  `length(x)`, and the quadratic form `γᵀQγ` is taken in the flattened coordinates. `HessianDFP` and
  the `Hessian(::_DFP, …)` methods are widened to match.
- **`_BFGS` and `_DFP` run on a *bare* `Manifold`.** `Q` is sized by the intrinsic dimension — the
  length of the flattening, 2 for `St(3, 1)` — while the gradient and the direction are horizontal
  lifts of the ambient shape, `3 × 3`. Four methods that the `NamedTuple` case had and the bare case
  did not sat on that boundary: `outer!` and `_mul!` for an `AbstractLieAlgHorMatrix`, `alloc_h` for a
  `Manifold`, and `_copyto!` of a point into a `GlobalSection`. Without them `_BFGS` on a bare
  manifold died in `outer!` with `AssertionError: axes(O, 1) == axes(x, 1)`. `ParameterHandling.flatten`
  gains the `GrassmannLieAlgHorMatrix` method that the Stiefel lift already had.
- **`_BFGS` and `_DFP` now actually update their inverse Hessian.** `state.ḡ` was refreshed at the end
  of the iteration, i.e. at the very iterate the next secant difference `γ = ∇f(x⁽ᵏ⁾) - ∇f(x⁽ᵏ⁻¹⁾)`
  was formed at, so `γ` was identically zero, `δᵀγ` was zero with it, and the guard around the `Q`
  update skipped it on **every** iteration. `Q` stayed the identity for the whole solve, i.e. both
  methods silently ran as steepest descent. Both caches now advance `ḡ` themselves, right after they
  have used it. On `sum(sin.(x).^2)` from `0.5·ones(3)` with `Backtracking`, `_BFGS` went from
  `1.1e-3` after exhausting all 1000 iterations to machine precision in six.
- **The `_DFP` update uses `γγᵀ` where it used to use `δδᵀ`.** DFP is
  `Q ← Q - Qγγᵀ Q/(γᵀQγ) + δδᵀ/(δᵀγ)` (nocedal2006numerical, eq. 6.15); the rank-one term that is
  subtracted was built from `cache.ΔxΔx`, which also left `cache.ΔgΔg` computed on the line above and
  never read.
- **The (quasi-)Newton directions are checked for descent.** `ensure_descent!` replaces a direction
  with `∇f⋅δ ≥ 0` by the steepest-descent direction. A Newton step only descends where the Hessian is
  positive definite, and up to SimpleSolvers 0.8 the `Bisection` and `Quadratic` line searches hid
  this by returning a *negative* step; they no longer do. Without the safeguard,
  `sum(sin.(x).^2)` from `ones(3)` — where `sin²` has second derivative `-0.83` — converges to `π/2`,
  which is a *maximum*. `Adam` and `MomentumMethod` are deliberately excluded: their direction is a
  moving average and is allowed not to descend on an individual step.
- **`AdamState` and `MomentumState` are zero-initialized.** They used `_similar`, i.e. uninitialized
  memory, which is read on the first `update!` — so the first step depended on whatever happened to be
  in memory, and `Adam` could throw a `DomainError` from `√` of a negative value.
- **`l2norm` of a `NamedTuple` combines the block norms in quadrature** instead of summing them, which
  overestimated the ℓ² norm of the flattened parameters by up to `√k` for `k` blocks — and with it
  every stopping criterion computed from it. Stopping behaviour on `NamedTuple` parameters changes.
- **`mul!(C, A, α)` and `rmul!` on the special matrix types return `C`.** They used to return the inner
  field (`C.S` / `C.A` / `C.B`), violating the LinearAlgebra contract. This made `MomentumMethod` and
  `Adam` unusable on a bare `Manifold`: for a `StiefelManifold` of size (3, 1), `_mul` degraded a
  (3, 3) `StiefelLieAlgHorMatrix` into a (2, 1) `Matrix`, and the two algorithms then died with
  `DimensionMismatch` and `CanonicalIndexError`. ([#17])
- **`ParameterHandling.flatten(T, ::Manifold)` round-trips the manifold type.** It reconstructed a
  `StiefelManifold` for every kind of manifold, so a `GrassmannManifold` came back Stiefel — with a
  different `rgrad` and a different retraction, and no error anywhere.
- **`Float32` parameter `NamedTuple`s are no longer silently promoted to `Float64`** by
  `ParameterHandling.flatten`, which defaults to `Float64`; the flattened vector was then incompatible
  with the parameters.
- **`GlobalSection` is more tightly parameterized** (`GlobalSection{T,AT<:AbstractArray{T},λT}`) and
  copies its `Y` on construction. `update_section!` gained a four-argument form
  `update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B, retraction)` that writes into a separate destination; the
  three-argument form forwards to it in place.
- **An unimplemented retraction now errors** and names the retraction and the argument type. The
  fallback had an empty body, so it returned `nothing` and failed further downstream with a message
  about `Nothing`.

### Changed (breaking) — dependencies

- **Zygote is dropped** in favour of **ForwardDiff** — the unified interface differentiates a scalar
  objective of a small parameter set, which is what ForwardDiff is good at — and **ParameterHandling**,
  whose `flatten` / `unflatten` pair turns a `NamedTuple` of arrays into the vector the Euclidean
  methods want.
- **SimpleSolvers is now `0.12`** (was `0.8`), and the **minimum Julia version is 1.10** (was 1.0),
  which is SimpleSolvers 0.10's floor and the oldest version CI has ever tested.

  SimpleSolvers 0.9 reworked `Options`, so several keywords that used to be accepted here are gone:
  `g_restol`, `x_abstol_break`, `x_reltol_break`, `f_reltol_break` and `g_restol_break`. This does not
  change stopping behaviour at default settings — every removed `*_break` field defaulted to `Inf`,
  and the gradient residual is now gated on `f_reltol`, whose default `√eps(T)` is the number
  `g_restol` used to default to. The successive change in `f` is gated on the new `f_suctol`, which
  inherited `f_reltol`'s former default.

  SimpleSolvers 0.12 then arrived with the three defects this package had reported against it —
  **D3** (`Bisection` reporting success when it cannot bracket), **D4** (`maxlog` on the line-search
  warnings being per session rather than per solve) and **D6** (the polynomial searches extrapolating
  `α` without bound), all now closed and gone from the catalogue below — and two breaking changes, of
  which one reaches this package:

  - **A `LinesearchMethod` implements `solve_with_status`, and `solve` is derived from it.** The
    generic `solve_with_status` used to derive itself from `solve` and now raises instead, so a
    third-party method that implements only `solve` is broken. [`DecayingStatic`](@ref) implemented
    both and its body has moved into `solve_with_status`; the custom `solve` is deleted and the
    derived one — the call plus `linesearch_warnings` — replaces it exactly. This is not
    housekeeping: the old direction meant a method reached through `solve` emitted its messages from
    inside *every iteration* of a solve, which is the one thing the `LinesearchMethod` contract
    promises does not happen, and 0.12 removed the `maxlog` caps that used to hold that back.
  - `DecayingStatic` also honours a caller's `params.αmax` now, as `Static` does. It has no ceiling
    of its own — the schedule is the caller's to fix — but a caller that says no step above a given
    length is admissible means a scheduled one too.

  Nothing else here changed for it. `solver_step!` already called `solve_with_status` and never
  `linesearch_warnings`, so this package was already on the quiet channel that 0.12 makes structural,
  and it reads the outcome rather than the log (see `linesearch_rejected`). The whole test suite
  passes under 0.12 with no other edit.

### Fixed

- **A manifold solve bounds the step a line search may take (issues A1b, A15 and B3).**
  `solver_step!` now hands the search `params.αmax = c⋅2π/‖δ‖` through the new
  `linesearch_parameters`, with `c` the new `DEFAULT_STEP_CEILING` (`1`, i.e. never more than one full
  turn), settable per solve as `Optimizer(x, F; step_ceiling = …)` and disabled with `Inf`. Euclidean
  parameters get no ceiling and need none — neither an `AbstractVector`, where the field is omitted,
  nor a `NamedTuple` of ordinary arrays, where it is `Inf`.

  On a `NamedTuple` the ceiling is derived **per block** and minimised over the blocks that live on a
  manifold (`_manifold_αmax`), because the one `α` scales every block and each manifold block needs
  `‖αδᵢ‖ ≤ 2πc`. A block that is an ordinary array contributes none: `2π` is the turn of a rotation,
  and a vector space has no such scale either to impose or to tighten its neighbours with.

  **A line search bounds its step by the merit, and on a compact manifold the merit is bounded**, so
  that test never fires: `φ(α)` at `α = 1e9` can be genuinely *lower* than at `α = 0`, and a search
  reporting a decrease there is telling the truth. On the SVD problem `Quadratic` returned
  `α = 4.3e7` on a direction of norm `5.54` — a step of `‖αδ‖ = 2.4e8` — and the solve then reported
  *convergence* from a point no longer on `St(20, 3)`.

  Nothing about the direction was wrong at that step (`‖δ‖` in the same range as the two before it,
  `λmax(Q) = 3.86`), and two iterations later, with `Q` restarted to the identity so the direction
  *was* `-∇f`, the same search returned `8.6e7`. Both of the fixes this entry's own catalogue entry
  proposed for itself — an exact `Cayley` differential, then damping the quasi-Newton update — were
  implemented or measured and ruled out. The bound had to be on the **step**, and it is not a property
  of `φ`: it is the `2π` of a rotation over `‖δ‖`, which changes at every solver step.

  Hence the split. SimpleSolvers 0.12 takes the per-call ceiling (issue D6, filed from here) and
  stops its bracketing *at* it rather than clamping afterwards, so the merit is measured at the step
  handed back; this package supplies the value, because choosing it is a statement about the geometry.
  Upstream is explicit that its own `65536` backstop cannot close this: at `‖δ‖ ≈ 5.5` it still permits
  `‖αδ‖ = 3.6e5`.

  Over the eight starting points of `scripts/retraction_accuracy.jl`, counting solves that end with
  both factors still on the manifold (`check ≤ 1e-12`), ceiling off → on:

  | combination | before | after | worst `check` |
  |---|---|---|---|
  | `_BFGS` `Quadratic` `Cayley` | 4 of 8 | **8 of 8** | `3.2e-1` → `6.1e-14` |
  | `_BFGS` `BierlaireQuadratic` `Cayley` | 4 of 8 | **8 of 8** | `5.5e-1` → `6.9e-14` |
  | `_BFGS` `Backtracking(expand)` `Geodesic` | 7 of 8 | **8 of 8** | `2.8e-12` → `6.0e-14` |
  | `_BFGS` `Backtracking` `Geodesic` | 7 of 8 | **8 of 8** | `2.8e-12` → `6.3e-14` |

  All twenty combinations of that sweep are now 8 of 8, and the runs that used to exhaust a 20 000
  cap finish in 176 and 281 iterations. **The last two rows were never part of A1b** and are the same
  defect: that `2.8e-12` is the `_BFGS` + `Backtracking` + `Geodesic` seed 2 that `svd_optim.jl`
  singled out as "the worst of the eight by two orders of magnitude" and attributed to accumulation
  over 147 iterations. That attribution was wrong — it is one over-long step — and bounding the step
  removes it. A *shrink-only* `Backtracking` getting there at all is the surprise, and the answer is
  that its expansion phase can exceed `α = 1` while `‖δ‖` is what makes `α = 1` too far: the mechanism
  is not confined to the searches that extrapolate polynomials.

  Consequences for the rest of the package, all measured: the worst `check` over the whole sweep is
  `2.5e-13`, so `MANIFOLD_TOLERANCE` at `1e-12` clears it and the `1e-11`-or-`ProjectedSkew` caveat
  in `svd_optim.jl` is retired; the worst `rg` is `3.8e-7`, giving `CONVERGED_GRADIENT_TOLERANCE` a
  factor of 27. **Cost on the pinned seed: none.** Every figure `svd_tables()` regenerates, on both
  retractions, is reproduced to the digit with the ceiling on and off — the ceiling does not bind on
  seed 1234 at all, and what it buys is on the other seven.

  That last sentence was not true of the ceiling as first written, and the reason is issue **A15**,
  which the review of [#44] closed along with **B3**, three paragraphs down. Both were opened by this
  same change and shut within it, which is why they are described here rather than in the catalogue.
  Deriving the bound from `2π` over the norm of the *whole* direction combines the blocks of a
  `NamedTuple` in quadrature, so each block paid for its neighbours: on this problem, where both
  blocks are manifolds, that tightened it by up to `√2` and bound three `Geodesic` cells (`_BFGS`
  `Quadratic` 111 → 120 iterations, `_BFGS` `BierlaireQuadratic` 130 → 113, `_DFP` `Quadratic`
  175 → 308). They looked like the price of bounding the step and were the price of a sloppy norm.
  Worse, `ArrayNamedTuple` is *any* `NamedTuple` of arrays, so a manifold-free one was bounded by a
  rotation its problem does not have — measured at 3 184 iterations against 1 for the same Euclidean
  problem written as a vector. Per-block fixes both, and the three cells above go back to their
  no-ceiling values.

  One interaction with upstream, and it is why `linesearch_rejected` takes the ceiling as an argument
  (issue **B3**). Where the ceiling binds hard a search can report `LINESEARCH_FLOOR` — a claim about
  the *direction*, which the one-argument form acts on by restarting `Q` — when what was established
  is only that no *permitted* step decreases the merit measurably. A `LINESEARCH_FLOOR` returned at
  the ceiling is therefore exempt and its step is taken; below the ceiling it is a rejection as
  before, and `LINESEARCH_EXHAUSTED` and `LINESEARCH_NO_DESCENT` are never exempt, being true of the
  direction whatever ceiling was in force. Upstream declined to flag the capped case on the
  `LinesearchStatus` on the grounds that a caller who set the ceiling can compare it against
  `steplength` itself (issue D7); this is that comparison.

- **`Adam` and `MomentumMethod` no longer take the step a line search has rejected.** `solver_step!`
  restarts and searches along steepest descent when the outcome is `LINESEARCH_FLOOR`,
  `LINESEARCH_EXHAUSTED` or `LINESEARCH_NO_DESCENT`, and the two methods whose direction carries state
  were exempt from it. The reasoning was that `ensure_descent!` exempts them — a moving average is
  *allowed* not to descend on an individual step — and it does not carry over: a rejected search
  returns `α = 1` untouched, so the exemption did not permit a non-descent step, it took the
  **longest** step available along one.

  On Rosenbrock from `(-1.2, 1)` with `MomentumMethod(0.1)`, the solve reaches `f = 7.8e-5` by
  iteration 400 and is then ratcheted to `Inf` by thirteen such steps:

  | iteration | outcome | `α` | `f` |
  |---|---|---|---|
  | 441 | `LINESEARCH_NO_DESCENT` | 1.0 | 4.97e-2 → 4.65e3 |
  | 443 | `LINESEARCH_NO_DESCENT` | 1.0 | 8.16e1 → 5.33e9 |
  | 453 | `LINESEARCH_NO_DESCENT` | 1.0 | 1.46e3 → 4.41e19 |

  Each multiplies `f` by between `1e3` and `1e42`; the steps in between descend and cannot make it
  back. Iterations and final `f` on that problem, at a cap of 10 000:

  | | `Backtracking` | `Backtracking(expand)` | `BierlaireQuadratic` | `StrongWolfe(c₂ = 0.1)` |
  |---|---|---|---|---|
  | `MomentumMethod(0.1)` before | 379, **`Inf`** | 457, **`Inf`** | 6, **`Inf`** | 3 136, 2.7e-16 |
  | after | 2 921, **1.9e-16** | 2 693, **1.6e-16** | 355, **1.3e-16** | 3 254, 2.6e-16 |
  | `Adam` before | 10 000 (cap), 3.0e-5 | 275, 1.4e-16 | 1 941, 4.0e-17 | 2 846, 2.3e-16 |
  | after | **453**, 8.6e-18 | 256, 2.0e-20 | **285**, 1.7e-16 | **294**, 5.0e-17 |

  `GradientMethod` is bit-identical throughout — it is not in `FirstOrderMethodWithState`, so it was
  never exempt, which makes it the control. So is every `Static` figure, since that search reports no
  outcome to act on: `Static(0.1)` still diverges on Rosenbrock for both first-order methods, which is
  an over-large fixed step on a badly scaled problem and a different matter.

  **This also retires most of issue A9.** `Adam` + `BierlaireQuadratic` on the two-sphere problem of
  `test/manifold_linesearch_tests.jl` used to run out all 1 000 iterations under both retractions
  while sitting `6.8e-6` from the minimiser — at the answer, with no criterion it could meet. It now
  terminates in 251–286, and all fourteen (line search, retraction) combinations terminate on a
  criterion, in 251–331 iterations. The worst distance over the fourteen is `5.0e-7` and that one is
  `Static`; every searching line search is at `1.8e-8` or better. What survives of A9 is the cost
  argument, which is unchanged: `Adam` needs 251–331 iterations there where `GradientMethod` and
  `MomentumMethod` need 9–64, so `default_linesearch` keeps `Static` for `AdamFamily`.

  The momentum recursion is untouched: `p ← αp + ∇f` is evaluated in `update!(::MomentumState, …)`
  from `gradient_array(cache)` *after* the step, so which direction the step was taken along does not
  enter it. `ensure_descent!`, which acts on the direction before the search, still exempts both
  methods.

  Two things the open-issue entry for this (A7) had recorded and that measuring it did not bear out,
  kept here because they are the reason the fix is what it is:

  - **its `194 of 200` count of `LINESEARCH_NO_DESCENT` does not reproduce.** On the same problem,
    method and line search it is 3 of the first 200 and 13 of all 457. The direction does not ascend
    on almost every step; the searches overwhelmingly report `LINESEARCH_DECREASED` and the solve
    genuinely descends for 400 iterations. What the thirteen do is multiply `f` by `1e3`..`1e42`
    *each*, and the descending steps in between cannot make that back.
  - **neither guard it proposed would have fired.** "A bound on `‖p‖` relative to `‖∇f‖`" never
    triggers: that ratio stays between `0.57` and `6.8`, because it is the gradient that explodes and
    the momentum follows it. "A count of *consecutive* `LINESEARCH_NO_DESCENT` outcomes" never
    triggers either — the thirteen events are isolated. The question was never how often the
    direction ascends, but whether the step is taken when it does.

- **The line-search slope is the derivative of the line-search merit under `Cayley` too.**
  ``\varphi'(\alpha) = \langle\nabla{}f(x(\alpha)), B\rangle`` holds only where
  ``\alpha \mapsto \mathrm{retract}(\alpha{}B)`` is a one-parameter subgroup. `Geodesic` is one and
  `Cayley` is not, so pairing against the direction `B` regardless gave a slope that was exact at
  ``\alpha = 0`` and drifted from there. Against a central difference of the merit the search itself
  evaluates, on a `St(6, 3)` problem:

  | ``\alpha`` | 0 | 0.25 | 0.5 | 1 | 2 |
  |---|---|---|---|---|---|
  | paired with ``B`` | exact | 2.2% | 8.9% | 36% | **143%** |
  | with ``D(\alpha)`` | exact | 4e-10 | 1e-9 | 4e-9 | 4e-10 |

  The second row is the central difference's own truncation error at `h = 1e-6`, i.e. the slope is
  now exact to what the check can resolve. The new `retraction_differential` supplies the generator
  that turns with the step: with ``M = (\mathbb{I} - \frac{\alpha}{2}B)^{-1}``,
  ``\frac{d}{d\alpha}\mathrm{Cayley}(\alpha{}B) = MBM``, so the velocity of the trial curve is
  ``W(\alpha)(M^TBM)E`` where ``W(\alpha)`` is the frame the trial point was built in. Both inverses
  go through `lift_factors` and the Woodbury identity exactly as `cayley` does, so no ``N \times N``
  matrix is formed and the cost is ``O(Nn^2 + n^3)``. For `Geodesic` — and for an ordinary array under
  either retraction, where the retraction is addition — ``D(\alpha) = B`` and nothing is computed.

  **What is bit-identical**, verified over the ten (method, line search) combinations of
  `scripts/retraction_accuracy.jl` on the pinned seed and eight starting points: every `Geodesic`
  figure in the package, in every column; both `Backtracking` rows under `Cayley`, because
  `Backtracking` evaluates ``\varphi'`` at ``\alpha = 0`` only, where ``D(0) = B`` is returned
  untouched — so the default path costs nothing for this, in accuracy or in work; and, on the SVD
  problem, every `BierlaireQuadratic` figure, which is what shows that search never asks for
  ``\varphi'`` away from ``\alpha = 0`` there.

  **What moves** is `Bisection`, `StrongWolfe` and `Quadratic` under `Cayley`. Iterations and
  objective evaluations on the SVD problem, seed 1234:

  | | before | after |
  |---|---|---|
  | `_BFGS` + `Bisection` | 92 / 54 970 | 114 / 67 020 |
  | `_BFGS` + `StrongWolfe(c₂ = 0.1)` | 139 / 8 354 | 135 / 7 870 |
  | `_DFP` + `Bisection` | 96 / 56 106 | 110 / 64 306 |
  | `_DFP` + `Quadratic` | 550 / 54 176 | 529 / 50 656 |

  `Bisection` taking more iterations for a *correct* slope than for a wrong one is not a regression —
  it bisects ``\varphi'``, so a different slope is a different sequence of brackets — and its spread
  over the eight starting points tightens, 88–129 to 102–124 for `_DFP`. The clearest gain is
  `_DFP` + `Quadratic`, whose spread goes from 168–1 211 to 164–735.

  **It did not close issue A1b**, which is what it was written for, and that was the more useful
  result. A1b proposed an exact `Cayley` differential as the likely fix for `_BFGS` +
  `Quadratic`/`BierlaireQuadratic` running to the iteration cap and ending off the manifold on two of
  eight starting points. One of those four cases was fixed — `Quadratic` on seed 2 went from the
  20 000-iteration cap at `check = 5.0e-2` to **90 iterations at `check = 6.4e-15`** — and three were
  not. Ruling the slope out is what moved the diagnosis onto the *step*, which is where the cause was;
  see the step-ceiling entry above. This differential is worth having on its own account and is not a
  fix for A1b.

  Two things from the review of [#40], both in the *Retractions* page rather than here. That
  `α ↦ Cayley(αB̄)` is not a one-parameter subgroup is now *shown* and not asserted: everything in
  sight is a rational function of `B̄`, so `Cay(αB̄)Cay(βB̄)` and `Cay((α+β)B̄)` differ by a `tsB̄²` in
  both factors, and on a `St(6, 3)` lift of norm 2.99 the gap is `1.28` at `α = β = 1` against `8e-16`
  for `exp`. And the mechanism is the *parameterisation*, not the approximation, which `N = 2`
  separates completely: there `Cayley(αJ) = exp(2·atan(α/2)·J)` to the last bit, so the curve is the
  geodesic exactly and the slope is still wrong — by `1 + α²/4`, which is `D(α)` at `J² = -I`.

  That case is also how to read the percentages: to leading order the error is `α²λ²/4` for an
  eigenvalue `±iλ` of `B̄`, so it depends on the lift as much as on the step, and the same measurement
  gives 4.5% / 18% / 72% / 288% on the `St(3, 1)` sphere against the 2.2% / 8.9% / 36% / 143% quoted
  above for `St(6, 3)`. Figures here now name their problem for that reason.

- **`GradientMethod` and `MomentumMethod` no longer throw on their own default line search.**
  `Optimizer(ones(3), x -> sum(x.^2); algorithm = GradientMethod())` followed by a `solve!` was a
  `MethodError`: `trial_slope`'s `AbstractVector` branch calls `gradient(cache)`, which only
  `NewtonOptimizerCache`, `BFGSCache` and `DFPCache` defined, so none of the three first-order
  methods could use *any* line search that evaluates ``\varphi'`` on Euclidean vector parameters.
  That became the *default* path in 0.2.0, when `default_linesearch` started returning
  `Backtracking(T; expand = true)` for everything outside `AdamFamily`. Manifold parameters were
  never affected — that branch of `trial_slope` allocates and never touches the cache.

  Defining the accessor is not on its own enough, and the second half is the interesting one.
  `trial_slope` evaluates the trial gradient *into* the cache — that is what makes it
  allocation-free — while `update!(::MomentumState, ...)` re-runs ``p \gets \alpha{}p + \nabla{}f(x_k)``
  from the same array afterwards. Sharing one array made the momentum accumulate the gradient at
  whatever trial step the search last probed. Worst relative error in the state's momentum over eight
  iterations of ``f(x) = \sum(x^2 + 0.1x^4 + 0.3\sin 3x)``:

  | | `Static` | `Backtracking` | `Bisection` | `Quadratic` | `BierlaireQuadratic` | `StrongWolfe` |
  |---|---|---|---|---|---|---|
  | shared array | 0 | 0 | **1.04** | **1.04** | **4.51** | **1.04** |
  | scratch array | 0 | 0 | 0 | 0 | 0 | 0 |

  `Backtracking` is exact by accident: it evaluates ``\varphi'`` once, at ``\alpha = 0``, where the
  trial gradient *is* ``\nabla{}f(x_k)``. So the three first-order caches carry a scratch array of
  their own, reached through `latest_gradient`, and `cache.g` stays what the direction and the state
  updates need it to be.

- **The gradient residual is measured at the iterate a solve returns.** `rg` was
  ``\|\nabla{}f(x_k)\|`` at the iterate the step *started* from, for every method. Under `Static`
  that is harmless — the direction is a scaled gradient, so a vanishing gradient means a vanishing
  step — and under a direction that carries momentum it is not. A line search accurate enough to
  drive ``\nabla{}f(x_1) \approx 0`` made `g_converged` fire while the momentum term was still moving
  the iterate: `MomentumMethod` + `Backtracking` on ``f(x) = 1 + x^2`` from ``x = 1`` stopped after
  two iterations at ``x = -0.2`` reporting `rg = 0`, where ``\|\nabla{}f(x)\| = 0.4`` and the momentum
  was ``2``. Final ``\|x\|`` from `ones(3)`, before and after:

  | objective | method | `Bisection` | `Quadratic` | `BierlaireQuadratic` |
  |---|---|---|---|---|
  | ``1 + \|x\|^2`` | `MomentumMethod` | 0.346 → **0** | 0.346 → **0** | 4.5e-4 → **1.1e-8** |
  | ``1 + \|x\|^2`` | `Adam` | 1.16 → **0** | 1.16 → **7.7e-16** | 1.2e-5 → **9.5e-13** |
  | ``\sum\sqrt{1+x^2}`` | `MomentumMethod` | 0.122 → **3.9e-16** | 0.122 → **7.9e-14** | 0.122 → **1.1e-9** |
  | ``\sum\sqrt{1+x^2}`` | `Adam` | 1.16 → **3.9e-16** | 1.16 → **7.7e-16** | 0.104 → **4.5e-9** |

  `Adam` was the worse of the two — at ``\|x\| = 1.16`` it had barely left `ones(3)` — and both
  reported convergence. `solver_step!` now refreshes `latest_gradient` at the accepted iterate. It
  never cost an *iteration*: across the 42 (objective, method, line search) combinations measured the
  count is equal or one lower.

  **And it costs no gradient evaluation either**, which is not obvious and was not free. The refresh
  computes ``\nabla{}f(x_{k+1})`` in the frame of `section(cache)`, and that is bit-for-bit what
  `update!(cache, state, …)` recomputes at the top of the next step — `update_section!`'s
  three-argument method has the body the two-argument one `update!(::MomentumState, …)` uses, so the
  cache's frame after a step *is* the state's frame after `update!`. `store_gradient!` therefore
  reuses it instead of evaluating again, guarded on `solution(cache) == x` and
  `section(cache) == section(state)` so that a caller which moves the iterate between steps falls
  back rather than silently reusing a stale value. On the SVD problem of
  `test/optimizer_convergence/svd_optim.jl`, `Adam` + `Static` over 2 000 iterations (`err = 1.7097`
  in every column, i.e. one trajectory):

  | | before this release | refresh, no reuse | as shipped |
  |---|---|---|---|
  | wall clock | 124 ms | 167 ms (1.35×) | **128 ms (1.03×)** |
  | ``\nabla{}f`` per iteration | 1 | 2 | **1** |

  `rgₐ` is formed from the same two arrays — ``\nabla{}f(x_{k+1}) - \nabla{}f(x_k)``, the successive
  difference `OptimizerStatus` prints it as — so the two `g` rows of a status are about one step
  rather than about two different ones. It used to come from `state.ḡ`, which is two iterates behind
  `cache.g` for these three methods and is uninitialised on the first: on
  ``f(x) = \sum(x^2 + 0.1x^4)`` that reported `rgₐ = 4.976` where the successive difference is
  `0.295`. The remaining half of that — `Δf̃` still reads `state.ḡ` — is open issue A10 below.

  `Newton`, `_BFGS` and `_DFP` are untouched — `latest_gradient` defaults to `gradient(cache)`,
  nothing refreshes it for them and `store_gradient!` is not on their path. Verified bit-for-bit: 266
  solves over Rosenbrock, the SVD problem and a two-sphere `NamedTuple`, across all seven line
  searches, both retractions and all six methods, reproduce `rg`, the iterate, the value and `check`
  to the last digit; `rgₐ` moves for the three first-order methods and for nothing else.

- **…and for the (quasi-)Newton methods too, where it was not the previous iterate but whichever
  point the line search last probed.** `latest_gradient` aliased `cache.g` on `NewtonOptimizerCache`,
  `BFGSCache` and `DFPCache`, and `trial_slope` evaluates the trial gradient into it, so `rg` reported
  ``\|\nabla{}f\|`` wherever ``\varphi'`` was last asked for. The entry above left these three alone
  so that its own tables stayed bit-identical; this is what that deferred, catalogued as A8. On
  Rosenbrock from ``(-1.2, 1)``, `rg` against ``\|\nabla{}f\|`` at the iterate the solve returns:

  | | `Backtracking(expand)` | `Bisection` | `Quadratic` | `StrongWolfe(c₂ = 0.1)` |
  |---|---|---|---|---|
  | `Newton` before | `4.97e-14` vs `0` | 1.0× | 1.0× | 1.0× |
  | `_BFGS` before | **5.78e4×** | 1.0× | 1.0× | 1.0× |
  | `_DFP` before | **299×** | 1.0× | 1.0× | 1.0× |
  | all three after | 1.00× | 1.00× | 1.00× | 1.00× |

  The three bracketing searches came out exact only because their last ``\varphi'`` evaluation happens
  to land at the accepted ``\alpha``; the default `Backtracking` evaluates it at ``\alpha = 0`` only,
  which is the iterate the step *started* from. So the same `rg` meant a different point depending on
  which line search ran, and was not comparable across the rows of any table in this package.

  **It costs `_BFGS` and `_DFP` no gradient evaluation.** Their `update!(cache, state, grad, x)`
  recomputed `global_rep(section(state), grad(x))`, which is bit-for-bit what the refresh at the end
  of the previous `solver_step!` produced, so it goes through the same `store_gradient!` the
  first-order caches use. `NewtonOptimizerCache` goes through it too — all six caches now do — which
  is what makes `g̃_is_current` mean something there; it changes no count, because the frames `Newton`
  compares never agree inside a solve (that is A13). Over the twenty combinations of the eight-seed
  sweep in `scripts/retraction_accuracy.jl` the objective-evaluation count rises by exactly 10 per
  solve — one gradient evaluation on that problem, `GradientAutodiff` costing ten objective calls for
  its 60 parameters — and 10 is per *solve*, not per iteration: it is the refresh at the last iterate,
  the one no `update!` follows and so the one nothing reuses. `Newton` does pay one gradient
  evaluation per iteration (Rosenbrock, 103 → 124 over 25 iterations), because
  `update!(::NewtonOptimizerState, …)` advances the state's section by the *gradient* rather than by
  the direction, so the frames do not match and the guard correctly declines to reuse. That is the
  path already marked "this will have to be removed later" and it is left alone.

  **What moves.** A correct `rg` gates `g_converged` correctly, so a solve stops when it has met the
  criterion instead of overshooting it. Iterations: 19 of the 20 sweep combinations are unchanged
  (`_BFGS` + `StrongWolfe(c₂ = 0.1)` under `Geodesic`, 136 → 135, is the exception), and on Rosenbrock
  every method loses either zero or one. The stale value overestimated the residual near a minimiser,
  so the worst `rg` over the eighteen converging combinations × eight starting points improves from
  `2.5e-7` to `2.0e-7` — `CONVERGED_GRADIENT_TOLERANCE` has a factor of 50 of headroom now rather than
  40. `test/descent_direction_tests.jl` asserted `F(x) < 1e-27` on `sum(sin²)`, which was measuring
  the overshoot; what the criterion guarantees is ``\|\nabla{}F\| \leq \sqrt{\varepsilon}``, i.e.
  ``F \approx \|\nabla{}F\|^2/4 \approx 5.6\times10^{-17}``, and the worst of its 48 combinations is
  `4.2e-17`. It asserts `1e-15`, still fifteen orders below the `F = 3` maximum it exists to exclude.

  `gradient_difference!` moves to the same successive difference for all three caches, which **retires
  the known issue that `|g(x) - g(x')|` is structurally zero for `Newton`**: `solver_step!` advanced
  `state.ḡ` at the very iterate the cache took its gradient at, so the default could not have produced
  anything else. For `_BFGS` and `_DFP` it was the `γ` of the secant pair, one step behind the `rg`
  printed next to it.

  Every objective-evaluation count for the SVD problem in `default_linesearch`'s docstring and in
  `svd_optim.jl` is re-measured and ten higher. The *historical* before/after tables further up in
  this file are left as they were measured: their "before" column belongs to a code state that no
  longer exists, and moving only the "after" column would make them incomparable.

- **`x_converged` no longer fires on a solve that has diverged.** `rxᵣ = rxₐ / l2norm(cache.x)`
  measures "the iterate stopped moving" only while ``\|x\|`` is bounded. On the divergence removed
  above, a step of ``\|\delta\| = 345`` at an iterate of magnitude ``10^{100}`` gave
  ``rx_r \approx 3.4\times10^{-98}``, far under `x_reltol`, which is `2eps`, and the solve reported
  success. This was catalogued as A4 and documented rather than fixed, on the grounds that closing it
  needs a threshold on ``\|x\|`` that no property of the problem supplies. Two guards that need none:

  - the denominator is the new `solution_scale` rather than `l2norm`. On a manifold ``Y^TY =
    \mathbb{I}`` makes ``\|Y\|_F = \sqrt{n}`` *exactly*, so the scale is a constant the geometry
    supplies and a measured norm that is not that constant is itself the divergence signal; that trace
    reads ``345/\sqrt{3} \approx 199`` instead. A `NamedTuple` combines in quadrature, using the
    nominal scale for its manifold blocks and the measured one for the rest, and for Euclidean
    parameters `solution_scale` *is* `l2norm`.
  - `x_converged` also requires `!f_increased`. A vanishing step is evidence of convergence only if
    the objective did not just go up, and in that trace it went `3.38 → 9.13 → 1.2e169`. This is the
    only guard the Euclidean case has.

  `f_increased` itself was `abs(f) > abs(f̄)`, which is not "f increased" for an objective that takes
  negative values — `-5 → -6` is a decrease and read as an increase. It is `f > f̄` now. Nothing acted
  on the flag before (`allow_f_increases` defaults to `true`), so this was invisible; `x_converged`
  acts on it now.

  Measured over 264 solves — Rosenbrock, `sum(sin²)`, two Euclidean objectives, the `St(3,1)` sphere
  and a manifold `NamedTuple`, over up to six methods and six line searches each — **every numeric
  column is bit-identical**, and the eight-seed SVD sweep reproduces to the digit in all twenty
  combinations.
  What changes is eight reported flags, all on `sum(sin²)` solves whose last step raised `f` by
  round-off around the minimum and all of which keep `g_converged`, so no solve loses a stop or an
  iteration; and fourteen spurious `f_increased` flags on the negative-valued objective.

  What is *not* closed, and is now documented on `convergence_measures` rather than in the catalogue:
  a Euclidean solve that runs away **downhill** has no scale to be measured against and would still be
  reported as converged. The failure is unreachable from a solve in any case (`linesearch_rejected`,
  `curvature_is_usable`), which is why the regression test in the new
  `test/optimizer_status_tests.jl` builds the state it produced out of the same cache and state a
  solve hands to `OptimizerStatus`.

- **The `Geodesic` retraction no longer silently leaves the manifold for a large step.** `geodesic`
  built the exponential by summing ``\mathfrak{A}(X) = \sum_{n\geq1}X^{n-1}/n!`` directly. That series
  converges everywhere but is only *accurate* for a small argument, and the argument here is not
  small: the factorisation ``\bar{B} = B'(B'')^T`` puts ``\frac{1}{4}A^2 - B^TB`` in a block of
  ``X = (B'')^TB'``, so ``\|X\| \approx \|\bar{B}\|^2/4`` while its spectral radius is only
  ``\approx\|\bar{B}\|``. The terms cancel, and nothing reported it. `check(geodesic(B))` on a random
  `StiefelLieAlgHorMatrix(20, 3)`:

  | `‖B̄‖` | 5.8 | 17.8 | 36.5 | 78.8 | 160 | 361 | 767 |
  |---|---|---|---|---|---|---|---|
  | before | `2.1e-15` | `2.6e-12` | `4.4e-7` | `8.3e10` | `4.2e55` | `1.4e168` | `NaN` |
  | after | `1.1e-15` | `3.5e-15` | `7.9e-15` | `2.5e-14` | `9.0e-15` | `3.6e-14` | `7.7e-14` |

  At `‖B̄‖ = 79` the "retracted" point was not on the Stiefel manifold in any sense. `Cayley` was never
  affected — it inverts a `2n × 2n` matrix rather than summing a series.

  The fix is scaling and squaring, and it stays at `2n × 2n` throughout: the low-rank form is closed
  under squaring, `(I + B'W(B'')ᵀ)² = I + B'(2W + WXW)(B'')ᵀ`, so squaring the *exponential* is one
  application of `W ↦ 2W + WXW` and no `N × N` matrix is ever squared. It is also **faster** than what
  it replaces, by 1.6× at `N = 200, n = 10` and 4.6× at `N = 500, n = 50`, because the scaled series
  converges in a handful of terms where the unscaled one ground through hundreds.

  Consequences elsewhere. In `test/optimizer_convergence/svd_optim.jl` the worst `check` over eight
  starting points was `2.45e-5` (`_BFGS` + `Backtracking` + `Geodesic`), five orders of magnitude past
  that file's `MANIFOLD_TOLERANCE` of `1e-12`; it is `2.8e-12` now. Every `Geodesic` iteration and
  evaluation count in this package moved by a few percent, since a more accurate exponential is a
  different trajectory — the tables in `svd_optim.jl` and in `default_linesearch`'s docstring are
  re-measured, and `scripts/retraction_accuracy.jl` regenerates them. The `Cayley` columns are
  unchanged to the digit.

  `Geodesic` is now also the *cheaper* of the two retractions for `N ≳ 50`, which reverses what its
  docstring used to say: `cayley` finishes with a product of two `N × N` matrices at `O(N³)` where
  `geodesic` only assembles `I + B'𝔄(X)(B'')ᵀ` at `O(N²n)`. At `N = 1000, n = 20` that is 2.5 ms against
  39 ms.

- **`check` works on a `GrassmannManifold`.** It was defined for `StiefelManifold` only, so the one
  function that measures distance from the manifold — the assertion every manifold test rests on —
  was a `MethodError` for the other of the two manifolds this package provides. It is now
  `check(::Manifold)`, since the representative of a `GrassmannManifold` point satisfies `YᵀY = I`
  just as the Stiefel one does. This is why the accuracy loss above went unnoticed for so long: half
  the retraction paths had nothing that could have caught it.

- **`Optimizer`'s constructors no longer nest three levels of `kwargs...`, which cost Julia 1.12
  fifteen minutes of compile time per specialization.** The test suite ran for 31–42 minutes on 1.12
  across all three CI operating systems, against 3–5 minutes on 1.10, 1.13 and nightly, and
  `test/optimizer_convergence/svd_optim.jl` accounted for the whole difference
  (`Optimizer Convergence | 70 70 16m01.9s`, every other testset normal). It was never the numerical
  work: the identical solves driven from a script take 6.94 s on 1.12 and 7.04 s on 1.13, with the same
  iteration and evaluation counts.

  The trigger is having the `Optimizer` constructor and the `solve!` call in the *same* inferred
  function body — which is what any ordinary `function solve_my_problem(...)` does. Neither half is
  slow alone on 1.12: 0.99 s for the constructor, 2.35 s for `solve!`, 908 s for the two together.
  Inference has to resolve the `Core.kwcall` chain before it knows the constructor's return type, and
  on 1.12 propagating that into `solve!` goes superlinear.

  `Options` is now built once in the outermost method and passed positionally from there, so the
  return type does not depend on which keywords were given, and `Optimizer(x, F; …)` reaches the inner
  constructor through one level of splatting rather than three:

  | construction + solve in one body | 1.13 | 1.12 |
  |---|---|---|
  | before | 4.35 s | **940.86 s** |
  | before, behind a `@noinline` boundary | 4.57 s | 925.27 s |
  | after | 4.15 s | **6.71 s** |

  A 140× improvement on 1.12 and no change elsewhere. The second row is worth noting: a barrier around
  the construction does not help, and neither does `@nospecialize` on the enclosing function — only
  flattening the chain does, which is why the fix is where it is rather than at the call site. The
  regression itself is upstream and already fixed in 1.13 and nightly.

  No API change: every keyword `Optimizer` accepted, it still accepts.
- **A line search that reports it could not decrease the merit no longer gets its step taken.**
  `solver_step!` called `SimpleSolvers.solve`, which returns the step length and nothing else, so
  `LINESEARCH_FLOOR`, `LINESEARCH_EXHAUSTED` and `LINESEARCH_NO_DESCENT` were indistinguishable from a
  successful search. It now calls `solve_with_status`, and on any of those three outcomes restarts the
  inverse Hessian (`restart!`), sets the direction to steepest descent and searches once more. See
  `linesearch_rejected` for why the trigger is the outcome and not `φ > φ₀`.

  This was the `check(Y) = 1e200` divergence recorded in `test/optimizer_convergence/svd_optim.jl` and
  in commit `8673007`. On one of eight starting points, `_BFGS` + `Bisection` + `Geodesic` stopped
  after 4 iterations off the manifold altogether and reported *convergence*: `Bisection` bisects `φ'`,
  so on a non-convex ray it settled on a stationary point that was a maximum, said so
  (`LINESEARCH_FLOOR`, `φ(1) = φ(0)` exactly), and the step was taken anyway. `f` went from 3.38 to
  9.13, the corrupted secant pair gave the next direction `‖δ‖ = 345`, and retracting a lift that
  large left the manifold; `x_converged` then fired because `‖δ‖/‖x‖ ≈ 1e-98` once `‖x‖` was at
  `1e100`. That starting point now converges in 121 iterations at `check(Y) = 6e-14`.

  It also closes a second hole, which had been diagnosed as a tolerance problem. `@test status(result).rg < 1e-5`
  failed on CI at `1.354e-5`, and the assertion was unguaranteed rather than unlucky: over the ten
  (method, line search, retraction) combinations that test runs and eight starting points each,
  `g_converged` is `false` in **all eighty** — every solve terminates on `f_converged`, and `‖∇f‖`
  there ranged over `1.5e-8 .. 1.8e-5`. With the restart in place the worst case over the same eighty
  runs is `2.9e-7`, so `1e-5` has a factor of 35 of headroom and is now a real bound. ([#33])
- **The quasi-Newton update enforces the curvature condition.** The guard was
  `!iszero(ΔxΔg) && !isnan(ΔxΔg)`, which admits a negative `δᵀγ` as readily as a positive one — and
  admits denominators that are zero to within round-off, which it then divides a rank-two correction
  by. On the SVD problem `δᵀγ` took the values `-12.8`, `-4.5e-16` and `+1.5e-15` on consecutive
  iterations; `λmax(Q)` went from 3 to 442 as a result, and `λmin(Q)` to `-398` from another starting
  point. Both BFGS and DFP preserve positive definiteness of `Q` only for `δᵀγ > 0`, so
  `curvature_is_usable` now requires `δᵀγ > 1e-8 ⋅ ‖δ‖‖γ‖`. The threshold is relative because an
  absolute one cannot tell `1.5e-15` on a problem scaled to `1e0` from a legitimate pairing on a
  problem scaled to `1e-15`; its value barely matters, since `1e-8` and `eps(T)` behave identically.

  This is where `_DFP`'s notorious sensitivity to its starting point came from: an ill-conditioned `Q`
  on this problem is a `Q` built from pairs that should have been rejected. With an expanding
  `Backtracking` over eight starting points its iteration count goes from `512..77_890` to `512..845`
  on `Geodesic`, a factor of 92 less spread. It costs `_DFP` a factor of seventeen on Rosenbrock
  (50 iterations to 851, both reaching `f ≈ 3e-24`) because it had been exploiting those invalid
  updates to compensate for its under-scaled direction; `_BFGS` is unaffected at 22 either way.
- **A non-finite iterate stops the solve.** `meets_stopping_criteria` reported `NaN` through an
  `@error` and then carried on, so one starting point spent all 100 000 iterations of a raised cap
  emitting that message once per iteration. The flags are now part of the stopping condition, and none
  of the three convergence flags is set by it, so `isconverged` still distinguishes the two cases.
  `contains_nan` becomes `contains_nonfinite` and tests `isfinite`: `NaN` is the *last* thing a
  diverging solve produces, and the one on the SVD problem passed through `f = 1.2e169` and
  `check(Y) = 1.07e200` — neither of them `NaN` — two iterations before it got there. The
  `OptimizerStatus` fields `x_isnan`, `f_isnan` and `g_isnan` are renamed to `x_nonfinite`,
  `f_nonfinite` and `g_nonfinite` to match.
- **Gradients and directions are paired in the intrinsic coordinates, not the ambient ones.** `dot` on
  an `AbstractLieAlgHorMatrix` is the ambient Frobenius product, which counts each of the lift's
  off-diagonal blocks twice and so comes out *exactly twice* the product of its free parameters — the
  coordinates `Q` is sized by, `outer!` flattens to, and a line search's `α` parameterizes. The new
  `_dot` takes the pairing there instead, at three sites: `trial_slope`, which was returning
  `2φ′(α)`; the `δᵀγ` of the quasi-Newton update, whose value has to be consistent with the flattened
  `T₁`, `T₂` and `γᵀQγ` it divides; and the predicted decrease `Δf̃`, which has to be comparable with
  `Δf`. `_BFGS` on the SVD problem improves from 176 to 113 iterations (`Geodesic`, `Backtracking`)
  and from 197 to 93 (`Cayley`, `Bisection`).

  `trial_slope` remains exact only for `Geodesic`: `Cayley` is not a one-parameter subgroup, so
  `⟨∇f(x(α)), B⟩` is its derivative at `α = 0` and drifts from it with the step (about 6% at
  `α = 0.5`). This is documented on `trial_slope` rather than fixed.
- **`_DFP` keeps its inverse Hessian symmetric.** `Q γγᵀ Q` is symmetric in exact arithmetic, but
  forming it as two separate `mul!`s is not, and the error accumulated: `‖Q - Qᵀ‖/‖Q‖` grew from 8e-16
  after five iterations to 1.6e-11 after twenty thousand, at which point `eigvals(Q)` returned complex
  numbers for a matrix that is by definition symmetric. `BFGSCache` never had this, because it adds
  `T₁ + T₂` with `T₂` built as the exact transpose of `T₁`. Symmetrizing is not what makes `_DFP`
  converge, though — see the analysis in `test/optimizer_convergence/svd_optim.jl`.
- `zeros(SkewSymMatrix, n)` threw a `MethodError` about `zero(::Type{SkewSymMatrix})` — the
  non-parametric method had been dropped — including for its only in-repo caller,
  `zeros(::Type{StiefelLieAlgHorMatrix}, N, n)`. Both are public API.
- `Optimizer(x, problem)` sized its default gradient with `length(x)`, which for a `NamedTuple` is the
  number of *entries* and not the length of its flattening, so the first step failed with a
  `DimensionMismatch`. The choice now lives in `default_gradient`.
- `OptimizerCache` for a method with no matching cache — in practice an `Adam{Float64}` handed
  `Float32` parameters — reported a bare `MethodError`; it now says that `Adam` has to be constructed
  with the element type of the parameters.
- `GrassmannLieAlgHorMatrix` gained explicit `copy`, `zero`, `copyto!`, `fill!` and `similar` methods.
  The generic `AbstractArray` fallbacks route through `setindex!`, which it does not define, and its
  `similar` handed a two-parameter concrete type to `zeros`, which has no such method.
- `NewtonOptimizerCache(x, problem)` passed eight values for seven fields and would have thrown on its
  first call. Nothing calls it — it carries a "we probably don't need this constructor" comment — which
  is how it stayed that way; its field list is now spelled out rather than positional.

### Added

- **`Optimizer` takes a `step_ceiling`**, and `DEFAULT_STEP_CEILING` documents what it is for; see the
  A1b entry under *Fixed*. `linesearch_parameters`, `step_αmax` and `_manifold_αmax` are the internal
  helpers behind it, all in `src/optimizers/linesearch_problem.jl` next to `trial_iterate!`, whose
  dispatch on the solution type they mirror.
- **`scripts/retraction_accuracy.jl` reports how many seeds end on the manifold**, through the new
  `on_the_manifold`. That is the statistic A1b was about and the one its `worst check` column only
  implied — one bad seed and four bad seeds can give the same worst `check`, and it is the count that
  moved from 4 to 8. `solve_once` and `svd_tables` also take a `step_ceiling` keyword, so the
  before/after comparison every table in this release quotes is regenerated by the named harness
  rather than recalled: `svd_tables(step_ceiling = Inf)` is the "before" column throughout.

  Writing the `_BFGS + StrongWolfe(c₂ = 0.1)` row into `svd_optim.jl`'s table while re-measuring
  closes half of open issue C8 — the row was already being computed by every run of the sweep and
  printed to nobody, so adding it cost a line rather than the sweep time the entry assumed.
- **`Geodesic` takes an algorithm for the matrix exponential.** `Geodesic(ScaledSquaring())` is the
  default and is the fix described under *Fixed*; `AugmentedPade()` gets ``\mathfrak{A}`` out of a
  block of `Base.exp`'s Padé approximant, and `ProjectedSkew()` exponentiates the lift in an
  orthonormal basis of its own range, where it is a small skew-symmetric matrix and the exponential
  can be built from an eigendecomposition. All three compute the same map, so the one-parameter
  subgroup property `Geodesic` relies on holds for each.

  They differ in what they trade. `ProjectedSkew` is the only one whose orthogonality is
  *structural* rather than the outcome of a series, so it is the only one whose `check` does not
  degrade with the step: over the eight SVD starting points its worst case is `1.9e-13` against
  `ScaledSquaring`'s `2.8e-12`, and in `Float32` it holds `1.3e-6` where the others reach `2.3e-5`.
  It costs 1.2× to 1.8× on the *typical* case, measured over eight `(N, n)` pairs from `10, 2` to
  `1000, 20`, and needs `qr` and `eigen`. (This read "about an order of magnitude", which no
  measurement supports and which understates how attractive the algorithm is; it was catalogued as
  C6 and the figures above are that entry's.)
  `ScaledSquaring` is the default because it is the fastest, the most accurate against
  `exp(Matrix(B))`, and — using nothing but matrix products and norms — the only one of the three
  that runs on a `KernelAbstractions` GPU backend.

  `TaylorSeries()` is the pre-0.2.0 behaviour. It is kept, and documented as unusable, so that the
  regression stays reproducible from the test suite rather than only from this file.

  The algorithm reaches every entry point, not just `retraction(::Geodesic, ::AbstractLieAlgHorMatrix)`:
  `geodesic(Y, Δ, algorithm)` takes it as an optional third argument, so the tangent-vector form is
  selectable too rather than being pinned to the default.

  Described, with the theory of both retractions and the advantages and disadvantages of each
  algorithm, in the new *Retractions* documentation page. Its accuracy tables are recomputed when the
  documentation is built rather than quoted, so they cannot go stale the way the *typical case* figure
  above did (C6). `scripts/retraction_accuracy.jl` sweeps the same eight lifts from the same seed, so
  the page and the script print the same rows — and both include `Cayley`, whose `check` drifts
  further with the size of the lift than any of the three usable algorithms.

- **`Options(store_trace = true)` does something.** `OptimizerResult` gains a `trace` of one
  `OptimizerTraceEntry` — `(iteration, f, rg)` — per iteration, reachable through `trace(result)` and
  empty unless the option asked for it. The option existed before and was accepted and ignored, by
  this package *and* by SimpleSolvers, where it is still a field of `Options` that nothing reads as of
  0.12 (open issue D2): code
  that set it got neither a trace nor an error.

  The entries come from the `OptimizerStatus` that `solve!` already computes on every iteration, so
  the cost when the option is unset is one `Bool` test per iteration.

  What it is for is a statistic that does not depend on the *phase* of an orbit. `Adam` at a fixed
  learning rate does not converge to the minimizer, it circles it at a distance of order `α`, so the
  error at any single iteration is a sample of an arbitrary phase and moves with the last bits of the
  floating-point arithmetic — across Julia 1.10, 1.12 and 1.13 the final-iterate error on the SVD
  problem spans a factor of 3.0. Averaging over the last five hundred iterations measures the orbit's
  *radius* instead, which is a property of `α` and the problem: the same three versions span 1.06.
  That is what `test/optimizer_convergence/svd_optim.jl` now asserts on, at a tolerance with 1.8× of
  margin above the worst correct value and more than 1000× below the value the known `Adam` bugs
  produce. The tolerance it replaced had a factor of 1.9 to work in altogether.
- **`AdamWithEuclideanDecay`**, `Adam` with the *decoupled* weight decay of Loshchilov and Hutter
  (2019) — the method usually called AdamW. The decay `λx` is applied to the *direction* after the
  moments have been formed, so it never enters `m₁` or `m₂` and is scaled by the line search's `α`,
  exactly as the paper's schedule scales it. It shares `Adam`'s cache and state:
  `OptimizerState(AdamWithEuclideanDecay(), ps)` returns an `AdamState`.

  The name says *Euclidean* because `λx` is the gradient of `λ‖x‖²/2`, which is *constant* on both
  manifolds of this package (`‖Y‖²_F = tr(YᵀY) = n`, both being compact), so its Riemannian gradient
  `rgrad(Y, λY)` vanishes identically. The method therefore decays the ordinary arrays of a
  `NamedTuple` and leaves the manifolds in it alone — the case it exists for, a network with Stiefel
  weights next to unconstrained ones ([#14]) — and on a *bare* manifold it *is* `Adam`, for every `λ`.
  That last case warns at construction rather than being silently ignored. Whether a *Riemannian*
  weight decay should exist alongside it is [#28]; the name `AdamW` is held in reserve for the answer
  and, for now, raises an explanatory error rather than aliasing `Adam`. Derived in the new
  *Weight Decay on Manifolds* documentation page.

  This is **not** the replacement for the `AdamWithDecay` of v0.1.0 (see *Removed*) — that decayed the
  learning rate, and its replacement is `DecayingStatic`. The two are unrelated and compose: passing
  `linesearch = DecayingStatic(…)` decays both the step and, with it, the decay.
- **`OptimizerSolution`**, the union of `AbstractVector`, `Manifold` and `ArrayNamedTuple`, so that a
  single method signature covers all three — together with `named_tuple_wrapper.jl`, which supplies
  the elementwise arithmetic (`_copy`, `_zero`, `_add!`, `_mul`, `norm`) that the Euclidean code
  performs on plain vectors.
- **Heterogeneous and non-`Float64` parameter `NamedTuple`s.** `ArrayTuple` and `GlobalSectionTuple`
  were written as `Tuple{Vararg{AT}} where {AT<:…}`, which Julia's diagonal rule makes homogeneous — a
  `NamedTuple` holding a `StiefelManifold` and an ordinary `Matrix` at the same time was therefore not
  an `ArrayNamedTuple`. They are covariant now.
- `GradientFunction(F, ∇F!, nt::NamedTuple)`, so that a hand-written gradient can be used for
  parameters stored in a `NamedTuple`.
- `GlobalSection` now travels with the optimizer state rather than living inside the cache, with the
  accessors and in-place methods that `solver_step!` needs.
- `default_gradient` and `default_linesearch`, which name the two choices `Optimizer` makes when the
  caller does not supply a gradient or a line search.
- **`DecayingStatic`**, a `LinesearchMethod` that takes no search but whose step decays geometrically
  from `η₁` to `η₂` over `n` iterations, reading the iteration number out of the line search
  parameters. This is the replacement for the `AdamWithDecay` of v0.1.0 (see *Removed*), and being a
  line search it composes with `GradientMethod` and `MomentumMethod` too. It is the weaker of the two
  ways to make `Adam` converge: a geometric schedule is summable and so stops short (`‖∇f‖ ≈ 1e-3`
  once `‖Δx‖ ≈ 1e-13`), where letting a searching line search pick the step gets `Adam` to `≈1e-7`.
- **`AdamOptimizerWithDecay(n_epochs, T; η₁, η₂, kwargs...)`**, a convenience pairing of `Adam` with a
  `DecayingStatic` line search, returned as a `NamedTuple` to splat into `Optimizer`:
  `Optimizer(x, F; AdamOptimizerWithDecay(1000)...)`. It defines no type and no schedule of its own —
  `kwargs...` goes to `Adam`, so `β₁`, `β₂` and `δ` keep `Adam`'s defaults rather than a copy of them.
  It exists so that GeometricMachineLearning's method of the same name — which bundles Adam's `ρ₁`,
  `ρ₂`, `δ` with `η₁`, `η₂`, `n_epochs` and computes the identical
  `γ = exp(log(η₂/η₁)/n_epochs)` — has one name to migrate to now that the direction and the step
  size live in different halves of the optimizer. The name carries over but the call does not always:
  GML takes `T` from `η₁`, whose default is a `Float32` literal, so a `Float32` network needs
  `AdamOptimizerWithDecay(n_epochs, Float32)`; and GML's positional step sizes and its `ρ₁`/`ρ₂`
  spelling are keywords `η₁`, `η₂`, `β₁`, `β₂` here. Note that this decays the *learning rate*, not
  the weights; `AdamWithEuclideanDecay` is the unrelated one, and the *Two unrelated decays* section
  of the weight-decay page sets them side by side.
- **A line search can take its trial step through the retraction.** `linesearch_problem` built its
  merit with `SimpleSolvers.compute_new_iterate!`, i.e. `xₖ + α·pₖ`, which on a manifold is undefined —
  and would be wrong even if it were not, since a step has to go through the retraction and the
  direction is a horizontal lift of a different shape than the point. `Static`, the one method that
  never evaluates the merit, was therefore the only line search that worked on manifold parameters.
  `trial_iterate!` now builds the trial point the way `solver_step!` does, and `trial_slope` pairs the
  gradient with the direction through `global_rep`; both dispatch on the solution type, so the
  `AbstractVector` path keeps its allocation-free `compute_new_iterate!`.

### Known issues

Tracked issues open against this release. For the defects and gaps found *while* building it —
verified, measured, and not fixed here — see [Open Issues](#open-issues) at the end of this file.

- Type piracy in `l2norm`, `ParameterHandling.flatten`, `Gradient` and `outer!`, catalogued in the
  source. `ArrayNamedTuple` and `GlobalSectionNamedTuple` are type *aliases* for `NamedTuple`, so
  dispatching on them is dispatching on Base; a wrapper `struct` would make most of it legal. ([#16])
- Bare `Manifold` parameters are only partially supported ([#27]). In particular a
  `GrassmannManifold` cannot be optimized over *at all*, as a bare point or inside a `NamedTuple`.
  **Closed in [0.2.2](#022)**; the catalogue entry it pointed at (A11) is described
  there, under the change that fixed it. What is left of [#27] is `mode = :finitediff`, which has no
  `GradientFiniteDifferences` method for a bare `Manifold` of either kind, and `default_gradient`,
  which has no `Manifold` method and silently takes its `AbstractArray` one (A20).
- `mode = :finitediff` throws a `MethodError` on `NamedTuple` parameters ([#24]) — and, as the
  entry above records, on a bare `Manifold` too.
- No documentation page describes the unified optimizer interface yet ([#25]).

## [0.1.0]

Initial release. Ports the manifold optimizer machinery from GeometricMachineLearning and the
Euclidean optimizer machinery from SimpleSolvers into one package.

## Open Issues

Defects and gaps found while working on this release that are **not** fixed by it, kept here so they
are tracked with the code rather than in a scratch file. A is correctness in this package, B its
observability, C its dead code and bookkeeping; D is upstream, E lists things reported during the
investigation that turned out not to be problems, and F the loose ends of the geodesic-retraction
review. Everything was verified directly — where a claim rests on a measurement, the measurement is
given. Entries A5, A6 and D5 come from the review of [#36]; A10 from the review of [#38]; C7
from the line-search work of this release, A12 and C8 from the review of [#40], A13, A14 and C9 from
the work on A4 and A8, A16, C10, C11, D7 and D8 from moving to SimpleSolvers 0.12 and closing A1b, and
A17, C12 and C13 from the review of [#44], A18, A19, C14 and C15 from the review of [#45] and A20
from the review of [#46]; the rest from unifying the optimizer hierarchies. A21 is older than any of
them — it comes from the MNIST port of [#14] and was catalogued only when that material left the
repository, which is this section's own case made once more: a finding kept in a file beside the work
that found it leaves with that work.

Of that last group, **D7 is the one worth reading**, and it is the one that no longer bites here: it
and B3 were two halves of a path on which a sound quasi-Newton direction is discarded because the
*caller* bounded the step, and the downstream half is closed. The rest are a default whose one
calibration problem stopped constraining it (A16), an obvious test that is not run (C10), an
explanatory gap now stated as one rather than guessed at (C11), a docs link lost to an upstream
inventory (D8), and three small things the review of [#44] left behind (A17, C12, C13).

The [#45] group has a shape of its own worth naming: three of its four entries are things that were
*not verified* rather than things that are wrong. A19 is a documented property that nothing here
exercises, C15 is a fix argued from inferred types where the measurement covers only part of it, and
C14 is a duplication whose cost depends on A19. Only A18 is a defect you can point at. That is the
residue of a review that widened a fix on the strength of a static argument, and the argument is
sound — but it is worth knowing which claims in [0.2.1](#021) rest on a measurement and which rest on
reading a type.

The [#46] group is one entry, and how it was found is the transferable part. That PR closed A11 and
stated what it left open under [#27]; the statement was taken from [#27]'s own text, which names two
bullets, and only the first of them was re-checked. Running the second against merged `main` is what
turned up A20. **A PR's account of what it leaves behind is a claim like any other in this file, and
the issue it is quoting is not the code** — where an entry says "what remains is X", X is worth
executing before it is written down.

**Only open entries are listed here.** A1, A1b, A2, A3, A4, A7, A8, A9, A15, B3, C6, D3, D4 and D6 were
in this catalogue and have been fixed; each is now described in [0.2.0](#020)
above, under the change that fixed it, and the entry here is gone. C3 and C4 went the same way in
[0.2.1](#021), A11 in [0.2.2](#022), and **D9** in 0.6.0 — it is the shortest-lived entry this file
has had, opened and closed within one release, and its account now sits under the change that closed
it. The labels are *not* reused and the surviving ones are not renumbered, so the gaps are
deliberate: A5, A6 and A10 mean what they have always meant, and a reference to A1b, A4, A11, A15,
B3, D6, D9 or C6 in a commit message still resolves to the right subject.

Each entry says what it would take to fix it — look for **What to do**. That used to live in a
`plan.md` that sequenced the catalogue into PRs and was never tracked here (it was excluded
locally), so it drifted from the catalogue twice and was invisible to anyone who cloned the
repository — which is the failure mode the paragraph above describes, one level up. It is folded into
the entries it planned and deleted. The one thing it had that no entry does is the sequencing: the
observability entries (B1, B2, C1) are one PR because they are one subject, C3 and C4 were one small
PR, and anything that moves a number wants its own PR and its own re-measurement.

A1b took three attempts and is worth remembering for the shape of it rather than for the fix. Two
remedies were proposed from a correct-looking diagnosis and both were implemented or measured before
being ruled out — an exact `Cayley` differential, then damping the quasi-Newton update — because the
thing that looked wrong (the retraction, then `Q`) was in each case an *amplifier* rather than the
cause. What settled it was instrumenting the one quantity nobody had looked at, `‖αδ‖`, and finding
that the direction was unremarkable and the step was nine orders of magnitude too long. **When two
plausible fixes in a row fail to move the symptom, stop proposing a third and go and measure the
thing that is actually large.**

**Six measurements in this catalogue and its tables have now failed to reproduce**, each corrected in
place or before its entry moved: A1b's first-order `check` table, A7's `194 of 200` outcome count,
A1b's step-by-step trace of the divergence — found in the review of [#40], and an order of magnitude
out on two of its three rows — the `_DFP + Backtracking` row of `svd_optim.jl` (see C8) and a `Cayley`
spread in `default_linesearch` that quoted the pinned value as its upper bound, both found while
re-measuring for A8; and, found while fixing A1b, `svd_optim.jl`'s attribution of its worst `Geodesic`
`check` to "accumulation over 147 iterations", which was one over-long step. The *conclusion* held
every time except the last, and the supporting figures never did, which is a consistent enough pattern
to state as a rule: **treat a number here as reproducible only where the harness that produced it is
named**, and prefer a figure that a committed script or test regenerates to one that was measured once
at a REPL. Where a measurement needs code that is not in the package — the A1b trace needed `α`,
`‖δ‖` and `λmax(Q)`, none of which is observable from outside `solver_step!` — say so and say what was
changed to get it. C9 records how much of this package's measurement apparatus is still in that
category.

This is the detailed catalogue. The short, issue-tracker-facing list is *Known issues* under
[0.2.0](#020) above; the two do not overlap.

Ordered by severity within each section, except that entries found after the first pass are appended
to their section rather than interleaved, so that the labels stay stable references.

---

### A. This package — correctness

#### A5. `geodesic(Y, Δ)` and `cayley(Y, Δ)` are not deterministic and consume global RNG state

**Severity: medium.** Found in the PR #36 review, when an equality assertion between two calls of the
same retraction failed. Pre-existing on `main`; PR #36 neither causes nor fixes it.

Both tangent-vector entry points open with `GlobalSection(Y)`, and `global_section(::StiefelManifold)`
(`src/manifolds/stiefel_manifold.jl:126-133`) completes `Y` with a **random** basis of its orthogonal
complement:

```julia
A = KernelAbstractions.allocate(backend, T, N, N - n)
randn!(A)
A = A - Y.A * (Y.A' * A)
typeof(Y.A)(qr!(A).Q)
```

Two consequences, both measured on `St(20, 3)`:

- **The same call twice gives different answers.** Consecutive `geodesic(Y, Δ)` differ by `1.2e-14`
  in Frobenius norm. That is round-off, not a wrong answer — the retracted point genuinely does not
  depend on the section, and *that* is what the difference measures — but it means no test can assert
  equality between two retractions of the same input, only `isapprox`. Anything downstream that
  hashes, caches or bitwise-compares a retracted point is unsound.
- **It perturbs the global RNG stream.** After `Random.seed!(11)`, taking one retraction changes the
  next `rand()`. So a seeded run is reproducible only if the number of retractions taken is also
  fixed — which it is not, under any line search that adapts its trial count. Verified directly:
  `seed!(11); geodesic(Y, Δ); rand()` ≠ `seed!(11); rand()`.

The optimizer path is mostly insulated, because the section is built once at initialization and
thereafter parallel-transported by `update_section!`. The exposure is the direct `geodesic(Y, Δ)` /
`cayley(Y, Δ)` API and anything built on it.

A section is not unique, so drawing one is legitimate; taking it from the *global* RNG without the
caller being able to see or supply it is the problem. An `rng` argument threaded through
`GlobalSection`, or a deterministic completion (a Householder completion of `Y`, which needs no
randomness at all), would close it.

#### A6. `ScaledSquaring` takes about twice the squarings it needs

**Severity: low**, and a refinement rather than a defect — from the PR #36 review, where it was
deliberately left alone because changing it invalidates every table in that PR.

`ScaledSquaring`'s own docstring states the cause: `X = (B'')ᵀB'` has `‖X‖ ≈ ‖B̄‖²/4` while its
spectral radius is only `≈ ‖B̄‖`, because the eigenvalues of `X` are the nonzero (purely imaginary)
eigenvalues of the skew `B̄`. The halving count is taken from the norm, `s = ⌈log₂(‖X‖₁/θ)⌉`, so
`s ≈ 2log₂‖B̄‖` where `log₂‖B̄‖` would do.

Each squaring is an error amplification, so this costs both time and accuracy. Measured forward error
is `1e-14` in `Float64`, which is why it is not urgent — but `Float32` `check` reaches `2.3e-5` over
the sweep, and halving `s` is the obvious lever if that ever becomes the constraint.

The constraint on any replacement bound is that `ScaledSquaring` is the default *because* it is free
of scalar indexing and dense LAPACK, which is what lets it run on a GPU backend (see the `opnorm₁`
docstring). That rules out reaching for the spectral radius directly — an eigenvalue computation
would give the tighter bound and forfeit the reason the algorithm was chosen.

`NativePade`, added in [#54], takes `s` from ``\|X\|_1`` in exactly the same way and inherits the
whole of this, so the entry now covers two algorithms and a fix would apply to both at once.

#### A10. `state.ḡ` is two iterates behind for the three first-order states

**Severity: low** — everything it still reaches is reported and not acted on. Found in the review of
[#38], which fixes the half of it that had become visible and leaves the rest. **Pre-existing on
`main`.**

`GradientState`, `MomentumState` and `AdamState` are advanced by `update!(state, opt, x)`, which runs
*after* the step. It writes the *post*-step iterate into `state.x` and the cache's *pre*-step gradient
into `state.g`, shifting the one before that into `state.ḡ`:

```
after update! at the end of step k:   state.x = xₖ   state.g = ∇f(xₖ₋₁)   state.ḡ = ∇f(xₖ₋₂)
```

So `state.g` does not belong to `state.x`, and `state.ḡ` is two iterates behind `cache.g` rather than
one. The quasi-Newton states do not have this: `update!(::BFGSCache, …)` advances `state.ḡ` itself,
inside the step, right after forming `γ` from it.

Two consumers, both in `OptimizerStatus`:

- `rgₐ = ‖cache.Δg‖`. **Fixed in [#38]**: the first-order caches now override `gradient_difference!`
  and take `latest_gradient - gradient`, which is the successive difference the status prints and
  needs no `state.ḡ`. On `f(x) = Σ(x² + 0.1x⁴)` from `[1.5, -0.8, 0.4]` with `MomentumMethod` +
  `Bisection` the old value was `4.976` at iteration three where the successive difference is
  `0.295`; on iteration one it differenced against the `_similar` memory these states never write,
  which is the same defect `test/optimizer_state_initialization.jl` exists to catch for the `Adam`
  moments.
- `Δf̃ = ⟨state.ḡ, δ⟩` (`optimizer_status.jl:82`), the first-order predicted decrease. **Not fixed.**
  It is a two-step-stale gradient paired with the current direction, so the prediction it makes is
  not one. Its only reader is `f_converged_strong`, which C1 records as computed and discarded — so
  whichever way C1 goes, this has to be settled with it, and settling it separately would be
  measuring a number nothing looks at.

The honest fix is upstream of both: `update!(state, opt, x)` should store the gradient that belongs
to the `x` it is storing. `latest_gradient` is exactly that gradient and is already in the cache. The
obstacle is that the same call site feeds the momentum recursion `p ← αp + ∇f(xₖ)`, which needs the
*pre*-step gradient and must keep getting `gradient_array(cache)` — so the two uses have to be
separated first, and `update!(::MomentumState, …)`'s argument list says they currently are not.

---

#### A12. The `Cayley` differential is recomputed per `φ'`, and its cost is unmeasured

**Severity: low**, and not a defect — a cost this release introduced and did not measure. Found in
the review of [#40], where `retraction_differential` was added.

Under `Cayley`, `trial_slope` now calls `retraction_differential` on every evaluation of ``\varphi'``.
That is `lift_factors`, a `StiefelProjection`, two ``2n\times{}2n`` solves and about six allocations —
``O(Nn^2 + n^3)``, the same order as the retraction itself — where before it was a `_dot` against an
array the cache already held. `Geodesic` returns ``\bar{B}`` untouched at every ``\alpha`` and
`Cayley` does at ``\alpha = 0``, so every `Geodesic` solve and the `Backtracking` default pay nothing;
what is unmeasured is a search that evaluates ``\varphi'`` many times per iteration, which on this
problem is `Bisection` at ≈580 objective evaluations per iteration.

**The iteration and evaluation counts in `svd_optim.jl` do not answer this.** They moved under the
change — `_BFGS + Bisection` under `Cayley` from 92 to 114 iterations — but they moved because the
trajectory changed, so they measure a different solve rather than the cost of a step. Nothing here
is a wall-clock measurement.

The obvious remedy if it does turn out to matter is not a cache but a shared factorisation:
`linesearch_problem`'s `d(α, params)` calls `trial_iterate!` and then `trial_slope` with the *same*
``\alpha``, and both go through `lift_factors` — the first on ``\alpha\bar{B}`` and the second on
``\bar{B}`` — so one line search evaluation factors the same lift twice. Fusing them would need
`trial_iterate!` to hand its factors on, which is a wider change to that interface than a cost
nobody has measured justifies.

---

#### A13. `Newton`'s state advances its frame by the gradient, not by the step

**Severity: low** — it costs an evaluation and not an answer. Found while fixing A8, which is what
made the difference visible: with the gradient reuse in place, `Newton` is the one method that cannot
have it.

`update!(state::NewtonOptimizerState, opt, x)` (`src/optimizers/optimizer.jl`, the line already
marked "this will have to be removed later") ends with

```julia
update_section!(state.section, gradient_array(cache(opt)), x -> retraction(opt.retraction, x))
```

i.e. it advances the state's `GlobalSection` by ``\nabla{}f`` where every other state
advances it by the direction the step was taken along. For Euclidean parameters `update_section!` is
``\Lambda^t.Y \gets \Lambda^{t-1}.Y + B``, so this is not a formality: after a step the cache's frame
holds ``Y + \delta`` and the state's holds ``Y + \nabla{}f``.

Two consequences, both of them about `store_gradient!`'s reuse guard, which requires
`section(cache) == section(state)`:

- **`Newton` pays one gradient evaluation per iteration that `_BFGS` and `_DFP` do not.** The frames
  do not match, so the guard declines the reuse and the cache evaluates ``\nabla{}f`` afresh at the
  point `refresh_latest_gradient!` has just evaluated it at. Measured on Rosenbrock from
  ``(-1.2, 1)`` with `Backtracking(expand)`: 103 gradient evaluations over 26 iterations before A8,
  124 over 25 after.
- **The reuse comes back by accident once the iteration has nowhere left to go.** Once ``\nabla{}f``
  and ``\delta`` have both gone to zero the two frames agree again and the guard fires. That is
  *correct* — on Euclidean parameters `global_rep` is the identity, so the value depends only on
  `solution(cache) == x`, which the guard also checks — but it means the branch taken is not a
  property anything should assert on. `test/optimizer_tests.jl` asserts on the gradient the direction
  is built from instead, and says so. Measured by calling `latest_gradient_is_current` at the top of
  each of 30 forced `solver_step!`s: the guard fires on 3 of the 30 on Rosenbrock — all of them past
  the iteration the solve stops at, which is why the evaluation count above does not move — and on 28
  of the 30 on ``\sum{}x^2``, where `Newton` reaches the minimiser in one step and both quantities are
  zero from then on.

The fix is to advance the state's section by `direction(cache(opt))` like every other state, after
which the reuse is available to `Newton` too and the extra evaluation goes away. It needs a
re-measurement of the `Newton` rows and nothing else — `NewtonOptimizerCache` is `AbstractArray`-only,
so no manifold path reaches this.

---

#### A14. `x_converged` still cannot see a Euclidean solve that diverges downhill

**Severity: low** — no measured solve reaches it. The remainder of A4, which is otherwise fixed
above; kept here because the hole is real and because the next person to look at
`convergence_measures` should find it named rather than have to re-derive it.

The two guards A4's fix installed are a *scale* (`solution_scale`, exact on a manifold because
``\|Y\|_F = \sqrt{n}``) and the *objective* (`x_converged` requires `!f_increased`). A Euclidean
solve whose ``\|x\|`` grows without bound while `f` decreases at every step defeats both: nothing
bounds ``\|x\|``, so ``\|x - x'\|/\|x'\|`` still goes to zero for a step that is not small, and the
objective never gives the second guard anything to act on. It would be reported as converged.

The divergences this package has actually produced all went *uphill* (`3.38 → 9.13 → 1.2e169`) or
straight to non-finite, so the guards cover them; and both of those causes are fixed at their source
(`linesearch_rejected`, `curvature_is_usable`). What would close it is a scale fixed at the start of
the solve rather than read off the current iterate — and the obvious candidate, ``\|x_0\|``, fails on
the two commonest starting points: it is `0` at the origin, and it is the wrong scale entirely for a
solve that legitimately travels a long way. That is the same "no property of the problem supplies a
threshold" this entry inherits from A4; it is narrower now, and it is not gone.

---

#### A16. `DEFAULT_STEP_CEILING = 1` is calibrated against one problem

**Severity: low.** From the step-ceiling work that closed A1b.

The *shape* of the ceiling is not in question — `c⋅2π/‖δ‖` is what the geometry gives, and `c = 1`
says "never more than one full turn", which is an argument rather than a fit. What is thin is the
evidence for that particular `c`. It rests on the SVD problem: over the converging solves there the
largest `‖αδ‖` is `2.03`, comfortably inside `2π`, and every one of the twenty combinations in the
sweep is unmoved or improved by the ceiling. Upstream's own table brackets it from the other side —
`αmax = 10` leaves 7 of 8 seeds on the manifold and `αmax = 1` leaves 8 of 8 — but that is the same
problem again.

The per-block ceiling that closed A15 makes this *thinner*, not less thin, and that is the reason to
keep the entry open. Since the ceiling stopped being tightened by a block's neighbours, it no longer
binds anywhere on the pinned seed at all — every figure in `svd_optim.jl`'s table is now identical
with the ceiling on and off. So the SVD problem no longer bounds `c` from below in any way: it says
only that `c = 1` is loose enough there, and nothing about where it would start to cost.

So there is no measurement of a problem on which `c = 1` is too *tight*, and one plausibly exists: a
solve whose direction is systematically under-scaled wants a large `α`, which is exactly the `_DFP`
story that motivated `Backtracking(expand = true)`. `_DFP` passes here, so the two do not collide on
this problem; nothing says they cannot.

**What to do**: measure `c ∈ {0.5, 1, 2, 4, Inf}` across the sweep and record where the iteration
counts start to move. If they move at `c = 1` on any row, the default is too tight and the entry
becomes a real defect; if they move only below it, the default is justified and this entry closes with
a table. `svd_tables(step_ceiling = …)` takes the keyword already, so this is a loop and not new code.
Note that the sweep can no longer answer the *upper* half of that question — see the paragraph above.

---

#### A17. `_manifold_αmax` pairs solution and direction blocks positionally

**Severity: low**, and an unasserted assumption rather than a live defect. From the review of [#44],
where the per-block ceiling that closed A15 was written.

`_manifold_αmax(values(sol), values(direction(cache)), c)` walks the two tuples in step and decides
from `sol`'s block whether `δ`'s block is a manifold direction. It therefore assumes the two
`NamedTuple`s have the same keys in the same order, and it checks nothing.

The assumption holds, and holds by construction: the cache's direction is built from the solution by
`_similar`, which is `Base.map` over the `NamedTuple`, and `map` throws
`ArgumentError: Named tuple names do not match.` on keys that differ *or are merely reordered*, then
rebuilds the result under the first argument's keys. Verified directly on a mixed
`(Y::StiefelManifold, W::Matrix, b::Vector)` problem: the direction comes back as
`(Y::StiefelLieAlgHorMatrix, W::Matrix, b::Vector)` with `keys` equal.

That argument is stronger than it was when this entry was written. It used to rest on `apply_toNT`'s
hand-rolled `@assert keys(ps[1]) == keys(p)`, and `@assert`'s own docstring warns that it "might be
disabled at various optimization levels"; Base's check is part of how `map` constructs the result and
cannot be compiled out. `apply_toNT` was
deleted in [0.5.0](#050), which is where all 30 of its call sites became
`map`.

What is not good about it is that this is the one place in the package that pairs two block
structures *without* going through the construction that checks. Everything else — `_copyto!`,
`_difference!`, the `GlobalSection` copies — is a `map` and would fail loudly. If a future cache
ever built its direction some other way, the ceiling would silently be derived from the wrong block,
which is a wrong `αmax` and not an error.

**What to do**: cheapest is one `@assert keys(sol) == keys(δ)` where the cache is constructed, so the
invariant is stated once and costs nothing per solver step. Routing `_manifold_αmax` itself through
`map` is the *obviously* correct version and is why it was not done: it builds a `NamedTuple`
of per-block ceilings, i.e. an allocation on every line-search call, to compute one scalar.

---

#### A18. `𝔄` and `𝔄exp` accept an `AbstractExponentialAlgorithm` they cannot serve

**Severity: low**, and a signature that is wider than the implementation rather than a wrong answer.
From the review of [#45], where `𝔄exp` was added.

`𝔄(X, algorithm)` is implemented for [`TaylorSeries`](@ref), [`ScaledSquaring`](@ref) and
[`AugmentedPade`](@ref). [`ProjectedSkew`](@ref) is the fourth `AbstractExponentialAlgorithm` and has
no `𝔄` method at all: it specialises `geodesic` directly, because it exponentiates the lift in an
orthonormal basis of its range rather than going through ``\mathfrak{A}`` — see the *Disadvantages*
paragraph on `docs/src/retractions.md`, which already tells readers that `𝔄(X, ProjectedSkew())` does
not exist.

The signatures `𝔄(B̂, B̄, ::AbstractExponentialAlgorithm)` and `𝔄exp(B̂, B̄, ::AbstractExponentialAlgorithm)`
nevertheless accept it, so `𝔄exp(B̂, B̄, ProjectedSkew())` dispatches, forwards, and dies one frame in
with a `MethodError` naming `𝔄` — not the function that was called, and not the fact that this
algorithm lives a level up. `𝔄exp` inherits the hole rather than adding one, and narrowing only
`𝔄exp` would put the two out of step, which is why it was left as it is.

**What to do**: either give `𝔄` a `ProjectedSkew` method that errors with the explanation — that it
is a `geodesic`-level algorithm, and to call `geodesic(B, ProjectedSkew())` — which fixes both
entry points at once and costs one method; or introduce the subtype of `AbstractExponentialAlgorithm`
that the three ``\mathfrak{A}``-level algorithms share and narrow both signatures to it, which makes
it a `MethodError` at the call site instead of a frame in. The first is cheaper and says more; the
second is the one that makes the type hierarchy match what is implemented.

---

#### A19. `ScaledSquaring`'s GPU claim is untested here

**Severity: unknown, which is the point.** From the review of [#45]. **Half of it is fixed** — see
*Fixed* under [0.4.2](#042) — and this is the half that is not.

The documentation states the property in three places and rests the default on it:
`docs/src/retractions.md` calls [`ScaledSquaring`](@ref) one of the two usable algorithms that run
unchanged on a `KernelAbstractions` GPU backend and the default because it is the cheaper of them,
and `docs/src/manifold_optimizers.md` repeats it. `GeometricOptimizers.opnorm₁` exists solely to keep
it: its docstring says `LinearAlgebra.opnorm(X, 1)` "is a scalar-indexing double loop, and scalar
indexing is exactly what a GPU array cannot serve."

Two things sat against that, both verified by reading. The first turned out to be a defect rather
than a doubt, and is fixed:

- `𝔄(A)` — the series `ScaledSquaring` sums, on the ``2n\times{}2n`` argument — opened with
  `Aⁿ = one(A)` and `𝔄A = one(A)`. `Base.one(::AbstractMatrix)` is `Base._one` in
  `base/abstractarray.jl`, which does `similar`, `fill!` and then **a scalar-indexed loop over the
  diagonal**. So the path was not "nothing but matrix products and norms"; it reached the same
  construct `opnorm₁` was written to avoid, one level down. `AbstractLieAlgHorMatrix` had a
  `Base.one` of its own that is a KernelAbstractions kernel precisely to avoid this — but `𝔄`'s
  argument is a bare matrix, so that method did not apply to it. (It was written for
  `StiefelLieAlgHorMatrix` alone and moved to the abstract type in [0.2.2](#022), which is why the
  Grassmann retraction was on the scalar-indexed path as well until then; that was *a piece of* this
  entry and not a fix for it.) **Fixed in [#54]**, which is where the claim also stopped being
  untested against an array type that errors on scalar indexing. What remains is the second bullet,
  and a `JLArray` does not settle it — it reproduces the *failure mode* of a GPU backend, not the
  backend.
- **No run in this repository exercises it**, and there is no GPU code left here to change that:
  `mnist_cuda.jl`, `mnist_metal.jl`, `mnist_metal_short.jl` and `metal_memory_probe.jl` were all of
  it, and they moved to [GMLDatasets.jl](https://github.com/JuliaGNI/GMLDatasets.jl) with the rest of
  the MNIST material. None of the five MNIST scripts passed `retraction`, so all of them took the
  `Optimizer` default, which is `Cayley()` — the same fact recorded under F below. The 6 h 53 min
  RTX 4090 run whose figures that package's documentation carries therefore never called `geodesic`,
  `𝔄` or `ScaledSquaring` at all.

What is *not* established is whether this actually breaks. CUDA.jl and Metal.jl error on disallowed
scalar indexing, which would make it a hard failure rather than a slow one, but no GPU was available
to the review and the claim is not being called false — only unverified, with a specific reason to
doubt it and a one-line way to find out.

**What to do**: run `geodesic(60 * rand(StiefelLieAlgHorMatrix{Float32}, 20, 3) |> gpu)` on a CUDA or
Metal backend, with `NativePade` alongside it. The identity half of this is done — `𝔄` builds it the
way `one(::AbstractLieAlgHorMatrix)` does, through the shared `unit_matrix` — so what is left is the
transcript, which is worth having given that three documentation passages depend on it and that the
one thing already found here was found by reading rather than by running. Do it together with C14,
which decides where the identity is assembled.

---

#### A20. `default_gradient` has no `Manifold` method and silently takes the `AbstractArray` one

**Severity: medium**, and narrow — it is unreachable through the constructor anyone actually calls.
From the review of [#46], found by checking that PR's claim about what it left open against merged
`main` instead of against the wording of [#27]. Pre-existing; [#46] neither causes nor fixes it.

`default_gradient` has two methods (`src/optimizers/optimizer.jl:175-176`):

```julia
default_gradient(problem::OptimizerProblem{T}, x::AbstractArray) where {T} = GradientAutodiff{T}(problem.F, length(x))
default_gradient(problem::OptimizerProblem, x::ArrayNamedTuple) = GradientAutodiff(problem.F, x)
```

`Manifold <: AbstractMatrix`, so a bare manifold takes the first. That builds the gradient from the
*length* and composes `problem.F` with a flat vector, where the point of `GradientAutodiff(F, ::Manifold)`
— added in [0.2.2](#022) at `src/utils.jl:30` — is that it rebuilds the manifold before
calling `F`. The `NamedTuple` method is already the delegating form, and the reason it exists is
recorded in `default_gradient`'s own docstring one screen up: a `Gradient` for a `NamedTuple` "has to
be constructed from `x` itself". A bare manifold is the same argument and did not get the same
treatment.

It does not raise where it is built. It raises at the first gradient evaluation, and only on an
objective that names its argument type — so an `F(Y) = -tr(Y'MY)` written without the annotation
computes something plausible off the flattened vector and never says anything. Reproduced on `main`
at 6166479:

```julia
F(Y::GrassmannManifold) = -tr(Y' * M * Y)
x = rand(GrassmannManifold{Float64}, 3, 1)

default_gradient(OptimizerProblem(F, x), x)(x)
# MethodError: no method matching F(::Vector{ForwardDiff.Dual{…}})

GradientAutodiff(F, x)(x)          # the method [#46] added, for comparison
# [-0.28344900891714625; -0.660038737272924; -0.4099740609429401;;]
```

The same on a `StiefelManifold`, which is why this is not a Grassmann entry: it is the half of [#27]'s
second bullet that survived A11.

Nothing in the test suite reaches it. `Optimizer(x, F; …)` builds its own gradient at
`src/optimizers/optimizer.jl:244` and never consults `default_gradient`; the constructor that does is
the lower-level `Optimizer(algorithm, problem, hessian, cache, linesearch; gradient = default_gradient(problem, cache.x), …)`
at `:156`, and `_optimizer` at `:201`. So the gap is real and unreachable by the documented entry
point at the same time, which is the reason it outlived a PR that fixed everything around it.

**What to do**: one method,

```julia
default_gradient(problem::OptimizerProblem, x::Manifold) = GradientAutodiff(problem.F, x)
```

next to the two above, and a test that goes through the `:156` constructor with an objective that
annotates its argument — the annotation is the part that matters, since without it the wrong gradient
is silent rather than loud. Worth doing together with the other half of [#27], `mode = :finitediff`
(`:246`), which has no `Manifold` method either and no `NamedTuple` one ([#24]); the three are one
subject, which is "every entry point that builds a gradient should agree about what a manifold is".

---

#### A21. The optimizer interface cannot hold GPU arrays

**Severity: medium**, and a regression rather than a gap: `GeometricMachineLearning`'s optimizers ran
on `CUDABackend()`, and the port of [#14] could not keep that. Found by that port, recorded in the
`MNIST_PORT.md` it wrote, and moved here when the MNIST material left (see [0.3.1](#031)
above) — this entry is that file's finding restated against current `src/`, not a new measurement.

**This entry loses the sharpest half of its first bullet in
[0.5.0](#050).** The scalar-indexing fallback it described was
`ParameterHandling`'s, and that dependency is gone: `NeuralNetworkParameters.flatten!` writes each leaf
with one `copyto!(v, doffs, x, firstindex(x), n)` over a range the layout already knows, and indexes no
element, so there is nothing left for a device array to fall through *to*. What is restated below is
what survives, which is a transfer per step rather than an error.

The parameters of a GPU run stay on the host. Two independent things put them there:

- **The per-step flattening, now as a transfer rather than a scalar-indexing fallback.**
  `(grad::Gradient{T})(nt::ArrayNamedTuple{T})` (`src/optimizers/named_tuple_wrapper.jl:21`) flattens
  on *every* gradient evaluation, and `flatten` allocates its destination as a
  `Vector{T}(undef, length(layout))` — a **host** vector, whatever the leaves are. `unflatten` is the
  same boundary in reverse: it slices `v[l.range]`, reshapes, and hands `rebuild` a host array, so the
  parameter set it returns is host-resident even if the one it was built from was not. A device run
  therefore pays a download and an upload of the whole parameter set per step, which is the cost
  measured below — a transfer that works rather than scalar indexing that does not, but the parameters
  still do not stay resident.
- **The state.** `_similar(a::Manifold{T}) = rand(manifold_constructor(a){T}, size(a)...)` at `:46`
  goes to `rand(manifold_type, N, n)` (`src/manifolds/abstract_manifold.jl:110`) and from there to
  `rand(CPU(), …)` at `:43` — always the host, whatever `a` is. It backs `x̄` and the `BFGS`/`DFP`
  caches, so even a flattening that stayed on the device would leave the state mixing host and device
  arrays. Untouched by the dependency change.

What this cost the run it was found in is small: the optimizer touches only the parameters — 154938
of them, 620 kB in `Float32`, so ≈1.2 MB uploaded and downloaded per step — against ≈3 GB of
device-side activations in the forward and backward passes, which stayed on the device throughout.
That is why the port left it, and it is a statement about that network and not about the interface.
A parameter set large enough to be worth keeping resident would pay the transfer on every step.

The rest of `src/` is written against `KernelAbstractions` and *looks* backend-agnostic; whether it
is has not been established, and A19 above is one specific reason to doubt it.

**What to do**: the flattening is now `NeuralNetworkParameters`' to fix rather than this package's —
what it needs is a `flatten` that allocates its destination on the backend of the parameters, and a
`FlatParameters` that keeps it there, which is an upstream request and not a method here. On this side
what is left is to thread the backend through `_similar` —
`rand(backend, MT{T}, N, n)` already exists (`src/manifolds/abstract_manifold.jl:70`, allocating
through `KernelAbstractions` at `:28`), and `KernelAbstractions.get_backend` is how `global_section`
and `Base.zero` already find the backend of a point they are given
(`src/manifolds/stiefel_manifold.jl:128,143`). Neither is large, and neither is worth doing blind:
the check is a GPU run of the optimizer, which nothing in this repository does any more, so this
should be closed together with A19 — one backend, one session, both claims settled.

---

### B. This package — observability

#### B1. A line search failure is invisible in the returned status

PR #35 makes `solver_step!` act on `LINESEARCH_FLOOR` / `LINESEARCH_EXHAUSTED` /
`LINESEARCH_NO_DESCENT`, but nothing records that it happened. `OptimizerStatus` has no field for it
(verified: no reference to the outcome in `optimizer_status.jl`), so a solve that needed a
quasi-Newton restart on half its iterations is indistinguishable, in the object the caller gets,
from one that never needed one. Only a `verbosity ≥ 2` log message with `maxlog = 1` shows it.

The `MomentumMethod` runaway is no longer the argument for this that this entry claimed when that was
open as A7. That divergence was 13 rejected outcomes
over 457 iterations rather than 194 over 200, and it is fixed by *acting* on them rather than by
counting them — so the counter is not a guard against anything, it is what would have made the
thirteen visible. It is still worth having on its own terms: after that fix, a solve that needed a
steepest-descent substitution on a quarter of its iterations is *still* indistinguishable, in the
object the caller gets, from one that never needed one.

**This entry has now lost its worked example, and did not lose its point.** It read that `_BFGS` +
`Quadratic` + `Cayley` on seed 8 of the SVD problem takes the steepest-descent branch on 4 780 of its
20 000 iterations and says nothing about any of them. That solve no longer exists: the step ceiling
that closed A1b brings that whole row inside 176 iterations. No replacement figure is quoted here
because there is no
way to get one from outside `solver_step!` — which *is* the defect, one level down, and it is why the
fix below would be its own instrument. Do not replace the number by reasoning about it.

**What to do**: one counter, `linesearch_restarts`,
and not one field per outcome — the question a caller has is "did this solve need help". Accumulate
it on the *state*, which persists across iterations as `iterations` already does, rather than
widening the `OptimizerStatus(state, cache, f; config)` constructor, which is where `solver_step!`
would otherwise have to thread it.

#### B2. `show_trace` and `extended_trace` are still accepted and ignored

PR #35 implements `store_trace`. The other two remain dead in this package *and* upstream (verified:
no reads of either in `SimpleSolvers/src/` outside `options.jl`'s struct, constructor and `show`).
Setting them gets neither output nor an error. Both are now cheap, since the per-iteration record
exists.

**What to do**: `show_trace` prints the record every `config.show_every`
iterations; `extended_trace` adds `rxₐ` and `Δf` to `OptimizerTraceEntry`. Small, and it retires the
last two silently-ignored options. Belongs in one PR with B1 and C1 — all three are the same subject,
the status object not saying enough.

---

### C. This package — dead code and bookkeeping

#### C1. `f_converged_strong` is computed and thrown away

`convergence_measures` computes it at `src/optimizers/optimizer_status.jl:312` and returns it at
`:319`; the `OptimizerStatus` constructor destructures it at `:135` and never stores it. Nothing else
in the package mentions it. It is `Δf ≤ f_mindec ⋅ Δf̃`, i.e. an Armijo-style sufficient-decrease test
on the *outer* iteration, so it plausibly belongs with the stall detection that `Options.max_stalls`
and `Options.f_stall_window` were meant to drive — both of which are also unread here.

Whichever way this goes, it has to be settled together with A10: `Δf̃` is its only input, and for the
three first-order methods that is a two-step-stale gradient paired with the current direction. Using
`f_converged_strong` without fixing A10 would be acting on a prediction that is not one; deleting it
retires A10's second consumer along with it.

**What to do**: either *use* it as the stall detector `Options.max_stalls` and
`Options.f_stall_window` were meant to drive — count consecutive iterations that fail it, stop after
`max_stalls` — or *delete* it from `convergence_measures`' return tuple and say in the docstring that
the outer-iteration Armijo test is not implemented. Deleting is the honest default. Using it is worth
more but is a behaviour change that needs its own measurement over the eight starting points, so it
must not ride along in an observability PR; split it out if that is the choice.

#### C2. `compute_direction!` for the quasi-Newton methods is dead

`src/optimizers/iterative_hessians/iterative_hessians_direction.jl:1-3` defines
`compute_direction!(opt, ::Union{BFGSState,DFPState})`. There is no call site (verified: the only
live callers are the `Newton` methods in `newton_optimizer_direction.jl`). The direction is formed
inline at the end of the cache `update!` instead — `bfgs_cache.jl`, `dfp_cache.jl`.

**What to do**: delete the file and its `include`. The alternative — routing
`solver_step!` through `compute_direction!` for symmetry with `Newton` — is a refactor with no
behavioural gain, and the inline form is what the `ḡ`-ordering fix depends on. Deleting is the
smaller and clearer change.

#### C5. `_DFP` + `Backtracking(expand = true)` is documented rather than run, on stale grounds

`test/optimizer_convergence/svd_optim.jl` excludes that pair because its iteration count ranged
`512..77_890` over eight starting points. The curvature condition in PR #35 brings that to
`385..1_118` (`Geodesic`) and `466..1_177` (`Cayley`), comfortably inside the 5 000 cap the file
already uses, so the stated reason no longer holds. Left out only because there is no CI measurement
of the post-fix spread yet — the original surprise was a factor of four between platforms.

**What to do**: add the pair to the driver loop in `svd_optim.jl` at the existing 5 000 cap and
rewrite the comment that explains why it is not run. Gate it on having seen one green CI run on
Linux *and* Windows, because the local spread does not measure the thing that surprised us.

---

#### C7. `ensure_descent!` is vacuous for `GradientMethod`

**Severity: low**, and currently harmless. Found while writing `steepest_descent!`, which exists
because of the same aliasing.

`ensure_descent!` tests `dot(rhs(cache), direction(cache)) > 0`. That works because `rhs` is
``-\nabla{}f`` on the (quasi-)Newton caches — but on the three first-order caches `rhs` is defined as
an *alias* for `direction` (`gradient_optimizer.jl:82`, `momentum_optimizer.jl:63`,
`adam_optimizer.jl:72`), so the test reads `dot(δ, δ) > 0` and is true for every `δ` that is neither
zero nor `NaN`.

Of the three, only `GradientCache` reaches it: `MomentumMethod` and `Adam` are `FirstOrderMethodWithState`
and `solver_step!` skips the call for them deliberately. So `GradientMethod` runs a safeguard that
cannot fire. It is harmless *today* because that method's direction already is ``-\nabla{}f``, which
always descends — the safeguard has nothing to catch. It is worth recording because the harmlessness
is a property of the direction and not of the guard: anything that changes what `GradientCache` puts
in `direction` would silently lose the check rather than start failing it.

The same aliasing had a second consequence that *was* live, and is fixed in this release: the
steepest-descent substitution after a rejected line search was written as
`_copyto!(direction(cache), rhs(cache))`, which is a no-op on these three caches. See
`steepest_descent!`.

---

#### C8. `svd_optim.jl`'s table and the script's `COMBINATIONS` are not the same ten rows

**Severity: low**, bookkeeping. Found in the review of [#40], while making the "regenerated by
`scripts/retraction_accuracy.jl`" claim in that table true.

Both are ten (method, line search) pairs and eight of the ten agree. The two that do not:

- **`_DFP + Backtracking`** is in the table and not in `COMBINATIONS`. That is deliberate — at 48 322
  iterations on the pinned seed it would dominate the runtime of every sweep, and its only purpose in
  the table is the `α = 1` ceiling argument below it — and it now says so in place. Its spread
  (`10_448..114_116`) is an older measurement at a cap high enough not to bind, which is why it
  exceeds the `SVD_MAX_ITERATIONS = 20_000` the rest of the column is quoted against.
- ~~**`_BFGS + StrongWolfe(c₂ = 0.1)`** is in `COMBINATIONS` and not in the table.~~ **Closed.** It
  was measured on every run of the sweep and printed to nobody; the row is in the table now
  (`135 / 135` iterations, `7 893 / 7 880` evaluations), written down while re-measuring for the step
  ceiling. The two options this entry offered were to add the row or drop it from `COMBINATIONS`, and
  adding it turned out to cost one line rather than the sweep time the entry assumed — it was already
  being computed.

So one of the two discrepancies is closed and the other is documented rather than removed. The general
point stands: a table that names a script as its source should be checkable against that script row by
row, or say which rows are exceptions and why. `svd_optim.jl` now marks its three non-regenerated
cells explicitly, which is what makes the remaining gap safe.

**The first bullet is why this is worth doing rather than filing.** Running the A8 sweep found that
the `_DFP + Backtracking` row had been *wrong* on `Geodesic` since the curvature-condition fix —
47 115 iterations and 1 177 919 evaluations where the same harness measures 48 322 and 1 208 157, and
where `default_linesearch`'s own table said 48 322 all along. Neither table is regenerated by the
sweep, so nothing compared them. Both are corrected and both now name the harness and the cap; a row
that no script regenerates is a row that goes stale silently.

---

#### C9. Most of the harnesses these figures come from are not in the repository

**Severity: low**, and the direct cause of every stale figure this catalogue has had to correct. The
sequencing plan this catalogue absorbed listed five measurement scripts under `/tmp/go_diag/` and
said they "should be moved somewhere durable before relying on them". One of the five was:
`scripts/retraction_accuracy.jl`, which regenerates the SVD tables and the exponential-accuracy
tables. The rest are gone, and each round of work since has added another.

What is quoted somewhere and has no committed harness:

- **the flag-and-residual probe** — 264 solves over Rosenbrock, ``\sum\sin^2``, two Euclidean
  objectives, the `St(3,1)` sphere and a manifold `NamedTuple`, across six methods and six line
  searches, reporting iterations, gradient-evaluation counts, `rg` against ``\|\nabla{}f\|`` computed
  outside the optimizer, and the four status flags. It is what showed A4 to be numerically inert and
  what measured A8's `5.8e4×` table, and it is the only thing that would catch either regressing.
- **the Rosenbrock iteration counts** that `ROSENBROCK_MAX_ITERATIONS` is set against.
- **the `Adam` cross-version statistic**, quoted in `svd_optim.jl` as a 1.06× spread over three Julia
  versions and reproducible only by running three Julia versions.
- **the 1.12 compile-time reproducer** (D1), which no longer exists at all.
- **the wall-clock timings** in `default_linesearch` — `0.155 s` against `0.246 s` on `Geodesic`,
  `0.205 s` against `0.451 s` on `Cayley`, and the "1.6× to 2.2× faster" they support. `svd_tables`
  reports iterations and evaluations and not time, so the step-ceiling round regenerated every other
  figure in that docstring and left these four untouched. They now say so in place, which is the
  minimum this entry asks for and not a fix.
- **the MNIST run**, as of [0.3.1](#031) above: the 6 h 53 min RTX 4090 figures that A19
  and A21 rest on, the ``\sqrt{1.8} \approx 1.342`` plateau and the per-configuration losses are
  still quoted here, while `distill_mnist_results.jl` and the five scripts that produced them are now
  in GMLDatasets.jl. This is the one entry on the list whose harness *exists* and is merely elsewhere,
  which makes it the mildest case and the easiest to get wrong: a figure quoted in this repository
  and regenerated in another one goes stale exactly as quietly, and nothing here will notice.

The pattern is C8's worked example: a number that no committed script regenerates is a number that
goes stale silently, and the ones this catalogue has caught were all of that kind. Either put them
under `scripts/` next to `retraction_accuracy.jl`, or accept the rule the preamble already states and
stop quoting figures that nothing can re-run.

---

#### C10. The eight-seed sweep is now inside `MANIFOLD_TOLERANCE` and is still not a test

**Severity: low**, and it is the cheapest open item here.

`svd_optim.jl` used to say that enabling the eight-seed sweep as a test would need either
`ProjectedSkew` or a tolerance of `1e-11`, because the worst `check` over the eight was `2.8e-12`
against a `MANIFOLD_TOLERANCE` of `1e-12`. The step ceiling removed that outlier — it was one
over-long step and not the accumulation the file attributed it to — and the worst `check` over all
twenty combinations and all eight seeds is now `2.5e-13`. **The stated obstacle is gone.**

What is in the suite instead is the four A1b cases at seeds 2 and 8, added with the ceiling. That
covers the defect that was found and not the twenty-by-eight surface the sweep measures, which is
where both A1b and the previously unnoticed `Geodesic` 7-of-8 rows were found in the first place —
neither by the pinned seed the suite otherwise runs.

**What to do**: run the sweep in CI, not in `Pkg.test()` — at `SVD_MAX_ITERATIONS = 20_000` it is
minutes rather than seconds, which is why it is not simply added to the driver loop. A scheduled
workflow asserting `on_the_manifold(...) == 8` for every row and `rg < CONVERGED_GRADIENT_TOLERANCE`
would have caught A1b years earlier than a sweep someone remembered to run. Gate on one green run per
platform first, as C5 asks for the same reason.

---

#### C11. The `Quadratic` `Geodesic`/`Cayley` gap is unexplained, and now explicitly so

**Severity: low**, an explanatory gap and not a defect. Recorded because a wrong explanation for it
was removed this round and nothing replaced it.

`_DFP` + `Quadratic` takes 175 iterations under `Geodesic` and 529 under `Cayley`. `default_linesearch`
used to attribute that to `trial_slope` being only first-order correct under `Cayley`; the exact
`retraction_differential` disproved it (the figure moved 550 → 529 and the gap stayed). This round
removed the second candidate too: the step ceiling turned out to have nothing to do with it either —
`175 vs 529` with the ceiling on is the same pair as with it off, since the per-block ceiling does not
bind on this seed. Both docstring and docs page now say the remainder is unexplained rather than
offering a third guess.

An intermediate version of the ceiling *did* move the `Geodesic` figure, to 308, and it would have
been easy to read that as the third explanation. It was an artefact of combining the two manifold
blocks in quadrature (issue A15) and went away when the ceiling became per-block. Worth recording as a
near miss of exactly the kind the A1b preamble warns about.

That is the right state to be in — see the preamble on A1b, where two plausible-looking explanations
in a row were the expensive part — but it is a loose end, and the pattern that resolved A1b applies:
stop proposing mechanisms and instrument the quantity that differs. Here that is the sequence of
brackets the fit is built on, which is not observable from outside SimpleSolvers.

**What to do**: nothing, unless the gap starts to matter. `Quadratic` is not a default under either
retraction and both figures converge. If it does matter, the measurement is the per-iteration `α`,
bracket width and fit residual under the two retractions from one starting point — the same
instrumentation A1b needed, and the same reason it is not in the package.

---

#### C12. `step_αmax` takes its element type from the ceiling alone

**Severity: low**, a sharp edge on an internal helper. From the review of [#44].

```julia
function step_αmax(c::T, δ) where {T}
    n = l2norm(δ)
    (isfinite(n) && n > zero(n)) ? c * T(2π) / T(n) : T(Inf)
end
```

`T` comes from `c` and from nothing else, so `step_αmax(1, δ)` on an integer ceiling throws an
`InexactError` at `T(2π)` rather than promoting. `T(n)` is the mirror image and is the worse of the
two, because it does not throw: with `c` a `Float32` and `δ` a `Float64` direction whose norm exceeds
`3.4e38` — finite in `Float64`, `Inf32` on conversion — the guard above sees a finite `n`, the
division underflows and `step_αmax` returns **`0.0f0`**. Measured: `step_αmax(1.0f0, [1e39, 1e39])` is
`0.0`. Upstream then raises an `ArgumentError`, correctly, because a ceiling of zero asks for a step
that violates `α > 0`; so the failure is loud, but it is raised for the wrong reason and blames the
caller for what is a conversion in this function.

Neither is reachable through an `Optimizer`. The struct stores `step_ceiling::T` and the constructor
writes `T(step_ceiling)`, so `c` arrives in the element type of the problem and `l2norm(δ)` is already
in it — the two types cannot differ; `manifold_linesearch_tests.jl` pins exactly that (`step_ceiling =
1` gives a `Float64` field). So this is about calling the helper directly, which the tests do.

**What to do**: `promote_type(typeof(c), typeof(l2norm(δ)))` and take the one `T` from that, which is
a one-line change, kills both halves, and leaves every existing assertion true — the `Float32` test
promotes to `Float32`. What is not worth doing is guarding the narrowing separately; the promotion is
the guard.

---

#### C13. `MANIFOLD_TOLERANCE` is defined three times

**Severity: low**, and the one on this list with a way to go wrong quietly. From the review of [#44].

`const MANIFOLD_TOLERANCE = 1e-12` appears in `test/optimizer_convergence/svd_optim.jl:19`,
`test/manifold_linesearch_tests.jl:45` and — added with the step ceiling —
`scripts/retraction_accuracy.jl:184`. Three copies of one number with no import path between them: a
script cannot `include` a test file that runs a suite as a side effect, and the constant is a property
of the tests rather than of the package, so it does not belong in `src/`.

The reason it matters more than ordinary duplication is what the third copy does. `on_the_manifold`
counts seeds against it, and that count is what every "8 of 8" in this release means. If the script's
copy and the suite's copy ever drift, the sweep and the tests will disagree about whether a solve is
on the manifold and nothing will say so — the sweep is not run in CI (see C10), so the disagreement
would surface as a table that no longer matches a passing suite.

**What to do**: one `test/manifold_tolerance.jl` holding the constant and a comment, `include`d by all
three. That the script reaches into `test/` is already true — it takes its matrix from
`test/optimizer_convergence/svd_matrix.jl` — so this adds no new coupling, only removes two copies.

---

#### C14. `geodesic` and `𝔄exp` assemble the same product independently

**Severity: low**, and a duplication that was created deliberately rather than found. From the review
of [#45], where `𝔄exp` was added.

Both compute ``\mathbb{I} + B'\mathfrak{A}(B', B'')(B'')^T``:

```julia
geodesic(B, algorithm) = manifold_type(B)(one(B) + B̂ * 𝔄(B̂, B̄, algorithm) * B̄')   # retractions.jl:128
𝔄exp(B̂, B̄, algorithm) = I + B̂ * 𝔄(B̂, B̄, algorithm) * B̄'                          # modified_exponential.jl
```

`𝔄exp` was added as a name for what `geodesic` already did, not as a replacement for the inline
expression, on the grounds that `geodesic` also takes the lift apart and wraps the result. Both of
those are one call each, so `manifold_type(B)(𝔄exp(lift_factors(B)..., algorithm))` is the whole of
it, and the reason not to fold them is thinner than it looked when the two lines were written a
commit apart.

Two things now live in two places rather than one. The **default algorithm** is the first: both say
`ScaledSquaring`, and they have to, because a lift retracted through `geodesic` and the same lift
exponentiated through `𝔄exp` are meant to agree — a testset asserts exactly that, which is a test
existing to catch a duplication rather than a defect. The **identity** is the second, and the two do
not spell it the same way: `geodesic` uses `one(B)`, the KernelAbstractions kernel on
`StiefelLieAlgHorMatrix`, and `𝔄exp` uses `I + …`, whose `LinearAlgebra` method writes the diagonal
by scalar indexing. Whether that difference costs anything is A19's question.

**What to do**: decide A19 first, since it decides how the identity should be built, then have
`geodesic` call `𝔄exp` and delete the inline expression. The default then has one home, and the
testset that pins the two together can go with it.

---

#### C15. The compile-time figures cover the first-order caches only

**Severity: low**, and a gap in evidence rather than in code. From the review of [#45].

The measurements in [0.2.1](#021) — 14.25 s / 14.48 s cold against a run that did not finish in seven
or in ten minutes — were taken on `GeometricMachineLearning`'s symplectic-autoencoder test against a
branch on which only `GradientCache`/`GradientState`, `MomentumCache`/`MomentumState` and
`AdamCache`/`AdamState` had been unbound. That test drives `Adam`, so those six are the whole of what
it exercises.

`BFGSCache`, `BFGSState`, `DFPCache`, `NewtonOptimizerCache`, `NewtonOptimizerState` and the `VT` of
`OptimizerResult` were unbound afterwards on the strength of their *inferred types* — for the
quasi-Newton three, `Base.return_types` showed the same coupled shape the six had, and worse for
`BFGSState`, which carried a free `T` across four parameters and the three-parameter `GlobalSection`
`UnionAll` under a `Vararg`; after, all five parameters are independent. That is a sound argument from
the same root cause, and it is not a measurement: no quasi-Newton or Newton solve was ever timed
through a function, before or after, so the claim that they hung is an inference and the claim that
they no longer do is untested.

**What to do**: the cheap version is the one already written — the repro in the release notes with
`algorithm = _BFGS()` and with `Newton()` on Euclidean parameters, cold, before and after, which is
two runs of an existing harness. Do it before quoting these numbers for anything but `Adam`. The
version worth more is C9's: a compile-time measurement that lives in `scripts/` rather than in a
`/tmp` file that is gone by the time anyone asks.

#### C16. `NativePade`'s threshold rests on a measurement, not on a backward-error criterion

**Severity: low**, and a gap in evidence rather than in code — the same shape as C15. From the review
of [#54], and it is what keeps [#52] open now that an algorithm has been added.

[#52] asked for the "proposed algorithm, Padé degree, scaling threshold, and backward-error criterion
… documented from primary references", and for a criterion "appropriate for this structured, strongly
non-normal argument rather than importing a threshold without validation". Three of those four are
done: the degree, the threshold, and the provenance of the coefficients as the ``[7/6]`` Padé
approximant of ``\exp`` rearranged, cited to [higham2005scaling, higham2008functions](@cite). The
fourth is not, and the docstring says so rather than implying otherwise.

What `θ = 1/2` actually rests on is (a) the Newton--Schulz residual bound
``\|\mathbb{I} - q_6\|_1 \leq \sum_k|q_k|\theta^k = 0.256``, hence ``0.256^{32} \approx 2e{-}19``,
and (b) a measured forward error — the eight-point norm sweep on both manifolds in both formats, plus
400 random ``6\times6`` arguments per ``\theta``. Both are forward statements about this family of
arguments. Neither is a backward-error criterion, and the ``\theta_m`` tables that would supply one
[higham2005scaling, almohy2010new](@cite) are derived for ``\exp`` rather than ``\varphi_1`` and are
stated in ``\|X\|`` — which is precisely the norm A6 records as uninformative here, since ``X``'s
lower-left block gives ``\|X\| \approx \|\bar{B}\|^2/4`` against a spectral radius of only
``\approx\|\bar{B}\|``. Importing such a table without adapting it is the thing [#52] asked not to
do, so it was not done; the honest position is that the threshold is validated and not derived.

Two smaller pieces of [#52] are open with it. Its comparison of candidate designs is empirical —
`NativePade` against `AugmentedPade` on cost, allocations, `check` and forward error — but the
theoretical comparison of direct ``\varphi_1`` evaluation against the low-rank and augmented
alternatives was not written; option (a) was chosen on the argument that it needs no solve, which is a
portability argument rather than a numerical one. And no *actual* GPU backend has been exercised: the
`JLArray` test settles scalar-index freedom, which is the failure mode, but A19 above is still open
for the backend itself.

**What to do**: derive a ``\varphi_1`` backward-error bound for a non-normal argument, or state
plainly in the docstring that no such bound is claimed and that the threshold is a measured one — the
latter is what is written today, and it is enough to use the algorithm honestly but not enough to
close [#52]'s first acceptance criterion. Whichever, it belongs with A19: one session that runs the
thing on a GPU and settles a bound is two of these entries.

---

### D. Upstream

#### D1. Julia 1.12: nested `kwargs...` feeding a call in the same inferred body

**Root-caused and worked around in this package by PR #35, but the behaviour is upstream.**

Constructing an `Optimizer` through three nested levels of `kwargs...` splatting and calling `solve!`
on it *in the same inferred function body* cost **940.86 s** of compile time on 1.12.6 against
**4.35 s** on 1.13.0-rc2. Neither half is slow alone (0.99 s for the constructor, 2.35 s for
`solve!`). Flattening to one level: 6.53 s. A `@noinline` barrier around the construction does not
help (925.27 s), and neither does `@nospecialize` on the enclosing function (965 s).

Effect on this project before the fix: the CI suite took 31–42 minutes on 1.12 on all three
operating systems, against 3–5 minutes on 1.10, 1.13 and nightly, with a single test file accounting
for the whole difference.

1.13 and nightly are unaffected, so this needs an upstream report only if 1.12 is still receiving
backports; if it is not, the value is documentation rather than a fix, and the warning on
`Optimizer(x, F)` already carries it.

**The reproducer is written down now — `scripts/nested_kwargs_cost.jl` — and it does not reproduce.**
That was the outstanding action here ("the reproducer this was measured with lived in `/tmp` and is
gone; it has to be rewritten, which is the case for writing it into `test/` or `scripts/` this time"),
and doing it produced a negative rather than a bug report. Four reconstructions, on both 1.12 patch
releases:

| | 1.12.6 | 1.12.7 | 1.13.0-rc3 |
|---|---|---|---|
| real `Optimizer`, one level | 3.44 | 3.13 | 3.36 |
| real `Optimizer`, three levels | 3.44 | 3.12 | 3.39 |
| real `Optimizer`, three levels, 3 algorithms × 2 retractions | — | 5.05 | 5.37 |
| real `Optimizer`, one level, same breadth | — | 5.05 | 5.34 |
| `Base`-only synthetic, 800-deep tree, three levels | — | 0.21 | 0.19 |
| `Base`-only synthetic, one level | — | 0.21 | 0.19 |

Against the 4.35 s / **940.86 s** this entry records for the first two rows. The nested and flat
columns are equal to the hundredth of a second everywhere, and the 1.13 figures agree with the
recorded 4.35 s to within a cold measurement's spread — so the harness is measuring the right thing;
it is the 940 that will not come back.

**So the description above is not sufficient**, four ways: "a constructor reached through N nested
`kwargs...` levels whose result is passed to a second function with a large call tree, both in one
inferred body" does not by itself produce the cliff. That is what anyone would work from, so knowing
it is incomplete is the useful part of this. The patch-release explanation is *ruled out* — 1.12.6 and
1.12.7 agree to within 0.3 s on every control — which was the cheap hypothesis and worth eliminating
first.

**What to do**, in order:

1. Nothing goes to JuliaLang yet. A bug report needs a reproducer and four negatives are not one.
2. Try the two ingredients none of the four varies: `Options(T; options_kwargs...)`, whose keyword
   defaults are computed from other keywords so inference has a dependency order to resolve, and the
   objective's own call tree (the SVD closure over a captured `A`, at the original's problem size
   rather than the harness's `St(20, 3)`).
3. If those are also negative, the remaining hypothesis is that something in this package between
   PR #35 and now removed the sensitivity — in which case **D1 is closeable** and the artefact worth
   having is whichever commit did it. Testing that means bisecting `#35..HEAD` with the nested shape
   reinstated at each step, which `scripts/nested_kwargs_cost.jl --with-package` is written to make
   mechanical.

The warning on `Optimizer(x, F)` stays either way: it records a measurement that was taken, and
nothing here shows the flattening is safe to undo — only that the cost it avoids cannot currently be
demonstrated.

#### D2. SimpleSolvers 0.12: three `Options` fields that nothing reads

`store_trace`, `show_trace` and `extended_trace` exist as fields of `SimpleSolvers.Options`
(`src/base/options.jl:458-460`), as constructor keywords (`:489-491`) and in its `show` (`:521-523`),
with **zero readers** anywhere in `SimpleSolvers/src/`. Anyone setting them on a SimpleSolvers solver
gets silence rather than a trace or an error. PR #35 implements `store_trace` at the
GeometricOptimizers level, which is arguably the right layer since `solve!` is ours, but the upstream
options remain misleading.

**What to do**: file against JuliaGNI/SimpleSolvers.jl, naming which of the two
defensible resolutions is preferred — implement the trace in SimpleSolvers' own solvers, or remove
the three fields and let each caller own its trace, which is what GeometricOptimizers now does.
Either is better than a field that accepts a value and discards it. If the first, this package's
local implementation should be retired in favour of it.

**Still open in 0.12**, and now confirmed from both sides: upstream's own open-issues list carries it
under *"Reported by GeometricOptimizers.jl, not addressed in 0.12.0"*, naming the same three fields
and the same two resolutions, and noting that both are breaking so neither belonged in a release
driven by something else. The header of this entry moved from 0.11 to 0.12 and nothing else in it did.

#### D5. Documenter does not catch an `@ref` to a method signature that no longer exists

From the PR #36 review. That PR replaced `geodesic(::StiefelLieAlgHorMatrix)` and
`geodesic(::GrassmannLieAlgHorMatrix)` with one method on `AbstractLieAlgHorMatrix`, and left a
docstring pointing at `[`geodesic(::StiefelLieAlgHorMatrix)`](@ref)` — a method the same PR deletes.

I expected the docs build to fail on it. It does not: `makedocs` completes with exit 0 and no
warning, because Documenter falls back to the **binding**-level docs for `geodesic` when the
signature matches no documented method. Verified by reintroducing the dead reference and rebuilding.

So the guarantee is weaker than it looks. A `@ref` to a *name* that does not exist is caught; a
`@ref` to a name that exists with a signature that does not is silently redirected to some other
method's page. This is worth knowing here specifically because the commit immediately before that one
on the same branch was "mend four dead doc links" — the build cannot be what certifies that work.
`checkdocs = :all` does not help: it checks that docstrings are *included*, not that references
resolve to what they name.

The `@extref` half behaves the *opposite* way and D8 below is the case: an external link that does not
resolve is a hard error and fails the build. So the two halves of the same feature have opposite
failure modes — a dead internal reference is silently redirected, a dead external one stops the
build — which is worth knowing when a docs build suddenly fails after an upstream release.

---

#### D7. A step ceiling that binds can be reported as `LINESEARCH_FLOOR`

**Severity: medium**, inherited from SimpleSolvers 0.12 with the step ceiling and carried in
upstream's own open-issues list. **Open upstream and without a consequence here**: the downstream
half was B3, and it is closed — see *Fixed* above. This is the part that is not this package's to
fix, and it stays listed because any *other* consumer of a capped search inherits it.

`SimpleSolvers.capped_status` classifies the step at `αmax` by the same round-off rule `τ` as any
other returned step. A merit that is *still falling* at the ceiling — which is what the capped case
means — but has fallen by less than `τ` over the whole admissible range therefore comes back as
`LINESEARCH_FLOOR`. That outcome is a claim about the **direction**, that no line search can make
progress along it, and consumers act on it: SimpleSolvers' own solver through `flag_stall!` and
`max_stalls`, and this package through `linesearch_rejected`, which throws `Q` away and re-searches
along steepest descent. What was actually established is only that no step the *caller permits*
decreases the merit measurably.

Upstream is explicit that this is the same shape as the two unearned floors 0.12 removed from
`Bisection`, reached through a third door, and that it is reachable exactly where the caller's ceiling
is tightest — the case the ceiling exists for. It was left standing because the alternatives are not
obviously better: `LINESEARCH_EXHAUSTED` would say "no step was found" about a step that was found and
returned, and closing it properly means a boolean on the `LinesearchStatus`, which upstream declined
on the grounds that the struct is copied per solver step and a caller who set the ceiling can compare
it against `steplength`.

Nothing measured here is affected at `DEFAULT_STEP_CEILING = 1`: all twenty sweep combinations
reproduce their no-ceiling iteration counts or improve on them, so the path is not being taken.

**What to do**: nothing, on either side. Upstream is where it is filed and the reasoning for leaving
it there is sound; downstream it is closed, because `linesearch_rejected` now takes the ceiling
`solver_step!` passed and exempts a `LINESEARCH_FLOOR` returned at it. That is what upstream meant by
"a caller who set the ceiling can compare it against `steplength`", and it needed no new field on the
status. The entry stays here as the record of *why* that comparison is in `linesearch_rejected` — a
reader who finds the exemption and not this will think it is guarding against nothing.

---

#### D8. `SimpleSolvers.solve_with_status` has no binding-level entry to link to

**Severity: low**, and it cost this package two documentation links this round.

SimpleSolvers documents `solve_with_status` per *method*, so its `objects.inv` carries entries like
`SimpleSolvers.solve_with_status-Union{Tuple{T}, Tuple{Linesearch{...}}}` and no bare
`SimpleSolvers.solve_with_status` binding. `solve_with_status!` does have one. A binding-level
`[`SimpleSolvers.solve_with_status`](@extref)` therefore cannot resolve, and — unlike the internal
`@ref` case of D5 — DocumenterInterLinks makes that a **hard error** that terminates the build.

This surfaced when SimpleSolvers' published docs were rebuilt for 0.12, and is *not* caused by
depending on 0.12: the inventory is fetched from the `stable` URL regardless of which version is
resolved, so `main` had the same broken build the moment those docs went up. Two docstrings here
carried the reference — `solver_step!` and `linesearch_rejected` — and both are now plain code, which
fixes the build and loses the link.

**What to do**: ask upstream for a binding-level docstring on `solve_with_status`, as
`solve_with_status!` already has; it is one `@docs` entry and it restores the link for every
downstream package. Failing that, this package can pin the local fallback inventory that
`docs/make.jl` already names — `docs/inventories/SimpleSolvers.toml`, which does not currently
exist — so that an upstream docs rebuild cannot break this build again. The second is worth doing
regardless: relying on a fetched inventory means a docs build that passes today can fail tomorrow with
no commit here.

---

### E. Reported and then withdrawn

Three things reported as problems during the investigation that are **not**:

- `test/grassmann_test_help.jl` is not orphaned — it is `include`d by
  `test/global_sections/global_sections.jl:8`, `test/global_sections/omega_functions.jl:8` and
  `test/retractions/retractions.jl:9`.
- `test/optimizers_problems.jl` is not dead code — `test/optimizer_tests.jl:12` includes it.
- `NaNMath` is used, at `test/optimizer_tests.jl:2`.

---

### F. Loose ends from the geodesic-retraction review

Not a defect in the code; a thing a later reader would otherwise have to rediscover.

- **The PR #36 description still says the MNIST scripts use the geodesic retraction.** The claim was
  corrected in `docs/src/manifold_optimizers.md`, but it also appears in the pull request body, where
  it is the stated justification for making `ScaledSquaring` the default. None of the five MNIST
  scripts passed `retraction`, so they all took the `Optimizer` default, which is `Cayley()`. The
  default is still the right choice — being free of dense LAPACK is reason enough — but if that PR
  body becomes a squashed commit message, the wrong reason goes into the history with it. The
  scripts have since moved to GMLDatasets.jl (see [0.3.1](#031)), which changes nothing
  about the PR body this entry is about.

(The second loose end here — that `svd_tables()` had never been re-run, so the iteration and
evaluation counts rested on the author's measurement alone — is closed. It was run on `main` before
the A8 work below and reproduced every one of the twenty rows to the digit, iterations, evaluations
and `check` alike. The two figures it did *not* confirm were the two that came from somewhere else,
and both are corrected: see C8.)

[#14]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/14
[#16]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/16
[#17]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/17
[#18]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/18
[#24]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/24
[#25]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/25
[#27]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/27
[#28]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/28
[#33]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/33
[#36]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/36
[#38]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/38
[#39]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/39
[#40]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/40
[#44]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/44
[#45]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/45
[#46]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/46
[#47]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/47
[#48]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/48
[#49]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/49
[#50]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/50
[#51]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/51
[#52]: https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/52
[#54]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/54
[#59]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/59
[#60]: https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/60
[0.1.0]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.1.0
[0.2.0]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.2.0
[0.2.1]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.2.1
[0.2.2]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.2.2
[0.3.0]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.3.0
[0.3.1]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.3.1
[0.4.0]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.4.0
[0.4.1]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.4.1
[0.4.2]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.4.2
[0.4.3]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.4.3
[0.5.0]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.5.0
[0.6.0]: https://github.com/JuliaGNI/GeometricOptimizers.jl/releases/tag/v0.6.0
[Unreleased]: https://github.com/JuliaGNI/GeometricOptimizers.jl/compare/v0.6.0...main
