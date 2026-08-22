# PR #57 Documentation Workplan

## Context

PR #57 (`issue-56-scaled-squaring-docs`) is still a draft. The latest review asks for a clearer,
more deliberate structure on the Retractions page, especially after **The exponential needs an
algorithm**. It specifically asks for this progression:

1. direct Taylor evaluation and why it fails;
2. Padé approximation and why it is the standard dense choice;
3. scaling and squaring as a framework that can use Taylor or Padé kernels;
4. `ProjectedSkew`; and
5. a numerical comparison of all methods.

The review also objects to internal shorthand such as “A1b” and to prose that reads like debugging
notes rather than user documentation.

PR #54 (`implement-pade-approximation-for-computing-geodesic`) remains open as of August 22, 2026.
It adds `NativePade`, a degree-6 diagonal Padé approximation for `𝔄`, followed by the same low-rank
modified-squaring recurrence used by `ScaledSquaring`. It uses Newton--Schulz inversion to remain
compatible with scalar-indexing-free backends. Because #54 is not merged into this branch, the
current edits do not add links, examples, or API claims for `NativePade` that would fail the present
documentation build.

## Changes Made

### `docs/src/retractions.md`

- Removed the unexplained “issue A1b” reference from the main narrative.
- Reorganized the exponential-algorithm discussion into numbered sections:
  - direct Taylor series;
  - Padé approximation;
  - scaling and modified squaring;
  - `ProjectedSkew`; and
  - numerical comparison.
- Explicitly separated approximation kernels (Taylor and Padé) from scaling and squaring.
- Clarified that conventional dense matrix-exponential routines usually combine Padé with scaling
  and squaring, while the current `ScaledSquaring` type combines Taylor with modified squaring.
- Kept the four-step derivation of `ScaledSquaring`, including the initial `1/2^s` factor and the
  recurrence using the original `X`.
- Recast `AugmentedPade` as the current Padé-based reference implementation rather than presenting
  it as the desired direct algorithm.
- Rewrote benchmark commentary to describe the measured results without broad claims based on a
  single sweep.
- Rewrote the selection guidance to distinguish mathematical properties, backend restrictions, and
  machine-dependent timing results.

### `src/retractions/exponential_algorithms.jl`

- Shortened the `AbstractExponentialAlgorithm` overview and removed the embedded one-row benchmark
  and prescriptive conclusions.
- Reworked the `ScaledSquaring` docstring around its Taylor kernel, modified-squaring recurrence,
  scaling threshold, backend portability, and possible overscaling.
- Simplified the `AugmentedPade`, `ProjectedSkew`, and `TaylorSeries` docstrings and removed informal
  or overly strong benchmark language.

### `src/retractions/modified_exponential.jl`

- Rephrased the cancellation warning for the unscaled Taylor kernel.
- Rephrased the reason `𝔄exp` defaults to `ScaledSquaring`.

### `src/retractions/retraction_types.jl`

- Removed the public-docstring reference to “issue A1b”.
- Reframed the Cayley/geodesic timing comparison as machine-dependent rather than universal.
- Replaced “cheaper alternative” wording with the mathematically relevant “rational alternative”.

## Literature Alignment Checked

- Higham (2005): conventional scaling and squaring scales by a power of two, evaluates a Padé
  approximant, and repeatedly squares.
- Al-Mohy and Higham (2010): overscaling can reduce accuracy; norm-based scaling choices can be more
  conservative than necessary.
- Skaflestad and Wright (2009): modified-squaring recurrences apply to matrix functions related to
  the exponential, including `φ`-functions.

The revised text no longer equates scaling and squaring with Padé. It states that Padé or Taylor may
be the small-argument kernel and describes the package's current `ScaledSquaring` implementation as
Taylor plus low-rank modified squaring.

## Where To Continue

1. Review `docs/src/retractions.md` from `## The exponential needs an algorithm` through
   `## Choosing one` for flow and mathematical notation.
2. Decide how PR #57 should relate to PR #54:
   - preferred: merge #54 first, rebase #57, then add `NativePade` to the algorithm table, usage
     examples, benchmark tables, and selection guidance;
   - temporary: keep #57 buildable on current `main` and avoid documenting an unavailable type.
3. After #54 is present, distinguish the two Padé paths clearly:
   - `NativePade`: direct Padé approximation of `𝔄` plus low-rank modified squaring;
   - `AugmentedPade`: `𝔄` extracted from a larger dense matrix exponential, retained mainly as an
     independent reference.
4. Recheck statements about Julia's `exp` implementation against the Julia version supported by the
   package. Avoid promising a fixed Padé degree unless the implementation guarantees it.
5. Consider shortening or removing historical benchmark tables from public API docstrings; keep the
   reproducible comparison in the guide and `scripts/retraction_accuracy.jl`. comment from benedict-96: remove everything that's outdated!
6. Run validation:

   ```sh
   git diff --check
   JULIA_DEPOT_PATH=/Users/benbradmin/.julia \
     ~/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia \
     --project=docs docs/make.jl
   ```

7. Inspect the rendered Retractions page, especially heading hierarchy, table widths, admonitions,
   and cross-references.
8. Open an issue regarding an implementation of ScaledSquaring+NativePade.

## Current Worktree

Modified by this review pass:

- `docs/src/retractions.md`
- `src/retractions/exponential_algorithms.jl`
- `src/retractions/modified_exponential.jl`
- `src/retractions/retraction_types.jl`

Pre-existing untracked files were left untouched:

- `mnist_parameters.jld2`
- `results/`
- `scripts/mnist_parameters.jld2`

`git diff --check` passed after the edits. The documentation build has not yet been run.

## Review Log

### August 22, 2026 — second documentation review

- Re-read the Retractions guide and the public docstrings changed on this branch.
- Confirmed that the guide separates approximation kernels from scaling-and-squaring recovery.
- Confirmed that `ScaledSquaring` is described as Taylor plus low-rank modified squaring and that
  `AugmentedPade` is described as the current dense-`exp` reference path.
- Confirmed that benchmark conclusions are qualified as machine-dependent.
- Confirmed that the pending `NativePade` API from PR #54 is not referenced by this branch.
- Ran `git diff --check`; no whitespace errors were reported.
- Per the requested workflow, did not run tests or build the documentation at this stage.
