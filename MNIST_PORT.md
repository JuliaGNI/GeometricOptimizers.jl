# Porting `transformer_mnist.jl` — status

State of the branch `mnist-script-and-adam-fixes` ([PR #14](https://github.com/JuliaGNI/GeometricOptimizers.jl/pull/14)),
which ports `GeometricMachineLearning/scripts/transformer_mnist.jl` to this package as
`scripts/mnist.jl` (host), `scripts/mnist_cuda.jl` and `scripts/mnist_metal.jl`, with
`scripts/mnist_metal_short.jl` as a bounded stand-in for the last of those.
Last updated 2026-08-08, after the first smoke run on the RTX 4090.

The port has to be *self-contained*: `GeometricMachineLearning` depends on
`GeometricOptimizers`, so `DataLoader`, `ClassificationTransformer`, `NeuralNetwork` and
`FeedForwardLoss` are not available and the network, the loss, the initialization and the
batching are written out explicitly. The target API is the one of
`test/optimizer_convergence/svd_optim.jl`: a flat `NamedTuple` of parameters,
`Optimizer(ps, F; algorithm, linesearch = Static(η))`, `OptimizerState(algorithm, ps)` and a
hand written minibatch loop over `solver_step!` (`solve!` cannot be used, as it optimizes a
*fixed* objective until it converges, whereas the objective changes with every batch).

## What was done

### Three limitations of the optimizer interface

These blocked the port and are fixed in `src/`:

1. **`ArrayNamedTuple` was homogeneous.** `ArrayTuple` and `GlobalSectionTuple` were written
   as `Tuple{Vararg{AT}} where {AT<:AbstractArray{T}}`; Julia's *diagonal rule* makes such a
   type homogeneous, so a `NamedTuple` that stores a `StiefelManifold` and an ordinary
   `Matrix` at the same time was not an `ArrayNamedTuple`. Both aliases are covariant now,
   which is what lets the script keep the Stiefel attention projections next to the plain
   `ResNetLayer` weights, biases and classification head — as in the original.
2. **Only `Float64` parameters.** `ParameterHandling.flatten` defaults to `Float64`, so
   `Float32` parameters were silently promoted and the flattened vector no longer matched the
   parameters. `ArrayNamedTuple`s are now flattened to their own element type, so the script
   can run in `Float32` like the original.
3. **No hand written gradients for `NamedTuple`s.** Added
   `GradientFunction(F, ∇F!, nt::NamedTuple)`. `∇F!` is called on the *flattened*
   parameters, i.e. on `ParameterHandling.flatten(nt)[1]`.

### Bugs in `Adam` and `MomentumMethod`

Found while getting the script to take its first step; all fixed in
`src/manifold_optimizers/`:

1. **Uninitialized moments.** `AdamState` initialized `m₁`, `m₂`, `m̃₂` and `MomentumState`
   initialized `p` with `_similar`, i.e. with uninitialized memory, and
   `initialize_state!(::OptimizerState)` is a no-op. They are read in the first call to
   `update!(::OptimizerCache, ...)` before they are ever written to, so the first optimizer
   step depended on whatever happened to be in memory. For `Adam` this made `_rac!` take the
   square root of negative values and throw `DomainError with -Inf`, which is how the bug was
   found. The `Adam` convergence test in `svd_optim.jl` passed only because the garbage
   happened to be benign there.
2. **Wrong recursion factors.** The moments are stored in bias-corrected form, so the factors
   have to be `(β - βᵗ)/(1 - βᵗ)` and not `β/(1 - βᵗ)`; the latter amplifies the moments by
   up to `1/(1 - β)` during the first iterations.
3. **Misplaced square root.** `_rac!` was applied to `m₂` instead of to `m̃₂ = m₂ + δ`, so
   the direction was `-m₁/(m₂ + δ)` instead of `-m₁/(√m₂ + δ)`, and the in-place square root
   corrupted the second moment that is stored in the state afterwards.

The formulas now agree with the implementation that was moved here from
`GeometricMachineLearning` (`git show ab94efb^:src/optimizers/adam_optimizer.jl`). On the SVD
test problem the relative error of `Adam` improves from `1.1e-3` to `1.7e-5` (`Geodesic`) and
from `1.7e-3` to `6.0e-5` (`Cayley`), i.e. `Adam` is now the best of the three algorithms
instead of the worst.

`test/optimizer_state_initialization.jl` is a new regression test: the states are
zero-initialized, after the first step `m₁ == ∇L` and `m₂ == ∇L ⊙ ∇L` (this pins down both
formula bugs), and a fixed seed reproduces a run exactly.

### Tests for the three interface limitations

`test/named_tuple_parameters.jl` (90 tests) covers the changes of the previous section, which
had no test of their own. It optimizes
``F(Y, W, b) = \frac{1}{2}\|YW + b - B\|^2`` over
`ps = (Y = rand(StiefelManifold{T}, 6, 3), W = randn(T, 3, 4), b = zeros(T, 6))`, i.e. over
the same *heterogeneous* shape that the script uses, for `T ∈ {Float64, Float32}`, all three
algorithms and both retractions:

- **the manifold property is preserved during the optimization** — `check(ps.Y)` is recorded
  after *every* step and has to stay below `100·eps(T)`. This is the test that the changes
  really needed. The tolerance is a round-off tolerance: the deviation grows like a random
  walk that the retraction keeps pulling back (`5`, `10`, `20`, `45` times `eps(T)` for `5`,
  `20`, `80`, `320` steps) and is `10·eps(T)` at the `20` steps that are taken, whereas a
  retraction that actually left the manifold would be off by the order of the step size,
  i.e. by `1e-1`.
- the heterogeneous `NamedTuple` is an `ArrayNamedTuple` and an `OptimizerSolution`
  (limitation 1),
- `ParameterHandling.flatten` returns a `Vector{T}` that round-trips (limitation 2),
- the hand written `∇F!` produces the same iterates as the default `ForwardDiff` gradient
  (limitation 3),
- the entries that are *not* on a manifold are updated as well.

Two things worth knowing when reading the file:

- **Every run has to seed the RNG.** The `GlobalSection` is drawn at random and the iterates
  depend on it — the difference is `O(η²)` per step and grew to a visible `5 %` over `20`
  steps while the hand written gradient was being compared to the default one. The same
  remark is already in `test/optimizer_state_initialization.jl`.
- **`Adam` has to be constructed with the element type of the parameters.** `Optimizer`
  converts a `MomentumMethod` to the element type of `x` but does not do the same for `Adam`,
  so an `Adam{Float64}` with `Float32` parameters misses
  `OptimizerCache(::Adam{T}, ::OptimizerSolution{T})` and falls through to
  `NewtonOptimizerCache`, which then fails with a `MethodError` — one that now says as much.
  All three scripts therefore construct the algorithm as `Adam(T)`, with `T = Float32`.

### The script

`scripts/mnist.jl` (with `scripts/Project.toml`; the `Manifest.toml` is git-ignored) builds
the same network as
`ClassificationTransformer(dl; n_heads = 7, L = 16, add_connection = false, Stiefel = Stiefel)`
and runs the same four trainings as the original. Points worth remembering:

- **The learning rate comes from the line search.** The optimizer *methods* only produce a
  direction, so the step size is `linesearch = Static(η)`, which is also what `Optimizer`
  defaults to for `GradientMethod`, `MomentumMethod` and `Adam`. `Adam`'s positional argument
  is the element type, not `η` (the `η` field it used to carry was never applied to the
  direction and is gone), and `MomentumMethod(α)`'s `α` is the momentum coefficient, not a
  step size. The scripts still pass `Static(learning_rate)` explicitly, so that the rate is
  readable at the call site rather than being a package default.
- **`Zygote`, not `ForwardDiff`.** The cost of `GradientAutodiff` scales with the number of
  parameters, of which there are 154938 here, so a hand written `∇F!` is passed to the
  `Optimizer`. Measured at the full configuration (`L = 16`, batch size 2048, `Float32`): a
  loss evaluation takes 0.84 s, a `Zygote` gradient 3.6 s (21 s for the first one, including
  compilation), a full optimizer step about 5 s.
- **`check_gradient` uses a directional derivative.** A central difference is useless here:
  in `Float32`, with `d` a random unit vector in 154938 dimensions, the directional
  derivative is of the order `‖g‖/√n ≈ 1e-4` while the cancellation error of the difference
  quotient is `eps(F)/2ε ≈ 5e-5` — the original check reported a meaningless 4.1 %. The check
  now uses `ForwardDiff.derivative` along `d`, which needs a single dual number, and reports
  a relative error of **4.9e-6** at the full configuration. `∇F!` is therefore correct.
- `split_and_flatten` reproduces the output documented for
  `GeometricMachineLearning.split_and_flatten` exactly.

### What the four configurations are for, and why one of them must fail

The three Stiefel configurations (`Adam`, gradient, momentum) compare *optimizers*. The
fourth, `regular weights, Adam`, compares something else: it is the **baseline**, the same
network with the attention projections left unconstrained, and it is expected to sit at chance
with a flat loss.

That is the published result of B. Brantner, *Generalizing Adam To Manifolds For Efficiently
Training Transformers* (§"Numerical Example: the Transformer"): the vision transformer with
unconstrained projections and none of the usual heuristics — layer normalization, dropout,
regularization, pre-training — "is not able to learn much, as the error rate is stuck at
around 1.34, which indicates a trivial prediction". With `L = 16` blocks and nothing
normalizing between them the gradient that reaches the early blocks vanishes, so the network
collapses onto a constant prediction `eᵢ` and stays there. Constraining the projections to the
Stiefel manifold is what removes the problem — `YᵀY = I`, so a block neither amplifies nor
damps what passes through it — which is the point the experiment exists to make: hard
geometric constraints in place of the heuristics.

The `1.34` is reproduced by the loss used here. `network_loss` is `norm(pred - out)/norm(out)`,
and over a batch of `k` one-hot targets a constant prediction is wrong on 9 of 10 images and
off by `√2` on each, so the loss is `√(2·0.9·k)/√k = √1.8 ≈ 1.342` — the paper's `√(9/10·2)`.

**So a flat loss and an accuracy of `≈0.10` in this run are the experiment working.** They
were briefly mistaken for an `Adam` bug on the non-manifold path after the three epoch `Metal`
run of 2026-08-07 (loss `1.327 → 1.330`, accuracy `0.1049`); that reading is wrong and the
numbers are the expected ones to three digits. `scripts/mnist_metal_short.jl` therefore judges
this configuration against the plateau rather than against an accuracy floor.

### The `CUDA` script

`scripts/mnist_cuda.jl` is the same script with the network on the GPU and `n_epochs = 500`,
as in the original. `CUDA` is part of `scripts/Project.toml`; it installs (but is not
functional) on machines without a CUDA device, so `mnist.jl` still works in the same
environment, and `mnist_cuda.jl` falls back to the host so that it can be tested there.

**The parameters stay on the host — the optimizer interface cannot hold GPU arrays.** This
is a regression compared to `GeometricMachineLearning`, whose optimizers did run on
`CUDABackend()` (which is what the original script uses). Two things block it:

1. `ParameterHandling.flatten` has no method for GPU arrays. The optimizer flattens its
   parameter `NamedTuple` on *every* step (`(grad::Gradient)(::ArrayNamedTuple)` in
   `src/optimizers/named_tuple_wrapper.jl`), and a `CuVector` misses
   `flatten(::Type{T}, ::Vector{R})` — it falls through to ParameterHandling's
   `flatten(::Type{T}, ::AbstractVector)`, which `map`s over the *elements* of the vector.
2. `_similar(::StiefelManifold)` is `rand(StiefelManifold{T}, size(a)...)`, i.e. always
   `rand(CPU(), ...)`. It is used for `x̄` and for the `BFGS`/`DFP` caches, so the state would
   mix host and device arrays. Everything else in `src/` is written against
   `KernelAbstractions` and looks backend-agnostic.

Neither matters for the script, because the optimizer only touches the 154938 parameters
(620 kB in `Float32`) while the forward and backward passes touch the whole batch. Per step
the parameters are uploaded and the gradient downloaded once, ≈1.2 MB, against ≈3 GB of
device-side activations. What is on the device is the batch, `predict`, the loss and the
`Zygote` gradient; the geometry (retraction, global section, `Adam`) runs on the host.

Verified at a reduced configuration (`L = 2`, batch size 64, 2 epochs, 256 images) against
`mnist.jl`: the `regular weights, Adam` run — the one without a random `GlobalSection` — is
identical digit for digit, and the Stiefel runs agree to `1e-5`, i.e. to the size of the
global-section noise. `check_gradient` compares the *device* `∇F!` to a host directional
derivative (`ForwardDiff.Dual`s cannot go through `cuBLAS`) and reports `2.8e-7`, so it also
checks the host/device split.

**The run reports to files, not to a terminal.** The RTX 4090 is reached over ssh, the run
outlives the session that started it, and 58000 optimizer steps are not something anybody
watches. So `mnist_cuda.jl` writes, alongside its output, a report (the environment — host,
GPU, driver, revision, package versions — the gradient checks, one line per epoch, and a
summary and verdict per configuration) and a CSV with one row per optimizer step, flushed per
line and per epoch respectively; the `.jld2` is rewritten after every configuration rather
than once at the end. A run that dies in the third configuration therefore leaves the first
two complete and every finished epoch of the third, and the two text files are readable
without Julia. A configuration that throws is caught, reported with its backtrace and does not
take the remaining ones down; a non-finite loss ends that configuration early.
`MNIST_N_EPOCHS`, `MNIST_BATCH_SIZE`, `MNIST_ACCURACY_EVERY`, `MNIST_REPORT`, `MNIST_LOSSES`,
`MNIST_OUTPUT` and `MNIST_PROGRESS` override the schedule and the paths;
`scripts/run_mnist_cuda.sh` wraps all of it for `screen`, stamps the output files with the
start time and prints what to copy off the machine.

The test accuracy and `‖YᵀY-I‖` are measured every 25 epochs rather than once at the end. Both
cost well under a percent of the run, and they answer what a single final number cannot:
whether a configuration was still improving when it stopped, and whether the drift off the
manifold accumulates with the step count or settles.

**Smoke run on the RTX 4090, 2026-08-08** (`--smoke`, two epochs per configuration). It found
two things, neither in the training and both fixed:

1. `Metal` in `scripts/Project.toml` made the environment unprecompilable on Linux — see the
   next section.
2. `CUDA.available_memory` no longer exists. 5.8 renamed it to `free_memory`, and the 6.x
   split into `CUDACore`/`CUDATools` forwards only the new name; the workstation runs 6.2.1.
   The call sits in the epoch loop, so all four configurations died at the same point, each
   after a complete first epoch. The accessor is resolved once now from whichever name the
   installed version has, and a further rename costs the memory number rather than the run.

What it established is what had never been tried: **the CUDA path itself executes.** Both
gradient checks pass on the device (`3.05e-07` Stiefel, `3.71e-05` regular), MNIST loads, all
four configurations build their optimizer and complete an epoch of 29 steps at batch size
2048, and the report, the CSV and the `.jld2` are all written. Epoch 1 learns: Stiefel
`1.125 → 0.980`, regular `1.307 → 1.267`. The two configurations that reused compiled code
took 30 s and 32 s for their 29 steps, i.e. ≈1.0 s/step against the 2.13–2.30 s/step of the
M4 Max, which puts the full run at ≈15 h rather than ≈36 h — provisional, because those 30 s
still contain compilation. Everything past the memory call is still unexecuted: the accuracy
evaluation, `orthonormality_error`, the epoch lines, the summaries and the verdict.

### Repetitions: one configuration several times

`scripts/mnist_cuda_repetitions.jl` trains *one* configuration `MNIST_REPETITIONS` times and
ends on a mean and a corrected sample standard deviation over the repetitions — for the test
accuracy, the final epoch loss, `‖YᵀY-I‖` and the wall clock — with the individual samples
printed next to each. `mnist_cuda.jl` is untouched and still runs each of the four
configurations exactly once, which is what a comparison of their learning curves needs.

It exists because of the observation in step 4 below: a single accuracy from an `Adam`
configuration is a sample and not a number. Everything below the run loop is `mnist_cuda.jl`'s
code — same network, initialization, objective, gradient, schedule and bars — with the seed
turned into an argument of `train`, so a repetition here is comparable to the corresponding run
there. Each repetition builds its own `Optimizer` and `OptimizerState`, so no cache and no
iteration counter survives from the previous one.

- `MNIST_REPETITIONS` (default 5) is how often each configuration is trained. One repetition is
  one configuration of the full run, i.e. ≈1:35 h for `Adam` on an RTX 4090, so the default is
  ≈8 h — about what `mnist_cuda.jl` takes for all four.
- `MNIST_CONFIGURATIONS` (default `adam-stiefel`) selects which ones, from `adam-stiefel`,
  `adam-regular`, `gradient`, `momentum`, or `all`. Unknown keys fail before MNIST is loaded.
- `MNIST_VARY_SEED` (default `1`) gives repetition `r` the seed `seed + r - 1`, so the spread is
  that of the method over initializations *and* over the nondeterminism, which is the number to
  quote. With `0` every repetition uses the same seed and the spread is the nondeterminism
  alone — the observation itself rather than a property of the optimizer.
- The remaining variables, the three output files and the `screen` wrapper
  (`scripts/run_mnist_repetitions.sh`, with `--smoke`, `--repeat N` and `-c LIST`) are those of
  `mnist_cuda.jl` and `run_mnist_cuda.sh`.

Every repetition is judged individually by the same `verdict`, and the statistics are reported
alongside those verdicts rather than instead of them: five repetitions whose mean clears the
accuracy floor but of which one collapsed is not the same outcome as five that all worked, and
only the individual verdicts distinguish the two. A repetition that throws is missing from the
statistics, and the count of samples printed with them says so.

Its loss CSV has one column more than the one `mnist_cuda.jl` writes (`repetition`, after
`configuration`), so `scripts/distill_mnist_results.jl` — which feeds the figures of the
documentation — does not read it. Those figures compare the four configurations and stay the
job of `mnist_cuda.jl`; this script answers how far the `Adam` number in them can be trusted.

### The `Metal` script

`scripts/mnist_metal.jl` is the same script again for the GPU of an Apple silicon Mac. The
host/device split, the network and the initialization are those of `mnist_cuda.jl`; unlike
`CUDA`, `Metal` does not *load* on platforms it does not support, so this one is macOS on
Apple silicon only (the other two still work in the same environment everywhere).

`Metal` is not a dependency of `scripts/Project.toml` — it cannot be *resolved* on Linux, and
a project that lists it leaves the environment unprecompilable there, which is how the first
attempt at `mnist_cuda.jl` on the RTX 4090 died (`KeyError: ... "Metal"` out of
`Base.Precompilation.scan_deps!`). The Mac adds it once with
`julia --project=scripts -e 'using Pkg; Pkg.add("Metal")'`; that writes `Metal` back into
`scripts/Project.toml`, and the line does not belong in a commit.

**Device memory is the one substantial difference to the `CUDA` script, and it is not
optional: without it the script takes the whole machine down.** An `MtlArray` needs *two*
independent things to be released, and neither works on its own:

1. **`Metal.@autoreleasepool`.** A kernel launch autoreleases its command buffer, and a
   command buffer retains every buffer it references. In a script the outermost pool never
   drains, so the command buffers of the forward and backward passes accumulate there and
   pin every intermediate. No amount of garbage collection frees them — the reference that
   keeps them alive is on the Objective-C side.
2. **`GC.gc(false)`.** Conversely the pool can only drop a buffer once the `MtlArray` owning
   it has been finalized, and Julia's collector is driven by the size of the *Julia* heap: an
   `MtlArray` is a few hundred bytes of Julia object in front of hundreds of megabytes of
   Metal buffer, so it does not fire anywhere near often enough by itself.

Measured over ten gradient steps at a batch size of 512 (M4 Max, `Metal` 1.10), reading
`Metal.device().currentAllocatedSize` — reproduce with `scripts/metal_memory_probe.jl`:

| per step | device memory |
| --- | --- |
| neither | `0.5` → `4.2` GB, climbing by `0.41` GB per step |
| `GC.gc(true)` | `0.4` → `4.1` GB, i.e. a *full* collection changes nothing |
| pool only | up to `12.4` GB between the collections that happen to occur |
| **pool and `GC.gc(false)`** | **`0.31` GB, flat, at no measurable cost in time** |

Every batch, every chunk of `accuracy` and the gradient check therefore run inside a
`device_scope`, which is exactly those two things plus a budget check.

**Why this is worse than it sounds.** The memory of an Apple GPU is unified and an allocation
does not *fail* when it runs out: the kernel starts compressing, then paging, and the process
is jetsammed long before Metal reports anything. So the usual catch-an-`OutOfMemoryError`-and-
retry-with-a-GC safety net (`Metal.alloc_buffer_with_retry`, and the whole design of the
`CUDA` memory pool) never triggers. At the batch size of 2048 the leak added ≈1.6 GB per step
and froze a 36 GB machine three times before it was found — the evidence is a `JetsamEvent`
in `/Library/Logs/DiagnosticReports` naming `julia` as `largestProcess`, *not* a kernel panic.
`Metal` 1.10 does have a `maybe_collect` (`src/memory_pressure.jl`), but it only engages above
75 % of `recommendedMaxWorkingSetSize` (27 GB here) and only calls `GC.gc(false)`, which is
useless against an Objective-C retain.

Because the failure mode is silent, the script also carries a `memory_budget` — two thirds of
the recommended working set — checked at the *in-step* high water mark inside `∇F!`. A leak
that comes back stops the script with an error instead of the machine.

Verified over a full epoch at the full configuration (batch size 2048, 29 batches): device
memory stays at `0.01`–`0.05` GB between steps, the in-step peak is `9.4` GB against the
`18` GB budget, `check_gradient` reports `6.1e-8` and the loss falls from `1.06` to `0.86`.
A step takes ≈2 s, so the four full runs would be ≈32 h on an M4 Max.

One shared fix came out of this: the cotangent of a `transpose(ps.Q[i])` is a lazy
`Transpose`, which `vec` turns into a wrapped array that a device cannot copy from. `_dense`
gained a `Union{Adjoint,Transpose}` method and `_write_gradient!` now goes through it, in all
three scripts.

### The short `Metal` run

`scripts/mnist_metal_short.jl` exists because the full run is a day and a half, and a day and
a half should not be started on the strength of three epochs. It runs the same four trainings
on a schedule *derived* at runtime: after the gradient check has compiled the forward and
backward passes it times a handful of real `Adam` steps, divides the wall clock it has been
given (`MNIST_TIME_BUDGET`, two hours by default) by the four configurations and picks the
`n_epochs` that fits, and a per-run deadline truncates a configuration that overshoots its
share so that the later ones still get their time. It therefore finishes inside its budget on
any machine.

It checks what a short run can check: the gradient against a host directional derivative for
both initializations, that the three constrained configurations reduce their loss and classify
above a floor, that the unconstrained one sits at the plateau of `1.342` it is supposed to sit
at, that the parameters stay finite and on the manifold, and that device memory stays inside
its budget — and it ends on a verdict per configuration rather than a table to be read by eye.
What it cannot answer is the accuracy question: a few hundred steps are a few per cent of the
schedule. Written 2026-08-08, syntax-checked, **not yet executed**.

## State

The full test suite passes (16 test sets). All four scripts run end to end; what none of them
has done yet is train to completion.

- `mnist.jl` on the host: all three algorithms take steps, the loss goes down over a single
  step (`1.128` → `1.07`–`1.09`), the parameters stay on the Stiefel manifold
  (`check ≈ 5e-6`), and `check_gradient` reports `4.9e-6`.
- `mnist_metal.jl`: three epochs of all four configurations on an M4 Max, 2026-08-07, exit 0,
  `check_gradient` `6.1e-8`, memory budget never tripped — the numbers are in the table above
  and under "the four configurations".
- `mnist_cuda.jl`: two epochs of all four configurations on an RTX 4090, 2026-08-08. The
  gradient checks and the first epoch of each configuration ran on the device; the run then
  died on `CUDA.available_memory`, which is fixed. **It has to be repeated** — the accuracy
  evaluation, the drift measurement, the epoch lines and the verdicts have never executed.
- `mnist_metal_short.jl`: never executed.

**No error is currently open.** What is *unverified* is the training itself, because the four
full runs have deliberately not been started yet.

## Next steps

1. **Confirm that the training converges.** *Largely answered, on the way to the `Metal`
   fix.* The earlier worry came from an eight-step smoke test in which the loss *increased*
   (`1.06` → `1.33`, i.e. above the `≈0.95` of a uniform prediction). Over a full epoch at the
   full configuration — 29 `Adam` steps, Stiefel weights, on the GPU — it turns around after
   roughly ten steps and falls monotonically to `0.86`, i.e. below the uniform-prediction
   level. So the rise is the transient it looked like and none of the three suspects (step
   size semantics, `L = 16` without normalization, the `norm(pred - out)/norm(out)` loss)
   needs to be chased. What is still unconfirmed is only the *accuracy* the full runs reach.

   This applies to the three configurations with weights on the Stiefel manifold. The fourth,
   `regular weights, Adam`, is the *baseline* and is not supposed to converge — see below.
2. ~~**Decide `n_epochs`.**~~ Settled: `scripts/mnist_cuda.jl` and `scripts/mnist_metal.jl`
   use the `500` of the original; `scripts/mnist.jl` stays at `5`, because on the host a step
   costs ≈5 s and 500 epochs × 29 batches would be ≈20 h *per configuration*.
3. **Repeat the smoke run on the RTX 4090**, now that the two failures of 2026-08-08 are
   fixed. It is `scripts/run_mnist_cuda.sh --smoke`, it takes ten minutes, and it is what
   exercises the paths the first attempt never reached (the accuracy evaluation, the drift
   measurement, the epoch lines, the verdicts) as well as giving the per-epoch time this
   machine should be extrapolated from.
4. **Then run the four configurations** (Stiefel/`Adam`, regular/`Adam`, Stiefel/gradient,
   Stiefel/momentum) and compare the accuracies with the original script. The RTX 4090 is the
   machine for this: ≈15 h by the smoke run's ≈1.0 s/step, against the ≈36 h the M4 Max would
   need at 2.13–2.30 s/step — and the M4 Max is the machine the user works on.

   Two things to hold on to when those numbers come back. **A single accuracy from either
   `Adam` configuration is a sample, not a number:** constrained `Adam` does not reproduce run
   to run, because at the first step its update is `m̂/(√v̂+δ) ≈ sign(g)` per coordinate and
   the gradient has vanished through the 16 unnormalized blocks, so coordinates sitting at ~0
   have their sign decided by the last ulp. Two host runs with the same seed, one epoch, gave
   accuracies `0.3761` and `0.3302` and drifts `4.6e-04` and `3.7e-05` off the manifold, while
   `GradientMethod` and `MomentumMethod` reproduced to four decimals. A small gap against the
   original is therefore not a regression, and an accuracy that is meant to be quoted comes
   from `scripts/run_mnist_repetitions.sh` (see *Repetitions* above), which trains the
   configuration five times and reports `mean ± deviation`. That is also why `mnist_cuda.jl` uses an
   `orthonormality_tolerance` of `1e-2` where `mnist_metal_short.jl` uses `1e-4` — one of
   those two samples already exceeded `1e-4` after 29 steps. **And the drift series is worth
   reading:** it is recorded every 25 epochs, so the full run answers what a single number
   cannot. If it grows with the step count rather than settling, that is a real finding, and
   `mnist_metal_short.jl`'s `1e-4` needs revisiting with it.
5. **Optional: `Enzyme` as a second gradient backend**, for comparison with `Zygote`. Not
   attempted so far.
6. **Optional, out of scope so far:** a bare `Manifold` (not wrapped in a `NamedTuple`)
   cannot be used with `Adam`. `AdamState(x::StiefelManifold)` stores its gradient as
   `_zero(x)`, whereas `AdamCache` uses the `StiefelLieAlgHorMatrix` form, and the
   `_copyto!` between the two hits a generic `copyto!` that needs `setindex!`. The existing
   test suite only combines bare manifolds with `GradientMethod`, so this is pre-existing;
   `test/optimizer_state_initialization.jl` documents it by testing the first `Adam` step
   only for the `NamedTuple` case.
7. **Optional:** `Optimizer` should convert `Adam` to the element type of the parameters, the
   way it already does for `MomentumMethod` (see the remark on `Adam{Float64}` above). One
   line, but it changes the public behaviour, so it is not part of this branch.
8. ~~`MomentumMethod` accumulates `p ← p + α∇L` and uses `-(∇L + p)` as the direction, i.e.
   `p` is a scaled *sum* of all past gradients rather than the usual exponentially weighted
   average.~~ Fixed on the `unify-interfaces` branch (issue #18): the recursion is
   `p ← αp + ∇L` with direction `-p`.
