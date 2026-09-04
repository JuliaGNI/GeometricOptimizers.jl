```@raw latex
The optimizers in the previous chapters are presented as single operations: a step is taken and the
parameters move. A step is in fact assembled from several distinct pieces of work, and a caller who
wants to know what a step \textit{cost} has to be able to tell them apart. This chapter describes the
mechanism the package offers for that, the \textit{step observer}, states the problem it solves, and
shows what else it can be used for.
```

# Observing the Phases of an Optimizer Step

## The problem: one step, several costs

One call to `solver_step!` followed by `GeometricOptimizers.update!` is not one operation. On
a manifold it is at least three:

1. the **gradient** — the reverse pass or the automatic differentiation of the objective, evaluated in
   the flattened coordinates;
2. the **optimizer state and direction** — the cache and state bookkeeping that turns that gradient
   into a direction in the [global tangent space](@ref "Global Tangent Spaces")
   ``\mathfrak{g}^\mathrm{hor}``: the moments of [`Adam`](@ref), the inverse-Hessian update of
   [`BFGS`](@ref), the secant pairs;
3. the **retraction and its application** — [`geodesic`](@ref) or [`cayley`](@ref) on the lift, the
   [`update_section!`](@ref) that re-bases the [`GlobalSection`](@ref), and the copy of the result
   back onto the parameters.

To which a searching line search adds a fourth: an **objective** evaluation for every trial step it
takes. [`Static`](@extref SimpleSolvers.Static) selects its fixed step without evaluating a trial.

The question a user of a geometric optimizer asks about this is a comparison. Standard `Adam` needs
only the first two of the three; the geometric version pays for the third as well. *How much* does it
pay? The answer is not a constant. The retraction is an ``N \times N`` matrix operation on the host
and a small number of kernels on a device, while the gradient is a whole reverse pass over a network;
on a CPU the geometry may well disappear next to automatic differentiation, and on a GPU it may not,
because the reverse pass parallelizes and much of the bookkeeping does not.

A stopwatch around the step cannot answer it. One number per step is one number: it cannot say which
fraction of a step was spent differentiating and which was spent retracting. And the split cannot be
recovered from outside the package either, because the boundaries are not at the outside:

* the step machinery applies the retraction in the ``\mathrm{NaN}`` guard, for the accepted step,
  once more if a rejected search is retried along the steepest-descent direction, and when
  `GeometricOptimizers.update!` advances the state's own section;
* a **searching** line search additionally applies it once per trial, inside
  [`GeometricOptimizers.trial_iterate!`](@ref), which is called from the merit function the
  [line search](@ref "Linesearches for Optimizers") owns, and evaluates the objective in the same
  place. `Static` performs neither of these trial operations.

A caller holding only `solver_step!` and `update!` sees none of those boundaries. It can time the two
calls, and that total is the sum of all four kinds of work, in a proportion that depends on how many
trials the line search happened to take.

## Why the optimizer does not time every step automatically

The obvious alternative — have the optimizer measure its own phases and hand back a table — is worse,
for three reasons.

*A timestamp on a device is not a timestamp.* GPU kernels are launched asynchronously, so the elapsed
time between two host clock readings around a launch is the time it took to *queue* the work, not to
do it. Making the reading meaningful requires synchronizing the device immediately before it. The
package cannot do that on the caller's behalf: synchronizing costs real time and serializes work that
would otherwise overlap, so an uninstrumented run would be slowed by measurement it never asked for.
Whoever is measuring has to decide when to synchronize.

*The clock is the caller's choice.* `time_ns`, a monotonic wall clock, CUDA events, or a counter of
calls rather than of seconds are all reasonable, and they are not interchangeable. What
[`PhaseTimer`](@ref) requires of one is only that it count in whole nanoseconds and never go
backwards; `Base.time` is not usable, since it returns seconds as a `Float64`.

*So is the bookkeeping.* The package supplies a basic event log and exclusive timer, but choices such
as keeping per-step vectors, reporting medians rather than means, or writing rows to a file remain
properties of an experiment, not of an optimizer.

## The fix: observed boundaries and ready-made recorders

An **observer** is a callable the caller installs on an [`Optimizer`](@ref). The optimizer notifies it
immediately before and immediately after each of the phases above:

```julia
observer(phase, event)
```

`phase` names the work and `event` is `:enter` or `:exit`. That is the whole callback interface. The
observer is told *when* a boundary is crossed and nothing else. [`EventLog`](@ref) and
[`PhaseTimer`](@ref) implement the common recording and timing cases; callers can still supply any
callable with this interface when they need different storage or behavior.

The default is [`NoStepObserver`](@ref), which does nothing, and for which the package calls the
observed operation directly rather than through the notification path. Supplying no observer
therefore costs nothing.[^1]

`EventLog()` records every `(phase, event)` pair. `PhaseTimer()` counts calls and accumulates
exclusive nanoseconds for every phase. Both accept `phases=:gradient` or an iterable such as
`phases=(:gradient, :retraction_application)` to record less. A GPU caller can construct
`PhaseTimer(synchronize=CUDA.synchronize)` so that every host timestamp follows a device
synchronization; `clock` is configurable as well. These are conveniences rather than restrictions:
a custom callable remains a valid observer.

[^1]: Measured on this package's own default step, the allocation figures for `GradientMethod`, `MomentumMethod`, `Adam` and `BFGS` are identical with and without the observer machinery present, and specialization on [`NoStepObserver`](@ref) removes the notification calls entirely.

### The protocol

Three properties are guaranteed, and a stack-based observer relies on all three.

**Every `:enter` is matched by exactly one `:exit`.** The exit notification is emitted from a
`finally` block, so it arrives even when the observed operation throws. An observer that pushes on
`:enter` and pops on `:exit` therefore cannot be left with a corrupted stack by a `NaN` in the
objective or a failure in the reverse pass.

**Notifications nest.** A gradient evaluation that happens inside a line-search trial is reported as
a `:gradient` pair *inside* the enclosing pair, never as a sibling. This is what makes *exclusive*
phase times possible: an observer can stop charging the enclosing phase when a nested one opens and
resume when it closes, so each phase is credited only with the work that belongs to it and the phases
sum to the whole without double counting.

**The optimizer's control flow does not depend on the observer.** The notifications are reports. An
observer cannot change the step, and a step behaves the same whether one is installed or not.

### The phases

| `phase` | What is inside it |
|:---|:---|
| `:gradient` | The gradient evaluation itself: the reverse pass, the automatic differentiation, or the user's in-place `∇F!`, over the flattened iterate. |
| `:objective` | An evaluation of the objective. |
| `:retraction_application` | Construction and application of a retraction: the exponential or Cayley map on the lift, the [`update_section!`](@ref) that follows it, and the copy of the retracted point onto the parameters. |

`:objective` is in that list mainly so that it can be *subtracted*. Objective evaluations are driven
by the line search, they are the caller's own function, and their cost has nothing to do with the
geometry; keeping them in a phase of their own is what stops them from being charged to the retraction
or to the gradient.

A caller may add phases of its own with [`observe_optimizer_phase`](@ref), which is the same helper
the package uses internally. Bracketing the whole optimization in one and letting the three inner
phases subtract themselves from it yields the second of the three costs above — the optimizer's own
bookkeeping — without the package needing a phase for it:

```julia
observe_optimizer_phase(observer, :bookkeeping) do
    solve!(x, OptimizerState(method, x), opt)
end
```

A caller who drives the loop itself rather than calling [`solve!`](@ref) brackets `solver_step!` and
`update!` the same way; the outer phase does not care what is inside it.

[`step_observer`](@ref) reads back the observer installed on an optimizer.

## Example: the events of one iteration

Installing an [`EventLog`](@ref) is the whole setup. Printed with one level of indentation per open
phase, the structure of an iteration becomes visible:

```@example observers
using GeometricOptimizers

loss(x) = sum(abs2, x)
x = [1.0, -2.0]

method = GradientMethod()
recorder = EventLog()
opt = Optimizer(x, loss; algorithm = method, max_iterations = 1, observer = recorder)

solve!(x, OptimizerState(method, x), opt)

function print_nested(events, depth = 0)
    for (phase, event) in events
        event === :exit && (depth -= 1)
        println("  "^depth, event === :enter ? "┌ " : "└ ", phase)
        event === :enter && (depth += 1)
    end
end

print_nested(recorder.events)
```

Reading the trace from the top: the leading `:objective` is [`solve!`](@ref) evaluating the objective
at the starting iterate. The first `:gradient` is the gradient the cache turns into a direction, and
the `:retraction_application`/`:objective` pair after it is the ``\mathrm{NaN}`` guard, which builds a
trial point and checks that the objective there is finite.

The middle of the trace belongs to `Backtracking`, the default line search: one
`:retraction_application` and one `:objective` for each trial step it takes, plus the pair in which a
`:gradient` is *nested* — that is the slope ``\varphi'(\alpha)`` the search asks for once per step, and
the nesting is what lets an exclusive timer charge the differentiation to `:gradient` rather than to
the retraction. Passing `linesearch = Static(0.1)` removes all of it: a fixed step takes no trials.

The last `:retraction_application` applies the accepted step, the `:gradient` after it is taken *at the
point the step ended at* — so that the convergence measures describe the iterate the step returns
rather than the one it started from — and the two trailing `:objective` pairs are `solve!` evaluating
the objective at the final iterate: once inside the loop, for the status that stopping is decided on,
and once after it, for the status the result carries. The trace entry and the returned result reuse
the value evaluated for the status at their own iterate rather than evaluating again.

## Example: exclusive time per phase

[`PhaseTimer`](@ref) keeps a stack of open phases; opening a nested phase stops the clock on its
parent and closing it starts the parent's clock again, so the accumulated times are mutually
exclusive. No recorder implementation is needed in user code:

A timer records two numbers per phase. `timer.calls[phase]` counts how many times the phase was
entered, and `timer.exclusive[phase]` accumulates the time spent in it, **in nanoseconds**. The first
`solve!` below is a warm-up whose measurements are thrown away by `empty!`, for the reason given
after the example:

```@example observers
y = [1.0, -2.0]
timer = PhaseTimer()
opt = Optimizer(y, loss; algorithm = method, max_iterations = 1, observer = timer)

solve!(y, OptimizerState(method, y), opt)   # warm-up: compiled, then discarded
empty!(timer)

y .= [1.0, -2.0]
observe_optimizer_phase(timer, :bookkeeping) do
    solve!(y, OptimizerState(method, y), opt)
end

for phase in sort!(collect(keys(timer.calls)))
    println(rpad(phase, 26), timer.calls[phase], " call(s)  ",
        timer.exclusive[phase], " ns")
end
```

`:bookkeeping` is the outer phase from above, executed, and it is what turns three measurements into
four. The nanoseconds now split the whole optimization into the price of the geometry
(`:retraction_application`), the price of differentiation (`:gradient`), the caller's own objective
(`:objective`), and — as the remainder left once the three inner phases have subtracted themselves —
the direction computation, the cache and state updates, the line search's control flow and the
convergence tests. Its call count is `1` because the caller opened it once: a count is a count of
*entries into the phase*, which is why `:objective` here is close to the number of line-search trials
and not to anything measured in time.

Two things are worth doing before believing the durations, neither of which the package can do for the
caller. The warm-up is the first: Julia compiles on first call, and on a run this short that
compilation lands in whichever phase was open when it happened — without the discarded run above,
`:bookkeeping` reads in the hundreds of milliseconds against microseconds for everything else, and the
comparison the measurement was wanted for is destroyed. The second is that on a device you must
synchronize before every timestamp — `PhaseTimer(synchronize=CUDA.synchronize)` does this — or the
intervals describe kernel launches rather than kernel execution.

Even warmed, these particular numbers are a two-parameter problem measured at `time_ns` resolution, so
read the shape rather than the digits.

## What else observers are good for

**Counting work rather than timing it.** The `:objective` pairs a step emits are one per line-search
trial plus the ``\mathrm{NaN}`` guard's, so their number is a property of the search and the problem
and is often more informative than its duration. The count over a whole run distinguishes a line
search that accepts its first trial from one that backtracks repeatedly.

**Progress and logging.** An observer that prints, or updates a progress bar, on `:enter` of
`:gradient` gives a per-step heartbeat for a long training run without the optimizer needing a
verbosity setting for it.

**Diagnosis.** When a step misbehaves, the event trace says where it went. A `:retraction_application`
that repeats several times before the first `:objective` is the ``\mathrm{NaN}`` guard shortening the
direction; a long run of `:objective` pairs is a line search that cannot find a decrease.

**Tests.** Because the protocol guarantees matched, nested pairs, an observer is a way for a test to
assert that a code path was taken at all — which is how this package's own
`test/optimizer_observer.jl` checks that every first-order method reports the same phase structure.

## Coverage

The phases above are emitted from the complete [`solve!`](@ref) loop, the step machinery, the
line-search merit and slope functions, and the `update!` methods of [`GradientMethod`](@ref),
[`MomentumMethod`](@ref), [`Adam`](@ref), [`ScalarMomentAdam`](@ref),
[`Newton`](@ref GeometricOptimizers.Newton), [`BFGS`](@ref) and [`DFP`](@ref). Every direct
objective evaluation made by `solve!` is reported as `:objective`, including the ones made for
[`GeometricOptimizers.OptimizerStatus`](@ref). Trace entries and the returned result carry the value
already evaluated for the status at the same iterate rather than evaluating again, so they
contribute no events of their own.

One boundary needs stating for a caller doing arithmetic on the totals:

* The slope ``\varphi'(\alpha)`` a differentiable line search asks for is reported as
  `:retraction_application`, because on a manifold that is what most of it is: the differential of the
  retraction, paired against the globally represented gradient. The gradient evaluation it needs is
  nested inside as a `:gradient` pair and is therefore subtracted from it by an exclusive timer. For a
  plain vector iterate there is no retraction differential and what remains under that label is an
  inner product. `Backtracking`, the default, asks for the slope once per step, at ``\alpha = 0``.
  A consequence for `calls` rather than for `exclusive`: `:retraction_application` is entered twice
  per slope request — once for the trial point and once for the slope itself — so its call count
  exceeds the number of retractions actually applied by one per request.

For a whole set of parameters, the observed gradient sits *inside* the
[`RiemannianGradient`](@ref) wrapper. `:gradient` therefore covers the flat gradient and the
differentiation, and the leaf-by-leaf projection onto the tangent spaces is charged to the enclosing
phase instead — which is also what keeps the `NetworkParameters` dispatch that the projection needs.
