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

To which the line search adds a fourth: an **objective** evaluation for every trial step it takes.

The question a user of a geometric optimizer asks about this is a comparison. Standard `Adam` needs
only the first two of the three; the geometric version pays for the third as well. *How much* does it
pay? The answer is not a constant. The retraction is an ``N \times N`` matrix operation on the host
and a small number of kernels on a device, while the gradient is a whole reverse pass over a network;
on a CPU the geometry may well disappear next to automatic differentiation, and on a GPU it may not,
because the reverse pass parallelizes and much of the bookkeeping does not.

A stopwatch around the step cannot answer it. One number per step is one number: it cannot say which
fraction of a step was spent differentiating and which was spent retracting. And the split cannot be
recovered from outside the package either, because the boundaries are not at the outside:

* the retraction is applied **more than once per step** — once per iteration of the ``\mathrm{NaN}``
  guard in `solver_step!`, once for the accepted step, once more if the line search rejects every
  trial and the step is retried along the steepest-descent direction, and once again in
  `GeometricOptimizers.update!` when the state's own section is advanced;
* it is also applied **once per line-search trial**, inside [`GeometricOptimizers.trial_iterate!`](@ref), which is called
  from the merit function the [line search](@ref "Linesearches for Optimizers") owns;
* the objective is evaluated once per trial in the same place.

A caller holding only `solver_step!` and `update!` sees none of those boundaries. It can time the two
calls, and that total is the sum of all four kinds of work, in a proportion that depends on how many
trials the line search happened to take.

## Why the package does not simply time itself

The obvious alternative — have the optimizer measure its own phases and hand back a table — is worse,
for three reasons.

*A timestamp on a device is not a timestamp.* GPU kernels are launched asynchronously, so the elapsed
time between two host clock readings around a launch is the time it took to *queue* the work, not to
do it. Making the reading meaningful requires synchronizing the device immediately before it. The
package cannot do that on the caller's behalf: synchronizing costs real time and serializes work that
would otherwise overlap, so an uninstrumented run would be slowed by measurement it never asked for.
Whoever is measuring has to decide when to synchronize.

*The clock is the caller's choice.* `time_ns`, a monotonic wall clock, CUDA events, or a counter of
calls rather than of seconds are all reasonable, and they are not interchangeable.

*So is the bookkeeping.* Accumulating per phase, keeping per-step vectors, reporting medians rather
than means, writing rows to a file — these are properties of an experiment, not of an optimizer.

## The fix: the package reports boundaries, the caller owns the clock

An **observer** is a callable the caller installs on an [`Optimizer`](@ref). The optimizer notifies it
immediately before and immediately after each of the phases above:

```julia
observer(phase, event)
```

`phase` names the work and `event` is `:enter` or `:exit`. That is the whole interface. The observer
is told *when* a boundary is crossed and nothing else — no duration, no unit, no storage. A caller
that wants seconds reads its own clock in the callback; a caller on a GPU synchronizes the device
first; a caller that only wants to count calls never reads a clock at all.

The default is [`NoStepObserver`](@ref), which does nothing, and for which the package calls the
observed operation directly rather than through the notification path. Supplying no observer
therefore costs nothing.[^1]

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
the package uses internally. Bracketing `solver_step!` and `update!` with an outer phase and letting
the three inner phases subtract themselves from it yields the second of the three costs above — the
optimizer's state and direction bookkeeping — without the package needing a phase for it:

```julia
observe_optimizer_phase(observer, :optimizer_state_direction) do
    solver_step!(x, state, opt)
    GeometricOptimizers.update!(state, opt, x)
end
```

[`step_observer`](@ref) reads back the observer installed on an optimizer.

## Example: the events of a single step

The smallest useful observer records what it is told. Printed with one level of indentation per open
phase, the structure of a step becomes visible:

```@example observers
using GeometricOptimizers
using GeometricOptimizers: increase_iteration_number!, initialize_state!, solver_step!,
                           update!

mutable struct EventLog
    events::Vector{Tuple{Symbol, Symbol}}
end

(recorder::EventLog)(phase, event) = (push!(recorder.events, (phase, event)); nothing)

function print_nested(events)
    depth = 0
    for (phase, event) in events
        event === :exit && (depth -= 1)
        println("  "^depth, event === :enter ? "┌ " : "└ ", phase)
        event === :enter && (depth += 1)
    end
end

x = [1.0, -2.0]
loss(x) = sum(abs2, x)

recorder = EventLog(Tuple{Symbol, Symbol}[])
method = GradientMethod()
opt = Optimizer(x, loss; algorithm = method, linesearch = Static(0.1),
    observer = recorder)
state = OptimizerState(method, x)
initialize_state!(state)

observe_optimizer_phase(recorder, :optimizer_state_direction) do
    increase_iteration_number!(state)
    solver_step!(x, state, opt)
    update!(state, opt, x)
end

print_nested(recorder.events)
```

Reading the trace: the first `:gradient` is the gradient at the current iterate, which the cache turns
into a direction. The `:retraction_application`/`:objective` pair after it is the ``\mathrm{NaN}``
guard building a trial point and checking that the objective there is finite. The next
`:retraction_application` applies the accepted step. The second `:gradient` is the one taken *at the
point the step ended at*, so that the convergence measures describe the iterate the step returns
rather than the one it started from, and the last two pairs are
`GeometricOptimizers.update!` recording the new objective value and advancing the state's
section.

The `Static(0.1)` line search takes no trials of its own. With a searching line search —
`Backtracking`, the default — the same trace additionally contains one `:objective` per trial and the
`:retraction_application` that built each trial point.

## Example: exclusive time per phase

This is the observer the decomposition asks for. It keeps a stack of open phases; opening a nested
phase stops the clock on its parent and closing it starts the parent's clock again, so the
accumulated times are mutually exclusive.

```@example observers
mutable struct PhaseTimer
    open::Vector{Tuple{Symbol, UInt64}}     # phase, when it last (re)started
    exclusive::Dict{Symbol, UInt64}         # accumulated nanoseconds, excluding nested phases
    calls::Dict{Symbol, Int}
end

PhaseTimer() = PhaseTimer(Tuple{Symbol, UInt64}[], Dict{Symbol, UInt64}(),
    Dict{Symbol, Int}())

# Charge the innermost open phase for everything since it last started.
function charge!(timer::PhaseTimer, t::UInt64)
    isempty(timer.open) && return nothing
    phase, since = timer.open[end]
    timer.exclusive[phase] = get(timer.exclusive, phase, UInt64(0)) + (t - since)
    nothing
end

function (timer::PhaseTimer)(phase::Symbol, event::Symbol)
    t = time_ns()                     # a GPU caller synchronizes the device here
    if event === :enter
        charge!(timer, t)             # the enclosing phase stops being charged
        timer.calls[phase] = get(timer.calls, phase, 0) + 1
        push!(timer.open, (phase, time_ns()))
    else
        charge!(timer, t)
        pop!(timer.open)
        isempty(timer.open) || (timer.open[end] = (timer.open[end][1], time_ns()))
    end
    nothing
end
```

Run over the same step, it reports how often each phase was entered and how much time belongs to it
alone:

```@example observers
y = [1.0, -2.0]
timer = PhaseTimer()
opt = Optimizer(y, loss; algorithm = GradientMethod(), linesearch = Static(0.1),
    observer = timer)
state = OptimizerState(GradientMethod(), y)
initialize_state!(state)

observe_optimizer_phase(timer, :optimizer_state_direction) do
    increase_iteration_number!(state)
    solver_step!(y, state, opt)
    update!(state, opt, y)
end

for phase in sort(collect(keys(timer.calls)))
    println(rpad(phase, 26), timer.calls[phase], " call(s)")
end
```

The durations are deliberately not printed here — on a single step of a two-parameter problem they are
dominated by the clock's own resolution, and the point of the example is the accounting rather than the
numbers. What the accounting gives is the comparison the measurement was wanted for: the time in
`:retraction_application` is the price of the geometry, the time in `:gradient` is the price of
differentiation, `:objective` is the line search's own cost, and what is left in the outer
`:optimizer_state_direction` is the optimizer's bookkeeping.

Two things are worth doing before believing such numbers, neither of which the package can do for the
caller. Discard a warm-up step: Julia compiles on first call, and on a short run that compilation can
exceed everything being measured. And on a device, synchronize before every timestamp — the marked
line above — or the intervals describe kernel launches rather than kernel execution.

## What else observers are good for

**Counting work rather than timing it.** The number of `:objective` pairs in a step is the number of
trials the line search took, which is a property of the search and the problem and is often more
informative than its duration. The same count over a whole run distinguishes a line search that
accepts its first trial from one that backtracks repeatedly.

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

The phases above are emitted from the step machinery: `solver_step!`, the line-search merit and slope
functions, and the `update!` methods of [`GradientMethod`](@ref), [`MomentumMethod`](@ref),
[`Adam`](@ref) and [`ScalarMomentAdam`](@ref).

Three boundaries need stating, and a caller doing arithmetic on the totals should know all three.

* [`solve!`](@ref) evaluates the objective for the [`GeometricOptimizers.OptimizerStatus`](@ref) and, when tracing, for
  each trace entry. Those evaluations are not inside an `:objective` phase, so an observer attached to
  a whole `solve!` will report fewer objective evaluations than the objective actually receives.
  Driving the loop directly — `solver_step!` and `update!`, as the examples above do — avoids the
  discrepancy.
* The end-of-iteration state update of the quasi-Newton methods, [`BFGS`](@ref) and [`DFP`](@ref),
  evaluates the objective and advances its section without notifying the observer. Their
  `solver_step!` is observed as usual; it is the `update!` that is not.
* The slope ``\varphi'(\alpha)`` a differentiable line search asks for is reported as
  `:retraction_application`, because on a manifold that is what most of it is: the differential of the
  retraction, paired against the globally represented gradient. The gradient evaluation it needs is
  nested inside as a `:gradient` pair and is therefore subtracted from it by an exclusive timer. For a
  plain vector iterate there is no retraction differential and what remains under that label is an
  inner product. `Backtracking`, the default, asks for the slope once per step, at ``\alpha = 0``.

For a whole set of parameters, the observed gradient sits *inside* the
[`RiemannianGradient`](@ref) wrapper. `:gradient` therefore covers the flat gradient and the
differentiation, and the leaf-by-leaf projection onto the tangent spaces is charged to the enclosing
phase instead — which is also what keeps the `NetworkParameters` dispatch that the projection needs.
