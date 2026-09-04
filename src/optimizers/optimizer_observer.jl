_observer_phases(::Nothing) = nothing
_observer_phases(phase::Symbol) = Set{Symbol}((phase,))
_observer_phases(phases) = Set{Symbol}(phases)

_records_phase(::Nothing, phase::Symbol) = true
_records_phase(phases, phase::Symbol) = phase in phases

"""
    EventLog(; phases=nothing)

Record optimizer phase notifications as `(phase, event)` tuples in `events`.

By default every phase is recorded. Pass a phase name or an iterable of phase
names as `phases` to retain only those phases. Call `empty!(log)` between runs
to reuse an existing log.

# Examples

```jldoctest
julia> recorder = EventLog();

julia> observe_optimizer_phase(recorder, :gradient) do
           nothing
       end

julia> recorder.events
2-element Vector{Tuple{Symbol, Symbol}}:
 (:gradient, :enter)
 (:gradient, :exit)
```
"""
struct EventLog{P}
    events::Vector{Tuple{Symbol, Symbol}}
    phases::P
end

function EventLog(; phases = nothing)
    selected_phases = _observer_phases(phases)
    EventLog{typeof(selected_phases)}(Tuple{Symbol, Symbol}[], selected_phases)
end

function (log::EventLog)(phase::Symbol, event::Symbol)
    _records_phase(log.phases, phase) && push!(log.events, (phase, event))
    nothing
end

function Base.empty!(log::EventLog)
    empty!(log.events)
    log
end

_no_synchronize() = nothing

"""
    PhaseTimer(; phases=nothing, clock=time_ns, synchronize=_no_synchronize)

Accumulate exclusive phase times in nanoseconds.

The `exclusive` dictionary stores the accumulated time for each phase and
`calls` stores how often that phase was entered. Nested phases are excluded
from their parent. By default every phase is recorded; pass a phase name or an
iterable of phase names as `phases` to record less. Unselected nested phases
are still excluded from a selected parent's time.

`clock` defaults to `time_ns` and must return a monotonically increasing count
of nanoseconds that `UInt64` accepts exactly. `Base.time` is therefore not a
usable clock: it returns seconds as a `Float64` and the conversion throws an
`InexactError`. Differences are taken in `UInt64`, so a clock that ever goes
backwards underflows rather than reporting a negative interval.

`synchronize` is called immediately before every clock
reading and defaults to a no-op; a GPU caller can pass its device
synchronization function. Call `empty!(timer)` between runs to reset its
accumulators.
"""
struct PhaseTimer{P, C, S}
    open::Vector{Tuple{Symbol, UInt64, Bool}}
    exclusive::Dict{Symbol, UInt64}
    calls::Dict{Symbol, Int}
    phases::P
    clock::C
    synchronize::S
end

function PhaseTimer(; phases = nothing, clock = time_ns, synchronize = _no_synchronize)
    selected_phases = _observer_phases(phases)
    PhaseTimer{typeof(selected_phases), typeof(clock), typeof(synchronize)}(
        Tuple{Symbol, UInt64, Bool}[], Dict{Symbol, UInt64}(), Dict{Symbol, Int}(),
        selected_phases, clock, synchronize)
end

@inline function _timestamp(timer::PhaseTimer)
    timer.synchronize()
    UInt64(timer.clock())
end

function _charge_open_phase!(timer::PhaseTimer, timestamp::UInt64)
    isempty(timer.open) && return nothing
    phase, since, record = timer.open[end]
    record || return nothing
    timer.exclusive[phase] = get(timer.exclusive, phase, UInt64(0)) + (timestamp - since)
    nothing
end

function (timer::PhaseTimer)(phase::Symbol, event::Symbol)
    timestamp = _timestamp(timer)
    if event === :enter
        _charge_open_phase!(timer, timestamp)
        record = _records_phase(timer.phases, phase)
        record && (timer.calls[phase] = get(timer.calls, phase, 0) + 1)
        push!(timer.open, (phase, timestamp, record))
    elseif event === :exit
        isempty(timer.open) &&
            throw(ArgumentError("cannot exit phase $phase: no phase is open"))
        open_phase = timer.open[end][1]
        open_phase === phase ||
            throw(ArgumentError("cannot exit phase $phase while phase $open_phase is open"))
        _charge_open_phase!(timer, timestamp)
        pop!(timer.open)
        if !isempty(timer.open)
            parent_phase, _, record = timer.open[end]
            timer.open[end] = (parent_phase, timestamp, record)
        end
    else
        throw(ArgumentError("optimizer phase event must be :enter or :exit, got $event"))
    end
    nothing
end

function Base.empty!(timer::PhaseTimer)
    empty!(timer.open)
    empty!(timer.exclusive)
    empty!(timer.calls)
    timer
end

"""
    NoStepObserver()

The default no-op observer used by [`Optimizer`](@ref).

Pass an [`EventLog`](@ref), a [`PhaseTimer`](@ref), or any callable `observer`
to `Optimizer(...; observer=observer)` to receive
`observer(phase, event)` notifications, where `event` is `:enter` or `:exit` and
`phase` is one of `:gradient`, `:objective`, or `:retraction_application`.
Notifications are properly nested and an `:exit` is emitted from a `finally`
block if the observed operation throws.

Callers that need an exclusive optimizer-state timer can bracket
`solver_step!` and the following `update!` with
[`observe_optimizer_phase`](@ref)`(observer, :optimizer_state_direction)`. A
stack-based observer can then pause the outer phase while any nested phase is
active. The optimizer itself reports boundaries rather than choosing a clock;
[`PhaseTimer`](@ref) reads the caller-selected clock and can synchronize a CUDA
device immediately before each timestamp.
"""
struct NoStepObserver end

@inline (::NoStepObserver)(phase, event) = nothing

"""
    observe_optimizer_phase(f, observer, phase)

Run `f()` between matching `observer(phase, :enter)` and
`observer(phase, :exit)` notifications. The exit notification is guaranteed
even when `f` throws.
"""
@inline observe_optimizer_phase(f, ::NoStepObserver, phase) = f()

@inline function observe_optimizer_phase(f, observer, phase)
    observer(phase, :enter)
    try
        f()
    finally
        observer(phase, :exit)
    end
end

# All gradient entry points used by SimpleSolvers ultimately reach one of its
# AbstractVector methods. Wrapping those two methods therefore observes an
# autodiff gradient, an in-place user gradient, and the vectorized call made by
# this package's Manifold/NetworkParameters adapters without changing any of
# their allocation or projection semantics.
struct ObservedGradient{T, GT <: Gradient{T}, OT} <: Gradient{T}
    gradient::GT
    observer::OT
end

@inline function (g::ObservedGradient{T})(x::AbstractVector{T}) where {T}
    observe_optimizer_phase(g.observer, :gradient) do
        g.gradient(x)
    end
end

@inline function (g::ObservedGradient{T})(
        dest::AbstractVector{T}, x::AbstractVector{T}) where {T}
    observe_optimizer_phase(g.observer, :gradient) do
        g.gradient(dest, x)
    end
end

# The wrapper reports the objective it mediates for the same reason it reports the gradient: the
# Newton state evaluates both through the gradient it was handed, so leaving this one unwrapped would
# make `:objective` undercount on that path. The closure exists only where an observer is installed.
function _objective(g::ObservedGradient)
    inner = _objective(g.gradient)
    x -> observe_optimizer_phase(g.observer, :objective) do
        inner(x)
    end
end

_observed_gradient(g::Gradient, ::NoStepObserver) = g
function _observed_gradient(g::Gradient{T}, observer) where {T}
    ObservedGradient{T, typeof(g), typeof(observer)}(g, observer)
end
