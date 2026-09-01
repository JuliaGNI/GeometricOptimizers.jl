"""
    NoStepObserver()

The default no-op observer used by [`Optimizer`](@ref).

Pass any callable `observer` to `Optimizer(...; observer=observer)` to receive
`observer(phase, event)` notifications, where `event` is `:enter` or `:exit` and
`phase` is one of `:gradient`, `:objective`, or `:retraction_application`.
Notifications are properly nested and an `:exit` is emitted from a `finally`
block if the observed operation throws.

Callers that need an exclusive optimizer-state timer can bracket
`solver_step!` and the following `update!` with
[`observe_optimizer_phase`](@ref)`(observer, :optimizer_state_direction)`. A
stack-based observer can then pause the outer phase while any nested phase is
active. The package deliberately reports boundaries rather than wall-clock
values so that a CUDA caller can synchronize the device immediately before
each timestamp.
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
struct ObservedGradient{T,GT<:Gradient{T},OT} <: Gradient{T}
    gradient::GT
    observer::OT
end

@inline function (g::ObservedGradient{T})(x::AbstractVector{T}) where {T}
    observe_optimizer_phase(g.observer, :gradient) do
        g.gradient(x)
    end
end


@inline function (g::ObservedGradient{T})(dest::AbstractVector{T}, x::AbstractVector{T}) where {T}
    observe_optimizer_phase(g.observer, :gradient) do
        g.gradient(dest, x)
    end
end

_observed_gradient(g::Gradient, ::NoStepObserver) = g
_observed_gradient(g::Gradient{T}, observer) where {T} =
    ObservedGradient{T,typeof(g),typeof(observer)}(g, observer)
