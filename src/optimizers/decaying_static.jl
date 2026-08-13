@doc raw"""
    DecayingStatic(T; η₁, η₂, n)

A [`SimpleSolvers.LinesearchMethod`](@extref) that takes no search at all, like
[`SimpleSolvers.Static`](@extref), but whose step size decays geometrically with the iteration
number:

```math
\alpha(t) = \gamma^t\eta_1, \qquad \gamma = \exp(\log(\eta_2/\eta_1)/n),
```

so that ``\alpha(0) = \eta_1`` and ``\alpha(n) = \eta_2``. It keeps decaying past `n`; the decay is
not floored at ``\eta_2``, because a floor is exactly what stops the iteration from converging (see
below).

# Why this exists

[`Adam`](@ref) produces a direction of magnitude ``\approx{}1`` per component whatever the gradient
is — that scale-freeness is the point of it. With a constant step ``\alpha`` the iterate therefore
does not converge to the minimizer but circles it at a distance of order ``\alpha``, so no
gradient-based stopping criterion is ever met and the solve runs until `max_iterations`. A step size
that goes to zero is what turns it back into a convergent iteration.

This is what the `AdamWithDecay` method of v0.1.0 did with its own `η₁`, `η₂` and `n_epochs` fields.
It was removed when the step size moved out of the [`OptimizerMethod`](@ref)s and into the line
search; this is the replacement, and being a `LinesearchMethod` it composes with
[`GradientMethod`](@ref) and [`MomentumMethod`](@ref) just as well.

# Examples

```jldoctest; setup = :(using GeometricOptimizers)
ls = DecayingStatic(; η₁ = 1e-2, η₂ = 1e-6, n = 1000)

# output

DecayingStatic from α = 0.01 to α = 1.0e-6 over 1000 iterations.
```

!!! info "The iteration number comes from the state"
    `solve` reads `iteration_number(params.state)`, which [`solver_step!`](@ref) puts into the line
    search's parameters. A `DecayingStatic` handed a `params` without a `state` therefore cannot
    work, and says so.
"""
struct DecayingStatic{T<:Number} <: LinesearchMethod{T}
    η₁::T
    η₂::T
    γ::T
    n::Int

    function DecayingStatic(::Type{T}=Float64; η₁=T(1.0e-2), η₂=T(1.0e-6), n::Integer=1000) where {T}
        @assert η₁ > 0 && η₂ > 0 "the step sizes have to be positive, got η₁ = $(η₁) and η₂ = $(η₂)"
        @assert η₂ ≤ η₁ "this decays, so η₂ = $(η₂) has to be at most η₁ = $(η₁)"
        @assert n > 0 "the decay horizon has to be positive, got n = $(n)"
        new{T}(T(η₁), T(η₂), T(exp(log(η₂ / η₁) / n)), Int(n))
    end
end

"""
    step_size(method, t)

The step size a [`DecayingStatic`](@ref) takes in iteration `t`, counted from `t = 0`.
"""
step_size(method::DecayingStatic{T}, t::Integer) where {T} = method.γ^t * method.η₁

function solve(ls::Linesearch{T,<:DecayingStatic}, ::T, params) where {T}
    hasproperty(params, :state) ||
        error("DecayingStatic needs the iteration number and therefore the `state` in the line " *
              "search parameters; `solver_step!` passes it, a bare `solve(ls, α)` does not.")
    step_size(method(ls), iteration_number(params.state))
end

# see the remark on `Static`: nothing is searched, so there is no decrease to report
solve_with_status(ls::Linesearch{T,<:DecayingStatic}, α::T, params) where {T} =
    LinesearchStatus(solve(ls, α, params), LINESEARCH_UNKNOWN)

function change_precision(::Type{T}, method::DecayingStatic) where {T}
    T ≠ eltype(method) || return method
    DecayingStatic(T; η₁=T(method.η₁), η₂=T(method.η₂), n=method.n)
end

Base.show(io::IO, alg::DecayingStatic) =
    print(io, "DecayingStatic from α = ", alg.η₁, " to α = ", alg.η₂, " over ", alg.n, " iterations.")
