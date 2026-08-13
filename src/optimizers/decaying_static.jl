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

@doc raw"""
    AdamOptimizerWithDecay(n_epochs, T; η₁, η₂, kwargs...)

[`Adam`](@ref) paired with a [`DecayingStatic`](@ref) line search whose learning rate decays
geometrically from `η₁` to `η₂` over `n_epochs`, returned as a `NamedTuple` to splat into
[`Optimizer`](@ref):

```julia
method = AdamOptimizerWithDecay(1000)
opt    = Optimizer(x, problem; method...)
solve!(x, OptimizerState(method.algorithm, x), opt)
```

The state has to be built from `method.algorithm`, so the pairing is worth binding to a name rather
than splatting it twice.

There is no new type here and no schedule of its own: this is exactly `Adam(T)` and
`DecayingStatic(T; η₁, η₂, n = n_epochs)`, under the one name that the two together used to have.

# Why it exists

`GeometricMachineLearning` carries an `AdamOptimizerWithDecay` that bundles Adam's `ρ₁`, `ρ₂`, `δ`
with a learning-rate schedule `η₁`, `η₂`, `n_epochs`, and computes the same
``\gamma = \exp(\log(\eta_2/\eta_1)/n)``. Since 0.2.0 the two halves of that live in different
places here — the direction in an [`OptimizerMethod`](@ref), the step size in a
`SimpleSolvers.LinesearchMethod` — which is the right split but leaves no single name to migrate
that method to. This is it, so GML can delete its own copy; see
[GeometricMachineLearning#230](https://github.com/JuliaGNI/GeometricMachineLearning.jl/pull/230).

!!! warning "This decays the learning rate, not the weights"
    Despite the shared word, this has nothing to do with [`AdamWithEuclideanDecay`](@ref). That one
    decays the *weights*, by ``\lambda{}x``, and leaves the learning rate alone; this one decays the
    *learning rate* and never touches a weight. They compose, and neither implies the other — see
    the *Two unrelated decays* section of the weight-decay page.

# Arguments

`η₁`, `η₂` and `n_epochs` go to the line search; everything else is forwarded to [`Adam`](@ref), so
`β₁`, `β₂` and `δ` — GML's `ρ₁`, `ρ₂` and `δ` — keep `Adam`'s own defaults rather than a second copy
of them. `T` is the element type of the parameters and is positional, as it is for `Adam` and
`DecayingStatic`.

!!! note "What a call migrated from `GeometricMachineLearning` has to change"
    GML's signature is
    `AdamOptimizerWithDecay(n_epochs, η₁ = 1f-2, η₂ = 1f-6, ρ₁ = 9f-1, ρ₂ = 9.9f-1, δ = 1f-8; T = typeof(η₁))`,
    so the *name* migrates but the call does not always:

    - **The default element type differs.** GML takes `T` from `η₁`, whose default is a `Float32`
      literal, so `AdamOptimizerWithDecay(1000)` is `Float32` there and `Float64` here. Pass the type
      — `AdamOptimizerWithDecay(1000, Float32)` — for a `Float32` network. Forgetting to is not
      silent: `OptimizerCache` rejects an `Adam{Float64}` handed `Float32` parameters and says so.
    - **The step sizes and moment coefficients are keywords here, not positional arguments**, and the
      coefficients are spelled `β₁`, `β₂` as everywhere else in this package. GML's
      `AdamOptimizerWithDecay(1000, 1f-3, 1f-8)` becomes
      `AdamOptimizerWithDecay(1000, Float32; η₁ = 1f-3, η₂ = 1f-8)`.

# Examples

```jldoctest; setup = :(using GeometricOptimizers)
AdamOptimizerWithDecay(1000).linesearch

# output

DecayingStatic from α = 0.01 to α = 1.0e-6 over 1000 iterations.
```
"""
function AdamOptimizerWithDecay(n_epochs::Integer, ::Type{T}=Float64; η₁=T(1.0e-2), η₂=T(1.0e-6),
        kwargs...) where {T}
    # the `η` defaults are written in `T` (and not as bare `Float64` literals) so that `γ` is computed
    # in `T`, i.e. so that this really is the `DecayingStatic(T; …)` it claims to be down to the last
    # bit; they are `DecayingStatic`'s defaults, which are also GML's.
    (algorithm=Adam(T; kwargs...), linesearch=DecayingStatic(T; η₁=η₁, η₂=η₂, n=n_epochs))
end
