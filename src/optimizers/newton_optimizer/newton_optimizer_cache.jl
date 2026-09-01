# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`. Newton
# is `AbstractArray`-only, so its bounds were never the expensive kind — it is unbounded so that
# every method's cache and state read the same way, and so that nobody has to work out per struct
# whether a given bound happens to be one of the costly ones. The invariant is enforced by the inner
# constructors' signatures, which is dispatch and therefore costs nothing.
"""
    NewtonOptimizerCache <: OptimizerCache

# Keys

- `x`: current iterate (this stores the guess called by the functions generated with [`linesearch_problem`](@ref)),
- `Δx`: direction of optimization step (difference between `x` and `x̄`); this is obtained by multiplying `rhs` with the inverse of the Hessian,
- `g`: gradient value (this stores the gradient associated with `x`),
- `g̃`: scratch for [`latest_gradient`](@ref) — the gradient at a line-search trial point, and at the
  accepted iterate once [`solver_step!`](@ref) has refreshed it; see [`GradientCache`](@ref) for why
  this is a field of its own and not an alias for `g`,
- `g̃_is_current`: whether `g̃` is the gradient at `x`; see [`store_gradient!`](@ref),
- `Δg`: gradient difference (difference between `g̃` and `g`); this is used for computing the [`OptimizerStatus`](@ref),
- `rhs`: the right hand side used to compute the update,
- `H`: the Hessian matrix evaluated at `x`,

Also compare this to [`SimpleSolvers.NonlinearSolverCache`](@extref).
"""
struct NewtonOptimizerCache{T, AT, HT, GS} <: OptimizerCache{T}
    x::AT
    Δx::AT
    g::AT
    g̃::AT
    g̃_is_current::Base.RefValue{Bool}
    Δg::AT
    rhs::AT
    H::HT

    section::GS

    function NewtonOptimizerCache(x::AT) where {T, AT <: AbstractArray{T}}
        h = zeros(T, length(x), length(x))
        section = GlobalSection(x)
        cache = new{T, AT, typeof(h), typeof(section)}(
            similar(x), similar(x), similar(x), similar(x),
            Ref(false), similar(x), similar(x), h, section)
        initialize!(cache, x)
        cache
    end

    # we probably don't need this constructor -- and nothing calls it, which is how it came to pass
    # eight values for seven fields (it would have thrown on the first call). The field list is
    # spelled out here so that the next person to reach for it gets a cache and not an arity error.
    function NewtonOptimizerCache(x::AT, problem::OptimizerProblem) where {
            T <: Number, AT <: AbstractArray{T}}
        g = Gradient(problem)(x)
        h = Hessian(problem)(x)
        section = GlobalSection(x)
        g̃ = similar(x)
        fill!(g̃, T(NaN))
        new{T, AT, typeof(h), typeof(section)}(
            copy(x), zero(x), g, g̃, Ref(false), zero(x), -g, h, section)
    end
end

function OptimizerCache(::Union{Newton, QuasiNewtonOptimizerMethod}, x::OptimizerSolution)
    NewtonOptimizerCache(x)
end

# This fallback used to be `OptimizerCache(::OptimizerMethod, x) = NewtonOptimizerCache(x)`, so a
# method that *does* have a cache of its own but missed its `OptimizerCache(::Adam{T},
# ::OptimizerSolution{T})` method on a `T` mismatch ended up building a `NewtonOptimizerCache`
# and died several frames later with a `MethodError` that named neither the method nor `T`.
function OptimizerCache(method::OptimizerMethod, x::OptimizerSolution{T}) where {T}
    error("there is no OptimizerCache for the method $(typeof(method)) and parameters of element type $(T). " *
          "Note that a method that carries parameters of its own has to be constructed with the element type of " *
          "the parameters, e.g. `Adam($(T))` for $(T) parameters (unlike `MomentumMethod`, `Adam` is not converted).")
end

section(cache::NewtonOptimizerCache) = cache.section

"""
    rhs(cache)

Return the right hand side of an instance of [`NewtonOptimizerCache`](@ref)
"""
rhs(cache::NewtonOptimizerCache) = cache.rhs

"""
    gradient(::NewtonOptimizerCache)

Return the stored gradient (array) of an instance of [`NewtonOptimizerCache`](@ref)
"""
gradient(cache::NewtonOptimizerCache) = cache.g
gradient_array(cache::NewtonOptimizerCache) = gradient(cache)
latest_gradient(cache::NewtonOptimizerCache) = cache.g̃
function refresh_latest_gradient!(cache::NewtonOptimizerCache, g::Gradient)
    _refresh_latest_gradient!(cache, g)
end
function latest_gradient_is_current(cache::NewtonOptimizerCache, state::OptimizerState, x::OptimizerSolution)
    _latest_gradient_is_current(cache, state, x)
end
function invalidate_latest_gradient!(cache::NewtonOptimizerCache)
    _invalidate_latest_gradient!(cache)
end
# `∇f(x_{k+1}) - ∇f(x_k)`, from the two gradients the cache holds. The default differences against
# `state.ḡ`, and for `Newton` that is *the same gradient* `cache.g` holds: `solver_step!` calls
# `update!(state, gradient(opt), x)` at the top of the step, at the iterate the cache takes its
# gradient at, so the difference the status printed was structurally zero. See `gradient_difference!`.
function gradient_difference!(cache::NewtonOptimizerCache, ::OptimizerState)
    _latest_gradient_difference!(cache)
end

"""
    direction(cache)

Return the direction of the gradient step (i.e. `Δx`) of an instance of [`NewtonOptimizerCache`](@ref).
"""
direction(cache::NewtonOptimizerCache) = cache.Δx

hessian(cache::NewtonOptimizerCache) = cache.H

solution(cache::NewtonOptimizerCache) = cache.x

@doc raw"""
    update!(cache::NewtonOptimizerCache, x, g, hes)

Update an instance of [`NewtonOptimizerCache`](@ref) based on `x`.

This is used in [`update!(::OptimizerState, ::AbstractVector)`](@ref).

This sets:
```math
\begin{aligned}
% \bar{x}^\mathtt{cache} & \gets x, \\
x^\mathtt{cache} & \gets x, \\
g^\mathtt{cache} & \gets g, \\
\mathrm{rhs}^\mathtt{cache} & \gets -g, \\
H^\mathtt{cache} & \gets H(x), \\
\delta^\mathtt{cache} & \gets (H^\mathtt{cache})^{-1}\mathrm{rhs}^\mathtt{cache},
\end{aligned}
```
where we wrote ``H`` for the Hessian (i.e. the input argument `hes`).
"""
function update!(cache::NewtonOptimizerCache, state::OptimizerState,
        g::Gradient, ∇²f::Hessian, x::AbstractVector)
    # first, and before the two `copyto!`s below: `store_gradient!` compares `solution(cache)` against
    # `x` and `section(cache)` against `section(state)`, which those overwrite. It is what the other
    # five caches use, and going through it rather than calling `g(gradient(cache), x)` directly is
    # what makes `g̃_is_current` mean anything here -- only `store_gradient!` clears it, so without
    # this the flag stayed `true` across a whole step while `trial_slope` overwrote `g̃`.
    store_gradient!(cache, state, g, x)
    copyto!(section(cache), section(state))
    copyto!(solution(cache), x)
    copyto!(rhs(cache), gradient(cache))
    rmul!(rhs(cache), -1)
    ∇²f(hessian(cache), x)
    direction(cache) .= hessian(cache) \ rhs(cache)
    cache
end

function initialize!(cache::NewtonOptimizerCache{T}, ::AbstractVector{T}) where {T}
    cache.x .= T(NaN)
    direction(cache) .= T(NaN)
    cache.g .= T(NaN)
    latest_gradient(cache) .= T(NaN)
    invalidate_latest_gradient!(cache)
    cache.rhs .= T(NaN)
    hessian(cache) .= T(NaN)
    cache
end
