# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
# The invariant is enforced by the inner constructor's signature, which is dispatch and therefore
# costs nothing.
"""
    BFGSCache

The [`OptimizerCache`](@ref) for the [`BFGS`](@ref) algorithm. Also see [`update!(::BFGSCache, ::OptimizerState, ::AbstractVector, ::AbstractVector`)](@ref).

`g̃` is the scratch for [`latest_gradient`](@ref) and `g̃_is_current` says whether it is the gradient at
`x`; see [`GradientCache`](@ref), which carries the same pair for the same reason, and
[`store_gradient!`](@ref).
"""
struct BFGSCache{T,VT,GT,MT,GS} <: OptimizerCache{T}
    x::VT    # current solution

    g::GT    # current gradient
    g̃::GT    # most recently evaluated gradient; see `latest_gradient`
    g̃_is_current::Base.RefValue{Bool}

    T1::MT
    T2::MT
    T3::MT
    ΔxΔg::MT
    ΔxΔx::MT

    rhs::GT
    Δx::GT
    Δg::GT

    section::GS

    function BFGSCache(x::AT) where {T,AT<:OptimizerSolution{T}}
        # `_zero(x)` is *not* redundant here, and dropping it is a real bug: `zero` of a manifold
        # element is its horizontal lift, whose free-parameter count is the intrinsic dimension and
        # not the size of the dense storage. For a `StiefelManifold(6, 3)` that is 12 against 18, and
        # `Q` has to be the former -- it multiplies gradients, which are lifts. `flatlength` then
        # counts without building the flat vector, which is what this used to allocate and discard.
        n = flatlength(_zero(x))
        q = zeros(T, n, n)
        section = GlobalSection(x)
        g = _zero(x)
        cache = new{T,AT,typeof(g),typeof(q),typeof(section)}(_copy(x), _similar(g), _similar(g), Ref(false), _similar(q), similar(q), similar(q), similar(q), similar(q), _similar(g), _similar(g), _similar(g), section)
        initialize!(cache, x)
        cache
    end
end

OptimizerCache(::BFGS, x::OptimizerSolution) = BFGSCache(x)

section(cache::BFGSCache) = cache.section

"""
    rhs(cache)

Return the right hand side of an instance of [`BFGSCache`](@ref)
"""
rhs(cache::BFGSCache) = cache.rhs

"""
    gradient(cache)

Return the stored gradient (array) of an instance of [`BFGSCache`](@ref)
"""
gradient(cache::BFGSCache) = cache.g
# the same array, as on `NewtonOptimizerCache` and the first-order caches: what `store_gradient!`
# writes the gradient the direction is built from into
gradient_array(cache::BFGSCache) = gradient(cache)
latest_gradient(cache::BFGSCache) = cache.g̃
refresh_latest_gradient!(cache::BFGSCache, g::Gradient) = _refresh_latest_gradient!(cache, g)
latest_gradient_is_current(cache::BFGSCache, state::OptimizerState, x::OptimizerSolution) =
    _latest_gradient_is_current(cache, state, x)
invalidate_latest_gradient!(cache::BFGSCache) = _invalidate_latest_gradient!(cache)

"""
    direction(cache)

Return the direction of the gradient step (i.e. `Δx`) of an instance of [`BFGSCache`](@ref).
"""
direction(cache::BFGSCache) = cache.Δx

solution(cache::BFGSCache) = cache.x

hessian(::BFGSCache) = error("BFGSCache does not store the Hessian, but it's inverse! Call inverse_hessian.")
inverse_hessian(::BFGSCache) = error("The inverse Hessian is stored in the state, not the cache!")

function update!(cache::BFGSCache, state::OptimizerState, x::OptimizerSolution)
    _copyto!(cache.x, x)
    _copyto!(direction(cache), state.s)
    outer!(cache.ΔxΔx, direction(cache), direction(cache))
    cache
end

# Type piracy: `outer!` is imported from `SimpleSolvers` and `ArrayNamedTuple` is an alias
# for Base's `NamedTuple`. See issue #16.
function outer!(m::AbstractMatrix{T}, arr1::ArrayNamedTuple{T}, arr2::ArrayNamedTuple{T}) where {T}
    v1, _ = flatten(arr1)
    v2, _ = flatten(arr2)
    outer!(m, v1, v2)
end

@doc raw"""
    outer!(m, g₁, g₂)

The outer product of two horizontal lifts, written into `m`.

Like the `ArrayNamedTuple` method above, this exists because the quasi-Newton `Q` is sized by the
*intrinsic* dimension of the parameters — the length of their flattening — while the direction and the
gradient are handed around in the *ambient* horizontal-lift representation. For a bare
`StiefelManifold` of size `(3, 1)` those are 2 and `3 × 3` respectively, so `SimpleSolvers.outer!`,
which indexes its arguments linearly against `axes(m)`, would assert on the mismatch. Flattening first
is what the `NamedTuple` case has always done; without the same method here `BFGS` and `DFP` cannot
run on a *bare* `Manifold` at all.
"""
function outer!(m::AbstractMatrix{T}, g₁::AbstractLieAlgHorMatrix{T}, g₂::AbstractLieAlgHorMatrix{T}) where {T}
    v1, _ = flatten(g₁)
    v2, _ = flatten(g₂)
    outer!(m, v1, v2)
end

@doc raw"""
    update!(cache, x, g)

Update the [`BFGSCache`](@ref) based on `x` and `g`.

# Extended help

The update rule used here can be found in [kochenderfer2019algorithms](@cite) and [nocedal2006numerical](@cite):

It does:

```math
\begin{aligned}
\delta & \gets x^{(k)} - x^{(k-1)}, \\
\gamma & \gets \nabla{}f^{(k)} - \nabla{}f^{(k-1)}, \\
T_1 & \gets \delta\gamma^TQ, \\
T_2 & \gets Q\gamma\delta^T, \\
T_3 & \gets (1 + \frac{\gamma^TQ\gamma}{\delta^T\gamma})\delta\delta^T,\\
Q & \gets Q - (T_1 + T_2 - T_3)/{\delta^T\gamma}
\end{aligned}
```
"""
function update!(cache::BFGSCache{T}, state::BFGSState{T}, x::OptimizerSolution{T}, g::GradientArrayOrNamedTuple{T}) where {T}
    update!(cache, state, x)
    _copyto!(gradient(cache), g)
    _copyto!(rhs(cache), g)
    _rmul!(rhs(cache), -one(T))
    _copyto!(direction(cache), state.s)
    _difference!(cache.Δg, gradient(cache), state.ḡ)

    # `_dot`, not `⋅`: every other quantity in the update below lives in the flattened coordinates --
    # `outer!` flattens before it forms `ΔxΔx` and `ΔxΔg`, `Q` is sized by the flattening, and `γᵀQγ` is
    # taken there -- so the `δᵀγ` they are all divided by has to be flattened too. `⋅` on a horizontal
    # lift is the ambient Frobenius product, i.e. exactly twice that, which left the `1 + γᵀQγ/δᵀγ`
    # coefficient mixing two different inner products. See `_dot`.
    ΔxΔg = _dot(cache.Δx, cache.Δg)

    # `curvature_is_usable` and not `!iszero(ΔxΔg) && !isnan(ΔxΔg)`: this update keeps `Q` positive
    # definite only for `δᵀγ > 0`, and the old guard admitted a negative pairing as readily as a
    # positive one -- as well as denominators of the order of `1e-16`, which it is about to divide a
    # rank-two correction by.
    if curvature_is_usable(ΔxΔg, cache.Δx, cache.Δg)
        outer!(cache.ΔxΔx, cache.Δx, cache.Δx)
        outer!(cache.ΔxΔg, cache.Δx, cache.Δg)
        mul!(cache.T1, cache.ΔxΔg, inverse_hessian(state))
        mul!(cache.T2, inverse_hessian(state), cache.ΔxΔg')
        Δg2 = flatten(cache.Δg)[1]
        γQγ = Δg2' * inverse_hessian(state) * Δg2
        cache.T3 .= (one(T) .+ γQγ ./ ΔxΔg) .* cache.ΔxΔx
        inverse_hessian(state) .-= (cache.T1 .+ cache.T2 .- cache.T3) ./ ΔxΔg
    end

    # `ḡ` has to still hold the gradient at the *previous* iterate while `Δg` is formed above, so it
    # is advanced here, right after it has been used. It used to be advanced in
    # `update!(::BFGSState, …)` at the end of the iteration instead -- which runs at the very iterate
    # the next `Δg` is computed at, so `Δg` was identically zero, `ΔxΔg` was zero with it, and the
    # guard above skipped the `Q` update on every single iteration.
    _copyto!(state.ḡ, gradient(cache))

    _mul!(direction(cache), inverse_hessian(state), rhs(cache))
    _copyto!(state.s, direction(cache))

    cache
end

# `store_gradient!` and not `global_rep(section(state), grad(x))` directly: the two are the same
# computation, and `solver_step!` has already done it at the end of the previous step -- see
# `store_gradient!`, which reuses `latest_gradient` when the pairing holds and evaluates afresh when
# it does not. This is what keeps `refresh_latest_gradient!` from costing `BFGS` a second gradient
# evaluation per iteration. It has to run *before* `update!(cache, state, x)` overwrites `cache.x`,
# which is what the pairing is checked against.
#
# `store_gradient!` writes into `gradient(cache)`, so the `g` handed on below *is* `cache.g` and the
# `_copyto!(gradient(cache), g)` in the four-argument method is a copy onto itself. That method is
# also called with a `g` of its own, from `update!(cache, state, x, g)` directly, which is why it
# keeps the copy.
function update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, x::OptimizerSolution)
    store_gradient!(cache, state, grad, x)
    update!(cache, state, x, gradient(cache))
end

update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, ::HessianBFGS, x::OptimizerSolution) = update!(cache, state, grad, x)

# `∇f(x_{k+1}) - ∇f(x_k)`, from the two gradients the cache holds. This used to return `cache.Δg`
# untouched -- the `γ` of the secant pair, i.e. `∇f(x_k) - ∇f(x_{k-1})`, which is one step behind the
# `rg` that `latest_gradient` now reports. See `gradient_difference!`.
gradient_difference!(cache::BFGSCache, ::OptimizerState) = _latest_gradient_difference!(cache)

function initialize!(cache::BFGSCache{T}, ::OptimizerSolution{T}) where {T}
    _fill!(solution(cache), T(NaN))
    _fill!(direction(cache), T(NaN))
    _fill!(gradient(cache), T(NaN))
    _fill!(latest_gradient(cache), T(NaN))
    invalidate_latest_gradient!(cache)
    _fill!(rhs(cache), T(NaN))
    cache
end
