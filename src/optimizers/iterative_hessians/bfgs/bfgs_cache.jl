"""
    BFGSCache

The [`OptimizerCache`](@ref) for the [`_BFGS`](@ref) algorithm. Also see [`update!(::BFGSCache, ::OptimizerState, ::AbstractVector, ::AbstractVector`)](@ref).
"""
struct BFGSCache{T,VT<:OptimizerSolution{T},GT,MT,GS<:GlobalSectionSingleOrNamedTuple{T}} <: OptimizerCache{T}
    x::VT    # current solution

    g::GT    # current gradient

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
        v, unflatten = ParameterHandling.flatten(_zero(x))
        q = zeros(T, length(v), length(v))
        section = GlobalSection(x)
        g = _zero(x)
        cache = new{T,AT,typeof(g),typeof(q),typeof(section)}(_copy(x), _similar(g), _similar(q), similar(q), similar(q), similar(q), similar(q), _similar(g), _similar(g), _similar(g), section)
        initialize!(cache, x)
        cache
    end
end

OptimizerCache(::_BFGS, x::OptimizerSolution) = BFGSCache(x)

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
    v1, _ = ParameterHandling.flatten(arr1)
    v2, _ = ParameterHandling.flatten(arr2)
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
is what the `NamedTuple` case has always done; without the same method here `_BFGS` and `_DFP` cannot
run on a *bare* `Manifold` at all.
"""
function outer!(m::AbstractMatrix{T}, g₁::AbstractLieAlgHorMatrix{T}, g₂::AbstractLieAlgHorMatrix{T}) where {T}
    v1, _ = ParameterHandling.flatten(g₁)
    v2, _ = ParameterHandling.flatten(g₂)
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
T_3 & \gets (1 + \frac{\gamma^TQ\gamma}{\delta^\gamma})\delta\delta^T,\\
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

    ΔxΔg = cache.Δx ⋅ cache.Δg

    if !iszero(ΔxΔg) && !isnan(ΔxΔg)
        outer!(cache.ΔxΔx, cache.Δx, cache.Δx)
        outer!(cache.ΔxΔg, cache.Δx, cache.Δg)
        mul!(cache.T1, cache.ΔxΔg, inverse_hessian(state))
        mul!(cache.T2, inverse_hessian(state), cache.ΔxΔg')
        Δg2 = ParameterHandling.flatten(cache.Δg)[1]
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

function update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, x::OptimizerSolution)
    update!(cache, state, x, global_rep(section(state), grad(x)))
end

update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, ::HessianBFGS, x::OptimizerSolution) = update!(cache, state, grad, x)

# `Δg` is the `γ` of the secant pair, already formed in `update!` above from the `ḡ` that has
# since been advanced, so `OptimizerStatus` must not recompute it. See `gradient_difference!`.
gradient_difference!(cache::BFGSCache, ::OptimizerState) = cache.Δg

function initialize!(cache::BFGSCache{T}, ::OptimizerSolution{T}) where {T}
    _fill!(solution(cache), T(NaN))
    _fill!(direction(cache), T(NaN))
    _fill!(gradient(cache), T(NaN))
    _fill!(rhs(cache), T(NaN))
    cache
end
