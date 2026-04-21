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

# strictly speaking this constitutes type piracy (`outer!` is imported from `SimpleSolvers`.)
function outer!(m::AbstractMatrix{T}, arr1::ArrayNamedTuple{T}, arr2::ArrayNamedTuple{T}) where {T}
    v1, _ = ParameterHandling.flatten(arr1)
    v2, _ = ParameterHandling.flatten(arr2)
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
    _difference!(cache.Δg, gradient(cache), state.ḡ)

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

    _mul!(direction(cache), inverse_hessian(state), rhs(cache))
    _copyto!(state.s, direction(cache))

    cache
end

function update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, x::OptimizerSolution)
    update!(cache, state, x, global_rep(section(state), grad(x)))
end

update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, ::HessianBFGS, x::OptimizerSolution) = update!(cache, state, grad, x)

function initialize!(cache::BFGSCache{T}, ::OptimizerSolution{T}) where {T}
    _fill!(solution(cache), T(NaN))
    _fill!(direction(cache), T(NaN))
    _fill!(gradient(cache), T(NaN))
    _fill!(rhs(cache), T(NaN))
    cache
end
