"""
    BFGSCache

The [`OptimizerCache`](@ref) for the [`_BFGS`](@ref) algorithm. Also see [`update!(::BFGSCache, ::OptimizerState, ::AbstractVector, ::AbstractVector`)](@ref).
"""
struct BFGSCache{T,VT,MT} <: OptimizerCache{T}
    x::VT    # current solution

    g::VT    # current gradient

    T1::MT
    T2::MT
    T3::MT
    ΔxΔg::MT
    ΔxΔx::MT

    rhs::VT
    Δx::VT
    Δg::VT

    function BFGSCache(x::AT) where {T,AT<:AbstractVector{T}}
        q = zeros(T, length(x), length(x))
        cache = new{T,AT,typeof(q)}(similar(x), similar(x), similar(q), similar(q), similar(q), similar(q), similar(q), similar(x), similar(x), similar(x))
        initialize!(cache, x)
        cache
    end
end

OptimizerCache(::_BFGS, x::AbstractVector) = BFGSCache(x)

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

function update!(cache::BFGSCache, state::OptimizerState, x::AbstractVector)
    cache.x .= x
    direction(cache) .= state.s
    outer!(cache.ΔxΔx, direction(cache), direction(cache))
    cache
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
function update!(cache::BFGSCache{T}, state::BFGSState{T}, x::AbstractVector{T}, g::AbstractVector{T}) where {T}
    update!(cache, state, x)
    gradient(cache) .= g
    rhs(cache) .= -g
    # cache.Δx .= cache.x .- state.x̄
    cache.Δx .= state.s
    cache.Δg .= gradient(cache) .- state.ḡ

    ΔxΔg = cache.Δx ⋅ cache.Δg

    if !iszero(ΔxΔg) && !isnan(ΔxΔg)
        outer!(cache.ΔxΔx, cache.Δx, cache.Δx)
        outer!(cache.ΔxΔg, cache.Δx, cache.Δg)
        mul!(cache.T1, cache.ΔxΔg, state.Q)
        mul!(cache.T2, state.Q, cache.ΔxΔg')
        γQγ = cache.Δg' * state.Q * cache.Δg
        cache.T3 .= (one(T) + γQγ ./ ΔxΔg) .* cache.ΔxΔx
        inverse_hessian(state) .-= (cache.T1 .+ cache.T2 .- cache.T3) ./ ΔxΔg
    end

    direction(cache) .= inverse_hessian(state) * rhs(cache)
    state.s .= direction(cache)

    cache
end

update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, x::AbstractVector) = update!(cache, state, x, grad(x))

update!(cache::BFGSCache, state::OptimizerState, grad::Gradient, ::HessianBFGS, x::AbstractVector) = update!(cache, state, grad, x)

function initialize!(cache::BFGSCache{T}, ::AbstractVector{T}) where {T}
    cache.x .= T(NaN)
    direction(cache) .= T(NaN)
    cache.g .= T(NaN)
    cache.rhs .= T(NaN)
    cache
end
