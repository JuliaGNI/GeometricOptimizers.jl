"""
    DFPCache <: OptimizerCache

The [`OptimizerCache`](@ref) corresponding to the [`_DFP`](@ref) method.
"""
struct DFPCache{T,VT,MT,GS<:GlobalSection{T}} <: OptimizerCache{T}
    x::VT    # current solution

    g::VT    # current gradient

    T1::MT
    T2::MT
    ΔgΔg::MT
    ΔxΔx::MT

    rhs::VT
    Δx::VT
    Δg::VT

    section::GS

    function DFPCache(x::AT) where {T,AT<:AbstractVector{T}}
        q = zeros(T, length(x), length(x))
        section = GlobalSection(x)
        cache = new{T,AT,typeof(q),typeof(section)}(similar(x), similar(x), similar(q), similar(q), similar(q), similar(q), similar(x), similar(x), similar(x), section)
        initialize!(cache, x)
        cache
    end
end

OptimizerCache(::_DFP, x::AbstractVector) = DFPCache(x)

section(cache::DFPCache) = cache.section

"""
    rhs(cache)

Return the right hand side of an instance of [`DFPCache`](@ref)
"""
rhs(cache::DFPCache) = cache.rhs

"""
    gradient(cache)

Return the stored gradient (array) of an instance of [`DFPCache`](@ref)
"""
gradient(cache::DFPCache) = cache.g

"""
    direction(cache)

Return the direction of the gradient step (i.e. `Δx`) of an instance of [`DFPCache`](@ref).
"""
direction(cache::DFPCache) = cache.Δx

solution(cache::DFPCache) = cache.x

hessian(::DFPCache) = error("DFPCache does not store the Hessian, but it's inverse! Call inverse_hessian.")
inverse_hessian(::DFPCache) = error("The inverse Hessian is stored in the state, not the cache!")

function update!(cache::DFPCache, state::OptimizerState, x::AbstractVector)
    cache.x .= x
    direction(cache) .= cache.x - state.x̄
    outer!(cache.ΔxΔx, direction(cache), direction(cache))
    cache
end

@doc raw"""
    update!(cache, x, g)

Update the [`DFPCache`](@ref) based on `x` and `g`.

# Extended help

The update rule used here can be found in [kochenderfer2019algorithms](@cite) and [nocedal2006numerical](@cite).
"""
function update!(cache::DFPCache{T}, state::DFPState{T}, x::AbstractVector{T}, g::AbstractVector{T}) where {T}
    update!(cache, state, x)
    gradient(cache) .= g
    rhs(cache) .= -g
    # cache.Δx .= cache.x .- state.x̄
    cache.Δx .= state.s

    cache.Δg .= gradient(cache) - state.ḡ

    ΔxΔg = cache.Δx ⋅ cache.Δg
    γQγ = cache.Δg' * state.Q * cache.Δg

    if !iszero(ΔxΔg) & !iszero(γQγ) & !isnan(ΔxΔg)
        outer!(cache.ΔxΔx, cache.Δx, cache.Δx)
        outer!(cache.ΔgΔg, cache.Δg, cache.Δg)
        # the DFP correction is `Q - Qγγᵀ Q/(γᵀQγ) + δδᵀ/(δᵀγ)` (nocedal2006numerical, eq. 6.15), so
        # the rank-one term that is subtracted is built from `γγᵀ`. This used to read `cache.ΔxΔx`,
        # i.e. `δδᵀ`, which left `cache.ΔgΔg` computed on the line above and never read.
        mul!(cache.T1, cache.ΔgΔg, state.Q)
        mul!(cache.T2, state.Q, cache.T1)
        state.Q .-= cache.T2 ./ γQγ
        state.Q .+= cache.ΔxΔx ./ ΔxΔg
    end

    # see the remark in `bfgs_cache.jl`: `ḡ` is advanced here, right after `Δg` has been formed from
    # it, and not at the end of the iteration, where it would be the gradient at the same iterate.
    state.ḡ .= gradient(cache)

    direction(cache) .= inverse_hessian(state) * rhs(cache)
    state.s .= direction(cache)

    cache
end

update!(cache::DFPCache, state::OptimizerState, grad::Gradient, x::AbstractVector) = update!(cache, state, x, grad(x))

update!(cache::DFPCache, state::OptimizerState, grad::Gradient, ::HessianDFP, x::AbstractVector) = update!(cache, state, grad, x)

# `Δg` is the `γ` of the secant pair, already formed in `update!` above from the `ḡ` that has
# since been advanced, so `OptimizerStatus` must not recompute it. See `gradient_difference!`.
gradient_difference!(cache::DFPCache, ::OptimizerState) = cache.Δg

function initialize!(cache::DFPCache{T}, ::AbstractVector{T}) where {T}
    cache.x .= T(NaN)
    direction(cache) .= T(NaN)
    cache.g .= T(NaN)
    cache.rhs .= T(NaN)
    cache
end
