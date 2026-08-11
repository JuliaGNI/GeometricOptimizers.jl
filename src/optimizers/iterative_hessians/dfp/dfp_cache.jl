"""
    DFPCache <: OptimizerCache

The [`OptimizerCache`](@ref) corresponding to the [`_DFP`](@ref) method.
"""
struct DFPCache{T,VT<:OptimizerSolution{T},GT,MT,GS<:GlobalSectionSingleOrNamedTuple{T}} <: OptimizerCache{T}
    x::VT    # current solution

    g::GT    # current gradient

    T1::MT
    T2::MT
    ΔgΔg::MT
    ΔxΔx::MT

    rhs::GT
    Δx::GT
    Δg::GT

    section::GS

    # The solution and the gradient are separate type parameters because on a manifold they are not
    # the same thing: `x` is a point of `St(N, n)` while the gradient, the direction and the secant
    # differences are horizontal lifts of shape `N × N`. `Q` is sized by neither -- it is sized by the
    # length of the *flattening*, the intrinsic dimension, which for a `NamedTuple` is emphatically
    # not `length(x)` (that is the number of entries).
    function DFPCache(x::AT) where {T,AT<:OptimizerSolution{T}}
        v, _ = ParameterHandling.flatten(_zero(x))
        q = zeros(T, length(v), length(v))
        section = GlobalSection(x)
        g = _zero(x)
        cache = new{T,AT,typeof(g),typeof(q),typeof(section)}(_copy(x), _similar(g), similar(q), similar(q), similar(q), similar(q), _similar(g), _similar(g), _similar(g), section)
        initialize!(cache, x)
        cache
    end
end

OptimizerCache(::_DFP, x::OptimizerSolution) = DFPCache(x)

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

function update!(cache::DFPCache, state::OptimizerState, x::OptimizerSolution)
    _copyto!(cache.x, x)
    _copyto!(direction(cache), state.s)
    outer!(cache.ΔxΔx, direction(cache), direction(cache))
    cache
end

@doc raw"""
    update!(cache, x, g)

Update the [`DFPCache`](@ref) based on `x` and `g`.

# Extended help

The update rule used here can be found in [kochenderfer2019algorithms](@cite) and [nocedal2006numerical](@cite).
"""
function update!(cache::DFPCache{T}, state::DFPState{T}, x::OptimizerSolution{T}, g::GradientArrayOrNamedTuple{T}) where {T}
    update!(cache, state, x)
    _copyto!(gradient(cache), g)
    _copyto!(rhs(cache), g)
    _rmul!(rhs(cache), -one(T))
    _copyto!(cache.Δx, state.s)

    _difference!(cache.Δg, gradient(cache), state.ḡ)

    ΔxΔg = cache.Δx ⋅ cache.Δg
    # `Q` lives in the flattened coordinates, so the quadratic form has to be taken there too
    Δg2 = ParameterHandling.flatten(cache.Δg)[1]
    γQγ = Δg2' * state.Q * Δg2

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
    _copyto!(state.ḡ, gradient(cache))

    _mul!(direction(cache), inverse_hessian(state), rhs(cache))
    _copyto!(state.s, direction(cache))

    cache
end

function update!(cache::DFPCache, state::OptimizerState, grad::Gradient, x::OptimizerSolution)
    update!(cache, state, x, global_rep(section(state), grad(x)))
end

update!(cache::DFPCache, state::OptimizerState, grad::Gradient, ::HessianDFP, x::OptimizerSolution) = update!(cache, state, grad, x)

# `Δg` is the `γ` of the secant pair, already formed in `update!` above from the `ḡ` that has
# since been advanced, so `OptimizerStatus` must not recompute it. See `gradient_difference!`.
gradient_difference!(cache::DFPCache, ::OptimizerState) = cache.Δg

function initialize!(cache::DFPCache{T}, ::OptimizerSolution{T}) where {T}
    _fill!(solution(cache), T(NaN))
    _fill!(direction(cache), T(NaN))
    _fill!(gradient(cache), T(NaN))
    _fill!(rhs(cache), T(NaN))
    cache
end
