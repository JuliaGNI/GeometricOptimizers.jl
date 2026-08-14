"""
    DFPCache <: OptimizerCache

The [`OptimizerCache`](@ref) corresponding to the [`_DFP`](@ref) method.

`g̃` is the scratch for [`latest_gradient`](@ref) and `g̃_is_current` says whether it is the gradient at
`x`; see [`GradientCache`](@ref), which carries the same pair for the same reason, and
[`store_gradient!`](@ref).
"""
struct DFPCache{T,VT<:OptimizerSolution{T},GT,MT,GS<:GlobalSectionSingleOrNamedTuple{T}} <: OptimizerCache{T}
    x::VT    # current solution

    g::GT    # current gradient
    g̃::GT    # most recently evaluated gradient; see `latest_gradient`
    g̃_is_current::Base.RefValue{Bool}

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
        cache = new{T,AT,typeof(g),typeof(q),typeof(section)}(_copy(x), _similar(g), _similar(g), Ref(false), similar(q), similar(q), similar(q), similar(q), _similar(g), _similar(g), _similar(g), section)
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
# see the remark in `bfgs_cache.jl`
gradient_array(cache::DFPCache) = gradient(cache)
latest_gradient(cache::DFPCache) = cache.g̃
refresh_latest_gradient!(cache::DFPCache, g::Gradient) = _refresh_latest_gradient!(cache, g)
latest_gradient_is_current(cache::DFPCache, state::OptimizerState, x::OptimizerSolution) =
    _latest_gradient_is_current(cache, state, x)
invalidate_latest_gradient!(cache::DFPCache) = _invalidate_latest_gradient!(cache)

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

    # see the remark in `bfgs_cache.jl`: `δᵀγ` is the denominator of an update whose every other term is
    # flattened, so it is `_dot` and not the ambient `⋅`
    ΔxΔg = _dot(cache.Δx, cache.Δg)
    # `Q` lives in the flattened coordinates, so the quadratic form has to be taken there too
    Δg2 = ParameterHandling.flatten(cache.Δg)[1]
    γQγ = Δg2' * state.Q * Δg2

    # see the remark in `bfgs_cache.jl`: `curvature_is_usable` is the curvature condition that keeps
    # `Q` positive definite, and it is what the bare `!iszero`/`!isnan` pair could not express
    if curvature_is_usable(ΔxΔg, cache.Δx, cache.Δg) && !iszero(γQγ) && !isnan(γQγ)
        outer!(cache.ΔxΔx, cache.Δx, cache.Δx)
        outer!(cache.ΔgΔg, cache.Δg, cache.Δg)
        # the DFP correction is `Q - Qγγᵀ Q/(γᵀQγ) + δδᵀ/(δᵀγ)` (nocedal2006numerical, eq. 6.15), so
        # the rank-one term that is subtracted is built from `γγᵀ`. This used to read `cache.ΔxΔx`,
        # i.e. `δδᵀ`, which left `cache.ΔgΔg` computed on the line above and never read.
        mul!(cache.T1, cache.ΔgΔg, state.Q)
        mul!(cache.T2, state.Q, cache.T1)
        # `Q γγᵀ Q` is symmetric in exact arithmetic, but forming it as two separate products is not
        # symmetric in floating point, and the error accumulates: unsymmetrized, `‖Q - Qᵀ‖/‖Q‖` grows
        # from 8e-16 after five iterations to 1.6e-11 after twenty thousand, and `eigvals(Q)` then
        # returns complex numbers. `BFGSCache` gets this for free because it adds `T₁ + T₂` where `T₂`
        # is built as the exact transpose of `T₁`; here the symmetrization has to be explicit.
        state.Q .-= (cache.T2 .+ cache.T2') ./ (2γQγ)
        state.Q .+= cache.ΔxΔx ./ ΔxΔg
    end

    # see the remark in `bfgs_cache.jl`: `ḡ` is advanced here, right after `Δg` has been formed from
    # it, and not at the end of the iteration, where it would be the gradient at the same iterate.
    _copyto!(state.ḡ, gradient(cache))

    _mul!(direction(cache), inverse_hessian(state), rhs(cache))
    _copyto!(state.s, direction(cache))

    cache
end

# see the remark in `bfgs_cache.jl`: this reuses the gradient `solver_step!` refreshed at the end of
# the previous step, has to run before `update!(cache, state, x)` overwrites `cache.x`, and leaves the
# four-argument method below copying `cache.g` onto itself
function update!(cache::DFPCache, state::OptimizerState, grad::Gradient, x::OptimizerSolution)
    store_gradient!(cache, state, grad, x)
    update!(cache, state, x, gradient(cache))
end

update!(cache::DFPCache, state::OptimizerState, grad::Gradient, ::HessianDFP, x::OptimizerSolution) = update!(cache, state, grad, x)

# see the remark in `bfgs_cache.jl`: the successive difference the status prints, and not the `γ` of
# the secant pair, which is one step behind it
gradient_difference!(cache::DFPCache, ::OptimizerState) = _latest_gradient_difference!(cache)

function initialize!(cache::DFPCache{T}, ::OptimizerSolution{T}) where {T}
    _fill!(solution(cache), T(NaN))
    _fill!(direction(cache), T(NaN))
    _fill!(gradient(cache), T(NaN))
    _fill!(latest_gradient(cache), T(NaN))
    invalidate_latest_gradient!(cache)
    _fill!(rhs(cache), T(NaN))
    cache
end
