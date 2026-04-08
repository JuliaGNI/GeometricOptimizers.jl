OptimizerCache(::GradientMethod, x::AbstractVector) = GradientCache(copy(x), similar(x), similar(x))
OptimizerCache(::GradientMethod, x::Manifold) = GradientCache(copy(x), zero(x), zero(x))
Hessian(::GradientMethod, ::OptimizerProblem, ::OptimizerSolution{T}) where {T} = NoHessian{T}()

struct NoHessian{T} <: Hessian{T} end

"""
    GradientCache <: OptimizerCache

Cache for the gradient optimizer.

# Fields
- `x::`[`Manifold`](@ref): the solution,
- `g`: the gradient (for the *manifold case* this is in [`AbstractLieAlgHorMatrix`](@ref) form),
- `δ`: the direction,
- `Δg`: difference in gradients,
- `section`: the [`GlobalSection`](@ref).
"""
struct GradientCache{T,MT<:OptimizerSolution{T},VT<:AbstractArray{T},ST<:GlobalSection{T}} <: OptimizerCache{T}
    x::MT
    g::VT
    δ::VT
    Δg::VT
    section::ST
end

function GradientCache(x::Manifold{T}, g::AT, δ::AT, Δg::AT) where {T,AT<:AbstractLieAlgHorMatrix}
    sec = GlobalSection(copy(x))
    GradientCache{T,typeof(x),typeof(g),typeof(sec)}(x, g, δ, Δg, sec)
end

function GradientCache(x::Manifold{T}, g::AT, δ::AT) where {T,AT<:AbstractLieAlgHorMatrix}
    Δg = similar(g)
    fill!(Δg, T(NaN))
    GradientCache(x, g, δ, Δg)
end

function GradientCache(x::Manifold{T}, g::AbstractLieAlgHorMatrix{T}) where {T}
    δ = similar(g)
    fill!(δ, T(NaN))
    GradientCache(x, g, δ)
end

function GradientCache(x::Manifold{T}) where {T}
    g = zero(x)
    fill!(g, T(NaN))
    GradientCache(x, g)
end

solution(cache::GradientCache) = cache.x
gradient_array(cache::GradientCache) = cache.g
direction(cache::GradientCache) = cache.δ
rhs(cache::GradientCache) = direction(cache)
section(cache::GradientCache) = cache.section

"""
    GradientState <: OptimizerState

State for the gradient optimizer.
"""
mutable struct GradientState{T,OT<:OptimizerSolution{T},GS<:GlobalSection{T,OT},VT<:AbstractArray{T}} <: OptimizerState{T}
    section::GS
    iterations::Int

    x::OT
    x̄::OT
    g::VT
    ḡ::VT
    f::T
    f̄::T
end

solution(state::GradientState) = state.x
previous_solution(state::GradientState) = state.x̄
gradient(state::GradientState) = state.g
previous_gradient(state::GradientState) = state.ḡ
value(state::GradientState) = state.f
previous_value(state::GradientState) = state.f̄

section(state::GradientState) = state.section

function GradientState(x::OST, g::AbstractArray{T}) where {T, OST<:OptimizerSolution{T}}
    _x = copy(x)
    _g = copy(g)
    gs = GlobalSection(_x)
    GradientState{T,typeof(_x),typeof(gs),typeof(_g)}(gs, 0, _x, rand(OST, size(x)...), _g, similar(_g), T(NaN), T(NaN))
end

GradientState(x::OptimizerSolution) = GradientState(x, zero(x))

OptimizerState(::GradientMethod, x...) = GradientState(x...)

function update!(state::GradientState{T}, gradient_array::AbstractArray{T}, x::Manifold{T}, f::Callable, retraction) where {T}
    copyto!(previous_solution(state), solution(state))
    copyto!(previous_gradient(state), gradient(state))
    state.f̄ = value(state)
    copyto!(solution(state), x)
    copyto!(gradient(state), gradient_array)
    state.f = f(x)

    update_section!(section(state), gradient_array, retraction)

    state
end

function update!(state::GradientState, opt::EuclideanOptimizer, x::Manifold)
    update!(state, direction(cache(opt)), x, problem(opt).F, opt.retraction)
end

# function compute_direction!(opt::EuclideanOptimizer{T,OM}, ::GradientState) where {T,OM<:GradientMethod}
#     direction(opt) .= rhs(opt)
# end

function update!(cache::GradientCache{T}, state::GradientState{T}, gradient::Gradient{T}, ::Hessian{T}, x::OptimizerSolution{T}) where {T}
    copyto!(section(cache), section(state))
    copyto!(gradient_array(cache), global_rep(section(state), gradient(x)))
    copyto!(solution(cache), x)
    copyto!(direction(cache), gradient_array(cache))
    rmul!(direction(cache), -1)

    cache
end
