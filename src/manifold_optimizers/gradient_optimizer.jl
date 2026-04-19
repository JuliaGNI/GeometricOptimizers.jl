OptimizerCache(::GradientMethod, x::OptimizerSolution) = GradientCache(_copy(x), _zero(x), _zero(x))
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
struct GradientCache{T,MT<:OptimizerSolution{T},VT<:GradientArrayOrNamedTuple{T},ST<:GlobalSectionSingleOrNamedTuple{T}} <: OptimizerCache{T}
    x::MT
    g::VT
    δ::VT
    Δg::VT
    section::ST
end

function GradientCache(x::OptimizerSolution{T}, g::AT, δ::AT, Δg::AT) where {T,AT<:GradientArrayOrNamedTuple{T}}
    sec = GlobalSection(_copy(x))
    GradientCache{T,typeof(x),typeof(g),typeof(sec)}(x, g, δ, Δg, sec)
end

function GradientCache(x::OptimizerSolution{T}, g::AT, δ::AT) where {T,AT<:GradientArrayOrNamedTuple{T}}
    Δg = _similar(g)
    _fill!(Δg, T(NaN))
    GradientCache(x, g, δ, Δg)
end

function GradientCache(x::OptimizerSolution{T}, g::GradientArrayOrNamedTuple{T}) where {T}
    δ = _similar(g)
    _fill!(δ, T(NaN))
    GradientCache(x, g, δ)
end

function GradientCache(x::OptimizerSolution{T}) where {T}
    g = _zero(x)
    _fill!(g, T(NaN))
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
mutable struct GradientState{T,OT<:OptimizerSolution{T},GS<:GlobalSectionSingleOrNamedTuple{T},VT<:GradientArrayOrNamedTuple{T}} <: OptimizerState{T}
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

function GradientState(x::OST, g::GradientArrayOrNamedTuple{T}) where {T,OST<:OptimizerSolution{T}}
    _x = _copy(x)
    _g = _copy(g)
    gs = GlobalSection(_x)
    GradientState{T,typeof(_x),typeof(gs),typeof(_g)}(gs, 0, _x, _similar(_x), _g, _similar(_g), T(NaN), T(NaN))
end

GradientState(x::OptimizerSolution) = GradientState(x, _zero(x))

OptimizerState(::GradientMethod, x...) = GradientState(x...)

function update!(state::GradientState{T}, gradient_array::GradientArrayOrNamedTuple{T}, direction::GradientArrayOrNamedTuple{T}, x::OptimizerSolution{T}, f::Callable, retraction) where {T}
    _copyto!(previous_solution(state), solution(state))
    _copyto!(previous_gradient(state), gradient(state))
    state.f̄ = value(state)
    _copyto!(solution(state), x)
    _copyto!(gradient(state), gradient_array)
    state.f = f(x)

    update_section!(section(state), direction, retraction)

    state
end

function update!(state::GradientState, opt::EuclideanOptimizer, x::OptimizerSolution)
    update!(state, gradient_array(cache(opt)), direction(cache(opt)), x, problem(opt).F, opt.retraction)
end

# function compute_direction!(opt::EuclideanOptimizer{T,OM}, ::GradientState) where {T,OM<:GradientMethod}
#     direction(opt) .= rhs(opt)
# end

function update!(cache::GradientCache{T}, state::GradientState{T}, gradient::Gradient{T}, ::Hessian{T}, x::OptimizerSolution{T}) where {T}
    _copyto!(section(cache), section(state))
    _copyto!(gradient_array(cache), global_rep(section(state), gradient(x)))
    _copyto!(solution(cache), x)
    _copyto!(direction(cache), gradient_array(cache))
    _rmul!(direction(cache), -1)

    cache
end
