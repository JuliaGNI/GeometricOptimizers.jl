OptimizerCache(::Adam{T}, x::OptimizerSolution{T}) where {T} = AdamCache(_copy(x), _zero(x), _zero(x))
Hessian(::Adam, ::OptimizerProblem, ::OptimizerSolution{T}) where {T} = NoHessian{T}()

"""
    AdamCache <: OptimizerCache

Cache for the gradient optimizer.

# Fields
- `x::`[`Manifold`](@ref): the solution,
- `g`: the gradient (for the *manifold case* this is in [`AbstractLieAlgHorMatrix`](@ref) form),
- `δ`: the direction,
- `Δg`: difference in gradients, needed for [`OptimizerStatus`](@ref),
- `section`: the [`GlobalSection`](@ref).
"""
struct AdamCache{T,MT<:OptimizerSolution{T},VT<:GradientArrayOrNamedTuple{T},ST<:GlobalSectionSingleOrNamedTuple{T}} <: OptimizerCache{T}
    x::MT
    g::VT
    δ::VT
    Δg::VT
    m₁::VT
    m₂::VT
    m̃₂::VT
    section::ST
end

first_moment(cache::AdamCache) = cache.m₁
second_moment(cache::AdamCache) = cache.m₂
_second_moment(cache::AdamCache) = cache.m̃₂

function AdamCache(x::OptimizerSolution{T}, g::AT, δ::AT, Δg::AT) where {T,AT<:GradientArrayOrNamedTuple{T}}
    sec = GlobalSection(_copy(x))
    m₁ = _similar(g)
    m₂ = _similar(g)
    m̃₂ = _similar(g)
    AdamCache{T,typeof(x),typeof(g),typeof(sec)}(x, g, δ, Δg, m₁, m₂, m̃₂, sec)
end

function AdamCache(x::OptimizerSolution{T}, g::AT, δ::AT) where {T,AT<:GradientArrayOrNamedTuple{T}}
    Δg = _similar(g)
    _fill!(Δg, T(NaN))
    AdamCache(x, g, δ, Δg)
end

function AdamCache(x::OptimizerSolution{T}, g::GradientArrayOrNamedTuple{T}) where {T}
    δ = _zero(g)
    AdamCache(x, g, δ)
end

function AdamCache(x::OptimizerSolution{T}) where {T}
    g = _zero(x)
    _fill!(g, T(NaN))
    AdamCache(x, g)
end

solution(cache::AdamCache) = cache.x
gradient_array(cache::AdamCache) = cache.g
direction(cache::AdamCache) = cache.δ
rhs(cache::AdamCache) = direction(cache)
section(cache::AdamCache) = cache.section

"""
    AdamState <: OptimizerState

State for the gradient optimizer.
"""
mutable struct AdamState{T,OT<:OptimizerSolution{T},GS<:GlobalSectionSingleOrNamedTuple{T},VT<:GradientArrayOrNamedTuple{T}} <: OptimizerState{T}
    section::GS
    iterations::Int

    x::OT
    x̄::OT
    g::VT
    ḡ::VT
    m₁::VT
    m₂::VT
    m̃₂::VT
    f::T
    f̄::T
end

solution(state::AdamState) = state.x
previous_solution(state::AdamState) = state.x̄
gradient(state::AdamState) = state.g
previous_gradient(state::AdamState) = state.ḡ
value(state::AdamState) = state.f
previous_value(state::AdamState) = state.f̄
first_moment(state::AdamState) = state.m₁
second_moment(state::AdamState) = state.m₂
_second_moment(state::AdamState) = state.m̃₂

section(state::AdamState) = state.section

function AdamState(x::OST, g::GradientArrayOrNamedTuple{T}) where {T,OST<:OptimizerSolution{T}}
    _x = _copy(x)
    _g = _copy(g)
    gs = GlobalSection(_x)
    AdamState{T,typeof(_x),typeof(gs),typeof(_g)}(gs, 0, _x, _similar(_x), _g, _similar(_g), _similar(_g), _similar(_g), _similar(_g), T(NaN), T(NaN))
end

AdamState(x::OptimizerSolution) = AdamState(x, _zero(x))

OptimizerState(::Adam, x...) = AdamState(x...)

function update!(state::AdamState{T}, gradient_array::GradientArrayOrNamedTuple{T}, direction::GradientArrayOrNamedTuple{T}, _first_moment::GradientArrayOrNamedTuple{T}, _second_moment::GradientArrayOrNamedTuple{T}, x::OptimizerSolution{T}, f::Callable, retraction) where {T}
    _copyto!(previous_solution(state), solution(state))
    _copyto!(previous_gradient(state), gradient(state))
    state.f̄ = value(state)
    _copyto!(solution(state), x)
    _copyto!(gradient(state), gradient_array)
    _copyto!(first_moment(state), _first_moment)
    _copyto!(second_moment(state), _second_moment)
    state.f = f(x)

    update_section!(section(state), direction, retraction)

    state
end

function update!(state::AdamState, opt::Optimizer, x::OptimizerSolution)
    update!(state, gradient_array(cache(opt)), direction(cache(opt)), first_moment(opt.cache), second_moment(opt.cache), x, problem(opt).F, opt.retraction)
end

function update!(cache::AdamCache{T}, state::AdamState{T}, gradient::Gradient{T}, β₁::T, β₂::T, δ::T, t::Integer, x::OptimizerSolution{T}) where {T}
    _copyto!(section(cache), section(state))
    _copyto!(gradient_array(cache), global_rep(section(state), gradient(x)))
    _copyto!(solution(cache), x)
    _t = t + 1
    fac₁₁ = β₁ / (1 - β₁^_t)
    fac₁₂ = (1 - β₁) / (1 - β₁^_t)
    fac₂₁ = β₂ / (1 - β₂^_t)
    fac₂₂ = (1 - β₂) / (1 - β₂^_t)
    _copyto!(first_moment(cache), _mul(fac₁₁, first_moment(state)))
    _add!(first_moment(cache), _mul(fac₁₂, gradient_array(cache)))
    _copyto!(second_moment(cache), _mul(fac₂₁, second_moment(state)))
    _add!(second_moment(cache), _mul(fac₂₂, _square(gradient_array(cache))))
    _copyto!(_second_moment(cache), second_moment(cache))
    _add!(_second_moment(cache), δ)
    _rac!(second_moment(cache))
    _copyto!(direction(cache), first_moment(cache))
    _div!(direction(cache), _second_moment(cache))
    _rmul!(direction(cache), -1)

    cache
end

function update!(cache::AdamCache{T}, state::AdamState{T}, gradient::Gradient{T}, method::Adam{T}, x::OptimizerSolution{T}) where {T}
    update!(cache, state, gradient, method.β₁, method.β₂, method.δ, state.iterations, x)
end
