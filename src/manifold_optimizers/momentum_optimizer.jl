OptimizerCache(::MomentumMethod{T}, x::OptimizerSolution{T}) where {T} = MomentumCache(_copy(x), _zero(x), _zero(x))
Hessian(::MomentumMethod, ::OptimizerProblem, ::OptimizerSolution{T}) where {T} = NoHessian{T}()

"""
    MomentumCache <: OptimizerCache

Cache for the gradient optimizer.

# Fields
- `x::`[`Manifold`](@ref): the solution,
- `g`: the gradient (for the *manifold case* this is in [`AbstractLieAlgHorMatrix`](@ref) form),
- `δ`: the direction,
- `Δg`: difference in gradients (used in [`OptimizerStatus`](@ref)),
- `section`: the [`GlobalSection`](@ref).
"""
struct MomentumCache{T,MT<:OptimizerSolution{T},VT<:GradientArrayOrNamedTuple{T},ST<:GlobalSectionSingleOrNamedTuple{T}} <: OptimizerCache{T}
    x::MT
    g::VT
    δ::VT
    Δg::VT
    section::ST
end

function MomentumCache(x::OptimizerSolution{T}, g::AT, δ::AT, Δg::AT) where {T,AT<:GradientArrayOrNamedTuple{T}}
    sec = GlobalSection(_copy(x))
    MomentumCache{T,typeof(x),typeof(g),typeof(sec)}(x, g, δ, Δg, sec)
end

function MomentumCache(x::OptimizerSolution{T}, g::AT, δ::AT) where {T,AT<:GradientArrayOrNamedTuple{T}}
    Δg = _similar(g)
    _fill!(Δg, T(NaN))
    MomentumCache(x, g, δ, Δg)
end

function MomentumCache(x::OptimizerSolution{T}, g::GradientArrayOrNamedTuple{T}) where {T}
    δ = _zero(g)
    MomentumCache(x, g, δ)
end

function MomentumCache(x::OptimizerSolution{T}) where {T}
    g = _zero(x)
    _fill!(g, T(NaN))
    MomentumCache(x, g)
end

solution(cache::MomentumCache) = cache.x
gradient_array(cache::MomentumCache) = cache.g
direction(cache::MomentumCache) = cache.δ
rhs(cache::MomentumCache) = direction(cache)
section(cache::MomentumCache) = cache.section

"""
    MomentumState <: OptimizerState

State for the gradient optimizer.
"""
mutable struct MomentumState{T,OT<:OptimizerSolution{T},GS<:GlobalSectionSingleOrNamedTuple{T},VT<:GradientArrayOrNamedTuple{T}} <: OptimizerState{T}
    section::GS
    iterations::Int

    x::OT
    x̄::OT
    g::VT
    ḡ::VT
    p::VT
    f::T
    f̄::T
end

solution(state::MomentumState) = state.x
previous_solution(state::MomentumState) = state.x̄
gradient(state::MomentumState) = state.g
previous_gradient(state::MomentumState) = state.ḡ
value(state::MomentumState) = state.f
previous_value(state::MomentumState) = state.f̄
momentum(state::MomentumState) = state.p

section(state::MomentumState) = state.section

function MomentumState(x::OST, g::GradientArrayOrNamedTuple{T}) where {T,OST<:OptimizerSolution{T}}
    _x = _copy(x)
    _g = _copy(g)
    gs = GlobalSection(_x)
    # as for [`AdamState`](@ref), the momentum has to be initialized with zeros: it is read
    # in the first call to `update!(::MomentumCache, ...)` before it is written to.
    MomentumState{T,typeof(_x),typeof(gs),typeof(_g)}(gs, 0, _x, _similar(_x), _g, _similar(_g), _zero(_g), T(NaN), T(NaN))
end

MomentumState(x::OptimizerSolution) = MomentumState(x, _zero(x))

OptimizerState(::MomentumMethod, x...) = MomentumState(x...)

function update!(state::MomentumState{T}, gradient_array::GradientArrayOrNamedTuple{T}, direction::GradientArrayOrNamedTuple{T}, α::T, x::OptimizerSolution{T}, f::Callable, retraction) where {T}
    _copyto!(previous_solution(state), solution(state))
    _copyto!(previous_gradient(state), gradient(state))
    state.f̄ = value(state)
    _copyto!(solution(state), x)
    _copyto!(gradient(state), gradient_array)
    # `p ← αp + ∇L`, i.e. the classic momentum recursion, which is what `update!(::MomentumCache,
    # ...)` anticipates when it forms the direction. Note that the decay has to be applied to `p`
    # and *not* to `∇L`: `p ← p + α∇L` (what this used to do) is an undamped accumulator that
    # grows without bound for a constant gradient instead of saturating at `∇L/(1 - α)`. See
    # issue #18.
    _rmul!(momentum(state), α)
    _add!(momentum(state), gradient_array)
    state.f = f(x)

    update_section!(section(state), direction, retraction)

    state
end

function update!(state::MomentumState, opt::Optimizer, x::OptimizerSolution)
    update!(state, gradient_array(cache(opt)), direction(cache(opt)), algorithm(opt).α, x, problem(opt).F, opt.retraction)
end

function update!(cache::MomentumCache{T}, state::MomentumState{T}, gradient::Gradient{T}, method::MomentumMethod{T}, x::OptimizerSolution{T}) where {T}
    _copyto!(section(cache), section(state))
    _copyto!(gradient_array(cache), global_rep(section(state), gradient(x)))
    _copyto!(solution(cache), x)
    # The direction is `-p` for the momentum `p ← αp + ∇L` of the step that is being taken. The
    # momentum stored in the state is still the one of the *previous* step, because
    # `update!(::MomentumState, ...)` runs after `solver_step!`, so the recursion is evaluated
    # here as well — with the same `α` and the same gradient, hence the same `p`.
    _copyto!(direction(cache), momentum(state))
    _rmul!(direction(cache), method.α)
    _add!(direction(cache), gradient_array(cache))
    _rmul!(direction(cache), -1)

    cache
end
