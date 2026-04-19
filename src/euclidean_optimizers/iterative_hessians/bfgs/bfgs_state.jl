"""
    BFGSState <: OptimizerState

The [`OptimizerState`](@ref) corresponding to the [`_BFGS`](@ref) method.

# Keys
- `x̄`
- `ḡ`
- `f̄`
- `Q`
"""
mutable struct BFGSState{T,AT<:OptimizerSolution{T},GT<:GradientArrayOrNamedTuple{T},MT<:AbstractMatrix{T},GS<:GlobalSectionSingleOrNamedTuple{T}} <: OptimizerState{T}
    x̄::AT
    s::GT
    ḡ::GT
    f̄::T
    Q::MT
    iterations::Int

    section::GS

    function BFGSState(x̄::AT, ḡ::GT, f̄::T, Q::MT) where {T,AT<:OptimizerSolution{T},GT<:GradientArrayOrNamedTuple{T},MT<:AbstractMatrix{T}}
        section = GlobalSection(x̄)
        state = new{T,AT,GT,MT,typeof(section)}(x̄, s, ḡ, f̄, Q, 0, section)
        initialize!(state, x̄)
        state
    end
end

section(state::BFGSState) = state.section

BFGSState(x̄::OptimizerSolution{T}, ḡ::GradientArrayOrNamedTuple{T}, f̄::T) where {T} = BFGSState(_copy(x̄), _copy(ḡ), f̄, alloc_h(x̄))
BFGSState(x̄::OptimizerSolution{T}, ḡ::GradientArrayOrNamedTuple{T}) where {T} = BFGSState(_copy(x̄), _copy(ḡ), zero(T))
BFGSState(x̄::OptimizerSolution) = BFGSState(_copy(x̄), _zero(x̄))

function alloc_h(x::ArrayNamedTuple{T}) where {T}
    v, _ = ParameterHandling.flatten(_zero(x))
    h = zeros(T, length(v), length(v))
    fill!(h, T(NaN))
end

OptimizerState(::_BFGS, x_args...) = BFGSState(x_args...)

inverse_hessian(state::BFGSState) = state.Q

function initialize!(state::BFGSState{T}, ::OptimizerSolution{T}) where {T}
    _fill!(state.x̄, T(NaN))
    _fill!(state.s, T(NaN))
    _fill!(state.ḡ, T(NaN))
    state.f̄ = NaN
    inverse_hessian(state) .= one(inverse_hessian(state))
    state.iterations = 0

    state
end

function update!(state::BFGSState, gradient::Gradient, x::XT, retraction) where {T,XT<:OptimizerSolution{T}}
    _copyto!(state.x̄, x)
    XT <: ArrayNamedTuple ? gradient(state.ḡ, x, state) : gradient(state.ḡ, x)
    state.f̄ = gradient.F(ParameterHandling.flatten(T, x)[1])

    update_section!(section(state), state.s, retraction)

    state
end

function _copyto!(sec::GlobalSection{T,AT,Nothing}, Y::AT) where {T,AT<:AbstractVector{T}}
    sec.Y .= Y
end

function update!(state::BFGSState{T}, direction::GradientArrayOrNamedTuple{T}, gradient::Gradient, x::XT, f::T, retraction) where {T,XT<:OptimizerSolution{T}}
    _copyto!(state.x̄, x)
    XT <: ArrayNamedTuple ? gradient(state.ḡ, x, state) : gradient(state.ḡ, x)
    state.f̄ = f

    _copyto!(state.s, direction)
    update_section!(section(state), state.s, retraction)
    _copyto!(state.section, x)

    state
end
