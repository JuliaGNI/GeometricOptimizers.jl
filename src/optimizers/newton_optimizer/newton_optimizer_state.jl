# The type parameters are deliberately unbounded; see [`NewtonOptimizerCache`](@ref) for why Newton
# follows the other methods here, and the warning in `optimizer_solution.jl` for what the rule is.
"""
    NewtonOptimizerState <: OptimizerState

The optimizer state is needed to update the [`Optimizer`](@ref). This is different from [`OptimizerStatus`](@ref) and [`OptimizerResult`](@ref) which serve as diagnostic tools.

# Keys

- `x`
- `x̄`
- `g`
- `ḡ`
- `f`
- `f̄`

The unbarred fields are the current iterate's and the barred ones the previous iterate's, as
[`update!`](@ref) maintains them.

`gradient` and `value` read `g` and `f` and are exported. The other four accessors are internal:
they are neither exported nor `Base.ispublic`, so they resolve only under a qualified name —
`GeometricOptimizers.solution` for `x`, `GeometricOptimizers.previous_solution` for `x̄`,
`GeometricOptimizers.previous_gradient` for `ḡ` and `GeometricOptimizers.previous_value` for `f̄`.
"""
mutable struct NewtonOptimizerState{T, AT, GT, GS} <: OptimizerState{T}
    iterations::Int

    x::AT
    x̄::AT
    g::GT
    ḡ::GT
    f::T
    f̄::T

    section::GS

    function NewtonOptimizerState(X::AT, G::GT) where {
            T, AT <: AbstractArray{T}, GT <: AbstractArray{T}}
        x = zero(X)
        x̄ = zero(X)
        g = zero(X)
        ḡ = zero(X)
        x .= T(NaN)
        x̄ .= T(NaN)
        g .= T(NaN)
        ḡ .= T(NaN)
        section = GlobalSection(x)
        new{T, AT, GT, typeof(section)}(0, x, x̄, g, ḡ, T(NaN), T(NaN), section)
    end

    NewtonOptimizerState(x) = NewtonOptimizerState(x, x)
end

section(state::NewtonOptimizerState) = state.section

OptimizerState(::Newton, x_args...) = NewtonOptimizerState(x_args...)

# The same scope check as on `OptimizerCache`, repeated here because `OptimizerState` is exported and
# `solve!(x, OptimizerState(method, x), opt)` is the documented pattern, so this is the entry point a
# manifold user reaches first. Without these two methods a `Manifold` reaches a `convert` that cannot
# turn a lift back into a point, and a parameter set finds no `NewtonOptimizerState` method at all.
# `Tuple{Newton, Manifold, Vararg}` is strictly more specific than `Tuple{Newton, Vararg}`, so
# neither method is ambiguous with the one above.
OptimizerState(::Newton, ::Manifold, args...) = throw(ArgumentError(_NEWTON_SCOPE))
OptimizerState(::Newton, ::NetworkParameters, args...) = throw(ArgumentError(_NEWTON_SCOPE))

function initialize!(state::NewtonOptimizerState{T}, x::AbstractVector{T}, g::AbstractVector{T}, f::T) where {T}
    state.iterations = 0
    state.x .= x
    state.g .= g
    state.f = f
    state.x̄ .= T(NaN)
    state.ḡ .= T(NaN)
    state.f̄ = T(NaN)
    section(state).Y .= x
end

function update!(state::NewtonOptimizerState{T}, x::AbstractVector{T}, g::AbstractVector{T}, f::T) where {T}
    state.x̄ .= state.x
    state.ḡ .= state.g
    state.f̄ = state.f
    state.x .= x
    state.g .= g
    state.f = f
    section(state).Y .= x
end

# The unbarred field is the current iterate's, which is what the unbarred accessor name means on
# every other `OptimizerState`. `NewtonOptimizerCache` is a separate type whose `x` means something
# else, so `solution(::NewtonOptimizerCache)` is `cache.x` and does not carry over to the state.
solution(state::NewtonOptimizerState) = state.x
previous_solution(state::NewtonOptimizerState) = state.x̄
gradient(state::NewtonOptimizerState) = state.g
previous_gradient(state::NewtonOptimizerState) = state.ḡ
value(state::NewtonOptimizerState) = state.f
previous_value(state::NewtonOptimizerState) = state.f̄

"""
    update!(state::NewtonOptimizerState, gradient, x)

Update an instance of [`NewtonOptimizerState`](@ref) based on `x` and `gradient`, where `g` is of type [`SimpleSolvers.Gradient`](@extref).

# Examples

If we only call `update!` once there are still `NaN`s for x̄, ḡ and f̄.
```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: NewtonOptimizerState)
f(x) = sum(x.^2)
x = [1., 2.]
state = NewtonOptimizerState(x)
grad = GradientAutodiff{Float64}(f, length(x))
update!(state, grad, x)

# output

NewtonOptimizerState{Float64, Vector{Float64}, Vector{Float64}, GlobalSection{Float64, Vector{Float64}, Nothing}}(0, [1.0, 2.0], [NaN, NaN], [2.0, 4.0], [NaN, NaN], 5.0, NaN, GlobalSection{Float64, Vector{Float64}, Nothing}([1.0, 2.0], nothing))
```
"""
function update!(state::NewtonOptimizerState, gradient::Gradient, x::AbstractVector)
    update!(state, x, gradient(x), _objective(gradient)(x))

    state
end
