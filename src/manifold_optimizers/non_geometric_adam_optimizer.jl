# Cayley ADAM, [li2020efficient](@cite) Algorithm 2, in this package's global tangent space
# representation. `ADAM_MATHS.md` derives the mapping line by line; the two identities the code below
# rests on are:
#
#   * the paper's auxiliary matrix `Ŵ = MXᵀ - ½X(XᵀMXᵀ)`, its skew-symmetrisation `W = Ŵ - Ŵᵀ` and the
#     projection `π_{T_X}(M) = WX` it defines (its lines 8-10, its equation (2)) are the package's
#     `Ω(Y, ·)`, and `global_rep(λ(Y), ·)` is that map conjugated into `𝔤ʰᵒʳ`: `W = λ(Y)Ḡλ(Y)ᵀ` for
#     `Ḡ = global_rep(λ(Y), ∇L)`. So the paper's three lines are the representation the package's
#     first-order caches are handed their gradient in already, and there is nothing to implement for
#     them. What the paper needs them for -- re-projecting the momentum onto the tangent space at the
#     new iterate -- is what `update_section!` does here instead.
#
#   * `v` is a *scalar*: a squared gradient norm rather than a squared gradient. That single number is
#     the whole of the difference from `Adam`, and the whole of what "non-geometric" names -- see the
#     docstring of [`NonGeometricAdam`](@ref).
#
# What is deliberately not ported is the paper's *retraction*: its lines 12-14 are two fixed-point
# iterations of equation (5), an approximation of the Cayley transform whose contraction condition is
# what its line 11 caps the step length for. This package retracts exactly ([`Cayley`](@ref) via
# Sherman-Morrison-Woodbury, or any other [`AbstractRetraction`](@ref)) and bounds the step for an
# unrelated reason, [`step_αmax`](@ref).

function _non_geometric_adam_check(x)
    x isa StiefelManifold || throw(ArgumentError("NonGeometricAdam supports exactly one StiefelManifold solution; ordinary arrays, Grassmann solutions, NamedTuples, and mixed trees are unsupported"))
    eltype(x) <: AbstractFloat || throw(ArgumentError("NonGeometricAdam requires floating-point Stiefel parameters"))
    x
end

OptimizerCache(::NonGeometricAdam, x) = throw(ArgumentError("NonGeometricAdam supports exactly one StiefelManifold solution"))
OptimizerCache(::NonGeometricAdam{T}, x::StiefelManifold{T}) where {T} = NonGeometricAdamCache(x)
Hessian(::NonGeometricAdam, ::OptimizerProblem, ::StiefelManifold{T}) where {T} = NoHessian{T}()

"""
    NonGeometricAdamCache <: OptimizerCache

Cache for [`NonGeometricAdam`](@ref).

The fields are [`AdamCache`](@ref)'s, except that the second moment is a scalar and there is
therefore no `m₂` to keep in a horizontal lift:

- `x::`[`StiefelManifold`](@ref): the solution,
- `g`: the gradient, in [`StiefelLieAlgHorMatrix`](@ref) form,
- `δ`: the direction,
- `Δg`: difference in gradients, needed for [`OptimizerStatus`](@ref),
- `g̃`: scratch for [`latest_gradient`](@ref); see [`GradientCache`](@ref),
- `g̃_is_current`: whether `g̃` is the gradient at `x`; see [`store_gradient!`](@ref),
- `m₁`: the first moment, a horizontal lift, stored bias-corrected,
- `m₂::T`: the second moment, the *scalar* of [li2020efficient](@cite), stored bias-corrected,
- `m̃₂::T`: `√(m₂ + δ)`, the scalar the direction is divided by,
- `section`: the [`GlobalSection`](@ref).
"""
mutable struct NonGeometricAdamCache{T,XT,VT,ST} <: OptimizerCache{T}
    x::XT
    g::VT
    δ::VT
    Δg::VT
    g̃::VT
    g̃_is_current::Base.RefValue{Bool}
    m₁::VT
    m₂::T
    m̃₂::T
    section::ST
end

function NonGeometricAdamCache(x::StiefelManifold{T}) where {T}
    _non_geometric_adam_check(x)
    sec = GlobalSection(x)
    g = global_rep(sec, zero(x.A))
    δ = _zero(g)
    Δg = _similar(g)
    _fill!(Δg, T(NaN))
    g̃ = _similar(g)
    _fill!(g̃, T(NaN))
    NonGeometricAdamCache{T,typeof(x),typeof(g),typeof(sec)}(
        _copy(x), g, δ, Δg, g̃, Ref(false), _zero(g), zero(T), zero(T), sec)
end

solution(cache::NonGeometricAdamCache) = cache.x
gradient(cache::NonGeometricAdamCache) = cache.g
gradient_array(cache::NonGeometricAdamCache) = cache.g
latest_gradient(cache::NonGeometricAdamCache) = cache.g̃
refresh_latest_gradient!(cache::NonGeometricAdamCache, g::Gradient) = _refresh_latest_gradient!(cache, g)
latest_gradient_is_current(cache::NonGeometricAdamCache, state::OptimizerState, x::OptimizerSolution) =
    _latest_gradient_is_current(cache, state, x)
invalidate_latest_gradient!(cache::NonGeometricAdamCache) = _invalidate_latest_gradient!(cache)
gradient_difference!(cache::NonGeometricAdamCache, ::OptimizerState) = _latest_gradient_difference!(cache)
direction(cache::NonGeometricAdamCache) = cache.δ
rhs(cache::NonGeometricAdamCache) = cache.δ
steepest_descent!(cache::NonGeometricAdamCache) = _steepest_descent_from_gradient!(cache)
section(cache::NonGeometricAdamCache) = cache.section
first_moment(cache::NonGeometricAdamCache) = cache.m₁
second_moment(cache::NonGeometricAdamCache) = cache.m₂
_second_moment(cache::NonGeometricAdamCache) = cache.m̃₂

"""
    NonGeometricAdamState <: OptimizerState

State for [`NonGeometricAdam`](@ref).

As for [`NonGeometricAdamCache`](@ref), `m₂` is a scalar and not a horizontal lift.
"""
mutable struct NonGeometricAdamState{T,OT,GS,VT} <: OptimizerState{T}
    section::GS
    iterations::Int
    x::OT
    x̄::OT
    g::VT
    ḡ::VT
    m₁::VT
    m₂::T
    f::T
    f̄::T
end

function NonGeometricAdamState(x::StiefelManifold{T}, g::GradientArrayOrNamedTuple{T}) where {T}
    _non_geometric_adam_check(x)
    _g = _copy(g)
    _g isa StiefelLieAlgHorMatrix || throw(ArgumentError("NonGeometricAdam requires a single Stiefel gradient"))
    _x = _copy(x)
    gs = GlobalSection(_x)
    # the first moment has to be initialized with zeros and not with `_similar`: it is read in the
    # first call to `update!(::NonGeometricAdamCache, ...)` before it is written to. Same for `m₂` --
    # the paper initializes it with `1` in its line 2 and with `0` in the authors' implementation;
    # see `ADAM_MATHS.md`.
    NonGeometricAdamState{T,typeof(_x),typeof(gs),typeof(_g)}(
        gs, 0, _x, _copy(_x), _g, _copy(_g), _zero(_g), zero(T), T(NaN), T(NaN))
end

NonGeometricAdamState(x::StiefelManifold{T}) where {T} =
    NonGeometricAdamState(x, global_rep(GlobalSection(x), zero(x.A)))
OptimizerState(::NonGeometricAdam, x::StiefelManifold) = NonGeometricAdamState(x)
OptimizerState(::NonGeometricAdam, x) = throw(ArgumentError("NonGeometricAdam supports exactly one StiefelManifold solution"))

solution(state::NonGeometricAdamState) = state.x
previous_solution(state::NonGeometricAdamState) = state.x̄
gradient(state::NonGeometricAdamState) = state.g
previous_gradient(state::NonGeometricAdamState) = state.ḡ
value(state::NonGeometricAdamState) = state.f
previous_value(state::NonGeometricAdamState) = state.f̄
first_moment(state::NonGeometricAdamState) = state.m₁
second_moment(state::NonGeometricAdamState) = state.m₂
section(state::NonGeometricAdamState) = state.section

function update!(state::NonGeometricAdamState{T}, gradient_array::StiefelLieAlgHorMatrix{T},
    direction::StiefelLieAlgHorMatrix{T}, _first_moment::StiefelLieAlgHorMatrix{T},
    _second_moment::Real, x::StiefelManifold{T}, f::Callable, retraction) where {T}
    _copyto!(state.x̄, state.x)
    _copyto!(state.ḡ, state.g)
    state.f̄ = state.f
    _copyto!(state.x, x)
    _copyto!(state.g, gradient_array)
    _copyto!(state.m₁, _first_moment)
    state.m₂ = T(_second_moment)
    state.f = f(x)
    update_section!(state.section, direction, retraction)
    state
end

function update!(state::NonGeometricAdamState, opt::Optimizer, x::OptimizerSolution)
    update!(state, gradient_array(cache(opt)), direction(cache(opt)), first_moment(opt.cache),
        second_moment(opt.cache), x, problem(opt).F, opt.retraction)
end

function update!(cache::NonGeometricAdamCache{T}, state::NonGeometricAdamState{T}, gradient::Gradient{T},
    method::NonGeometricAdam{T}, x::StiefelManifold{T}) where {T}
    # first, and before the two `_copyto!`s below; see `store_gradient!`
    store_gradient!(cache, state, gradient, x)
    _copyto!(section(cache), section(state))
    _copyto!(solution(cache), x)
    t = state.iterations
    @assert t ≥ 1 "the bias-corrected NonGeometricAdam moments are undefined before the first iteration (t = $(t)); `increase_iteration_number!` has to be called before the step"

    # Both moments are stored bias-corrected, as [`Adam`](@ref)'s are, hence the `- β^t` in the
    # numerators of `fac₁₁` and `fac₂₁`. This is the paper's lines 4-7 rearranged and not a departure
    # from them: substituting `m = (1 - β₁ᵗ)m̂` into its recursion gives exactly these factors, and its
    # `r` of line 7 is then `√(v̂ + ε)` alone, the `(1 - β₁ᵗ)` of `r` having been absorbed.
    fac₁₁ = (method.β₁ - method.β₁^t) / (1 - method.β₁^t)
    fac₁₂ = (1 - method.β₁) / (1 - method.β₁^t)
    fac₂₁ = (method.β₂ - method.β₂^t) / (1 - method.β₂^t)
    fac₂₂ = (1 - method.β₂) / (1 - method.β₂^t)
    _copyto!(first_moment(cache), _mul(fac₁₁, first_moment(state)))
    _add!(first_moment(cache), _mul(fac₁₂, gradient_array(cache)))
    # The line that makes this method the paper's and not `Adam`'s: `‖Ḡ‖²` where `Adam` has `Ḡ ⊙ Ḡ`.
    # `l2norm` of a horizontal lift is the norm of its free parameters, `√(½‖A‖² + ‖B‖²)` in the
    # ambient Frobenius norm -- the norm this package measures gradients and steps with everywhere
    # else, and not the ambient `‖∇L‖²` of the authors' implementation; see `ADAM_MATHS.md`.
    cache.m₂ = fac₂₁ * second_moment(state) + fac₂₂ * l2norm(gradient_array(cache))^2
    # `√(m₂ + δ)` and not `√m₂ + δ`: the paper adds `ε` *inside* the square root (its line 7, and
    # `vnew_hat.add(epsilon).sqrt()` in the authors' implementation), where [`Adam`](@ref) adds it
    # outside. On a scalar the distinction is one number, but it is the paper's number.
    cache.m̃₂ = √(cache.m₂ + method.δ)
    # the direction is `-m̂/√(v̂ + ε)`, i.e. minus the paper's `W` of its line 9. It carries no learning
    # rate: that is the line search's `α` (see [`default_linesearch`](@ref)), and the cap the paper's
    # line 11 puts on `α` is [`step_αmax`](@ref) here.
    _copyto!(direction(cache), first_moment(cache))
    _rmul!(direction(cache), -one(T) / cache.m̃₂)

    cache
end
