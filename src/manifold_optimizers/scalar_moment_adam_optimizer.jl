# Cayley ADAM, [li2020efficient](@cite) Algorithm 2, in this package's global tangent space
# representation. The docstring of [`ScalarMomentAdam`](@ref) maps every symbol of the source's
# pseudocode onto the code below and records where the port departs from it; the two identities the
# code rests on are:
#
#   * the paper's auxiliary matrix `Ŵ = MXᵀ - ½X(XᵀMXᵀ)`, its skew-symmetrisation `W = Ŵ - Ŵᵀ` and the
#     projection `π_{T_X}(M) = WX` it defines (its lines 8-10, its equation (2)) are the package's
#     `Ω(Y, ·)`, and `global_rep(λ(Y), ·)` is that map conjugated into `𝔤ʰᵒʳ`: `W = λ(Y)Ḡλ(Y)ᵀ` for
#     `Ḡ = global_rep(λ(Y), ∇L)`. So the paper's three lines are the representation the package's
#     first-order caches are handed their gradient in already, and there is nothing to implement for
#     them. What the paper needs them for -- re-projecting the momentum onto the tangent space at the
#     new iterate -- is what `update_section!` does here instead. The derivation is on the
#     [Optimizer Methods](@ref "Standard Neural Network Optimizers") page; `test/scalar_moment_adam.jl` pins it
#     against a literal transcription of the source's formula.
#
#   * `v` is a *scalar*: a squared gradient norm rather than a squared gradient. That single number is
#     the whole of the difference from `Adam`, and it is what the method is named for.
#
# What is deliberately not ported is the paper's *retraction*: its lines 12-14 are two fixed-point
# iterations of equation (5), an approximation of the Cayley transform whose contraction condition is
# what its line 11 caps the step length for. This package retracts exactly ([`Cayley`](@ref) via
# Sherman-Morrison-Woodbury, or any other [`AbstractRetraction`](@ref)) and bounds the step for an
# unrelated reason, [`step_αmax`](@ref).

const _SCALAR_MOMENT_ADAM_SCOPE = "ScalarMomentAdam supports exactly one StiefelManifold solution; " *
                                 "ordinary arrays, Grassmann solutions, NamedTuples and mixed parameter trees are unsupported"

_scalar_moment_adam_eltype_message(method::ScalarMomentAdam{T}, x::StiefelManifold{S}) where {T,S} =
    "ScalarMomentAdam($(T)) cannot optimize StiefelManifold{$(S)} parameters. Like `Adam`, " *
    "`ScalarMomentAdam` carries parameters of its own and is not converted by `Optimizer`, so it has " *
    "to be constructed with the element type of the parameters: `ScalarMomentAdam($(S))`."

# The scope check, as an error rather than a `MethodError`. The second argument has to be typed
# `::OptimizerSolution` and *not* left as `Any`: the fallback in
# `optimizers/newton_optimizer/newton_optimizer_cache.jl` is
# `OptimizerCache(::OptimizerMethod, ::OptimizerSolution{T})`, and against an untyped second argument
# neither method is more specific, so every unsupported `x` raised an *ambiguity* `MethodError`
# instead of the `ArgumentError` this method exists to raise. That covered `Optimizer(x, F;
# algorithm = ScalarMomentAdam())` for an `AbstractVector`, a `NamedTuple`, a `GrassmannManifold` and
# an element-type-mismatched `StiefelManifold` alike -- every path a caller reaches this by.
# `test/scalar_moment_adam.jl` pins all four.
OptimizerCache(::ScalarMomentAdam, ::OptimizerSolution) = throw(ArgumentError(_SCALAR_MOMENT_ADAM_SCOPE))
# The element-type mismatch gets its own message: `ScalarMomentAdam()` is `Float64` and, as for
# [`Adam`](@ref), is not converted by [`Optimizer`](@ref), so `Float32` parameters are the likeliest
# way to arrive here and "supports exactly one StiefelManifold" would be a lie about the reason.
OptimizerCache(method::ScalarMomentAdam, x::StiefelManifold) = throw(ArgumentError(_scalar_moment_adam_eltype_message(method, x)))
OptimizerCache(::ScalarMomentAdam{T}, x::StiefelManifold{T}) where {T} = ScalarMomentAdamCache(x)
Hessian(::ScalarMomentAdam, ::OptimizerProblem, ::StiefelManifold{T}) where {T} = NoHessian{T}()

"""
    ScalarMomentAdamCache <: OptimizerCache

Cache for [`ScalarMomentAdam`](@ref).

The fields are [`AdamCache`](@ref)'s, except that `m₂` is a single number rather than an element of
`𝔤ʰᵒʳ`:

- `x::`[`StiefelManifold`](@ref): the solution,
- `g`: the gradient, in [`StiefelLieAlgHorMatrix`](@ref) form,
- `δ`: the direction,
- `Δg`: difference in gradients, needed for [`OptimizerStatus`](@ref),
- `g̃`: scratch for [`latest_gradient`](@ref); see [`GradientCache`](@ref),
- `g̃_is_current`: whether `g̃` is the gradient at `x`; see [`store_gradient!`](@ref),
- `m₁`: the first moment, a horizontal lift, stored bias-corrected,
- `m₂::T`: the second moment, the *scalar* of [li2020efficient](@cite), stored bias-corrected,
- `m̃₂::T`: `√(m₂ + method.δ)`, the scalar the direction is divided by. The `δ` in it is
  [`ScalarMomentAdam`](@ref)'s regularizer and not the `δ` field above, which is the direction,
- `section`: the [`GlobalSection`](@ref).
"""
mutable struct ScalarMomentAdamCache{T,XT,VT,ST} <: OptimizerCache{T}
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

function ScalarMomentAdamCache(x::StiefelManifold{T}) where {T}
    sec = GlobalSection(x)
    g = global_rep(sec, zero(x.A))
    δ = _zero(g)
    Δg = _similar(g)
    _fill!(Δg, T(NaN))
    g̃ = _similar(g)
    _fill!(g̃, T(NaN))
    ScalarMomentAdamCache{T,typeof(x),typeof(g),typeof(sec)}(
        _copy(x), g, δ, Δg, g̃, Ref(false), _zero(g), zero(T), zero(T), sec)
end

solution(cache::ScalarMomentAdamCache) = cache.x
gradient(cache::ScalarMomentAdamCache) = cache.g
gradient_array(cache::ScalarMomentAdamCache) = cache.g
latest_gradient(cache::ScalarMomentAdamCache) = cache.g̃
refresh_latest_gradient!(cache::ScalarMomentAdamCache, g::Gradient) = _refresh_latest_gradient!(cache, g)
latest_gradient_is_current(cache::ScalarMomentAdamCache, state::OptimizerState, x::OptimizerSolution) =
    _latest_gradient_is_current(cache, state, x)
invalidate_latest_gradient!(cache::ScalarMomentAdamCache) = _invalidate_latest_gradient!(cache)
gradient_difference!(cache::ScalarMomentAdamCache, ::OptimizerState) = _latest_gradient_difference!(cache)
direction(cache::ScalarMomentAdamCache) = cache.δ
rhs(cache::ScalarMomentAdamCache) = cache.δ
steepest_descent!(cache::ScalarMomentAdamCache) = _steepest_descent_from_gradient!(cache)
section(cache::ScalarMomentAdamCache) = cache.section
first_moment(cache::ScalarMomentAdamCache) = cache.m₁
second_moment(cache::ScalarMomentAdamCache) = cache.m₂
_second_moment(cache::ScalarMomentAdamCache) = cache.m̃₂

"""
    ScalarMomentAdamState <: OptimizerState

State for [`ScalarMomentAdam`](@ref).

As for [`ScalarMomentAdamCache`](@ref), `m₂` is a scalar and not a horizontal lift.
"""
mutable struct ScalarMomentAdamState{T,OT,GS,VT} <: OptimizerState{T}
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

function ScalarMomentAdamState(x::StiefelManifold{T}, g::GradientArrayOrNamedTuple{T}) where {T}
    _g = _copy(g)
    _g isa StiefelLieAlgHorMatrix || throw(ArgumentError("ScalarMomentAdam requires a single Stiefel gradient"))
    _x = _copy(x)
    gs = GlobalSection(_x)
    # the first moment has to be initialized with zeros and not with `_similar`: it is read in the
    # first call to `update!(::ScalarMomentAdamCache, ...)` before it is written to. Same for `m₂` --
    # the paper initializes it with `1` in its line 2 and with `0` in the authors' implementation,
    # and `0` is what is followed; see the docstring of [`ScalarMomentAdam`](@ref).
    ScalarMomentAdamState{T,typeof(_x),typeof(gs),typeof(_g)}(
        gs, 0, _x, _copy(_x), _g, _copy(_g), _zero(_g), zero(T), T(NaN), T(NaN))
end

ScalarMomentAdamState(x::StiefelManifold{T}) where {T} =
    ScalarMomentAdamState(x, global_rep(GlobalSection(x), zero(x.A)))
OptimizerState(::ScalarMomentAdam, x::StiefelManifold) = ScalarMomentAdamState(x)
# The gradient-supplying form, as [`Adam`](@ref) has through its `OptimizerState(::Adam, x...)`. It is
# what [`ScalarMomentAdamState`](@ref)'s two-argument constructor is for, and without this method it
# reached the generic `OptimizerState(::OptimizerMethod, args...)` and was told that
# `OptimizerState` is "not implemented for ScalarMomentAdam", which was untrue.
OptimizerState(::ScalarMomentAdam, x::StiefelManifold, g) = ScalarMomentAdamState(x, g)
OptimizerState(::ScalarMomentAdam, x) = throw(ArgumentError(_SCALAR_MOMENT_ADAM_SCOPE))

solution(state::ScalarMomentAdamState) = state.x
previous_solution(state::ScalarMomentAdamState) = state.x̄
gradient(state::ScalarMomentAdamState) = state.g
previous_gradient(state::ScalarMomentAdamState) = state.ḡ
value(state::ScalarMomentAdamState) = state.f
previous_value(state::ScalarMomentAdamState) = state.f̄
first_moment(state::ScalarMomentAdamState) = state.m₁
second_moment(state::ScalarMomentAdamState) = state.m₂
section(state::ScalarMomentAdamState) = state.section

function update!(state::ScalarMomentAdamState{T}, gradient_array::StiefelLieAlgHorMatrix{T},
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

function update!(state::ScalarMomentAdamState, opt::Optimizer, x::OptimizerSolution)
    update!(state, gradient_array(cache(opt)), direction(cache(opt)), first_moment(opt.cache),
        second_moment(opt.cache), x, problem(opt).F, opt.retraction)
end

@doc raw"""
    _squared_gradient_norm(method::ScalarMomentAdam, cache, gradient, x)

The ``\lVert\cdot\rVert^2`` that [`ScalarMomentAdam`](@ref)'s second moment accumulates, selected by
`method.ambient_norm`.

`false` — the default — is `l2norm(gradient_array(cache))^2`, the squared norm of the horizontal
lift's free parameters, ``\frac{1}{2}\lVert{}A\rVert_F^2 + \lVert{}B\rVert_F^2``. It is free: the
lift is already in the cache, it is the norm the rest of the package measures gradients and steps with
(`rg` in [`OptimizerStatus`](@ref), [`step_αmax`](@ref)), and it makes [`Adam`](@ref) and
`ScalarMomentAdam` consume an identical gradient, so a comparison between the two differs in the
second moment and in nothing else.

`true` is [li2020efficient](@cite)'s own quantity, the squared Frobenius norm of the *ambient
Euclidean* gradient (`torch.norm(g)**2` on `p.grad` in the authors' implementation). It costs one
extra gradient evaluation per step, which is why it is not the default: `Gradient` applied to a
[`Manifold`](@ref) returns `rgrad(Y, ∇L)` and the ambient ``\nabla{}L`` is not part of the optimizer
protocol, so it has to be recovered from the flattened closure `GradientAutodiff(F, ::Manifold)`
builds — which is what `gradient(vec(x))` is. [`store_gradient!`](@ref)'s reuse does not help here,
because what it caches is the lift and not ``\nabla{}L``.

The two are not interchangeable up to a constant. `rgrad` annihilates the normal component, so for
``\nabla{}L = YS`` with ``S`` symmetric the lift — and with it the whole second moment — is zero
while ``\lVert\nabla{}L\rVert_F`` is not: the ratio is bounded above and has no positive lower
bound. Reach for `ambient_norm = true` to reproduce the source, and leave it `false` to compare
against [`Adam`](@ref).
"""
function _squared_gradient_norm(method::ScalarMomentAdam, cache::ScalarMomentAdamCache,
    gradient::Gradient, x::StiefelManifold)
    method.ambient_norm ? sum(abs2, gradient(vec(x))) : l2norm(gradient_array(cache))^2
end

function update!(cache::ScalarMomentAdamCache{T}, state::ScalarMomentAdamState{T}, gradient::Gradient{T},
    method::ScalarMomentAdam{T}, x::StiefelManifold{T}) where {T}
    # first, and before the two `_copyto!`s below; see `store_gradient!`
    store_gradient!(cache, state, gradient, x)
    _copyto!(section(cache), section(state))
    _copyto!(solution(cache), x)
    t = state.iterations
    @assert t ≥ 1 "the bias-corrected ScalarMomentAdam moments are undefined before the first iteration (t = $(t)); `increase_iteration_number!` has to be called before the step"

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
    # The line that makes this method the paper's and not `Adam`'s: a squared *norm* where `Adam` has
    # `Ḡ ⊙ Ḡ`. Which norm is `method.ambient_norm`'s business; see `_squared_gradient_norm`.
    cache.m₂ = fac₂₁ * second_moment(state) +
               fac₂₂ * _squared_gradient_norm(method, cache, gradient, x)
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
