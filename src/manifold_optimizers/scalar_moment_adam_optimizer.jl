# Cayley ADAM, [li2020efficient](@cite) Algorithm 2, in this package's global tangent space
# representation. The [Optimizer Methods](@ref "Standard Neural Network Optimizers") page maps every
# symbol of the source's pseudocode onto the code below and the docstring of
# [`ScalarMomentAdam`](@ref) records where the port departs from it; the two identities the code rests
# on are:
#
#   * the paper's auxiliary matrix `Ŵ = MXᵀ - ½X(XᵀMXᵀ)`, its skew-symmetrisation `W = Ŵ - Ŵᵀ` and the
#     projection `π_{T_X}(M) = WX` it defines (its lines 8-10, its equation (2)) are the package's
#     `Ω(Y, ·)`, and `global_rep(λ(Y), ·)` is that map conjugated into `𝔤ʰᵒʳ`: `W = λ(Y)Ḡλ(Y)ᵀ` for
#     `Ḡ = global_rep(λ(Y), ∇L)`. So the paper's three lines are the representation the package's
#     first-order caches are handed their gradient in already, and there is nothing to implement for
#     them. What the paper needs them for -- re-projecting the momentum onto the tangent space at the
#     new iterate -- is what `update_section!` does here instead. The derivation is on that page, and
#     `test/scalar_moment_adam.jl` pins it against a literal transcription of the source's formula.
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

function _scalar_moment_adam_eltype_message(
        method::ScalarMomentAdam{T}, x::StiefelManifold{S}) where {
        T, S}
    "ScalarMomentAdam($(T)) cannot optimize StiefelManifold{$(S)} parameters. Like `Adam`, " *
    "`ScalarMomentAdam` carries parameters of its own and is not converted by `Optimizer`, so it has " *
    "to be constructed with the element type of the parameters: `ScalarMomentAdam($(S))`."
end

# The scope check, as an error rather than a `MethodError`. The second argument has to be typed
# `::OptimizerSolution` and *not* left as `Any`: the fallback in
# `optimizers/newton_optimizer/newton_optimizer_cache.jl` is
# `OptimizerCache(::OptimizerMethod, ::OptimizerSolution{T})`, and against an untyped second argument
# neither method is more specific, so every unsupported `x` raised an *ambiguity* `MethodError`
# instead of the `ArgumentError` this method exists to raise. That covered `Optimizer(x, F;
# algorithm = ScalarMomentAdam())` for an `AbstractVector`, a `NamedTuple`, a `GrassmannManifold` and
# an element-type-mismatched `StiefelManifold` alike -- every path a caller reaches this by.
# `test/scalar_moment_adam.jl` pins all four.
function OptimizerCache(::ScalarMomentAdam, ::OptimizerSolution)
    throw(ArgumentError(_SCALAR_MOMENT_ADAM_SCOPE))
end
# The element-type mismatch gets its own message: `ScalarMomentAdam()` is `Float64` and, as for
# [`Adam`](@ref), is not converted by [`Optimizer`](@ref), so `Float32` parameters are the likeliest
# way to arrive here and "supports exactly one StiefelManifold" would be a lie about the reason.
function OptimizerCache(method::ScalarMomentAdam, x::StiefelManifold)
    throw(ArgumentError(_scalar_moment_adam_eltype_message(method, x)))
end
function OptimizerCache(::ScalarMomentAdam{T}, x::StiefelManifold{T}) where {T}
    ScalarMomentAdamCache(x)
end
function Hessian(::ScalarMomentAdam, ::OptimizerProblem, ::StiefelManifold{T}) where {T}
    NoHessian{T}()
end

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
mutable struct ScalarMomentAdamCache{T, XT, VT, ST} <: OptimizerCache{T}
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
    # `_zero(x)` and not `global_rep(sec, zero(x.A))`: `Base.zero(::StiefelManifold)` already returns
    # the zero of `𝔤ʰᵒʳ` for that manifold, so lifting a zero matrix through the section is a matrix
    # product whose answer is the array `_zero` hands back for free. `OptimizerCache(::Adam{T}, x)`
    # passes `_zero(x)` for the same reason.
    g = _zero(x)
    δ = _zero(g)
    Δg = _similar(g)
    _fill!(Δg, T(NaN))
    g̃ = _similar(g)
    _fill!(g̃, T(NaN))
    ScalarMomentAdamCache{T, typeof(x), typeof(g), typeof(sec)}(
        _copy(x), g, δ, Δg, g̃, Ref(false), _zero(g), zero(T), zero(T), sec)
end

solution(cache::ScalarMomentAdamCache) = cache.x
gradient(cache::ScalarMomentAdamCache) = cache.g
gradient_array(cache::ScalarMomentAdamCache) = cache.g
latest_gradient(cache::ScalarMomentAdamCache) = cache.g̃
function refresh_latest_gradient!(cache::ScalarMomentAdamCache, g::Gradient)
    _refresh_latest_gradient!(cache, g)
end
function latest_gradient_is_current(
        cache::ScalarMomentAdamCache, state::OptimizerState, x::OptimizerSolution)
    _latest_gradient_is_current(cache, state, x)
end
function invalidate_latest_gradient!(cache::ScalarMomentAdamCache)
    _invalidate_latest_gradient!(cache)
end
function gradient_difference!(cache::ScalarMomentAdamCache, ::OptimizerState)
    _latest_gradient_difference!(cache)
end
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
mutable struct ScalarMomentAdamState{T, OT, GS, VT} <: OptimizerState{T}
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

function ScalarMomentAdamState(x::StiefelManifold{T}, g::GradientStorage{T}) where {T}
    g isa StiefelLieAlgHorMatrix ||
        throw(ArgumentError("ScalarMomentAdam requires a single Stiefel gradient"))
    _g = _copy(g)
    _x = _copy(x)
    gs = GlobalSection(_x)
    # the first moment has to be initialized with zeros and not with `_similar`: it is read in the
    # first call to `update!(::ScalarMomentAdamCache, ...)` before it is written to. Same for `m₂` --
    # the paper initializes it with `1` in its line 2 and with `0` in the authors' implementation,
    # and `0` is what is followed; see the docstring of [`ScalarMomentAdam`](@ref).
    ScalarMomentAdamState{T, typeof(_x), typeof(gs), typeof(_g)}(
        gs, 0, _x, _copy(_x), _g, _copy(_g), _zero(_g), zero(T), T(NaN), T(NaN))
end

# `_zero(x)` for the same reason as in the cache above -- and here it also saves a whole
# `GlobalSection`, i.e. the QR of a random `N×N` matrix, built only to map a zero matrix to the zero
# lift and then discarded (the inner constructor builds the section it keeps itself). This is
# `AdamState(x) = AdamState(x, _zero(x))`.
ScalarMomentAdamState(x::StiefelManifold{T}) where {T} = ScalarMomentAdamState(x, _zero(x))

# The same three-method shape as `OptimizerCache` above, and for the same reason: the scope and the
# element type are what this method is narrow about, so both have to be *said* rather than left to a
# `MethodError`. `Adam`'s `OptimizerState(::Adam, x...)` is permissive here and `ScalarMomentAdam` is
# deliberately not -- `Adam` accepts every `OptimizerSolution` there is, so for it there is nothing
# to reject; this one accepts a single `StiefelManifold{T}` and nothing else, and a state built for
# the wrong element type would otherwise be handed to a step that cannot use it. `CHANGELOG.md`
# promises the check on this path as well as on `Optimizer`'s, and until this it was only on
# `Optimizer`'s: `OptimizerState(ScalarMomentAdam(), rand(StiefelManifold{Float32}, 4, 2))` returned
# a `ScalarMomentAdamState{Float32}` for a `Float64` method.
function OptimizerState(::ScalarMomentAdam{T}, x::StiefelManifold{T}) where {T}
    ScalarMomentAdamState(x)
end
function OptimizerState(method::ScalarMomentAdam, x::StiefelManifold)
    throw(ArgumentError(_scalar_moment_adam_eltype_message(method, x)))
end
# The gradient-supplying form, as [`Adam`](@ref) has through its `OptimizerState(::Adam, x...)`. It is
# what [`ScalarMomentAdamState`](@ref)'s two-argument constructor is for, and without this method it
# reached the generic `OptimizerState(::OptimizerMethod, args...)` and was told that
# `OptimizerState` is "not implemented for ScalarMomentAdam", which was untrue.
function OptimizerState(::ScalarMomentAdam{T}, x::StiefelManifold{T}, g) where {T}
    ScalarMomentAdamState(x, g)
end
function OptimizerState(method::ScalarMomentAdam, x::StiefelManifold, g)
    throw(ArgumentError(_scalar_moment_adam_eltype_message(method, x)))
end
# `x, args...` and not just `x`: the gradient-supplying form has to reject an unsupported `x` with
# the scope message too, and a `Vararg` tail is less specific than every `StiefelManifold` method
# above, so it catches exactly what they do not.
function OptimizerState(::ScalarMomentAdam, x, args...)
    throw(ArgumentError(_SCALAR_MOMENT_ADAM_SCOPE))
end

solution(state::ScalarMomentAdamState) = state.x
previous_solution(state::ScalarMomentAdamState) = state.x̄
gradient(state::ScalarMomentAdamState) = state.g
previous_gradient(state::ScalarMomentAdamState) = state.ḡ
value(state::ScalarMomentAdamState) = state.f
previous_value(state::ScalarMomentAdamState) = state.f̄
first_moment(state::ScalarMomentAdamState) = state.m₁
second_moment(state::ScalarMomentAdamState) = state.m₂
section(state::ScalarMomentAdamState) = state.section

function update!(
        state::ScalarMomentAdamState{T}, gradient_array::StiefelLieAlgHorMatrix{T},
        direction::StiefelLieAlgHorMatrix{T}, _first_moment::StiefelLieAlgHorMatrix{T},
        _second_moment::Real, x::StiefelManifold{T}, f::Callable, retraction,
        observer = NoStepObserver()) where {T}
    _copyto!(state.x̄, state.x)
    _copyto!(state.ḡ, state.g)
    state.f̄ = state.f
    _copyto!(state.x, x)
    _copyto!(state.g, gradient_array)
    _copyto!(state.m₁, _first_moment)
    state.m₂ = T(_second_moment)
    state.f = observe_optimizer_phase(observer, :objective) do
        f(x)
    end
    observe_optimizer_phase(observer, :retraction_application) do
        update_section!(state.section, direction, retraction)
    end
    state
end

function update!(state::ScalarMomentAdamState, opt::Optimizer, x::OptimizerSolution)
    update!(
        state, gradient_array(cache(opt)), direction(cache(opt)), first_moment(opt.cache),
        second_moment(opt.cache), x, problem(opt).F, opt.retraction, step_observer(opt))
end

"""
    _squared_gradient_norm(method::ScalarMomentAdam, cache, gradient, x)

The `‖·‖²` that [`ScalarMomentAdam`](@ref)'s second moment accumulates, selected by
`method.ambient_norm`.

`false` — the default — squares the horizontal lift already in the cache; `true` squares
[li2020efficient](@cite)'s own ambient Euclidean gradient, at one extra gradient evaluation per step.
`Gradient` applied to a [`Manifold`](@ref) returns `rgrad(Y, ∇L)` and the ambient `∇L` is not part of
the optimizer protocol, so `true` recovers it from the flattened closure
`GradientAutodiff(F, ::Manifold)` builds — which is what `gradient(vec(x))` is. `store_gradient!`'s
reuse does not help, because what it caches is the lift.

The two are not interchangeable up to a constant, and which to reach for when is on the
[Optimizer Methods](@ref "Standard Neural Network Optimizers") page.
"""
function _squared_gradient_norm(method::ScalarMomentAdam, cache::ScalarMomentAdamCache,
        gradient::Gradient, x::StiefelManifold)
    method.ambient_norm ? sum(abs2, gradient(vec(x))) : l2norm(gradient_array(cache))^2
end

function update!(cache::ScalarMomentAdamCache{T},
        state::ScalarMomentAdamState{T}, gradient::Gradient{T},
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
