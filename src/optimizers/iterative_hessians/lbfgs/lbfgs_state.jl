# The type parameters are deliberately unbounded; see the warning in `optimizer_solution.jl`.
# The invariant is enforced by the inner constructor's signature, which is dispatch and therefore
# costs nothing.
@doc raw"""
    LBFGSState <: OptimizerState

The [`OptimizerState`](@ref) of [`LBFGS`](@ref).

# Keys

The keys of [`BFGSState`](@ref) but `Q`, with the same meaning:
- `x`, `x̄`: the current and the previous iterate,
- `s`: the previous direction, in the coordinates of the lift,
- `ḡ`: the gradient the last step was built from,
- `f`, `f̄`: the objective at `x` and at `x̄`,

and in place of `Q` the memory of the two-loop recursion, all in the flattened coordinates of the
lift, whose length is ``d``:
- `S`, `Y`: `memory` vectors each, the secant pairs ``(\delta_i, \gamma_i)``, used as a ring buffer
  whose newest entry is `newest`, holding `npairs` pairs,
- `ρ`: the ``1/\delta_i^T\gamma_i``, and `α` the scratch of the first loop,
- `θ`: the scaling of the initial matrix ``H_0 = \theta{}M^{-1}``; see [`LBFGS`](@ref),
- `metric_section`: a copy of `section` at the start of the solve or the last restart, where ``M``
  is taken for `initial = :bfgs`,
- `q`, `z`: two flat vectors of length ``d``,
- `metric_workspace`: the buffers of the conjugate-gradient solve with ``M``, and `nothing` where
  ``M`` is the identity.

So the method holds ``(2\,\mathtt{memory} + 2)d`` numbers in the vectors, and ``7d`` more in
`metric_workspace` where ``M`` is not the identity, in place of the ``d\times{}d`` matrix `Q`.

`LBFGSState(x; memory = 10)` builds it; `OptimizerState(LBFGS(; memory), x)` passes the method's
`memory`.
"""
mutable struct LBFGSState{T, AT, GT, GS, VT, WT} <: OptimizerState{T}
    x::AT
    x̄::AT
    s::GT
    ḡ::GT
    f::T
    f̄::T
    iterations::Int

    section::GS
    metric_section::GS

    S::Vector{VT}
    Y::Vector{VT}
    ρ::Vector{T}
    α::Vector{T}
    θ::T
    newest::Int
    npairs::Int

    q::VT
    z::VT
    metric_workspace::WT

    function LBFGSState(x̄::AT,
            ḡ::GT,
            memory::Integer) where {T, AT <: OptimizerSolution{T},
            GT <: GradientStorage{T}}
        memory ≥ 1 || throw(ArgumentError("LBFGS needs `memory ≥ 1`, got $(memory)."))
        section = GlobalSection(x̄)
        metric_section = GlobalSection(x̄)
        backend = _solution_backend(x̄)
        d = _flat_length(x̄)
        flat() = KernelAbstractions.allocate(backend, T, d)
        S = [flat() for _ in 1:memory]
        Y = [flat() for _ in 1:memory]
        q = flat()
        workspace = _metric_workspace(x̄)
        state = new{T, AT, GT, typeof(section), typeof(q), typeof(workspace)}(
            _copy(x̄), x̄, _similar(ḡ), ḡ, T(NaN), T(NaN), 0, section, metric_section,
            S, Y, fill(T(NaN), memory), fill(T(NaN), memory), one(T), 0, 0,
            q, flat(), workspace)
        _poison!(state)
    end
end

function LBFGSState(x::OptimizerSolution{T}; memory::Integer = 10) where {T}
    LBFGSState(_copy(x), _zero(x), memory)
end

OptimizerState(method::LBFGS, x::OptimizerSolution) = LBFGSState(x; memory = method.memory)

# The length of the flattened coordinates, the `d` of the docstring: that of the lift `_zero(x)` and
# not of the point; see `QuasiNewtonCache`. A plain vector is its own flattening.
_flat_length(x::AbstractVector) = length(x)
_flat_length(x::OptimizerSolution) = flatlength(_zero(x))

# The buffers of `_metric_solve!`: two lifts and their flat mirrors, which `_metric_apply!` reads and
# writes, and the three vectors of the conjugate-gradient method. `nothing` where the metric is the
# identity, where nothing is solved.
_metric_workspace(::AbstractVector) = nothing

function _metric_workspace(x::OptimizerSolution{T}) where {T}
    _metric_is_identity(x) && return nothing
    B = _zero(x)
    e = FlatParameters(_flatten_on_backend(T, B)...)
    (B = B, MB = _zero(x), e = e, Me = similar(e), r = similar(parent(e)),
        p = similar(parent(e)), Mp = similar(parent(e)))
end

@doc raw"""
    _metric_solve!(z, b, λY, workspace)

Write ``M^{-1}b`` into the flat vector `z`, for the Gram matrix ``M`` of the metric at the section
`λY`; see [`GeometricOptimizers._metric_inverse!`](@ref). A copy where `workspace` is `nothing`, i.e.
where ``M`` is the identity.

Otherwise the conjugate-gradient method, which applies ``M`` with
[`GeometricOptimizers._metric_apply!`](@ref) and so forms neither ``M`` nor ``M^{-1}``. ``M`` is
symmetric positive definite, so the method terminates in at most ``d`` steps in exact arithmetic; it
stops when the residual is below ``\varepsilon\|b\|`` or after ``10d`` steps. Round-off makes it take
more than ``d`` steps where ``M`` is ill-conditioned: stopped after ``d = 34`` steps, the solve was
5% off at a section with ``\kappa(M) = 6\cdot10^3``, which made the path of `LBFGS` depend on the
section. On a symplectic point with
``2N = 256``, ``2n = 30`` (``d = 7245``) it took 251 steps at the section `global_section` builds.
"""
_metric_solve!(z, b, λY, ::Nothing) = copyto!(z, b)

function _metric_solve!(z::AbstractVector{T}, b::AbstractVector{T}, λY,
        workspace::NamedTuple) where {T}
    (; r, p, Mp) = workspace
    fill!(z, zero(T))
    copyto!(r, b)
    copyto!(p, r)
    rr = dot(r, r)
    tolerance = (eps(T) * sqrt(rr))^2
    iszero(rr) && return z
    for _ in 1:(10 * length(z))
        _metric_apply_flat!(Mp, λY, p, workspace)
        a = rr / dot(p, Mp)
        z .+= a .* p
        r .-= a .* Mp
        rr_new = dot(r, r)
        rr_new ≤ tolerance && break
        p .= r .+ (rr_new / rr) .* p
        rr = rr_new
    end
    z
end

# `Mp ← M p` in the flat coordinates, through the lifts of `workspace`; the column loop of
# `_metric_inverse!` does the same for one unit vector at a time.
function _metric_apply_flat!(Mp, λY, p, workspace)
    copyto!(parent(workspace.e), p)
    unflatten!(workspace.B, workspace.e)
    _metric_apply!(workspace.MB, λY, workspace.B)
    copyto!(Mp, parent(flatten!(workspace.Me, workspace.MB)))
end

# Between the parameters' representation and a flat vector, through the cache's flat buffers; see
# `_flat_scratch`. A plain vector is its own flattening.
_to_flat!(v, g, ::Nothing) = copyto!(v, g)
_to_flat!(v, g, flat) = copyto!(v, parent(flatten!(flat.rhs, g)))
_from_flat!(g, v, ::Nothing) = copyto!(g, v)
function _from_flat!(g, v, flat)
    copyto!(parent(flat.direction), v)
    unflatten!(g, flat.direction)
end

@doc raw"""
    restart!(state::LBFGSState)

Empty the memory of the two-loop recursion and take the metric for the initial matrix ``H_0`` at the
current section, so that the next direction is the steepest descent of the metric, as after
[`restart!(::BFGSState)`](@ref). [`solver_step!`](@ref) calls it when a line search reports that it
could not decrease the merit — see [`linesearch_rejected`](@ref).
"""
function restart!(state::LBFGSState)
    _reset_memory!(state)
    state
end

function _reset_memory!(state::LBFGSState{T}) where {T}
    state.npairs = 0
    state.newest = 0
    state.θ = one(T)
    state.metric_workspace === nothing || _copyto!(state.metric_section, state.section)
    state
end

# As `_poison!(::BFGSState)`: the first secant pair is `NaN`, `curvature_is_usable` rejects it, and
# the first direction is the steepest descent of the metric.
function _poison!(state::LBFGSState{T}) where {T}
    _fill!(state.s, T(NaN))
    _fill!(state.ḡ, T(NaN))
    state.f = T(NaN)
    state.f̄ = T(NaN)
    _reset_memory!(state)
end

# The pair `(δ, γ)`, flat, with `δᵀγ = ΔxΔg`, over the oldest one once `memory` are held.
function _push_pair!(state::LBFGSState{T}, δ, γ, ΔxΔg::T, method::LBFGS) where {T}
    j = mod1(state.newest + 1, length(state.S))
    copyto!(state.S[j], δ)
    copyto!(state.Y[j], γ)
    state.ρ[j] = inv(ΔxΔg)
    state.newest = j
    state.npairs = min(state.npairs + 1, length(state.S))
    if method.initial === :scaled
        # `θ = δᵀγ / γᵀM⁻¹γ`, with `z` as scratch
        _metric_solve!(state.z, state.Y[j], _initial_section(method, state),
            state.metric_workspace)
        state.θ = ΔxΔg / dot(state.Y[j], state.z)
    end
    state
end

# The section `M` of `H₀ = θM⁻¹` is taken at; see `LBFGS`.
function _initial_section(method::LBFGS, state::LBFGSState)
    method.initial === :scaled ? state.section : state.metric_section
end

# `d ← H r` for the `H` of the two-loop recursion [nocedal2006numerical, Algorithm 7.4], with `r` and
# `d` in the parameters' representation and `H₀ = θM⁻¹` for `M` at the section `λY`.
function _two_loop!(d, state::LBFGSState, r, flat, λY)
    (; S, Y, ρ, α, q, z) = state
    m = length(S)
    _to_flat!(q, r, flat)
    for i in 0:(state.npairs - 1)
        j = mod1(state.newest - i, m)
        α[j] = ρ[j] * dot(S[j], q)
        q .-= α[j] .* Y[j]
    end
    _metric_solve!(z, q, λY, state.metric_workspace)
    z .*= state.θ
    for i in (state.npairs - 1):-1:0
        j = mod1(state.newest - i, m)
        β = ρ[j] * dot(Y[j], z)
        z .+= (α[j] - β) .* S[j]
    end
    _from_flat!(d, z, flat)
end

# `d ← M⁻¹ r` at the section `λY`, with `r` and `d` in the parameters' representation.
function _metric_descent!(d, state::LBFGSState, r, λY, flat)
    if state.metric_workspace === nothing
        _copyto!(d, r)
    else
        _to_flat!(state.q, r, flat)
        _metric_solve!(state.z, state.q, λY, state.metric_workspace)
        _from_flat!(d, state.z, flat)
    end
    d
end

# The step path of `update!(::BFGSState, …)`, which this repeats for the same fields.
function update!(state::LBFGSState{T}, direction::GradientStorage{T}, gradient::Gradient,
        x::XT, f::T, retraction,
        observer = NoStepObserver(), workspace = nothing) where {
        T, XT <: OptimizerSolution{T}}
    _copyto!(state.x̄, state.x)
    _copyto!(state.x, x)
    # `ḡ` is advanced by the cache, right after the secant pair is formed; see `update!(::BFGSState, …)`
    state.f̄ = state.f
    state.f = f

    _copyto!(state.s, direction)
    observe_optimizer_phase(observer, :retraction_application) do
        update_section!(section(state), state.s, retraction, workspace)
    end
    _copyto!(state.section, x)

    state
end

# The cache update of `BFGS`, with the update of `Q` replaced by the push of the pair and `Q rhs` by
# the two-loop recursion; see `update!(::QuasiNewtonCache, ::BFGSState, …)`.
function update!(cache::QuasiNewtonCache{T}, state::LBFGSState{T},
        x::OptimizerSolution{T}, g::GradientStorage{T}) where {T}
    _copyto!(cache.x, x)
    _copyto!(direction(cache), state.s)
    _copyto!(gradient(cache), g)
    _copyto!(rhs(cache), g)
    _rmul!(rhs(cache), -one(T))
    _difference!(cache.Δg, gradient(cache), state.ḡ)

    ΔxΔg = _dot(cache.Δx, cache.Δg)
    if curvature_is_usable(ΔxΔg, cache.Δx, cache.Δg)
        δ, γ = _flat_secant(cache)
        _push_pair!(state, δ, γ, ΔxΔg, cache.method)
    end
    _copyto!(state.ḡ, gradient(cache))

    _two_loop!(direction(cache), state, rhs(cache), cache.flat,
        _initial_section(cache.method, state))
    _copyto!(state.s, direction(cache))

    cache
end

# As the `BFGSState` methods: the substitute for a direction that does not descend is the steepest
# descent of the metric at the current section, and the direction after a restart is `H₀ rhs`.
function ensure_descent!(cache::QuasiNewtonCache, state::LBFGSState,
        method::LBFGS, config::Options)
    δ = direction(cache)
    r = rhs(cache)
    if !(_dot(r, δ) > 0)
        config.verbosity ≥ 2 &&
            @warn "the $(method) direction is not a descent direction; using the steepest-descent direction for this step." maxlog = 1
        _metric_descent!(δ, state, r, state.section, cache.flat)
    end

    cache
end

function steepest_descent!(cache::QuasiNewtonCache, state::LBFGSState)
    _metric_descent!(direction(cache), state, rhs(cache), state.metric_section, cache.flat)

    cache
end

# `LBFGS` forms no `Q`; see `_inverse_hessian_scratch`.
function _inverse_hessian_scratch(::LBFGS, backend, ::Type{T}, n::Integer) where {T}
    (nothing, nothing, nothing, nothing, nothing)
end
