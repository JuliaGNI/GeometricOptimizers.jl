function _non_geometric_adam_check(x)
    x isa StiefelManifold || throw(ArgumentError("NonGeometricAdam supports exactly one StiefelManifold solution; ordinary arrays, Grassmann solutions, NamedTuples, and mixed trees are unsupported"))
    eltype(x) <: AbstractFloat || throw(ArgumentError("NonGeometricAdam requires floating-point Stiefel parameters"))
    x
end

function _non_geometric_adam_ambient(section::GlobalSection, B::StiefelLieAlgHorMatrix)
    n = B.n
    Matrix(section) * Matrix(B)[:, 1:n]
end

OptimizerCache(::NonGeometricAdam, x) = throw(ArgumentError("NonGeometricAdam supports exactly one StiefelManifold solution"))
OptimizerCache(::NonGeometricAdam{T}, x::StiefelManifold{T}) where {T} = NonGeometricAdamCache(x)
Hessian(::NonGeometricAdam, ::OptimizerProblem, ::StiefelManifold{T}) where {T} = NoHessian{T}()

mutable struct NonGeometricAdamCache{T,XT,MT,VT,ST} <: OptimizerCache{T}
    x::XT
    g::VT
    δ::VT
    Δg::VT
    g̃::VT
    g̃_is_current::Base.RefValue{Bool}
    m₁::MT
    m₂::MT
    m̃₂::MT
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
    NonGeometricAdamCache{T,typeof(x),typeof(x.A),typeof(g),typeof(sec)}(
        _copy(x), g, δ, Δg, g̃, Ref(false), zero(x.A), zero(x.A), zero(x.A), sec)
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

mutable struct NonGeometricAdamState{T,OT,MT,GS,VT} <: OptimizerState{T}
    section::GS
    iterations::Int
    x::OT
    x̄::OT
    g::VT
    ḡ::VT
    m₁::MT
    m₂::MT
    f::T
    f̄::T
end

function NonGeometricAdamState(x::StiefelManifold{T}, g::GradientArrayOrNamedTuple{T}) where {T}
    _non_geometric_adam_check(x)
    _g = _copy(g)
    _g isa StiefelLieAlgHorMatrix || throw(ArgumentError("NonGeometricAdam requires a single Stiefel gradient"))
    _x = _copy(x)
    gs = GlobalSection(_x)
    NonGeometricAdamState{T,typeof(_x),typeof(_x.A),typeof(gs),typeof(_g)}(
        gs, 0, _x, _copy(_x), _g, _copy(_g), zero(x.A), zero(x.A), T(NaN), T(NaN))
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

function update!(state::NonGeometricAdamState, gradient_array::StiefelLieAlgHorMatrix,
    direction::StiefelLieAlgHorMatrix, _first_moment::AbstractMatrix, _second_moment::AbstractMatrix,
    x::StiefelManifold, f::Callable, retraction)
    _copyto!(state.x̄, state.x)
    _copyto!(state.ḡ, state.g)
    state.f̄ = state.f
    _copyto!(state.x, x)
    _copyto!(state.g, gradient_array)
    _copyto!(state.m₁, _first_moment)
    _copyto!(state.m₂, _second_moment)
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
    store_gradient!(cache, state, gradient, x)
    _copyto!(section(cache), section(state))
    _copyto!(solution(cache), x)
    t = state.iterations
    t ≥ 1 || throw(ArgumentError("the first NonGeometricAdam update requires iteration number 1"))

    G = _non_geometric_adam_ambient(section(state), gradient_array(cache))
    _copyto!(cache.m₁, method.β₁ .* state.m₁ .+ (one(T) - method.β₁) .* G)
    _copyto!(cache.m₂, method.β₂ .* state.m₂ .+ (one(T) - method.β₂) .* (G .* G))
    _copyto!(cache.m̃₂, cache.m₂ ./ (one(T) - method.β₂^t))
    D = (cache.m₁ ./ (one(T) - method.β₁^t)) ./ (sqrt.(cache.m̃₂) .+ method.δ)
    _copyto!(direction(cache), global_rep(section(state), -D))
    cache
end
