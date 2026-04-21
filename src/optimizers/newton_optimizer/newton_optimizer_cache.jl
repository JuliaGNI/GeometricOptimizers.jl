"""
    NewtonOptimizerCache <: OptimizerCache

# Keys

- `x`: current iterate (this stores the guess called by the functions generated with [`linesearch_problem`](@ref)),
- `Δx`: direction of optimization step (difference between `x` and `x̄`); this is obtained by multiplying `rhs` with the inverse of the Hessian,
- `g`: gradient value (this stores the gradient associated with `x` called by the *derivative part* of [`linesearch_problem`](@ref)),
- `Δg`: gradient difference (difference between `g` and `ḡ`); this is used for computing the [`OptimizerStatus`](@ref),
- `rhs`: the right hand side used to compute the update,
- `H`: the Hessian matrix evaluated at `x`,

Also compare this to [`SimpleSolvers.NonlinearSolverCache`](@extref).
"""
struct NewtonOptimizerCache{T,AT<:AbstractArray{T},HT<:AbstractMatrix{T},GS<:GlobalSection{T}} <: OptimizerCache{T}
    x::AT
    Δx::AT
    g::AT
    Δg::AT
    rhs::AT
    H::HT

    section::GS

    function NewtonOptimizerCache(x::AT) where {T,AT<:AbstractArray{T}}
        h = zeros(T, length(x), length(x))
        section = GlobalSection(x)
        cache = new{T,AT,typeof(h),typeof(section)}(similar(x), similar(x), similar(x), similar(x), similar(x), h, section)
        initialize!(cache, x)
        cache
    end

    # we probably don't need this constructor
    function NewtonOptimizerCache(x::AT, problem::OptimizerProblem) where {T<:Number,AT<:AbstractArray{T}}
        g = Gradient(problem)(x)
        h = Hessian(problem)(x)
        section = GlobalSection(x)
        new{T,AT,typeof(h),typeof(section)}(copy(x), copy(x), zero(x), g, -g, zero(x), h, section)
    end
end

OptimizerCache(::OptimizerMethod, x::OptimizerSolution) = NewtonOptimizerCache(x)

section(cache::NewtonOptimizerCache) = cache.section

"""
    rhs(cache)

Return the right hand side of an instance of [`NewtonOptimizerCache`](@ref)
"""
rhs(cache::NewtonOptimizerCache) = cache.rhs

"""
    gradient(::NewtonOptimizerCache)

Return the stored gradient (array) of an instance of [`NewtonOptimizerCache`](@ref)
"""
gradient(cache::NewtonOptimizerCache) = cache.g
gradient_array(cache::NewtonOptimizerCache) = gradient(cache)

"""
    direction(cache)

Return the direction of the gradient step (i.e. `Δx`) of an instance of [`NewtonOptimizerCache`](@ref).
"""
direction(cache::NewtonOptimizerCache) = cache.Δx

hessian(cache::NewtonOptimizerCache) = cache.H

solution(cache::NewtonOptimizerCache) = cache.x

@doc raw"""
    update!(cache::NewtonOptimizerCache, x, g, hes)

Update an instance of [`NewtonOptimizerCache`](@ref) based on `x`.

This is used in [`update!(::OptimizerState, ::AbstractVector)`](@ref).

This sets:
```math
\begin{aligned}
% \bar{x}^\mathtt{cache} & \gets x, \\
x^\mathtt{cache} & \gets x, \\
g^\mathtt{cache} & \gets g, \\
\mathrm{rhs}^\mathtt{cache} & \gets -g, \\
H^\mathtt{cache} & \gets H(x), \\
\delta^\mathtt{cache} & \gets (H^\mathtt{cache})^{-1}\mathrm{rhs}^\mathtt{cache},
\end{aligned}
```
where we wrote ``H`` for the Hessian (i.e. the input argument `hes`).
"""
function update!(cache::NewtonOptimizerCache, state::OptimizerState, g::Gradient, ∇²f::Hessian, x::AbstractVector)
    copyto!(section(cache), section(state))
    copyto!(solution(cache), x)
    g(gradient(cache), x)
    copyto!(rhs(cache), gradient(cache))
    rmul!(rhs(cache), -1)
    ∇²f(hessian(cache), x)
    direction(cache) .= hessian(cache) \ rhs(cache)
    cache
end

function initialize!(cache::NewtonOptimizerCache{T}, ::AbstractVector{T}) where {T}
    cache.x .= T(NaN)
    direction(cache) .= T(NaN)
    cache.g .= T(NaN)
    cache.rhs .= T(NaN)
    hessian(cache) .= T(NaN)
    cache
end
