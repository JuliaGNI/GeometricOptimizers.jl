function compute_direction!(direction::AbstractVector{T}, cache::NewtonOptimizerCache{T}) where {T}
    direction .= solve(LU(), hessian(cache), rhs(cache))
end

function compute_direction!(cache::NewtonOptimizerCache{T}) where {T}
    compute_direction!(direction(cache), cache)
end

function compute_direction!(opt::Optimizer{T, Newton}) where {T}
    compute_direction!(cache(opt))
end
compute_direction!(opt::Optimizer{T, Newton}, ::NewtonOptimizerState) where {T} = compute_direction!(opt)
