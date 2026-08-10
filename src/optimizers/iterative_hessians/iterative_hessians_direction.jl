function compute_direction!(opt::Optimizer{T,IOM}, state::Union{BFGSState,DFPState}) where {T,IOM<:QuasiNewtonOptimizerMethod}
    direction(opt) .= inverse_hessian(state) * rhs(opt)
end
