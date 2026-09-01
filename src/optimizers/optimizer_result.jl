
"""
    OptimizerTraceEntry

One iteration's worth of the trace [`solve!`](@ref) records when
[`SimpleSolvers.Options`](@extref)`.store_trace` is set.

# Keys

- `iteration`: the iteration number the entry was taken at,
- `f`: the objective there,
- `rg`: the gradient residual ``\\|\\nabla{}f\\|`` there.

See [`trace`](@ref).
"""
struct OptimizerTraceEntry{YT, XT}
    iteration::Int
    f::YT
    rg::XT
end

# `VT` is deliberately unbounded; see the warning in `optimizer_solution.jl`. This is on the return
# path of every `solve!`, so it is one of the types an inferring caller has to intersect. `OST` keeps
# its bound: `OptimizerStatus{XT,YT}` is a plain two-parameter struct with no union expansion behind
# it, and it is what ties `T` and `YT` to the status the result reports.
"""
    OptimizerResult

Serves as a diagnostic tool for the [`Optimizer`](@ref) and is the return argument of [`solve!`](@ref).

# Keys

- `status::`[`OptimizerStatus`](@ref): current status of the optimization,
- `x`: solution,
- `f`: function value at solution,
- `trace`: one [`OptimizerTraceEntry`](@ref) per iteration if `Options.store_trace` was set, and
  empty otherwise. See [`trace`](@ref).

"""
mutable struct OptimizerResult{T, YT, VT, OST <: OptimizerStatus{T, YT}}
    status::OST

    x::VT    # current solution
    f::YT    # current function

    trace::Vector{OptimizerTraceEntry{YT, T}}
end

function OptimizerResult(status::OptimizerStatus{T, YT}, x::OptimizerSolution{T}, f::YT) where {
        T, YT}
    OptimizerResult(status, x, f, OptimizerTraceEntry{YT, T}[])
end

status(result::OptimizerResult) = result.status

solution(result::OptimizerResult) = result.x
Base.minimum(result::OptimizerResult) = result.f

@doc raw"""
    trace(result)

The per-iteration record [`solve!`](@ref) kept, or an empty vector if it was not asked to keep one.

`Options(store_trace = true)` is what asks. Before this existed the option was accepted and silently
ignored — by this package *and* by SimpleSolvers 0.11, where it is a field of
[`SimpleSolvers.Options`](@extref) that nothing reads — so code that set it got no trace and no
error either.

# Examples

There is one entry per iteration, and the last one agrees with the status the solve reports:

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: trace, iteration_number, status)
julia> f(x) = sum(x .^ 2);

julia> x = [1.0, 2.0];

julia> state = OptimizerState(Newton(), x);

julia> result = solve!(x, state, Optimizer(x, f; algorithm = Newton(), store_trace = true));

julia> length(trace(result)) == iteration_number(state)
true

julia> last(trace(result)).rg == status(result).rg
true

julia> first(trace(result)).iteration
1
```

Without the option there is no trace, and no error either:

```jldoctest; setup = :(using GeometricOptimizers; using GeometricOptimizers: trace; f(x) = sum(x .^ 2))
julia> x = [1.0, 2.0];

julia> result = solve!(x, OptimizerState(Newton(), x), Optimizer(x, f; algorithm = Newton()));

julia> isempty(trace(result))
true
```

# Implementation

The entries are taken from the [`OptimizerStatus`](@ref) that [`solve!`](@ref) already computes on
every iteration, so nothing extra is evaluated and the cost when `store_trace` is unset is one
`Bool` test per iteration.

What it is for: a statistic that has to be *phase-independent*. `Adam` at a fixed learning rate does
not converge to the minimizer, it orbits it at a distance of order ``\alpha``, so the error at any
one iteration is a sample of an arbitrary phase on that orbit and moves with the last bits of the
floating-point arithmetic. Averaging over a stretch of the orbit measures its radius instead, which
is a property of ``\alpha`` and the problem — across Julia 1.10, 1.12 and 1.13 the final-iterate
error on the problem in `test/optimizer_convergence/svd_optim.jl` spans a factor of 3.0 and the mean
over the last five hundred iterations spans 1.06.
"""
trace(result::OptimizerResult) = result.trace
