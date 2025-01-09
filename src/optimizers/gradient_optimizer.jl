@doc raw"""
    Gradient(η)

Make an instance of a gradient optimizer. 

This is the simplest neural network optimizer. It has no cache and computes the final velocity as:
```math
    \mathrm{velocity} \gets - \eta\nabla_\mathrm{weight}L.
```

# Implementation

The operations are done as memory efficiently as possible.
This means the provided ``\nabla_WL`` is mutated via:
```julia
rmul!(∇L, -method.η)
```
"""
struct Gradient{T<:Real} <: OptimizerMethod{T}
    η::T
    Gradient(η = 1e-2) = new{typeof(η)}(η)
end

function update!(o::Optimizer{<:Gradient}, ::GradientCache, B::AbstractVecOrMat)
    rmul!(B, -o.method.η)
end