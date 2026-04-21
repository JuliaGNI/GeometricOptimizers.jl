"""
    OptimizerCache

See e.g. [`NewtonOptimizerCache`](@ref) and [`BFGSCache`](@ref).

# Extended help

!!! todo
    `OptimizerCache`s are only used during [`solver_step!`](@ref)s. Outside of these, [`OptimizerState`](@ref)s are used to communicate information between different iterations. This may still have to be enforced consistently.
"""
abstract type OptimizerCache{T} end
