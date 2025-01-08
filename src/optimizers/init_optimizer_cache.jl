# @doc raw"""
#     init_optimizer_cache(method, x)
# 
# Initialize the cache corresponding to the weights `x` for a specific method.
# 
# # Implementation
# 
# Wrapper for the functions `setup_adam_cache`, `setup_momentum_cache`, `setup_gradient_cache`, `setup_bfgs_cache`.
# These appear outside of `optimizer_caches.jl` because the `OptimizerMethods` first have to be defined.
# """
init_optimizer_cache(::Gradient, x) = setup_gradient_cache(x)
init_optimizer_cache(::Momentum, x) = setup_momentum_cache(x)
init_optimizer_cache(::Union{Adam, AdamWithDecay}, x) = setup_adam_cache(x)
init_optimizer_cache(::BFGS, x) = setup_bfgs_cache(x)