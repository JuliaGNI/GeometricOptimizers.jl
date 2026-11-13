# Optimizer Methods API

The optimizer methods, with their states and caches: the first-order methods on manifolds, Newton,
and the quasi-Newton methods BFGS and DFP.

```@autodocs
Modules = [GeometricOptimizers]
Pages = ["optimizers/optimizer_methods.jl", "manifold_optimizers/gradient_optimizer.jl",
         "manifold_optimizers/momentum_optimizer.jl", "manifold_optimizers/adam_optimizer.jl",
         "manifold_optimizers/scalar_moment_adam_optimizer.jl",
         "manifold_optimizers/adam_with_euclidean_decay_optimizer.jl",
         "manifold_optimizers/direct_momentum_optimizer.jl",
         "manifold_optimizers/direct_scalar_moment_adam_optimizer.jl",
         "manifold_optimizers/factored_stiefel_adam_optimizer.jl",
         "optimizers/composite_method.jl",
         "newton_optimizer/newton_optimizer_cache.jl", "newton_optimizer/newton_optimizer_state.jl",
         "iterative_hessians/iterative_hessians.jl", "iterative_hessians/quasi_newton_cache.jl",
         "bfgs/bfgs_state.jl", "dfp/dfp_state.jl"]
```
