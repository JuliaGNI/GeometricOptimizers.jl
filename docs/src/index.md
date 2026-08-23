```@meta
CurrentModule = GeometricOptimizers
```

# GeometricOptimizers

Documentation for [GeometricOptimizers](https://github.com/JuliaGNI/GeometricOptimizers.jl).

The package optimizes on *homogeneous spaces* — the [Stiefel](@ref "The Stiefel Manifold") and
[Grassmann](@ref "The Grassmann Manifold") manifolds — with the standard neural network optimizers,
by keeping the optimizer cache in a [global tangent space](@ref "Global Tangent Spaces") that does
not depend on the current iterate. [Optimization on Homogeneous Spaces](@ref) is the place to start;
the [Manifolds](@ref "Basic Concepts from General Topology") chapter builds up the geometry it rests
on, and [Optimizer Methods](@ref "Standard Neural Network Optimizers") covers gradient descent,
momentum and Adam themselves.

Neural networks are not here. [`GeometricMachineLearning`](@extref GeometricMachineLearning
:doc:`index`) builds them on top of this package and re-exports most of what the
[API Reference](@ref) lists.
