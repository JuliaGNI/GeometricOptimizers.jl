module GeometricOptimizers

    using KernelAbstractions
    using Random
    using LinearAlgebra: Adjoint, qr!, norm, I, mul!, rmul!
    import LinearAlgebra
    import ChainRulesCore
    using ChainRulesCore: ProjectTo
    # we use the Vcat function from LazyArrays
    import LazyArrays

    include("utils.jl")

    export Manifold, StiefelManifold, GrassmannManifold
    export rgrad
    include("manifolds/abstract_manifold.jl")
    include("manifolds/stiefel_manifold.jl")
    include("manifolds/grassmann_manifold.jl")

    export SkewSymMatrix, SymmetricMatrix
    include("special_matrices/skew_symmetric.jl")
    include("special_matrices/symmetric.jl")
    include("special_matrices/stiefel_projection.jl")

    export StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix
    include("horizontal_component_for_homogeneous_spaces/abstract_lie_algebra_horizontal.jl")
    include("horizontal_component_for_homogeneous_spaces/stiefel_lie_algebra_horizontal.jl")
    include("horizontal_component_for_homogeneous_spaces/grassmann_lie_algebra_horizontal.jl")
    include("horizontal_component_for_homogeneous_spaces/constructor_for_stiefel_projection.jl")

    export GlobalSection
    include("global_sections/global_sections.jl")
    include("global_sections/omega_functions.jl")

    include("retractions/modified_exponential.jl")
    include("retractions/retraction_types.jl")
    include("retractions/retractions.jl")

    # optimizer methods I
    include("optimizers/optimizer_method.jl")

    # optimizer caches
    include("optimizers/optimizer_caches.jl")
    include("optimizers/bfgs_cache.jl")

    # optimizer
    export Optimizer
    include("optimizers/optimizer.jl")

    # optimizer methods II
    include("optimizers/gradient_optimizer.jl")
    include("optimizers/momentum_optimizer.jl")
    include("optimizers/adam_optimizer.jl")
    include("optimizers/adam_optimizer_with_learning_rate_decay.jl")
    include("optimizers/bfgs_optimizer.jl")

    include("optimizers/init_optimizer_cache.jl")
end
