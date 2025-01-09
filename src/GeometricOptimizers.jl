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

    export SkewSymMatrix, SymmetricMatrix, LowerTriangular, UpperTriangular
    include("special_matrices/skew_symmetric.jl")
    include("special_matrices/symmetric.jl")
    include("special_matrices/stiefel_projection.jl")
    include("special_matrices/triangular.jl")
    include("special_matrices/lower_triangular.jl")
    include("special_matrices/upper_triangular.jl")

    export StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix
    include("lie_algebras/abstract_lie_algebra_horizontal.jl")
    include("lie_algebras/stiefel_lie_algebra_horizontal.jl")
    include("lie_algebras/grassmann_lie_algebra_horizontal.jl")
    include("lie_algebras/stiefel_projection.jl")

    export GlobalSection, global_rep
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
    include("optimizers/adam_optimizer_with_decay.jl")
    include("optimizers/bfgs_optimizer.jl")

    include("optimizers/init_optimizer_cache.jl")
end
