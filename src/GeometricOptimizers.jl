module GeometricOptimizers

    using KernelAbstractions
    using Random
    using LinearAlgebra: Adjoint, qr!, norm, I
    import LinearAlgebra
    import ChainRulesCore
    using ChainRulesCore: ProjectTo

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
end
