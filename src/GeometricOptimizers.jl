module GeometricOptimizers

    export Manifold, StiefelManifold, GrassmannManifold
    include("manifolds/abstract_manifold.jl")
    include("manifolds/stiefel_manifold.jl")
    include("manifolds/grassmann_manifold.jl")

end
