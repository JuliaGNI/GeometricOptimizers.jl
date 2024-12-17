using GeometricOptimizers
using SafeTestsets
using Test

@testset "GeometricOptimizers.jl" begin
    begin @safetestset "Stiefel Manifold             " include("manifolds/stiefel_manifold.jl") end
    begin @safetestset "Grassmann Manifold           " include("manifolds/grassmann_manifold.jl") end
end
