using GeometricOptimizers
using SafeTestsets
using Test

begin @safetestset "Stiefel Manifold             " include("manifolds/stiefel_manifold.jl") end
begin @safetestset "Grassmann Manifold           " include("manifolds/grassmann_manifold.jl") end
begin @safetestset "Retractions                  " include("retractions/retractions.jl") end
