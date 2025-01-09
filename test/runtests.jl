using GeometricOptimizers
using SafeTestsets
using Test

begin @safetestset "Stiefel Manifold             " include("manifolds/stiefel_manifold.jl") end
begin @safetestset "Grassmann Manifold           " include("manifolds/grassmann_manifold.jl") end
begin @safetestset "Stiefel Projection           " include("special_matrices/stiefel_projetion.jl") end
begin @safetestset "Skew-Symmetric Matrix        " include("special_matrices/skew_symmetric.jl") end
begin @safetestset "Symmetric Matrix             " include("special_matrices/symmetric_matrix.jl") end
begin @safetestset "Grassmann Lie Alg Hor        " include("lie_algebras/grassmann_lie_algebra_horizontal.jl") end
begin @safetestset "Stiefel Lie Alg Hor          " include("lie_algebras/stiefel_lie_algebra_horizontal.jl") end
begin @safetestset "Retractions                  " include("retractions/retractions.jl") end
begin @safetestset "Ω functions                  " include("global_sections/omega_functions.jl") end
begin @safetestset "Global global_sections       " include("global_sections/global_sections.jl") end
begin @safetestset "Optimizer Convergence        " include("optimizer_convergence/svd_optim.jl") end