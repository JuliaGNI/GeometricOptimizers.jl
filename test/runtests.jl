using GeometricOptimizers
using SafeTestsets
using Test

begin
    @safetestset "Exports                      " include("exports.jl")
end
begin
    @safetestset "Stiefel Manifold             " include("manifolds/stiefel_manifold.jl")
end
begin
    @safetestset "Grassmann Manifold           " include("manifolds/grassmann_manifold.jl")
end
begin
    @safetestset "Stiefel Projection           " include("special_matrices/stiefel_projetion.jl")
end
begin
    @safetestset "Skew-Symmetric Matrix        " include("special_matrices/skew_symmetric.jl")
end
begin
    @safetestset "Symmetric Matrix             " include("special_matrices/symmetric_matrix.jl")
end
begin
    @safetestset "Triangular Matrices          " include("special_matrices/triangular.jl")
end
begin
    @safetestset "Scalar mul! Return Value     " include("special_matrices/scalar_mul_return_value.jl")
end
begin
    @safetestset "Optimizer Primitives         " include("special_matrices/optimizer_primitives.jl")
end
begin
    @safetestset "Grassmann Lie Alg Hor        " include("lie_algebras/grassmann_lie_algebra_horizontal.jl")
end
begin
    @safetestset "Stiefel Lie Alg Hor          " include("lie_algebras/stiefel_lie_algebra_horizontal.jl")
end
begin
    @safetestset "Retractions                  " include("retractions/retractions.jl")
end
begin
    @safetestset "Exponential Accuracy         " include("retractions/exponential_accuracy.jl")
end
begin
    @safetestset "Ω functions                  " include("global_sections/omega_functions.jl")
end
begin
    @safetestset "Global global_sections       " include("global_sections/global_sections.jl")
end
begin
    @safetestset "Optimizer Convergence        " include("optimizer_convergence/svd_optim.jl")
end
begin
    @safetestset "Optimizers                   " include("optimizer_tests.jl")
end
begin
    @safetestset "Optimizer Problems           " include("optimizer_problems.jl")
end
begin
    @safetestset "Optimizer Status             " include("optimizer_status_tests.jl")
end
begin
    @safetestset "Descent Direction            " include("descent_direction_tests.jl")
end
begin
    @safetestset "Quasi-Newton Secant Pair     " include("quasi_newton_secant_tests.jl")
end
begin
    @safetestset "Manifold Line Search         " include("manifold_linesearch_tests.jl")
end
begin
    @safetestset "Manifold Optimizers         " include("manifold_optimizers_with_new_interface.jl")
end
begin
    @safetestset "Grassmann Optimizers         " include("grassmann_optimizer_tests.jl")
end
begin
    @safetestset "Optimizer State Init         " include("optimizer_state_initialization.jl")
end
begin
    @safetestset "Optimizer Step Formulas      " include("optimizer_step_formulas.jl")
end
begin
    @safetestset "Adam + Euclidean decay       " include("adam_with_euclidean_decay.jl")
end
begin
    @safetestset "Adam + decaying step         " include("adam_optimizer_with_decay.jl")
end
begin
    @safetestset "Scalar-moment Adam           " include("scalar_moment_adam.jl")
end
begin
    @safetestset "NamedTuple Parameters        " include("named_tuple_parameters.jl")
end
begin
    @safetestset "NeuralNetworkParameters      " include("neural_network_parameters_protocol.jl")
end
begin
    @safetestset "changebackend                " include("changebackend.jl")
end
