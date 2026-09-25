using GeometricOptimizers
using SafeTestsets
using Test

# `Pkg.test(test_args = ["metal"])` runs the Metal testset alone and stops. Asked for that way it
# also runs off Apple silicon, where it fails rather than passing with nothing run; `metal.jl`
# says why. A full run takes it last instead, because a failing top-level testset ends the file.
if "metal" in ARGS
    @safetestset "Metal                        " include("metal.jl")
    exit()
end

begin
    @safetestset "Exports                      " include("exports.jl")
end
begin
    @safetestset "Aqua: piracy and compat      " include("aqua_tests.jl")
end
begin
    @safetestset "Own-vs-own ambiguities       " include("ambiguities.jl")
end
begin
    @safetestset "Container/section copies     " include("container_section_copy.jl")
end
begin
    @safetestset "Stiefel Manifold             " include("manifolds/stiefel_manifold.jl")
end
begin
    @safetestset "Grassmann Manifold           " include("manifolds/grassmann_manifold.jl")
end
begin
    @safetestset "Manifold Broadcast           " include("manifolds/broadcast.jl")
end
begin
    @safetestset "Symplectic Stiefel Manifold  " include("manifolds/symplectic_stiefel_manifold.jl")
end
begin
    @safetestset "Symplectic on a backend      " include("manifolds/symplectic_backend.jl")
end
begin
    @safetestset "Symplectic SR Decomposition  " include("decompositions/symplectic_sr.jl")
end
begin
    @safetestset "Backend default eltype       " include("default_eltype.jl")
end
begin
    @safetestset "Backend eltype check         " include("backend_eltype_check.jl")
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
    @safetestset "ProjectTo natural cotangent  " include("special_matrices/project_to.jl")
end
begin
    @safetestset "Triangular Matrices          " include("special_matrices/triangular.jl")
end
begin
    @safetestset "Mutating Return Values       " include("special_matrices/scalar_mul_return_value.jl")
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
    @safetestset "Optimizer phase observer     " include("optimizer_observer.jl")
end
begin
    @safetestset "Composite optimizer method   " include("composite_method.jl")
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
    @safetestset "Manifold Optimizers          " include("manifold_optimizers_with_new_interface.jl")
end
begin
    @safetestset "Grassmann Optimizers         " include("grassmann_optimizer_tests.jl")
end
begin
    @safetestset "Optimizer State Init         " include("optimizer_state_initialization.jl")
end
begin
    @safetestset "Optimizer State Accessors    " include("optimizer_state_accessors.jl")
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
    @safetestset "Flat Parameters              " include("flat_parameters.jl")
end
begin
    @safetestset "NeuralNetworkParameters      " include("neural_network_parameters_protocol.jl")
end
begin
    @safetestset "Container Parameters         " include("network_parameters_optimizer.jl")
end
begin
    @safetestset "Flat Buffer Allocations      " include("flat_buffer_allocations.jl")
end
begin
    @safetestset "changebackend                " include("changebackend.jl")
end
begin
    @safetestset "similar keeps the backend    " include("similar_backend.jl")
end
begin
    @safetestset "rgrad matches the backend    " include("gradient_backend.jl")
end
begin
    @safetestset "copyto! crosses backends     " include("device_copyto.jl")
end
begin
    @safetestset "device orthonormalization    " include("device_orthonormalization.jl")
end
begin
    @safetestset "device multiply              " include("device_multiply.jl")
end
begin
    @safetestset "device products and sums     " include("device_products.jl")
end
begin
    @safetestset "mixed-backend refusal        " include("mixed_backend_refusal.jl")
end
# Last, so that a Metal failure hides no host result.
if Sys.isapple() && Sys.ARCH === :aarch64
    @safetestset "Metal                        " include("metal.jl")
end
