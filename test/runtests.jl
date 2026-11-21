using ParallelTestRunner
using GeometricOptimizers

const TESTS = [
    "core" => [
        "Exports                      " => "integration/exports.jl",
        "Public surface               " => "integration/public_surface.jl",
        "Container/section copies     " => "optimizers/container_section_copy.jl",
        "Stiefel Manifold             " => "manifolds/stiefel_manifold.jl",
        "Direct Cayley and metrics    " => "manifolds/direct_cayley_and_metrics.jl",
        "Grassmann Manifold           " => "manifolds/grassmann_manifold.jl",
        "Manifold Broadcast           " => "manifolds/broadcast.jl",
        "Symplectic Stiefel Manifold  " => "manifolds/symplectic_stiefel_manifold.jl",
        "Symplectic on a backend      " => "manifolds/symplectic_backend.jl",
        "Biorthogonal Manifold        " => "manifolds/biorthogonal_manifold.jl",
        "Symplectic SR Decomposition  " => "decompositions/symplectic_sr.jl",
        "Backend default eltype       " => "integration/default_eltype.jl",
        "Backend eltype check         " => "integration/backend_eltype_check.jl",
        "Stiefel Projection           " => "special_matrices/stiefel_projection.jl",
        "Skew-Symmetric Matrix        " => "special_matrices/skew_symmetric.jl",
        "Symmetric Matrix             " => "special_matrices/symmetric_matrix.jl",
        "ProjectTo natural cotangent  " => "special_matrices/project_to.jl",
        "Triangular Matrices          " => "special_matrices/triangular.jl",
        "Mutating Return Values       " => "special_matrices/scalar_mul_return_value.jl",
        "Optimizer Primitives         " => "special_matrices/optimizer_primitives.jl",
        "Elementwise primitives       " => "optimizers/named_tuple_wrapper.jl",
        "Error-only methods           " => "integration/error_only_methods.jl",
        "Grassmann Lie Alg Hor        " => "lie_algebras/grassmann_lie_algebra_horizontal.jl",
        "Stiefel Lie Alg Hor          " => "lie_algebras/stiefel_lie_algebra_horizontal.jl",
        "Symplectic Lie Alg Hor       " => "lie_algebras/symplectic_lie_algebra_horizontal.jl",
        "Lift families                " => "lie_algebras/lift_families.jl",
        "Retractions                  " => "retractions/retractions.jl",
        "Symplectic retractions       " => "retractions/symplectic_retractions.jl",
        "Exponential Accuracy         " => "retractions/exponential_accuracy.jl",
        "In-place 𝔄                   " => "retractions/modified_exponential.jl",
        "Retraction workspace         " => "retractions/retraction_workspace.jl",
        "Ω functions                  " => "global_sections/omega_functions.jl",
        "Global global_sections       " => "global_sections/global_sections.jl",
        "Optimizer phase observer     " => "integration/optimizer_observer.jl",
        "Retraction records script    " => "integration/retraction_records.jl",
        "Optimizer Problems           " => "optimizers/optimizer_problems.jl",
        "Optimizer Status             " => "optimizers/optimizer_status_tests.jl",
        "Descent Direction            " => "optimizers/descent_direction_tests.jl",
        "Quasi-Newton Secant Pair     " => "optimizers/quasi_newton_secant_tests.jl",
        "Trial slope in a workspace   " => "optimizers/linesearch_problem.jl",
        "BFGS state update            " => "optimizers/iterative_hessians/bfgs/bfgs_state.jl",
        "Manifold Optimizers          " => "integration/manifold_optimizers_with_new_interface.jl",
        "Optimizer State Init         " => "manifold_optimizers/optimizer_state_initialization.jl",
        "Optimizer State Accessors    " => "integration/optimizer_state_accessors.jl",
        "Optimizer State Semantics    " => "integration/optimizer_state_semantics.jl",
        "Optimizer Step Formulas      " => "manifold_optimizers/optimizer_step_formulas.jl",
        "Adam + Euclidean decay       " => "manifold_optimizers/adam_with_euclidean_decay_optimizer.jl",
        "Adam + decaying step         " => "optimizers/decaying_static.jl",
        "Scalar-moment Adam           " => "manifold_optimizers/scalar_moment_adam_optimizer.jl",
        "Direct Stiefel optimizers    " => "manifold_optimizers/direct_stiefel_optimizers.jl",
        "Training step                " => "integration/training_optimizer.jl",
        "Adam step allocations        " => "integration/adam_step_allocations.jl",
        "Composite optimizer method   " => "optimizers/composite_method.jl",
        "Flat Parameters              " => "integration/flat_parameters.jl",
        "NeuralNetworkParameters      " => "integration/neural_network_parameters_protocol.jl",
        "Container Parameters         " => "integration/network_parameters_optimizer.jl",
        "Flat Buffer Allocations      " => "integration/flat_buffer_allocations.jl"
    ],
    "quality" => [
        "Aqua: piracy and compat      " => "quality/aqua.jl",
        "JET                          " => "quality/jet.jl",
        "ExplicitImports              " => "quality/explicit_imports.jl",
        "Docstring attachment         " => "quality/docstring_attachment.jl"
    ],
    "backends" => [
        "changebackend                " => "integration/changebackend.jl",
        "similar keeps the backend    " => "integration/similar_backend.jl",
        "rgrad and the backend        " => "integration/rgrad_backend.jl",
        "Default gradient             " => "integration/default_gradient.jl",
        "Storage gradient             " => "parameter_protocol.jl",
        "== through the storage       " => "integration/storage_equality.jl",
        "copyto! crosses backends     " => "integration/device_copyto.jl",
        "device orthonormalization    " => "integration/device_orthonormalization.jl",
        "device multiply              " => "integration/device_multiply.jl",
        "device products and sums     " => "integration/device_products.jl",
        "mixed-backend refusal        " => "integration/mixed_backend_refusal.jl",
        "device solve!                " => "integration/device_solve.jl"
    ],
    "optimizers" => [
        "Optimizers                   " => "integration/optimizer_tests.jl",
        "Manifold Line Search         " => "integration/manifold_linesearch_tests.jl",
        "Grassmann Optimizers         " => "integration/grassmann_optimizer_tests.jl",
        "Symplectic Optimizers        " => "integration/symplectic_optimizer_tests.jl",
        "Biorthogonal Optimizers      " => "integration/biorthogonal_optimizer_tests.jl"
    ],
    "verification" => [
        "Optimizer Convergence        " => "verification/svd_optim.jl"
    ],
    "slow" => [
        "Own-vs-own ambiguities       " => "ambiguities.jl"
    ],
    "doctests" => [
        "Doctests                     " => "quality/doctests.jl"
    ],
    "metal" => [
        "Metal                        " => "devices/metal.jl"
    ]
]

const SELECTED = filter(!startswith("--"), ARGS)
const GROUPS = isempty(SELECTED) ?
               (Sys.isapple() && Sys.ARCH === :aarch64 ?
                [
    "core", "quality", "backends", "optimizers", "verification", "slow", "metal"] :
                ["core", "quality", "backends", "optimizers", "verification", "slow"]) :
               SELECTED
const UNKNOWN = setdiff(SELECTED, first.(TESTS))
isempty(UNKNOWN) ||
    error("no test group ", join(UNKNOWN, ", "), " in runtests.jl; its groups are ",
        join(first.(TESTS), ", "))

runtests(GeometricOptimizers, filter(startswith("--"), ARGS);
    testsuite = Dict(label => :(include($(joinpath(@__DIR__, path))))
    for (group, files) in TESTS if group in GROUPS for (label, path) in files),
    serial = [label
              for (group, files) in TESTS if group in ("metal", "cuda")
              for (label, _) in files],
    serial_position = :after,
    retries = 0,
    exeflags = ["--threads=$(Threads.nthreads())"],
    max_default_jobs = max(1, Sys.CPU_THREADS ÷ Threads.nthreads()))
