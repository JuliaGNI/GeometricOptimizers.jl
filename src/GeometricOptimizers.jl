module GeometricOptimizers

using Base: Callable
using GeometricBase: AbstractProblem, SolverMethod, AbstractSolver
using SimpleSolvers: Options
using SimpleSolvers: AbstractSolverState, Linesearch, LinesearchMethod, LinesearchProblem, LU
import SimpleSolvers: outer!
using SimpleSolvers: x_abstol, x_reltol, f_abstol, f_reltol, f_suctol, f_mindec
import SimpleSolvers: Gradient, GradientAutodiff, GradientFiniteDifferences
using SimpleSolvers: HessianAutodiff, HessianFunction

import SimpleSolvers: Hessian, GradientFunction, HessianAutodiff, alloc_h
export GradientAutodiff, GradientFunction, GradientFiniteDifferences
# `RiemannianGradient` is this package's own `Gradient`, and exported because it is what `Optimizer`
# stores for a parameter set and therefore what `gradient(opt)` returns. A caller supplying its own
# gradient does not need to wrap it -- `Optimizer` does that -- but does need the name to dispatch on.
# See `utils.jl`.
export RiemannianGradient

# `Static` is exported because it is how a fixed learning rate is specified: the optimizer
# methods only produce a direction, see `default_linesearch`.
using SimpleSolvers: Static, Backtracking, Quadratic, BierlaireQuadratic, Bisection, StrongWolfe
export Static, Backtracking, Quadratic, BierlaireQuadratic, Bisection, StrongWolfe
export DecayingStatic
# `AdamOptimizerWithDecay` is a convenience pairing of `Adam` with `DecayingStatic`, not a method
export AdamOptimizerWithDecay

export Options

import SimpleSolvers: update!, direction, linesearch_problem, compute_new_iterate!, cache, l2norm
import SimpleSolvers: change_precision, solve_with_status
using SimpleSolvers: method, LinesearchStatus, LINESEARCH_UNKNOWN
# The ceiling on the step a line search may return, SimpleSolvers 0.12's half of issue A1b. Read by
# `DecayingStatic`, which has to honour a caller's ceiling itself; supplied by `linesearch_parameters`.
using SimpleSolvers: linesearch_αmax
# `solver_step!` decides whether to keep the step a line search returned, so it needs the outcome
# and not only the step length. See `linesearch_rejected`.
using SimpleSolvers: outcome, steplength
using SimpleSolvers: LINESEARCH_FLOOR, LINESEARCH_EXHAUSTED, LINESEARCH_NO_DESCENT
export update!

using Printf

using KernelAbstractions
using Random
using LinearAlgebra: Adjoint, qr, qr!, norm, I, mul!, rmul!, dot, ⋅
using LinearAlgebra: Diagonal, Hermitian, eigen
import LinearAlgebra
import ChainRulesCore
using ChainRulesCore: ProjectTo
# we use the Vcat function from LazyArrays
import LazyArrays

# `NeuralNetworkParameters` owns the parameter container and the walks over it. This package supplies
# the leaf protocol for its own structured matrices (`src/parameter_protocol.jl`) and uses the flat
# form everywhere a quasi-Newton method needs coordinates.
#
# Deliberately *not* re-exported: `flatten` and `unflatten` are that package's names, and a downstream
# package doing `using GeometricMachineLearning, GeometricOptimizers` would meet them twice.
# The list is what this package actually uses, so that it says which walks these optimizers are
# written in terms of. `ext/AbstractNeuralNetworksExt.jl` imports `mapstorage` for itself.
using NeuralNetworkParameters: NetworkParameters, params,
                               parameterlayout, flatlength,
                               flatten, flatten!, unflatten, unflatten!,
                               FlatParameters, ParameterSet,
                               mapparameters, mapparameters!,
                               foldparameters, foldstorage,
                               parameter_eltype,
                               register_parameter_type!
import NeuralNetworkParameters: freeparameters, rebuild, parameter_metadata

# `metric`, `check` and `Ω` join `rgrad` in being public: they are the geometry a caller works in,
# not implementation detail, and a downstream package that defines its own manifold layers on top of
# these types has to be able to extend them. `GeometricMachineLearning` reached all four through
# `GeometricOptimizers.`-qualified names or through copies of its own until 0.5; see
# GeometricMachineLearning#234.
export Manifold, StiefelManifold, GrassmannManifold
export rgrad, metric, check, Ω
include("manifolds/abstract_manifold.jl")
include("manifolds/stiefel_manifold.jl")
include("manifolds/grassmann_manifold.jl")

export SkewSymMatrix, SymmetricMatrix, LowerTriangular, UpperTriangular
export AbstractTriangular, StiefelProjection
include("special_matrices/skew_symmetric.jl")
include("special_matrices/symmetric.jl")
include("special_matrices/stiefel_projection.jl")
include("special_matrices/triangular.jl")
include("special_matrices/lower_triangular.jl")
include("special_matrices/upper_triangular.jl")
include("special_matrices/vector_storage_matrix.jl")

export StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix, AbstractLieAlgHorMatrix
include("lie_algebras/abstract_lie_algebra_horizontal.jl")
include("lie_algebras/stiefel_lie_algebra_horizontal.jl")
include("lie_algebras/grassmann_lie_algebra_horizontal.jl")
include("lie_algebras/stiefel_projection.jl")

# The leaf protocol, once `Manifold`, `VectorStorageMatrix` and `AbstractLieAlgHorMatrix` all exist.
include("parameter_protocol.jl")

# The whole global-section interface is public. Everything a caller needs in order to take one
# optimizer step by hand — build the section, lift a gradient into `𝔤ʰᵒʳ`, transport the section
# along the step, and evaluate it at the distinct element — is here, and a package that walks a
# parameter tree itself (as `GeometricMachineLearning` does for a neural network) needs all of it.
export GlobalSection, global_section, global_rep
export apply_section, apply_section!, update_section!
include("global_sections/global_sections.jl")
include("global_sections/omega_functions.jl")

# The retraction interface, likewise: `Geodesic` and `Cayley` are what a caller passes as
# `retraction = …`, and `geodesic`/`cayley`/`retraction` are what applies one.
export AbstractRetraction, Geodesic, Cayley
export geodesic, cayley, retraction
include("retractions/exponential_algorithms.jl")
include("retractions/modified_exponential.jl")
include("retractions/retraction_types.jl")
include("retractions/retractions.jl")

export Optimizer,
    OptimizerProblem,
    OptimizerMethod,
    OptimizerSolution,
    OptimizerState, isaOptimizerState,
    NewtonOptimizerState,
    HessianAutodiff,
    HessianBFGS,
    HessianDFP

import SimpleSolvers: solve!, solve
# `Newton`, `BFGS` and `DFP` are the optimizer methods, and `BFGSState`/`DFPState` are the states
# that go with them -- as `GradientState`, `MomentumState` and `AdamState` are exported with the
# first-order methods further down, and `NewtonOptimizerState` above. `OptimizerMethod`, their
# common supertype, is exported above so that a caller can dispatch on "any method". `DFPState` is
# an alias for `BFGSState`. Only the *caches* stay internal, for every method alike: they are
# `solver_step!` scratch, and nothing outside a step should be reading one.
export solve!, solve, value, gradient, Newton, BFGS, DFP, BFGSState, DFPState

include("optimizer_solution.jl")
include("optimizers/optimizer_problems.jl")
include("optimizers/optimizer_methods.jl")

include("optimizers/optimizer_state.jl")
include("optimizers/optimizer_cache.jl")
include("optimizers/descent_direction.jl")
include("optimizers/optimizer_status.jl")
include("optimizers/optimizer_result.jl")
include("optimizers/iterative_hessians/iterative_hessians.jl")
include("optimizers/iterative_hessians/bfgs/hessian_bfgs.jl")
include("optimizers/iterative_hessians/dfp/hessian_dfp.jl")
include("optimizers/newton_optimizer/newton_optimizer_cache.jl")
include("optimizers/newton_optimizer/newton_optimizer_state.jl")

include("optimizers/linesearch_problem.jl")
include("optimizers/decaying_static.jl")

include("optimizers/iterative_hessians/bfgs/bfgs_state.jl")
include("optimizers/iterative_hessians/dfp/dfp_state.jl")

include("optimizers/iterative_hessians/bfgs/bfgs_cache.jl")
include("optimizers/iterative_hessians/dfp/dfp_cache.jl")

# OptimizerSolution is defined in here for example. This should probably be moved to a separate file.
include("utils.jl")

include("optimizers/optimizer.jl")
include("optimizers/iterative_hessians/iterative_hessians_direction.jl")
include("optimizers/newton_optimizer/newton_optimizer_direction.jl")

include("optimizers/named_tuple_wrapper.jl")

export GradientMethod, GradientState
export MomentumMethod, MomentumState
export Adam, AdamState, ScalarMomentAdam, ScalarMomentAdamState
# `AdamWithEuclideanDecay` shares `Adam`'s cache and state, so there is no state to export
# alongside it; `AdamW` is exported so that the name errors with an explanation instead of an
# `UndefVarError` (see its docstring)
export AdamWithEuclideanDecay, AdamW

include("manifold_optimizers/gradient_optimizer.jl")
include("manifold_optimizers/momentum_optimizer.jl")
include("manifold_optimizers/adam_optimizer.jl")
include("manifold_optimizers/scalar_moment_adam_optimizer.jl")
include("manifold_optimizers/adam_with_euclidean_decay_optimizer.jl")

# Teach `NeuralNetworkParameters.load` how to rebuild each of these types from a file, which has no
# prototype to rebuild against. Was the extension's `__init__` before the package became a hard
# dependency; the registry lives in that package's main module, not its HDF5 extension, so this costs
# nothing when HDF5 is not loaded.
function __init__()
    register_parameter_type!("StiefelManifold", (S, md) -> StiefelManifold(_dense(S)))
    register_parameter_type!("GrassmannManifold", (S, md) -> GrassmannManifold(_dense(S)))
    register_parameter_type!("SymmetricMatrix", (S, md) -> SymmetricMatrix(_vector(S, md)...))
    register_parameter_type!("SkewSymMatrix", (S, md) -> SkewSymMatrix(_vector(S, md)...))
    register_parameter_type!("LowerTriangular", (S, md) -> LowerTriangular(_vector(S, md)...))
    register_parameter_type!("UpperTriangular", (S, md) -> UpperTriangular(_vector(S, md)...))
    # These two index positionally where the six above go by name, because they can: the older layout
    # covered five types and never a lift, so their `storage` is only ever the `Tuple` this protocol
    # wrote, in the order `parent` returned. A `NamedTuple` from that layout records no key order, so
    # anything that might meet one has to reach for a field name.
    register_parameter_type!("StiefelLieAlgHorMatrix",
        (S, md) -> StiefelLieAlgHorMatrix(S[1], S[2], md.N, md.n))
    register_parameter_type!("GrassmannLieAlgHorMatrix",
        (S, md) -> GrassmannLieAlgHorMatrix(S[1], md.N, md.n))
end

end
