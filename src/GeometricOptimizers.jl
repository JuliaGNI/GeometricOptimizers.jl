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
# `solver_step!` decides whether to keep the step a line search returned, so it needs the outcome
# and not only the step length. See `linesearch_rejected`.
using SimpleSolvers: outcome, steplength
using SimpleSolvers: LINESEARCH_FLOOR, LINESEARCH_EXHAUSTED, LINESEARCH_NO_DESCENT
export update!

using Printf

using KernelAbstractions
using Random
using LinearAlgebra: Adjoint, qr!, norm, I, mul!, rmul!, dot, ⋅
import LinearAlgebra
import ChainRulesCore
using ChainRulesCore: ProjectTo
# we use the Vcat function from LazyArrays
import LazyArrays

import ParameterHandling, ForwardDiff

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

export Optimizer,
    OptimizerProblem,
    OptimizerState, isaOptimizerState,
    NewtonOptimizerState,
    NewtonOptimizer,
    BFGSOptimizer,
    DFPOptimizer,
    HessianAutodiff,
    HessianBFGS,
    HessianDFP

import SimpleSolvers: solve!, solve
export solve!, solve, value, gradient, Newton

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
export Adam, AdamState
# `AdamWithEuclideanDecay` shares `Adam`'s cache and state, so there is no state to export
# alongside it; `AdamW` is exported so that the name errors with an explanation instead of an
# `UndefVarError` (see its docstring)
export AdamWithEuclideanDecay, AdamW

include("manifold_optimizers/gradient_optimizer.jl")
include("manifold_optimizers/momentum_optimizer.jl")
include("manifold_optimizers/adam_optimizer.jl")
include("manifold_optimizers/adam_with_euclidean_decay_optimizer.jl")

end
