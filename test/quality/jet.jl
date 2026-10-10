using GeometricOptimizers
using GeometricOptimizers: _dot, _zero, l2norm, solution_scale, _manifold_αmax,
                           _flat_secant,
                           _flat_mul!, retraction_matrix!, lift_factors!, update_section!,
                           retraction_workspace, GlobalSection, 𝔄, OptimizerCache,
                           inverse_hessian, direction, rhs, increase_iteration_number!,
                           solver_step!, value, problem, cache, config, update!,
                           initialize_state!, OptimizerStatus, unit_matrix, _poisson_tensor,
                           map_to_lo, map_to_up, map_to_S, map_to_Skew, _lmul_into!, _ladd,
                           write_ones_kernel!, write_poisson_blocks_kernel!,
                           assign_ones_for_stiefel_projection_kernel!, lo_mat_mul_kernel!,
                           up_mat_mul_kernel!, symmetric_mat_mul_kernel!,
                           skew_mat_mul_kernel!, addition_kernel!, assign_S_val_kernel!,
                           assign_Skew_val_kernel!
using JLArrays: JLArrays, JLBackend
using NeuralNetworkParameters: NeuralNetworkParameters, NetworkParameters
using KernelAbstractions: KernelAbstractions, CPU
using JET
using Test
import Random

include("../helpers/eltypes.jl")

# The entry points are the functions of `src/` under `@allocated` in
# `test/integration/flat_buffer_allocations.jl` and the functions that launch a kernel. Each has one line per
# element type at which a test in `test/` outside `test/quality/` calls the method that the
# `@allocated` call or the kernel launch reaches, at the argument types of one such call; another
# container type at the same element type has no line. An element type that reaches the method only
# through another function has no line: `𝔄(B̂, B̄, algorithm)` is reached in `Float32` only through
# a `Float32` geodesic, and `_dot` of a `Float32` and a `Float64` set takes another method. The
# launchers are internal, and no test calls them directly; their element types are those at which
# the tests reach them. `dot(γ, Q, γ)`, `outer!` and `l2norm` of a `NetworkParameters` are measured
# too, and are methods of `LinearAlgebra`, `SimpleSolvers` and `GeometricBase`.
const GO = (GeometricOptimizers,)

# A function that hands a value to a fold of `NeuralNetworkParameters` dispatches on it in the fold's
# frame, so its lines keep the reports of that package's frames too: a barrier on the initial value
# of `_dot`'s or `_manifold_αmax`'s fold gives a report there and none in this package.
const GO_FOLD = (GeometricOptimizers, NeuralNetworkParameters)

# A kernel launch is one varargs method of KernelAbstractions, so a launch argument that is not
# inferred dispatches in `KernelAbstractions.__run`, not here. The launcher lines keep the reports of
# KernelAbstractions' and JLArrays' frames, all but the dispatch of `KernelAbstractions.zeros` in its
# own `init_kernel`, which every launcher that allocates with it has and no launcher can remove.
# A barrier on a launch argument of `map_to_lo`, `_ladd` or `_poisson_tensor` gives a report in
# `KernelAbstractions.__run`.
function launch_reports(f, types)
    reports = JET.get_reports(JET.report_opt(f, types;
        target_modules = (GeometricOptimizers, KernelAbstractions, JLArrays)))
    filter(r -> !occursin("init_kernel", sprint(show, r)), reports)
end

Random.seed!(1234)

const N, n = 6, 3

lift(T) = StiefelLieAlgHorMatrix(SkewSymMatrix(rand(T, n, n)), rand(T, N - n, n), N, n)
function container(T)
    NetworkParameters((L1 = (A = lift(T),), L2 = (W = rand(T, 3, 4), b = rand(T, 5))))
end
function wide()
    NetworkParameters(NamedTuple{ntuple(i -> Symbol(:p, i), 369)}(
        ntuple(i -> randn(Float32, 4, 4), 369)))
end

const Lift64 = typeof(lift(Float64))
const GrassmannLift64 = typeof(rand(GrassmannLieAlgHorMatrix{Float64}, N, n))
const GrassmannLift32 = typeof(rand(GrassmannLieAlgHorMatrix{Float32}, N, n))
const Container64 = typeof(container(Float64))
const Wide32 = typeof(wide())
function stiefel_set(T)
    NetworkParameters((L1 = (Y = rand(StiefelManifold{T}, N, n),),
        L2 = (W = rand(T, n, 4), b = rand(T, N))))
end

# the body of `solve!`'s loop, as `test/integration/flat_buffer_allocations.jl` measures it
function _step!(x, state, opt)
    increase_iteration_number!(state)
    solver_step!(x, state, opt)
    f = value(problem(opt), x)
    OptimizerStatus(state, cache(opt), f; config = config(opt))
    update!(state, opt, x, f)

    f
end

function step_types(algorithm)
    x = randn(12)
    opt = Optimizer(x, v -> sum(abs2, v); algorithm = algorithm, max_iterations = 10_000)
    state = OptimizerState(algorithm, x)
    initialize_state!(state)
    (typeof(x), typeof(state), typeof(opt))
end

const QN = let ps = NetworkParameters((
        L1 = (Y = rand(StiefelManifold{Float64}, N, n),), L2 = (
            W = randn(n, 4), b = zeros(N))))
    c = OptimizerCache(BFGS(), ps)
    state = OptimizerState(BFGS(), ps)
    (cache = typeof(c), Q = typeof(inverse_hessian(state)),
        direction = typeof(direction(c)),
        rhs = typeof(rhs(c)), flat = typeof(c.flat))
end

const RW = let Y = rand(StiefelManifold{Float64}, N, n), ws = retraction_workspace(Y)
    (ws = typeof(ws), Λ = typeof(GlobalSection(Y)), B̂ = typeof(ws.B̂),
        B̄ᵗ = typeof(ws.B̄ᵗ'), algorithm = typeof(Geodesic().algorithm))
end

# JET drops the reports of a kernel body when it analyses the launcher, so each `@kernel` also
# gets a line that analyses the generated `cpu_<kernel>` function directly, at the
# `CompilerMetadata` context that the launcher builds. This uses internals of
# KernelAbstractions (`launch_config`, `mkcontext`, `blocks`, `Kernel.f`).
function kernel_body(kernel, ndrange, args...)
    k = kernel(CPU())
    nd, _, iterspace, dynamic = KernelAbstractions.launch_config(k, ndrange, nothing)
    ctx = KernelAbstractions.mkcontext(
        k, first(KernelAbstractions.blocks(iterspace)), nd, iterspace, dynamic)
    k.f, (typeof(ctx), map(typeof, args)...)
end

function kernel_body_reports(kernel, ndrange, args...)
    f, types = kernel_body(kernel, ndrange, args...)
    JET.get_reports(JET.report_opt(f, types;
        target_modules = (JET.AnyFrameModule(GeometricOptimizers),)))
end

# JET gives no report for a dynamic dispatch on the array that a kernel body writes, a statement
# with two line entries. The optimised IR of the body holds it: a `:call` whose callee is not
# a builtin or an intrinsic dispatches at run time, where a static call is an `:invoke`. This counts
# those calls, with Base only, so it runs where JET does not.
function kernel_body_dynamic_calls(kernel, ndrange, args...)
    f, types = kernel_body(kernel, ndrange, args...)
    code = first(only(code_typed(f, types; optimize = true))).code
    count(code) do statement
        Meta.isexpr(statement, :call) || return false
        callee = statement.args[1]
        callee isa GlobalRef && (callee = getfield(callee.mod, callee.name))
        !(callee isa Core.Builtin || callee isa Core.IntrinsicFunction)
    end
end

@testset "JET" begin
    if isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
        # the functions under `@allocated` in test/integration/flat_buffer_allocations.jl
        @test isempty(JET.get_reports(JET.report_opt(_dot, (Lift64, Lift64); target_modules = GO_FOLD)))
        @test isempty(JET.get_reports(JET.report_opt(_dot, (Wide32, Wide32); target_modules = GO_FOLD)))
        @test isempty(JET.get_reports(JET.report_opt(l2norm, (Lift64,); target_modules = GO_FOLD)))
        @test isempty(JET.get_reports(JET.report_opt(l2norm, (GrassmannLift32,); target_modules = GO_FOLD)))
        @test isempty(JET.get_reports(JET.report_opt(solution_scale, (Lift64,); target_modules = GO_FOLD)))
        @test isempty(JET.get_reports(JET.report_opt(solution_scale, (Container64,); target_modules = GO_FOLD)))
        @test isempty(JET.get_reports(JET.report_opt(solution_scale, (Wide32,); target_modules = GO_FOLD)))
        @test isempty(JET.get_reports(JET.report_opt(
            _manifold_αmax, (Container64, Container64, Float32); target_modules = GO_FOLD)))
        @test isempty(JET.get_reports(JET.report_opt(
            _manifold_αmax, (Wide32, Wide32, Float32); target_modules = GO_FOLD)))
        # the manifold arm, `_block_αmax(::Manifold, δ, c)` and `step_αmax`: a set with a Stiefel
        # leaf one level down, as `test/integration/network_parameters_optimizer.jl` calls it. A
        # barrier on the leaf `yᵢ` in the closure of `_manifold_αmax` gives no report here, and needs
        # none: the optimiser splits the call on `isa(_, Manifold)` across the two methods of
        # `_block_αmax`, so the optimised IR holds no dynamic call, the return type is `T`, and the
        # call allocates 0 bytes.
        for T in REAL_ELTYPES
            ps = stiefel_set(T)
            @test isempty(JET.get_reports(JET.report_opt(
                _manifold_αmax, (typeof(ps), typeof(_zero(ps)), T); target_modules = GO_FOLD)))
        end
        @test isempty(JET.get_reports(JET.report_opt(_flat_secant, (QN.cache,); target_modules = GO)))
        @test isempty(JET.get_reports(JET.report_opt(
            _flat_mul!, (QN.direction, QN.Q, QN.rhs, QN.flat); target_modules = GO)))
        @test isempty(JET.get_reports(JET.report_opt(
            retraction_matrix!, (RW.ws, Cayley, Lift64); target_modules = GO)))
        @test isempty(JET.get_reports(JET.report_opt(
            retraction_matrix!, (RW.ws, typeof(Geodesic()), Lift64); target_modules = GO)))
        @test isempty(JET.get_reports(JET.report_opt(
            lift_factors!, (RW.ws, Lift64); target_modules = GO)))
        @test isempty(JET.get_reports(JET.report_opt(
            lift_factors!, (RW.ws, GrassmannLift64); target_modules = GO)))
        @test isempty(JET.get_reports(JET.report_opt(
            update_section!, (RW.Λ, RW.Λ, Lift64, Cayley, RW.ws); target_modules = GO)))
        @test isempty(JET.get_reports(JET.report_opt(
            𝔄, (RW.B̂, RW.B̄ᵗ, RW.algorithm); target_modules = GO)))
        for algorithm in (BFGS(), DFP(), GradientMethod())
            @test isempty(JET.get_reports(JET.report_opt(
                _step!, step_types(algorithm); target_modules = GO)))
        end

        # the functions that launch a kernel; the backend arm of `unit_matrix`,
        # `_poisson_tensor` and `StiefelProjection` is analysed on a JLArray, because a `CPU`
        # dispatches to a host arm that launches nothing (one test reaches the `StiefelProjection`
        # backend arm on a `CPU` through `invoke`, at the same element types)
        for T in REAL_ELTYPES
            @test isempty(launch_reports(_poisson_tensor, (JLBackend, Type{T}, Int)))
            @test isempty(launch_reports(unit_matrix, (JLBackend, Type{T}, Int)))
            @test isempty(launch_reports(StiefelProjection, (JLBackend, Type{T}, Int, Int)))
        end
        for T in (Float32, Float64, ComplexF64, Int)
            @test isempty(launch_reports(map_to_lo, (Matrix{T},)))
            @test isempty(launch_reports(map_to_up, (Matrix{T},)))
        end
        for T in (Float32, Float64, ComplexF64, BigFloat)
            @test isempty(launch_reports(map_to_S, (Matrix{T},)))
            @test isempty(launch_reports(map_to_Skew, (Matrix{T},)))
        end
        for T in REAL_ELTYPES,
            MT in (StrictlyLowerTriangular, StrictlyUpperTriangular)

            @test isempty(launch_reports(
                _lmul_into!, (Matrix{T}, MT{T, Vector{T}}, Matrix{T})))
        end
        for T in (Float32, Float64, ComplexF64), MT in (SymmetricMatrix, SkewSymMatrix)

            @test isempty(launch_reports(
                _lmul_into!, (Matrix{T}, MT{T, Vector{T}}, Matrix{T})))
        end
        for T in (Float32, Float64, ComplexF64)
            @test isempty(launch_reports(_ladd, (SkewSymMatrix{T, Vector{T}}, Matrix{T})))
        end

        # the kernel bodies, at the element types the launchers above are reached at
        @test isempty(kernel_body_reports(write_poisson_blocks_kernel!, 2, zeros(Float32, 4, 4), 2))
        for T in REAL_ELTYPES
            @test isempty(kernel_body_reports(write_ones_kernel!, 3, zeros(T, 3, 3)))
            @test isempty(kernel_body_reports(
                assign_ones_for_stiefel_projection_kernel!, 3, zeros(T, 6, 3)))
            @test isempty(kernel_body_reports(
                lo_mat_mul_kernel!, (4, 2), zeros(T, 4, 2), rand(T, 6), rand(T, 4, 2), 4))
            @test isempty(kernel_body_reports(
                up_mat_mul_kernel!, (4, 2), zeros(T, 4, 2), rand(T, 6), rand(T, 4, 2), 4))
        end
        for T in (Float32, Float64, ComplexF64)
            @test isempty(kernel_body_reports(
                symmetric_mat_mul_kernel!, (4, 2), zeros(T, 4, 2), rand(T, 10), rand(T, 4, 2), 4))
            @test isempty(kernel_body_reports(
                skew_mat_mul_kernel!, (4, 2), zeros(T, 4, 2), rand(T, 6), rand(T, 4, 2), 4))
            @test isempty(kernel_body_reports(
                addition_kernel!, (4, 4), zeros(T, 4, 4), rand(T, 6), rand(T, 4, 4)))
        end
        for T in (Float32, Float64, ComplexF64, BigFloat)
            @test isempty(kernel_body_reports(
                assign_S_val_kernel!, 3, zeros(T, 6), ones(T, 3, 3), 3))
        end
        for T in (Float32, Float64, ComplexF64, Int, BigFloat)
            @test isempty(kernel_body_reports(
                assign_Skew_val_kernel!, 2, zeros(T, 3), ones(T, 3, 3), 3))
        end
    else
        @test_skip "JET does not work on Julia $(VERSION)"  # aviatesk/JET.jl#681
    end
end

# the two kernel bodies that write a diagonal, at the arguments of their `kernel_body_reports` lines
@testset "a kernel body that writes a diagonal holds no dynamic call, $T" for T in REAL_ELTYPES
    @test kernel_body_dynamic_calls(write_ones_kernel!, 3, zeros(T, 3, 3)) == 0
    @test kernel_body_dynamic_calls(
        assign_ones_for_stiefel_projection_kernel!, 3, zeros(T, 6, 3)) == 0
end
