# What the in-place `𝔄!`, `retraction_differential!` and `apply_section!` save, and what a whole
# `solve!` costs with them: the before-and-after figures of the CHANGELOG entry on issues #77 and
# #73 and on `apply_section!` without a workspace.
#
# Run it against a project in which `GeometricOptimizers` is the tree to be measured, as a driver
# that starts the cold processes itself:
#
#     julia --startup-file=no scripts/in_place_retraction_cost.jl <project> <T> <lift> [runs]
#
# `<T>` is `Float32` or `Float64`, `<lift>` is `Stiefel` or `Grassmann`, and `runs` is 5 by
# default. The driver runs this file `runs` times, each in a fresh process with `--project=<project>`,
# and prints the median of each figure over the runs. A tree from before these functions existed is
# measured on the allocating functions they replace, so one script gives both columns: the `before`
# project develops the tree of the base commit, the `after` project the branch.
#
# ## What one process measures
#
#   * **item 1, issue #77:** `𝔄` of the `20 × 20` argument `X = B̄ᵗB̂` of a lift with `N = 20`,
#     `n = 10`, for `ScaledSquaring`, `NativePade` and `AugmentedPade` -- `𝔄!(ws, X, algorithm)`
#     where it exists, `𝔄(X, algorithm)` where not -- and the whole geodesic retraction
#     `retraction_matrix!(ws, Geodesic(algorithm), B)` of the same lift, which takes it.
#   * **item 2, issue #73:** the `Cayley` differential of a lift with `N = 6`, `n = 3` at
#     `α = 0.5`: `retraction_differential!(D, ws, Cayley(), B, α)` where it exists,
#     `retraction_differential(Cayley(), B, α)` where not.
#   * **item 3:** `apply_section!(Y, λY, Y₂)` into a separate `N × N` destination at `N = 400`,
#     `n = 3`, and one `BFGS` state `update!` one iteration into a solve at `N = 400`, which is
#     where the state's `update_section!` runs.
#   * **item 4:** one whole `solve!` with `BFGS` and `Geodesic()`, Brockett's `tr(XᵀAXD)` on
#     `N = 20`, `n = 3` (`D = I` on the Grassmann manifold, where the objective has to be invariant
#     under `X ↦ XO`), from `Random.seed!(1234)`, because `GlobalSection` draws its completion from
#     the global RNG.
#
# Each figure is `@allocated` inside a function after one warm-up call, and the time per call of the
# fastest of 30 batches of calls after it, with the collector off and BLAS on one thread.

using LinearAlgebra: BLAS, Diagonal, I, qr, tr
using Printf
using Statistics: median

const SCRIPT = @__FILE__

function driver(project, T, lift, runs)
    lines = map(1:runs) do _
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project $SCRIPT worker $T $lift`
        readlines(cmd)
    end
    println(first(lines[1]))
    rows = [split(l, '\t') for l in lines[1][2:end]]
    @printf("%-46s %14s %14s\n", "$T, $lift, median of $runs cold processes", "bytes",
        "time (μs)")
    for (k, row) in enumerate(rows)
        bytes = median([parse(Float64, split(lines[r][k + 1], '\t')[2]) for r in 1:runs])
        time = median([parse(Float64, split(lines[r][k + 1], '\t')[3]) for r in 1:runs])
        @printf("%-46s %14.0f %14.3f\n", row[1], bytes, time)
    end
end

if length(ARGS) >= 3 && ARGS[1] != "worker"
    driver(ARGS[1], ARGS[2], ARGS[3], length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 5)
    exit()
end

using GeometricOptimizers
using GeometricOptimizers: 𝔄, lift_factors!, retraction_matrix!, retraction_workspace,
                           retraction_differential, GlobalSection, update!, solver_step!,
                           increase_iteration_number!, initialize_state!, problem, value,
                           ScaledSquaring, NativePade, AugmentedPade
import Pkg
import Random

BLAS.set_num_threads(1)

const GO = GeometricOptimizers
const T = eval(Symbol(ARGS[2]))
const STIEFEL = ARGS[3] == "Stiefel"
const LT = STIEFEL ? StiefelLieAlgHorMatrix : GrassmannLieAlgHorMatrix
const MT = STIEFEL ? StiefelManifold : GrassmannManifold

measured(f::F, args...) where {F} = (f(args...); @allocated f(args...))

# The fastest of 30 batches, per call, each batch about a millisecond long: a batch that another
# process on the machine interrupted is slower and not faster, so the minimum is what the code costs.
#
# The collector is off while the batches run. When it collects depends on the heap that the rows
# before this one left, and that differs between the two trees: with it on, the `AugmentedPade` row,
# whose work is one `exp` in both trees, differed by up to 6 % between them, in either direction from
# one set of runs to the next. What the collector would cost follows the bytes, the other column.
function per_call(f::F, args...) where {F}
    f(args...)
    batch = clamp(ceil(Int, 1e-3 / @elapsed(f(args...))), 1, 10_000)
    GC.gc()
    GC.enable(false)
    t = minimum(1:30) do _
        t₀ = time_ns()
        for _ in 1:batch
            f(args...)
        end
        (time_ns() - t₀) / batch / 1e3
    end
    GC.enable(true)
    t
end

function row(name, f::F, args...) where {F}
    println(
        name, '\t', measured(f, args...), '\t', per_call(f, args...))
end

function version(name)
    string(Pkg.dependencies()[findfirst(p -> p.name == name,
        Pkg.dependencies())].version)
end
println("# Julia ", VERSION, ", GeometricOptimizers ", pkgdir(GO), ", SimpleSolvers ",
    version("SimpleSolvers"), ", NeuralNetworkParameters ", version("NeuralNetworkParameters"),
    ", BLAS threads ", BLAS.get_num_threads())

# item 1
let B = rand(Random.Xoshiro(1), LT{T}, 20, 10),
    ws = retraction_workspace(rand(MT{T}, 20, 10))

    lift_factors!(ws, B)
    X = ws.B̄ᵗ * ws.B̂
    for algorithm in (ScaledSquaring(), NativePade(), AugmentedPade())
        name = string(nameof(typeof(algorithm)))
        if isdefined(GO, :𝔄!)
            row("item 1: 𝔄!, $name", GO.𝔄!, ws, X, algorithm)
        else
            row("item 1: 𝔄, $name", 𝔄, X, algorithm)
        end
        row("item 1: Geodesic retraction, $name",
            retraction_matrix!, ws, Geodesic(algorithm), B)
    end
end

# item 2
let B = rand(Random.Xoshiro(2), LT{T}, 6, 3), ws = retraction_workspace(rand(MT{T}, 6, 3))
    if isdefined(GO, :retraction_differential!)
        row(
            "item 2: Cayley differential, α = 0.5", GO.retraction_differential!, similar(B),
            ws, Cayley(), B, T(0.5))
    else
        row("item 2: Cayley differential, α = 0.5", retraction_differential, Cayley(), B,
            T(0.5))
    end
end

# item 3
let N = 400, λY = GlobalSection(rand(Random.Xoshiro(3), MT{T}, N, 3)),
    Y₂ = rand(Random.Xoshiro(4), MT{T}, N, N), Y = MT(zeros(T, N, N))

    row("item 3: apply_section!, N = 400", apply_section!, Y, λY, Y₂)
end

let N = 400
    Random.seed!(1234)
    x = rand(Random.Xoshiro(5), MT{T}, N, 3)
    F(Z) = sum(abs2, Z .- T(0.3)) + sum(sin.(Z))
    opt = Optimizer(x, F; algorithm = BFGS(), retraction = Geodesic())
    state = OptimizerState(BFGS(), x)
    initialize_state!(state)
    increase_iteration_number!(state)
    solver_step!(x, state, opt)
    f = value(problem(opt), x)
    row("item 3: BFGS state update!, N = 400", update!, state, opt, x, f)
end

# item 4
const N, n = 20, 3
const Q = Matrix(qr(randn(Random.Xoshiro(11), N, N)).Q)
const A = T.(Q * Diagonal(1:N) * Q')
const D = STIEFEL ? T.(Matrix(Diagonal([3, 2, 1]))) : Matrix{T}(I, n, n)
brockett(X) = tr(X' * A * X * D)

function solve_once()
    x = rand(Random.Xoshiro(4), MT{T}, N, n)
    Random.seed!(1234)
    opt = Optimizer(
        x, brockett; algorithm = BFGS(), retraction = Geodesic(), max_iterations = 100)
    state = OptimizerState(BFGS(), x)
    solve!(x, state, opt)
    x
end

row("item 4: solve!, BFGS, Geodesic, N = 20, n = 3", solve_once)
