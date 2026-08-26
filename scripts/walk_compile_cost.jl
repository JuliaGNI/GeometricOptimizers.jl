# What `NeuralNetworkParameters`' walks cost to *compile*, as a function of the shape of the
# parameter set.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/walk_compile_cost.jl
#
# Every walk across the children of one branch there — `_flatten_children!`, `_unflatten_children`,
# `_map_zip`, `_promote_eltypes` and six more — used to be an `@inline`d `Base.tail` chain, which kept
# them type stable and allocation free at *run* time and cost one specialisation per child at compile
# time, over argument types each as long as the branch. Inference on that grew as the cube of the
# width, which is what open issue **D9** was. `NeuralNetworkParameters` 0.2.2 writes the bodies out as
# `@generated` flat expansions instead, from this package's report, and the rows below are what that
# left. `Base.map` is kept as the floor: it never had the cliff, because past 32 fields Base drops to
# the `Any32` loop fallback, and it is what a consumer that wrote its own walk to get around D9 was
# paying instead.
#
# Two shapes with the same number of leaves therefore behave completely differently:
#
#   * FLAT   — one level, `k` entries. The shape `GMLDatasets`' MNIST transformer uses
#              (`scripts/geometric_optimizers/mnist.jl`, `parameter_layout`): 3·7·16 attention
#              projections + 2·16 ResNet parameters + 1 classification weight = 369 entries.
#   * NESTED — a level of layers, each with a handful of weights. What a network normally looks like,
#              and what a `NetworkParameters` normally holds.
#
# Kept after D9 closed, and that is deliberate: it is this package's half of the answer to open issue
# C9 ("most of the harnesses these figures come from are not in the repository"), and it is what would
# catch a *return* of the cliff — from a new walk here, or from a change upstream. The figures the
# 0.6.0 entry quotes are the ones this script prints.
#
# It reports *first-call* time, which is compilation plus a negligible run, and every shape gets a
# process of its own — see the note above `main`. Within one table the order still matters, since a
# `flatten` that has already compiled the walk for a branch shape leaves less for `mapparameters` to
# do, so the rows are printed cheapest-first and `flatten` is last.

using GeometricOptimizers
using GeometricOptimizers: _dot, l2norm, solution_scale
using NeuralNetworkParameters
using NeuralNetworkParameters: mapparameters, parameterlayout

const T = Float32

# The leaves are all the same tiny matrix: this measures the *shape* of the tree, not the size of the
# arrays in it, and a 4×4 keeps the run time next to nothing beside the compilation.
_leaf() = randn(T, 4, 4)

flat_set(k::Integer) = NamedTuple{Tuple(Symbol("p", i) for i in 1:k)}(Tuple(_leaf() for _ in 1:k))

nested_set(nblocks::Integer, nleaves::Integer) =
    NamedTuple{Tuple(Symbol("L", b) for b in 1:nblocks)}(
        Tuple(flat_set(nleaves) for _ in 1:nblocks))

# `time()` and not `@elapsed`: the point is the very first call, and `@elapsed` in a loop would
# report the second.
function first_call(f, args...)
    t = time()
    f(args...)
    round(time() - t; digits = 2)
end

# `Base.map` reaches the entries of one level, so on a flat set it is the floor to beat and on a nested
# one it hands `zero` a whole layer and raises -- which is the claim this whole file is about, so it is
# reported rather than skipped. Written out because the script used to call it unguarded on both shapes
# and so died on its first table, before printing anything.
function _map_row(ps)
    try
        string(first_call(x -> map(zero, x), ps), " s")
    catch e
        string(nameof(typeof(e)), " (`map` hands `zero` a whole layer)")
    end
end

function table(name, ps)
    println(name)
    println("  map(zero, ·)         : ", _map_row(ps))
    println("  mapparameters(zero)  : ", first_call(x -> mapparameters(zero, x), ps), " s")
    println("  mapparameters(copy)  : ", first_call(x -> mapparameters(copy, x), ps), " s")
    # The three folds this package writes for itself, because `NeuralNetworkParameters` has no *zipped*
    # fold and `foldparameters` walks one tree. They are `Base.tail` recursions, which is the shape
    # open issue D9 was about — so they are measured here rather than assumed cheap. What keeps them
    # cheap, and made `mapparameters` expensive before 0.2.2, is the `@inline`: these carry none.
    println("  l2norm               : ", first_call(l2norm, ps), " s")
    println("  solution_scale       : ", first_call(solution_scale, ps), " s")
    println("  _dot(·, ·)           : ", first_call((a, b) -> _dot(a, b), ps, ps), " s")
    println("  parameterlayout      : ", first_call(parameterlayout, ps), " s")
    println("  flatten              : ", first_call(flatten, ps), " s")
    # what it costs to wrap a bare set, which the nested table arrives already wrapped for
    ps isa NetworkParameters ||
        println("  NetworkParameters(·) : ", first_call(NetworkParameters, ps), " s")
end

# `flatten` on the flat set is the expensive one and dominates the whole script; the nested set is
# seconds. Both are printed so the two can be compared at the same leaf count.
const FLAT_ENTRIES = 369
const NESTED_BLOCKS, NESTED_LEAVES = 16, 24     # 384 leaves, ≤24 children at any one level

# What the *package* pays, which is the figure to gate a change to these walks on. `OptimizerCache`
# and `OptimizerState` are the entry point that reaches the elementwise primitives — `_zero`, `_copy`,
# `_similar`, `_fill!` — without going through `flatten`, which is how `GeometricMachineLearning` uses
# this package. Every other entry point builds a `Gradient` first and so pays for `flatten` anyway.
# Note which shape is which. The flat set is passed as a bare `NamedTuple`, because that is what it is
# and what this package has always accepted. The nested set has to be a `NetworkParameters`: a nested
# plain `NamedTuple` is not an `OptimizerSolution` and never was, its values being branches rather than
# arrays — the container is what makes nesting a solution at all.
function cache_construction(name, ps)
    println(name)
    println("  OptimizerCache(Adam)  : ",
            first_call(x -> GeometricOptimizers.OptimizerCache(Adam(T), x), ps), " s")
    println("  OptimizerState(Adam)  : ",
            first_call(x -> GeometricOptimizers.OptimizerState(Adam(T), x), ps), " s")
end

# ---------------------------------------------------------------------------------------------------
# Every table below runs in a **fresh process**, one shape at a time, and that is the whole
# methodology rather than a detail.
#
# This script used to run both shapes in one session, nested first. Compilation is shared across a
# session, so the second table was measuring what the first had already compiled — and the figure that
# came out of it, `mapparameters(zero, ·)` at "0.00 s" on the 369-entry flat set, was the argument for
# retiring this package's own walk in favour of upstream's. Measured cold, the same call costs 1.51 s
# against `Base.map`'s 0.01 s, and building one `OptimizerCache` on that set went from 1.57 s to
# 71.16 s. A 45× regression read as an improvement, for exactly as long as nobody ran the tables in
# the other order.
#
# So there is no ordering to get right any more, and no `--cold-…` flag to remember: a table is cold or
# it is not printed. `--in-process` is what the child processes are invoked with, and is not for
# interactive use.
# ---------------------------------------------------------------------------------------------------

const SHAPES = ("nested", "flat")

_shape(name) = name == "flat" ? flat_set(FLAT_ENTRIES) :
               NetworkParameters(nested_set(NESTED_BLOCKS, NESTED_LEAVES))

_label(name) = name == "flat" ? string("FLAT, ", FLAT_ENTRIES, " entries") :
               string("NESTED container, ", NESTED_BLOCKS, " × ", NESTED_LEAVES, " = ",
                      NESTED_BLOCKS * NESTED_LEAVES, " leaves")

function run_one(mode, name)
    ps = _shape(name)
    mode == "caches" ? cache_construction(_label(name), ps) : table(_label(name), ps)
end

function fan_out(mode)
    for name in SHAPES
        run(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project())
             $(@__FILE__) --in-process $mode $name`)
    end
end

function main(args)
    if "--in-process" in args
        i = findfirst(==("--in-process"), args)
        return run_one(args[i + 1], args[i + 2])
    end
    fan_out("--caches" in args ? "caches" : "table")
end

main(ARGS)
