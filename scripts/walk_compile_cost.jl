# What `NeuralNetworkParameters`' walks cost to *compile*, as a function of the shape of the
# parameter set.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/walk_compile_cost.jl
#
# Every recursion in that package — `_flatten_children!`, `_unflatten_children`, `_map_zip`,
# `_promote_eltypes` — is an `@inline`d `Base.tail` chain, which is what keeps them type stable and
# allocation free at *run* time. The cost is at compile time and it is superlinear in the number of
# children **at one level**: a chain of `k` nested inlined frames is inlined into one body. `Base.map`
# does not have it, because past 32 fields Base drops to the `Any32` loop fallback.
#
# Two shapes with the same number of leaves therefore behave completely differently:
#
#   * FLAT   — one level, `k` entries. The shape `GMLDatasets`' MNIST transformer uses
#              (`scripts/geometric_optimizers/mnist.jl`, `parameter_layout`): 3·7·16 attention
#              projections + 2·16 ResNet parameters + 1 classification weight = 369 entries.
#   * NESTED — a level of layers, each with a handful of weights. What a network normally looks like,
#              and what a `NetworkParameters` normally holds.
#
# Feeds open issue **D9**, and is this package's half of the answer to open issue C9 ("most of the
# harnesses these figures come from are not in the repository"): the numbers D9 quotes are the ones
# this script prints.
#
# It reports *first-call* time, which is compilation plus a negligible run. Each shape is measured in
# the order written, and the order matters — a `flatten` that has already compiled the tail chain for
# a given tuple type leaves much less for `mapparameters` to do. That is why `map` and `mapparameters`
# are measured *before* `flatten` here, and why the cold `mapparameters` figure D9 quotes comes from a
# session that never calls `flatten` at all (`--cold-mapparameters`).

using GeometricOptimizers
using GeometricOptimizers: _mapleaves
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

function table(name, ps)
    println(name)
    println("  map(zero, ·)         : ", first_call(x -> map(zero, x), ps), " s")
    println("  _mapleaves(zero)     : ", first_call(x -> _mapleaves(zero, x), ps), " s")
    println("  _mapleaves(copy)     : ", first_call(x -> _mapleaves(copy, x), ps), " s")
    println("  mapparameters(zero)  : ", first_call(x -> mapparameters(zero, x), ps), " s")
    println("  parameterlayout      : ", first_call(parameterlayout, ps), " s")
    println("  flatten              : ", first_call(flatten, ps), " s")
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

function main(args)
    if "--caches" in args
        cache_construction(string("NESTED container, ", NESTED_BLOCKS, " × ", NESTED_LEAVES, " = ",
                                  NESTED_BLOCKS * NESTED_LEAVES, " leaves"),
                           NetworkParameters(nested_set(NESTED_BLOCKS, NESTED_LEAVES)))
        cache_construction(string("FLAT NamedTuple, ", FLAT_ENTRIES, " entries"),
                           flat_set(FLAT_ENTRIES))
        return
    end

    if "--cold-mapparameters" in args
        # `mapparameters` on a wide-flat set in a session that has compiled nothing else for that
        # shape. This is the figure to quote when asking what widening a primitive from `map` to
        # `mapparameters` costs on its own.
        ps = flat_set(FLAT_ENTRIES)
        println("FLAT, ", FLAT_ENTRIES, " entries, cold (no `flatten` in this session)")
        println("  map(zero, ·)         : ", first_call(x -> map(zero, x), ps), " s")
        println("  _mapleaves(zero)     : ", first_call(x -> _mapleaves(zero, x), ps), " s")
        println("  mapparameters(zero)  : ", first_call(x -> mapparameters(zero, x), ps), " s")
        return
    end

    table(string("NESTED, ", NESTED_BLOCKS, " × ", NESTED_LEAVES, " = ",
                 NESTED_BLOCKS * NESTED_LEAVES, " leaves"),
          nested_set(NESTED_BLOCKS, NESTED_LEAVES))
    table(string("FLAT, ", FLAT_ENTRIES, " entries"), flat_set(FLAT_ENTRIES))
end

main(ARGS)
