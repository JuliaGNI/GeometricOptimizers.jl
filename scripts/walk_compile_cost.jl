# What `NeuralNetworkParameters`' walks, and the entry points of this package built on them, cost to
# *compile*, as a function of the shape of the parameter set **and of what that shape is held in**.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/walk_compile_cost.jl              # the walks
#     julia --project=. scripts/walk_compile_cost.jl --caches      # what a cache costs to build
#     julia --project=. scripts/walk_compile_cost.jl --gradient    # what the objective-gradient path costs
#
# ## The two axes, and why there are two
#
# A parameter set varies independently in its *shape* and in its *wrapping*, and this script sweeps
# both because a version of it that swept only the first got a diagnosis wrong:
#
#   * FLAT   — one level, `k` entries. The shape `GMLDatasets`' MNIST transformer uses
#              (`scripts/geometric_optimizers/mnist.jl`, `parameter_layout`): 3·7·16 attention
#              projections + 2·16 ResNet parameters + 1 classification weight = 369 entries.
#   * NESTED — a level of layers, each with a handful of weights. What a network normally looks like.
#   * BARE   — a plain `NamedTuple`, which is what this package has always accepted and what
#              `GMLDatasets` hands it.
#   * WRAPPED — inside a `NetworkParameters`, which is what a consumer holds and what
#              `GeometricMachineLearning` is moving to.
#
# **Until 0.2.3 this script compared flat-bare against nested-wrapped and read the difference as a cost
# of nesting.** It is not, and it never was. `NeuralNetworkParameters` D21: `LeafLayout` carried a
# `prototype` field nothing read, whose type parameter put each leaf's *concrete array type* into the
# layout type of every branch above it — 1849 nodes in the type tree of a 369-leaf wrapped layout
# against 742 without. `_layout(::NetworkParameters, ::Int)` is where a caller paid for it, inferring
# through the child walk's whole return type to wrap its result, so **only the wrapped column ever
# showed it**. Take the parameter out and the two columns agree at every shape.
#
# So the reading has inverted, and this is the sentence to keep: **agreeing columns are the pass
# condition.** A gap reappearing between BARE and WRAPPED at the same leaf count is what this sweep
# now exists to catch, and it would be an upstream layout regression rather than anything about depth.
# Do not read two agreeing columns as the fork failing to isolate the rows — that is what the process
# fan-out below is for, and it is checked separately by the fact that the *first* row of each process
# is as expensive as it is.
#
# ## The other cliff, which is a different one
#
# Every walk across the children of one branch — `_flatten_children!`, `_unflatten_children`,
# `_map_zip`, `_promote_eltypes` and six more — used to be an `@inline`d `Base.tail` chain, which kept
# them type stable and allocation free at *run* time and cost one specialisation per child at compile
# time, over argument types each as long as the branch. Inference on that grew as the cube of the
# width, which is what open issue **D9** was. `NeuralNetworkParameters` 0.2.2 writes the bodies out as
# `@generated` flat expansions, from this package's report, and splits `_map_zip` at 32 children.
# `Base.map` is kept as the floor: it never had the cliff, because past 32 fields Base drops to the
# `Any32` loop fallback, and it is what a consumer that wrote its own walk to get around D9 was paying
# instead.
#
# D9 and D21 are separate and neither is the other: D9 was the walk and is fixed in 0.2.2, D21 was the
# layout *type* and is fixed in 0.2.3. This script is kept after both closed, and that is deliberate:
# it is this package's half of the answer to open issue C9 ("most of the harnesses these figures come
# from are not in the repository"), and it is what would catch either cliff returning. The figures the
# 0.6.0 entry quotes are the ones this script prints — every one of them, which an earlier version of
# this file could not say: the `_zero`/`_copy`/`_similar`/`GlobalSection` rows of that table were not
# in the harness at all.
#
# ## How it measures
#
# It reports *first-call* time, which is compilation plus a negligible run, and **every shape gets a
# process of its own** — see the note above `main`. Within one table the rows are cumulative and the
# order is deliberate: they are printed cheapest-first, a row that shares a specialisation with one
# above it reads only what it adds, and `flatten` is last. That is why `map(zero, ·)` comes before the
# primitives written over `mapparameters` and why `parameterlayout` comes before `flatten`.
#
# A row that raises is printed as the exception rather than skipped. Several of them are *supposed* to:
# `map` hands `zero` a whole layer on a nested set, and the elementwise primitives turn a nested plain
# `NamedTuple` away at the door, because `ArrayNamedTuple{T}` is flat by construction and a nested
# bare set is not an `OptimizerSolution`. Those cells are the claim, so they are reported.
#
# ## The third axis is the Julia version, and one row used to swing 50× across it
#
# **Run this on all three supported versions**, and this is the row that made that necessary rather than
# merely thorough. The four folds over a parameter set — `l2norm`, `solution_scale`, `_dot`, `αmax` —
# were `Base.tail` recursions this package wrote for itself, because upstream's `foldparameters` walked
# one tree. At flat 369 the first three cost about 0.65, 0.65 and 1.47 s on 1.11.9 and **26, 26 and
# 57 s** on 1.12.7 and **35, 35 and 71 s** on 1.13.0-rc3, while the *nested* set of the same leaf count
# cost 0.2–1.4 s on every one of them. That was issue #70: the width of one branch, with 1.11 the only
# version that was cheap.
#
# `NeuralNetworkParameters` 0.2.4 added the zipped fold and all four are upstream's now. Measured, flat
# 369, seconds, and the columns agree — which is the point:
#
#   |                | 1.11.9 | 1.12.7 | 1.13.0-rc3 |
#   |----------------|--------|--------|------------|
#   | l2norm         |  0.87  |  1.27  |    0.95    |
#   | solution_scale |  0.50  |  0.48  |    0.52    |
#   | _dot           |  1.71  |  2.04  |    1.59    |
#   | αmax           |  0.03  |  0.03  |    0.03    |
#
# Note what did *not* improve: on 1.11 the generated fold is a shade dearer than the recursion it
# replaced, 0.87 against 0.69 for `l2norm` and 1.71 against 1.47 for `_dot`. That is the trade, and it
# is worth stating rather than rounding away — a third of a second on the version that was already cheap,
# against two orders of magnitude on the two that were not.
#
# Everything else — including `parameterlayout` and `flatten`, which is the pair the two-column sweep
# exists for — agrees across versions to within the spread of a single cold measurement.
#
# One practical note on running it across versions: `Manifest.toml` is not tracked and there are no
# per-version manifests, so `Pkg.instantiate()` under a different Julia will *install what is pinned*
# rather than re-resolve, and a manifest resolved on 1.13 does not load on 1.11 (`PrecompileTools`
# fails with `UndefVarError: StaticData`). Delete the manifest before switching down.
#
# And read the load average before believing a figure. Process-per-cell isolates compilation, not the
# machine: see the note above `cache_construction` for a run where one cell drove the box into swap and
# the next three cells' figures were nonsense that did not reproduce.

using GeometricOptimizers
using GeometricOptimizers: _dot, _zero, _copy, _similar, l2norm, solution_scale, _manifold_αmax
using NeuralNetworkParameters
using NeuralNetworkParameters: mapparameters, parameterlayout

const T = Float32

# The leaves are all the same tiny matrix: this measures the *shape* of the tree, not the size of the
# arrays in it, and a 4×4 keeps the run time next to nothing beside the compilation.
_leaf() = randn(T, 4, 4)

flat_set(k::Integer) = NamedTuple{Tuple(Symbol("p", i) for i in 1:k)}(Tuple(_leaf() for _ in 1:k))

# Distinct keys per block, deliberately: sixteen branches that are sixteen distinct `NamedTuple` types
# is the shape a network has, where sixteen of one type would be compiled once and flatten the sweep.
#
# **This changed in the revision that added the wrapping axis**, and the figures recorded before it were
# taken with `flat_set(nleaves)` repeated, i.e. sixteen branches of one type. It moves less than a cold
# measurement's spread on the two rows that can be compared: `parameterlayout` at 16 × 24 wrapped on
# 0.2.2 reads 14.28 s here against the 13.91 s recorded then, and `OptimizerCache(Adam)` 85.70 s against
# 87.56 s. Named because it is a change to what the numbers are *of*, which is the class of thing this
# whole file is about.
_block(b::Integer, k::Integer) =
    NamedTuple{Tuple(Symbol("L", b, "p", i) for i in 1:k)}(Tuple(_leaf() for _ in 1:k))

nested_set(nblocks::Integer, nleaves::Integer) =
    NamedTuple{Tuple(Symbol("L", b) for b in 1:nblocks)}(
        Tuple(_block(b, nleaves) for b in 1:nblocks))

# `time()` and not `@elapsed`: the point is the very first call, and `@elapsed` in a loop would
# report the second.
#
# **`invokelatest` and not a direct `f(args...)`, and that is the measurement rather than a detail.**
# Compiling `first_call` itself infers through the call in its body, so with a direct call the
# inference the figure is meant to report is spent while `first_call` is being compiled — *before*
# `t = time()` runs — and what gets printed is the leftover. `NeuralNetworkParameters`' copy of this
# harness read 0.00 s for every width and every column on Julia 1.13 until it was written this way.
# `invokelatest` makes the call opaque, so the caller's own compilation has nothing to do first.
#
# Same lesson as the process-per-shape fix below, from the other end: arrange the harness so the cost
# cannot have been paid where the clock is not looking.
function first_call(f, args...)
    t = time()
    Base.invokelatest(f, args...)
    round(time() - t; digits = 2)
end

# A row that raises is a result, not a gap — see the header. The exception's *name* and a word on what
# it means, because the type alone ("MethodError") does not say which of the several intended failures
# it is.
function row(label, note, f, args...)
    s = try
        string(first_call(f, args...), " s")
    catch e
        string(nameof(typeof(e)), isempty(note) ? "" : string(" (", note, ")"))
    end
    println("  ", rpad(label, 21), ": ", s)
end

const _NOT_A_SOLUTION = "a nested bare `NamedTuple` is not an `OptimizerSolution`"

function table(name, ps)
    println(name)
    # `Base.map` reaches the entries of one level, so on a flat set it is the floor to beat and on a
    # nested one it hands `zero` a whole layer and raises -- which is the claim this whole file is
    # about, so it is reported rather than skipped.
    row("map(zero, ·)", "`map` hands `zero` a whole layer", x -> map(zero, x), ps)
    # The four elementwise primitives the 0.6.0 changelog table quotes, which is why they are here:
    # they are what `GeometricMachineLearning` reaches while building a cache, and until this revision
    # no committed harness printed them.
    row("_zero", _NOT_A_SOLUTION, _zero, ps)
    row("_copy", _NOT_A_SOLUTION, _copy, ps)
    row("_similar", _NOT_A_SOLUTION, _similar, ps)
    row("GlobalSection", "", GlobalSection, ps)
    # The walk itself, underneath those four.
    row("mapparameters(zero)", "", x -> mapparameters(zero, x), ps)
    row("mapparameters(copy)", "", x -> mapparameters(copy, x), ps)
    # The four folds over a parameter set, which are `NeuralNetworkParameters`' zipped `foldparameters`
    # and `foldstorage` as of 0.2.4. Until then this package wrote them itself, as `Base.tail`
    # recursions, because upstream's fold walked one tree — and those cost 26 to 71 s to compile at flat
    # 369 on Julia 1.12 and 1.13 against 0.65 to 1.47 s on 1.11.9, which is issue #70 and the reason
    # these rows exist. They are still measured rather than assumed cheap, because that is the whole
    # lesson: the rows that swung 50× across the version axis are these.
    #
    # `αmax` is here because it was the *fourth* such fold and #70's own count said three. It is also
    # the only one on the per-iteration path, through `linesearch_parameters`.
    row("l2norm", "", l2norm, ps)
    row("solution_scale", "", solution_scale, ps)
    row("_dot(·, ·)", "", (a, b) -> _dot(a, b), ps, ps)
    row("αmax", "", (a, b) -> _manifold_αmax(a, b, one(T)), ps, ps)
    # The layout, and then the flattening that builds one. This pair is D21's, and the two columns of
    # the sweep are expected to agree on it.
    row("parameterlayout", "", parameterlayout, ps)
    row("flatten", "", flatten, ps)
    # what it costs to wrap a bare set, which the wrapped tables arrive already wrapped for
    ps isa NetworkParameters || row("NetworkParameters(·)", "", NetworkParameters, ps)
end

const FLAT_ENTRIES = 369
const NESTED_BLOCKS, NESTED_LEAVES = 16, 24     # 384 leaves, ≤24 children at any one level

# What the *package* pays, which is the figure to gate a change to these walks on.
#
# `OptimizerCache`/`OptimizerState` on `Adam` are the entry point that reaches the elementwise
# primitives — `_zero`, `_copy`, `_similar`, `_fill!`, `GlobalSection` — *without* going through
# `flatten`, which is how `GeometricMachineLearning` uses this package. That distinction is load
# bearing and was got wrong: the 0.6.0 changelog attributed a nested `OptimizerCache(Adam)` figure to
# `NeuralNetworkParameters.parameterlayout`, and `AdamCache` never calls it.
#
# `BFGS` follows because its cache *does* flatten — `_flat_scratch` builds a `FlatParameters` and so
# stores a `ParameterLayout` in the cache's own type — and `BFGSState` sizes `Q` with `flatlength`.
# Coming after `Adam` it reads the increment over the walks already compiled, which is the honest figure
# for a solve that has both.
#
# **A `Gradient` row belongs in `--gradient` and not in this table**, for two reasons and one of them is
# a warning. It is the biggest single compilation this package can provoke — `ForwardDiff` over a
# `Dual`-typed `unflatten` — so most of what it times is not this package's, and a process that has done
# it is not a process to measure anything else in. A run of this table with it as the first row, on
# `NeuralNetworkParameters` 0.2.2, produced 956 s for the `OptimizerCache(Adam)` row below it and 523 s
# for a `GradientAutodiff` that measures 6.46 s alone; those figures do not reproduce and the machine had
# gone to swap. Process-per-cell is not enough on its own when a cell can exhaust the machine for the
# next one, which is the third form of the harness lesson in this file. Keep the big one in a mode of its
# own and read the load average.
function cache_construction(name, ps)
    println(name)
    row("OptimizerCache(Adam)", _NOT_A_SOLUTION,
        x -> GeometricOptimizers.OptimizerCache(Adam(T), x), ps)
    row("OptimizerState(Adam)", _NOT_A_SOLUTION,
        x -> GeometricOptimizers.OptimizerState(Adam(T), x), ps)
    row("OptimizerCache(BFGS)", _NOT_A_SOLUTION,
        x -> GeometricOptimizers.OptimizerCache(BFGS(), x), ps)
    row("OptimizerState(BFGS)", _NOT_A_SOLUTION,
        x -> GeometricOptimizers.OptimizerState(BFGS(), x), ps)
end

# ---------------------------------------------------------------------------------------------------
# Every table below runs in a **fresh process**, one cell of the shape × wrapping grid at a time, and
# that is the whole methodology rather than a detail.
#
# This script used to run its shapes in one session, nested first. Compilation is shared across a
# session, so the second table was measuring what the first had already compiled — and the figure that
# came out of it, `mapparameters(zero, ·)` at "0.00 s" on the 369-entry flat set, was the argument for
# retiring this package's own walk in favour of upstream's. Measured cold, the same call costs 1.51 s
# against `Base.map`'s 0.01 s, and building one `OptimizerCache` on that set went from 1.57 s to
# 71.16 s. A 45× regression read as an improvement, for exactly as long as nobody ran the tables in
# the other order.
#
# So there is no ordering to get right any more, and no `--cold-…` flag to remember: a table is cold or
# it is not printed. The grid is four cells rather than two as of this revision, so the sweep costs
# about twice what it did; nothing is dropped for that, because a column that is not run is a column
# whose agreement cannot be checked. `--in-process` is what the child processes are invoked with, and
# is not for interactive use.
# ---------------------------------------------------------------------------------------------------

# The one cell that cannot exist outside the walks table: a nested bare `NamedTuple` is not an
# `OptimizerSolution`, its values being branches rather than arrays, so there is no cache and no
# `Gradient` to build for it. It is swept in the walks table, where it is perfectly well defined and is
# half of what shows the wrapper is not the variable, and the rows that turn it
# away there say so. The container is what makes nesting a solution at all.
const CELLS = (("flat", false), ("flat", true), ("nested", false), ("nested", true))

_bare(name) = name == "flat" ? flat_set(FLAT_ENTRIES) : nested_set(NESTED_BLOCKS, NESTED_LEAVES)

_shape(name, wrapped) = wrapped ? NetworkParameters(_bare(name)) : _bare(name)

function _label(name, wrapped)
    shape = name == "flat" ? string("FLAT, ", FLAT_ENTRIES, " entries") :
            string("NESTED, ", NESTED_BLOCKS, " × ", NESTED_LEAVES, " = ",
                   NESTED_BLOCKS * NESTED_LEAVES, " leaves")
    string(shape, wrapped ? ", in a NetworkParameters" : ", bare NamedTuple")
end

@doc raw"""
The objective-gradient path, alone in a process because nothing survives sharing one with it.

`GradientAutodiff(F, ps)` flattens `ps`, captures the layout, and hands `ForwardDiff` the closure
`_x -> F(unflatten(layout, _x))`. Compiling that means compiling `unflatten` over a `Dual` element
type, which is the widest thing this package asks of a layout, and it is what an `Optimizer` built on a
parameter set pays before its first iteration. It is also where `NeuralNetworkParameters` 0.2.3 pays off
most: at flat 369 the figure is **17.21 s bare and 6.46 s wrapped** on 0.2.2, against **4.81 s and
4.49 s** on 0.2.3. The two 0.2.2 cells differ for the reason the `flatten` row of `table()` does — on
0.2.2 the bare and wrapped shapes split the same total between `parameterlayout` and `flatten`, and this
path pays whichever half is left.

`l2norm` as the objective because it is a real one over a whole parameter set and it is this package's.

Alone in a process, and alone in its mode: see the note above `cache_construction` for what sharing one
with it did to the rows below it.
"""
function gradient_construction(name, ps)
    println(name)
    row("GradientAutodiff", "", x -> GradientAutodiff(l2norm, x), ps)
end

function run_one(mode, name, wrapped)
    ps = _shape(name, wrapped)
    label = _label(name, wrapped)
    mode == "caches"   ? cache_construction(label, ps) :
    mode == "gradient" ? gradient_construction(label, ps) : table(label, ps)
end

function fan_out(mode)
    for (name, wrapped) in CELLS
        mode != "table" && name == "nested" && !wrapped && continue
        run(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project())
             $(@__FILE__) --in-process $mode $name $wrapped`)
    end
end

function main(args)
    if "--in-process" in args
        i = findfirst(==("--in-process"), args)
        return run_one(args[i + 1], args[i + 2], args[i + 3] == "true")
    end
    fan_out("--caches" in args ? "caches" : "--gradient" in args ? "gradient" : "table")
end

main(ARGS)
