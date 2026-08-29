# What a quasi-Newton cache's flat buffers *carry* — in its type, and in memory it keeps alive — as
# opposed to what they cost to compile.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/cache_type_cost.jl
#
# `_flat_scratch` (`src/optimizers/iterative_hessians/iterative_hessians.jl`) gives `BFGSCache` and
# `DFPCache` four `NeuralNetworkParameters.FlatParameters` buffers, and `FlatParameters{T,DT,LT}` carries
# its `ParameterLayout` as `LT`. So the layout type is a type parameter of the cache itself, by way of
# `flat::FT`, and whatever the layout type holds the cache's signature holds too.
#
# Until `NeuralNetworkParameters` 0.2.3 that was every leaf's *concrete array type*: `LeafLayout{N,P}`
# had a `prototype` field nothing read, and `P` propagated up through every branch of the layout. The
# field also kept the leaf array itself alive, so a cache retained the parameter set its buffers were
# sized from. `LeafLayout{N}` is the shape alone now, and this script is the difference.
#
# ## It measures `_flat_scratch` and not a whole cache, deliberately
#
# A `BFGSCache` holds seven ``n \\times n`` matrices, where ``n`` is `flatlength(_zero(x))`. The two
# interesting parameter sets here are a *wide* one (369 leaves, where a per-leaf type parameter shows up
# in the type) and a *big* one (one 400 × 400 leaf, where a retained reference shows up in the bytes) —
# and those are ``n = 5904`` and ``n = 160400``, i.e. 2 GB and 1.6 TB of `Q`. An earlier version of this
# file built the caches and appeared to work, because `zeros` is lazily backed and `OptimizerCache` never
# writes to `Q`. That is not a measurement anyone should run twice.
#
# `_flat_scratch(T, _zero(ps))` is the whole of what the cache contributes to either figure, so it is
# what is built. `FT` on the cache is exactly `typeof` of what it returns.
#
# ## The two figures
#
#   * **type nodes** — the size of the type tree, counted rather than printed. `length(string(T))`
#     cannot be used for this: Julia elides long types when printing, so a wide layout prints at about
#     seven characters per child either way and the elision hides exactly the difference being looked
#     for. `NeuralNetworkParameters`' `scripts/leaf_layout_cost.jl` documents the same trap.
#   * **retained bytes** — `Base.summarysize` of the buffers, against the floor of four vectors of
#     `flatlength` elements. The excess is the layouts, and on the *big* set it is the whole question:
#     a layout that holds its leaves adds the leaves.

using GeometricOptimizers
using GeometricOptimizers: _zero, _flat_scratch
using NeuralNetworkParameters
using NeuralNetworkParameters: NetworkParameters, flatlength

const T = Float64

type_nodes(x) = x isa DataType ? 1 + sum(type_nodes(p) for p in x.parameters; init = 0) : 1

# Distinct keys, and 1 × 1 leaves so that the *width* is the variable and `n` stays small.
wide_set(k::Integer) = NamedTuple{Tuple(Symbol("p", i) for i in 1:k)}(
    Tuple(fill(T(i), 1, 1) for i in 1:k))

# Alternating `Matrix` and `Vector`: two distinct leaf *shapes*, so this is the set that would show a
# `LeafLayout` parameter keyed on anything but the shape. It is also the shape upstream's issue #15 used
# to argue that leaf diversity was not the driver — which was right, and beside the point.
mixed_set(k::Integer) = NamedTuple{Tuple(Symbol("p", i) for i in 1:k)}(
    Tuple(isodd(i) ? fill(T(i), 1, 1) : fill(T(i), 2) for i in 1:k))

# Two leaves, 160 400 elements: the retention question, with no width to confuse it.
big_set() = (W = fill(one(T), 400, 400), b = fill(one(T), 400))

function report(label, ps)
    g = _zero(ps)
    flat = _flat_scratch(T, g)
    n = flatlength(g)
    floor_bytes = 4 * n * sizeof(T)
    total = Base.summarysize(flat)
    println(rpad(label, 44),
            " flatlength ", lpad(n, 7),
            " | type nodes ", lpad(type_nodes(typeof(flat)), 6),
            " | bytes ", lpad(total, 9),
            " | over floor ", lpad(total - floor_bytes, 9))
end

println("`_flat_scratch`, i.e. `FT` on BFGSCache/DFPCache. Floor is 4 × flatlength × ", sizeof(T), " B.")
println()
# Every row is wrapped, because `_zero` takes a `NetworkParameters` and a whole set of parameters
# reaches this package no other way. There is no bare/wrapped axis to sweep here: the wrap shares the
# leaf arrays, so `_zero` walks the same leaves and `_flat_scratch` sizes the same `FT` either way.
report("wide 369 × (1 × 1)", NetworkParameters(wide_set(369)))
report("wide 128, alternating Matrix/Vector", NetworkParameters(mixed_set(128)))
report("big: one 400 × 400 leaf and one 400", NetworkParameters(big_set()))
