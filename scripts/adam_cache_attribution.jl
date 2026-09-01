# Where the ~88 s of a cold `OptimizerCache(Adam(T), ps)` goes, for a *nested* container.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/adam_cache_attribution.jl
#
# This exists because the figure was attributed wrongly for a release. The 0.6.0 changelog said the cost
# was `NeuralNetworkParameters.parameterlayout` and "the price of the property that package exists to
# provide". `AdamCache` never calls `parameterlayout`: it reaches `_zero`, `_similar`, `_fill!` and
# `GlobalSection` and stops. `parameterlayout` on this shape is 0.87 s of the 88.59 s, and upstream's
# 0.2.3 — which took that call from 14.28 s to 0.87 s — moved the cache figure not at all.
#
# So the rows below are the composition, cheapest-first and **cumulative in one process**: each is what
# it adds to the ones above it. That is the whole method. The answer it gives is that every piece the
# constructor calls costs under a second and a half once the pieces above it are compiled, and the
# constructor body that calls them costs eighty-five — which is `AdamCache`'s four-argument method
# taking a `GlobalSection`, four `_similar`s, an `_fill!` and a ten-field `new` with four large type
# parameters, all in one inferred body.
#
# **Run it on 1.13 as well, because that column is the proof.** There, every individual piece is
# *dearer* — `_zero` 2.41 s against 1.26, `_copy` 1.51 against 0.57 — and the constructor body is
# **3.54 s against 81.49 s**. So the cost is not the arguments, not their types and not the work: it is
# what 1.11's inference does with that one composition.
#
# That shape — cheap parts, expensive composition, and one Julia version far worse than the others — is
# open issue **D1**'s.
#
# ## It is fixed, and this file is what measured the fix
#
# The paragraph that stood here said no fix was attempted, on the grounds that D1's `@noinline` control
# did nothing there (925 s against 940 s) and what worked was flattening the nesting. Both halves of
# that turned out to be needed:
#
#   * The ten-field `new` moved into `_adam_cache`, reached with every member already computed and
#     passed in, so the frame infers from its own signature. **Alone: 84.37 s against 81.49, i.e.
#     nothing.**
#   * `@noinline` on `_adam_cache` as well. **Together: 2.91 s.**
#
# So the annotation is what stops inference re-crossing the boundary the split created, and D1's
# negative result was about *where* the barrier goes — it put one around the construction, leaving the
# composition intact on the far side. The row to watch below is `AdamCache(x, g, δ, Δg)`.
#
# `OptimizerCache(Adam)` on this shape, before against after, both on the branch that made the change:
# **88.23 → 6.87 s** on 1.11.9, 12.65 → 17.35 on 1.12.7 and 15.14 → 20.65 on 1.13.0-rc3. The last two
# are the cost of the annotation on versions that did not need it, and they are why this file wants
# running on all three rather than on the floor alone.
#
# `MomentumCache` has the same shape and has not been measured. It is not in this file and not in the
# sweep; that is the next thing to point this script at.
#
# `walk_compile_cost.jl --caches` is where the top-line figures come from and it forks a process per
# shape. This file deliberately does *not*: the rows only mean anything in one process, in order, since
# what is being measured is what each step adds.

using GeometricOptimizers
using GeometricOptimizers: _zero, _copy, _similar, _fill!, AdamCache, OptimizerCache, OptimizerState
using NeuralNetworkParameters: NetworkParameters, parameterlayout, flatten

const T = Float32

# 16 × 24 = 384 leaves, and distinct keys per block so the sixteen branches are sixteen distinct
# `NamedTuple` types — which is what a network has, where sixteen of one type would be compiled once.
_leaf() = randn(T, 4, 4)
_block(b, k) = NamedTuple{Tuple(Symbol("L", b, "p", i) for i in 1:k)}(Tuple(_leaf() for _ in 1:k))
nested(o, k) = NamedTuple{Tuple(Symbol("L", b) for b in 1:o)}(Tuple(_block(b, k) for b in 1:o))

# `invokelatest` for the reason `walk_compile_cost.jl` gives at length: a direct call lets the caller's
# own compilation infer through it before the clock starts.
function fc(label, f, args...)
    t = time()
    r = Base.invokelatest(f, args...)
    println("  ", rpad(label, 34), ": ", round(time() - t; digits = 2), " s")
    r
end

const NBLOCKS, NLEAVES = 16, 24

ps = NetworkParameters(nested(NBLOCKS, NLEAVES))
println("NESTED ", NBLOCKS, " × ", NLEAVES, " = ", NBLOCKS * NLEAVES,
        " leaves in a NetworkParameters, cumulative in one process:")

g   = fc("_zero(ps)",                _zero, ps)
      fc("_fill!(g, NaN)",           (a, b) -> _fill!(a, b), g, T(NaN))
      fc("_zero(g)",                 _zero, g)
sim = fc("_similar(g)",              _similar, g)
xc  = fc("_copy(ps)",                _copy, ps)
      fc("GlobalSection(_copy(ps))", x -> GlobalSection(_copy(x)), ps)
      fc("AdamCache(x, g, δ, Δg)",   (a, b, c, d) -> AdamCache(a, b, c, d), xc, g, _zero(g), sim)
      fc("AdamCache(x, g, δ)",       (a, b, c) -> AdamCache(a, b, c), _copy(ps), _zero(ps), _zero(ps))
      fc("OptimizerCache(Adam, ps)", x -> OptimizerCache(Adam(T), x), ps)
      fc("OptimizerState(Adam, ps)", x -> OptimizerState(Adam(T), x), ps)
# The two the old diagnosis blamed, measured last so their figures are what they add to everything
# above — which is the fairest possible reading for that diagnosis, and they are still under 2 s.
      fc("parameterlayout(ps)",      parameterlayout, ps)
      fc("flatten(ps)",              flatten, ps)
