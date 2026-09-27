# The lift sweep that every retraction measurement in this repository is taken over.
#
# `retraction_accuracy.jl` prints it as tables for the documentation and `retraction_records.jl`
# writes it as CSV for an external experiment. Both have to sweep the *same* lifts from the *same*
# seed, or a row of one is not the row of the other -- and `docs/src/retractions.md` recomputes the
# accuracy tables when the documentation is built, which is a third reader of these numbers.
#
# This file holds definitions only; both scripts `include` it.

using GeometricOptimizers: StiefelLieAlgHorMatrix
import Random

# The scales every accuracy table sweeps over. All of them use the *same* eight, drawn from the same
# seed, so that the three readers above print the same rows rather than rows that resemble each
# other.
const SCALES = (0.1, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0, 120.0)

const SWEEP_SEED = 1234

# `Random.seed!` on the default stream and not a local `Xoshiro`, which is what this was before it
# moved here and what it has to stay. The accuracy figures in
# `src/retractions/exponential_algorithms.jl`, the note on `Cayley` in
# `src/retractions/retraction_types.jl` and the tables `docs/src/retractions.md` recomputes when the
# documentation is built are all *these* lifts; drawing them from a different stream would move every
# published number by an amount that looks like a change in the algorithms.
"""
    sweep(T, N, n; scales = SCALES, seed = SWEEP_SEED)

A sweep of horizontal lifts of increasing norm, all drawn from the same seed.
"""
function sweep(T, N, n; scales = SCALES, seed::Integer = SWEEP_SEED)
    Random.seed!(seed)
    [T(s) * rand(StiefelLieAlgHorMatrix{T}, N, n) for s in scales]
end
