# The lift sweep of the retraction measurements. `retraction_accuracy.jl` prints it as tables and
# `retraction_records.jl` writes it as CSV; both include this file, so a row of one is the same lift
# as the row of the other. This file holds definitions only.
#
# `docs/src/exponential_algorithms.md` draws the same sweep for its accuracy tables with its own copy
# of these lines: `Random.seed!(1234)` on the default stream and the same eight scales. A change here
# that the page does not repeat makes the page's tables differ from this script's.

using GeometricOptimizers: StiefelLieAlgHorMatrix
import Random

const SCALES = (0.1, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0, 120.0)

const SWEEP_SEED = 1234

"""
    sweep(T, N, n; seed = SWEEP_SEED)

A horizontal lift for each of `SCALES`, in that order, each drawn after the one before it from the
default stream seeded with `seed`.
"""
function sweep(T, N, n; seed::Integer = SWEEP_SEED)
    Random.seed!(seed)
    [T(s) * rand(StiefelLieAlgHorMatrix{T}, N, n) for s in SCALES]
end
