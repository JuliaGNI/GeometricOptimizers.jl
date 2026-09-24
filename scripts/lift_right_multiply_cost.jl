# What `C * B` costs on the host, for a plain matrix `C` and a horizontal lift `B`, in the two forms
# `_rmul` can take: `-transpose(B * transpose(C))`, whose block products take views of a lazy
# `Transpose`, and `-transpose(B * permutedims(C))`, which copies `C` first. A device array's product
# does not serve a view of a `Transpose`, which is why the package takes the second form; this
# script says what that costs where the first one works.
#
# Run in a cold process per variant, with BLAS on one thread:
#
#     julia --startup-file=no --project=<env with GeometricOptimizers and Chairmarks> \
#         scripts/lift_right_multiply_cost.jl lazy
#     julia --startup-file=no --project=<same env> scripts/lift_right_multiply_cost.jl copy

using Chairmarks
using GeometricOptimizers
using LinearAlgebra
using Random

BLAS.set_num_threads(1)

const VARIANT = only(ARGS)
function product(C, B)
    VARIANT == "lazy" ? -transpose(B * transpose(C)) : -transpose(B * permutedims(C))
end

rng = Random.Xoshiro(1)
for (N, n) in ((6, 3), (50, 5), (200, 10), (1000, 20))
    B = rand(rng, StiefelLieAlgHorMatrix{Float64}, N, n)
    C = randn(rng, 4, N)
    product(C, B)
    t = @b product($C, $B)
    println(rpad(VARIANT, 5), " N = ", lpad(N, 4), "  n = ", lpad(n, 2), "  ",
        round(t.time * 1e6; digits = 2), " μs  ", t.bytes, " B")
end
