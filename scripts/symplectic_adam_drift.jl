# The runs of `model(SymplecticStiefelManifold, …)` in `scripts/adam_pullback_severity.jl` that end
# with `check(Y) ≥ 1e-8` under `gradient_rep!`: is `check` already that large at the start, and how
# does it compare with the bound `8eps(T)κ²` of the test suite? Each run that ends at `check ≥ 1e-8`
# ends below the bound of its final point: 2 of 50 at `(12, 4)` and 4 of 50 at `(40, 6)`, from
# starts with `κ` up to 4.4e9 and iterates with `κ` up to 1.3e8.
#
#   julia --startup-file=no --project=test scripts/symplectic_adam_drift.jl
using GeometricOptimizers
using GeometricOptimizers: check
using LinearAlgebra: cond
import Random
const T = Float64
for (N, n) in ((12, 4), (40, 6))
    for seed in 1:50
        Random.seed!(1000seed + N)
        Y = rand(SymplecticStiefelManifold{T}, N, n)
        Y₀ = Matrix(rand(SymplecticStiefelManifold{T}, N, n))
        c₀, κ₀ = check(Y), cond(Y.A)
        opt = TrainingOptimizer(Y; algorithm = Adam(), linesearch = T(1e-2))
        κmax = κ₀
        for _ in 1:1000
            optimization_step!(Y, opt, 2 * (Matrix(Y) - Y₀))
            κmax = max(κmax, cond(Y.A))
        end
        c = check(Y)
        c < 1e-8 && continue
        println(
            (N, n), " seed ", seed, ": check start ", round(c₀; sigdigits = 2), ", end ",
            round(c; sigdigits = 2), "; cond start ", round(κ₀; sigdigits = 3), ", max ",
            round(κmax; sigdigits = 3), ", end ", round(cond(Y.A); sigdigits = 3),
            "; 8eps·κ_end² ", round(8eps(T) * cond(Y.A)^2; sigdigits = 2))
    end
end
