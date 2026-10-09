using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, check, increase_iteration_number!,
                           solver_step!,
                           isconverged
using SimpleSolvers
using SimpleSolvers: l2norm, Bisection
using Test
import Random
include("../helpers/eltypes.jl")

# The headline feature of the unified interface: a *bare* `Manifold` can be handed to
# `Optimizer` as the set of parameters, exactly like a `Vector` or a `NamedTuple`.
# `test/integration/flat_parameters.jl` covers the `NamedTuple` case; this file covers the bare
# manifold.
#
# The problem is the smallest one that still has a manifold in it: minimize the distance to
# `[0, 0, 1.2]` over `St(3, 1)`, i.e. over the unit sphere in R³. The minimizer is the
# normalized target, `[0, 0, 1]`.

Random.seed!(1234)

const TARGET = [0.0, 0.0, 1.2]
# the normalized target, in closed form: exact in either precision
minimizer(::Type{T}) where {T} = StiefelManifold(T[0; 0; 1;;])

# `x₀` is deliberately far from the minimizer: it is on the equator relative to it.
initial_point(::Type{T}) where {T} = StiefelManifold(T[0.0; sqrt(0.5); sqrt(0.5);;])

objective(::Type{T}) where {T} =
    let target = T.(TARGET)
        x::StiefelManifold -> l2norm(vec(x), target)
    end

function optimize(::Type{T}, algorithm; α = 0.1, retraction = Geodesic(),
        linesearch = Static(T(α))) where {T}
    Random.seed!(1234)
    f = objective(T)
    x = initial_point(T)
    optimizer = Optimizer(
        x, f; algorithm = algorithm, linesearch = linesearch, retraction = retraction)
    result = solve!(x, OptimizerState(algorithm, x), optimizer)
    x, f, result
end

# `check` measures the deviation from the manifold, so this is a round-off tolerance. The
# values observed below are at most `3 * eps(T)`, in both precisions and with both retractions.
manifold_tolerance(::Type{T}) where {T} = 10 * eps(T)

# The distance to the minimizer is an error on the argument, so it is a multiple of `√eps(T)`: a
# minimizer is accurate to the root of the objective's precision. The worst distance observed below
# is `0.12 √eps(T)`, in both precisions and with both retractions.
@testset "a bare Manifold can be optimized with the unified interface, $T" for T in REAL_ELTYPES,
    retraction in (Geodesic(), Cayley())

    x, f, result = optimize(T, GradientMethod(); retraction = retraction)

    @test x isa StiefelManifold{T}                          # the type is preserved ...
    @test eltype(x) == T
    @test check(x) < manifold_tolerance(T)                  # ... and so is the manifold
    @test isconverged(result.status)                        # it converged ...
    @test isapprox(x, minimizer(T); atol = sqrt(eps(T)))    # ... to the minimizer
    @test f(x) < f(initial_point(T))                        # and it improved on the start
end

# The old interface required `GradientState(x)` to be built explicitly; `OptimizerState`
# dispatching on the algorithm is what replaces it. Both have to give the same result.
@testset "OptimizerState(GradientMethod(), x) agrees with GradientState(x), $T" for T in REAL_ELTYPES
    Random.seed!(1234)
    f = objective(T)
    x = initial_point(T)
    optimizer = Optimizer(x, f; algorithm = GradientMethod(), linesearch = Static(T(0.1)))
    result = solve!(x, GradientState(x), optimizer)

    x′, _, _ = optimize(T, GradientMethod())
    @test eltype(x) == T
    @test x ≈ x′
    # and the run converged to the minimizer; the tolerance is the one of the testset above
    @test isconverged(result.status)
    @test isapprox(x, minimizer(T); atol = sqrt(eps(T)))
end

# The two stateful algorithms used to fail outright on a *bare* manifold, which is why this
# file once exercised only `GradientMethod`. The cause was issue #17: the scalar `mul!`
# methods of the custom matrix types ended on `mul!(C.B, A.B, α)` (or `C.S`), so they returned
# that inner field instead of the destination `C`. The mutation was right, only the return
# value was wrong, which is fatal for `_mul(α, a) = _rmul!(_copy(a), α)`. For a
# `StiefelManifold` of size `(3, 1)` that turned a `(3, 3)` `StiefelLieAlgHorMatrix` into a
# `(2, 1)` `Matrix`, and from there `MomentumMethod` threw `DimensionMismatch` and `Adam`
# threw `CanonicalIndexError`. `GradientMethod` escaped it because it never calls `_mul`.
#
# #17 is fixed, so these are ordinary `@test`s now. The contract itself is pinned separately,
# per type, in `test/special_matrices/scalar_mul_return_value.jl`.
#
# The stateful algorithms take looser tolerances than `GradientMethod` above, on both counts.
# They carry state across steps, so they accumulate more round-off: the deviation from the
# manifold peaks at ~17 * eps(T) rather than the <= eps(T) that `GradientMethod` achieves.
stateful_manifold_tolerance(::Type{T}) where {T} = 100 * eps(T)

# `MomentumMethod` converges as far as `GradientMethod` does: the worst distance to the
# minimizer observed below is `0.42 √eps(T)` (Float64) and `0.15 √eps(T)` (Float32).
convergence_tolerance(::Type{T}, ::MomentumMethod) where {T} = 10 * sqrt(eps(T))

# `Adam` reaches the same tolerance, but only with a line search that searches. Its direction is
# `-m₁/(√m₂ + δ)`, of magnitude ≈ 1 per component whatever the gradient is, so with a fixed `α` every
# step it takes is ≈ `α` however close to the minimizer it already is: it circles the minimizer at
# that distance instead of converging, and the `Float64`/`Cayley` case used to run out its 1000
# iterations doing so. Letting the line search pick the step is what fixes that -- the distance to the
# minimizer goes from 1.6e-3 to 1.3e-8 -- and it then meets the same tolerance as the other two.
convergence_tolerance(::Type{T}, ::Adam) where {T} = 10 * sqrt(eps(T))

# The list below selects the stateful algorithms, which is why `BFGS` and `DFP` are absent although
# both accept a bare manifold. `Newton` is absent for a second reason: it is out of scope on a
# manifold, not overlooked. It builds the exact Hessian and there is no Riemannian one, so
# `Optimizer` and `OptimizerState` both reject it there; `test/integration/optimizer_tests.jl` pins that.
@testset "the stateful algorithms accept a bare Manifold too, $T" for T in REAL_ELTYPES,
    retraction in (Geodesic(), Cayley())

    for algorithm in (MomentumMethod(; α = T(0.1)), Adam())
        # `Adam` gets a searching line search; see `convergence_tolerance` above. `Bisection`
        # rather than `Backtracking` because Adam's direction is not required to descend, and a
        # sufficient-decrease search reports that on every step where it does not.
        linesearch = algorithm isa Adam ? Bisection(T) : Static(T(0.1))
        x, f, result = optimize(T, algorithm; retraction = retraction, linesearch = linesearch)

        @test x isa StiefelManifold{T}                              # type preserved ...
        @test eltype(x) == T
        @test check(x) < stateful_manifold_tolerance(T)             # ... and the manifold
        @test isconverged(result.status)                            # converged ...
        @test isapprox(x, minimizer(T); atol = convergence_tolerance(T, algorithm)) # ... to the minimizer
        @test f(x) < f(initial_point(T))                            # and improved on the start
    end
end
