using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, check, increase_iteration_number!, solver_step!
using SimpleSolvers
using SimpleSolvers: l2norm
using Test
import Random

# The headline feature of the unified interface: a *bare* `Manifold` can be handed to
# `Optimizer` as the set of parameters, exactly like a `Vector` or a `NamedTuple`.
# `test/named_tuple_parameters.jl` covers the `NamedTuple` case; this file covers the bare
# manifold, which is the one the old `optimization_step!` interface used to handle
# separately.
#
# The problem is the smallest one that still has a manifold in it: minimize the distance to
# `[0, 0, 1.2]` over `St(3, 1)`, i.e. over the unit sphere in R³. The minimizer is the
# normalized target, `[0, 0, 1]`.

Random.seed!(1234)

const TARGET = [0.0, 0.0, 1.2]
const MINIMIZER = StiefelManifold([0.0; 0.0; 1.0;;])

# `x₀` is deliberately far from the minimizer: it is on the equator relative to it.
initial_point(::Type{T}) where {T} = StiefelManifold(T[0.0; sqrt(0.5); sqrt(0.5);;])

objective(::Type{T}) where {T} = let target = T.(TARGET)
    x::StiefelManifold -> l2norm(vec(x), target)
end

function optimize(::Type{T}, algorithm; α=0.1, retraction=Geodesic()) where {T}
    Random.seed!(1234)
    f = objective(T)
    x = initial_point(T)
    optimizer = Optimizer(x, f; algorithm=algorithm, linesearch=Static(T(α)), retraction=retraction)
    solve!(x, OptimizerState(algorithm, x), optimizer)
    x, f
end

# `check` measures the deviation from the manifold, so this is a round-off tolerance. The
# values observed below are `0` (Geodesic) and `eps(T)` (Cayley).
manifold_tolerance(::Type{T}) where {T} = 10 * eps(T)

@testset "a bare Manifold can be optimized with the unified interface" begin
    for T in (Float64, Float32), retraction in (Geodesic(), Cayley())
        x, f = optimize(T, GradientMethod(); retraction=retraction)

        @test x isa StiefelManifold{T}                          # the type is preserved ...
        @test check(x) < manifold_tolerance(T)                  # ... and so is the manifold
        @test isapprox(x, MINIMIZER; atol=sqrt(eps(T)))         # it found the minimizer
        @test f(x) < f(initial_point(T))                        # and it improved on the start
    end
end

# The old interface required `GradientState(x)` to be built explicitly; `OptimizerState`
# dispatching on the algorithm is what replaces it. Both have to give the same result.
@testset "OptimizerState(GradientMethod(), x) agrees with GradientState(x)" begin
    for T in (Float64, Float32)
        Random.seed!(1234)
        f = objective(T)
        x = initial_point(T)
        optimizer = Optimizer(x, f; algorithm=GradientMethod(), linesearch=Static(T(0.1)))
        solve!(x, GradientState(x), optimizer)

        x′, _ = optimize(T, GradientMethod())
        @test x ≈ x′
    end
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
# manifold peaks at ~14 * eps(T) rather than the <= eps(T) that `GradientMethod` achieves. And
# after the same number of steps `Adam` in particular is still an order of magnitude further
# from the minimizer (worst case observed is ~1.7e-3, for `Float32` with `Cayley`).
stateful_manifold_tolerance(::Type{T}) where {T} = 100 * eps(T)
convergence_tolerance(::Type{T}) where {T} = 10 * sqrt(eps(T))

@testset "the stateful algorithms accept a bare Manifold too" begin
    for T in (Float64, Float32), retraction in (Geodesic(), Cayley())
        for algorithm in (MomentumMethod(T(0.1)), Adam(T(0.01)))
            x, f = optimize(T, algorithm; retraction=retraction)

            @test x isa StiefelManifold{T}                              # type preserved ...
            @test check(x) < stateful_manifold_tolerance(T)             # ... and the manifold
            @test isapprox(x, MINIMIZER; atol=convergence_tolerance(T)) # it found the minimizer
            @test f(x) < f(initial_point(T))                            # and improved on the start
        end
    end
end
