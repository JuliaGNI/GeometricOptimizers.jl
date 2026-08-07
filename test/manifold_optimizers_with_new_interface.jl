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

# Known limitation, and the reason this file only exercises `GradientMethod` above: on a
# *bare* manifold neither of the two stateful algorithms works. This is pre-existing and is
# not touched by this PR — on a `NamedTuple` of parameters all three algorithms work, which
# is what `test/named_tuple_parameters.jl` covers.
#
# Both failures have one cause, tracked in issue #17: the scalar `mul!` methods of the custom
# matrix types end on `mul!(C.B, A.B, α)` (or `C.S`), so they return that inner field instead
# of the destination `C`. The mutation is right, only the return value is wrong, which is
# fatal for `_mul(α, a) = _rmul!(_copy(a), α)`. For a `StiefelManifold` of size `(3, 1)` that
# turns a `(3, 3)` `StiefelLieAlgHorMatrix` into a `(2, 1)` `Matrix`, and from there
# `MomentumMethod` throws `DimensionMismatch` and `Adam` throws `CanonicalIndexError`.
# `GradientMethod` escapes it because it never calls `_mul`.
#
# These are `@test_broken` rather than deleted so that the fix is noticed here: they will
# report as `Unbroken` as soon as #17 is closed, at which point they become ordinary `@test`s.
@testset "the stateful algorithms do not yet accept a bare Manifold" begin
    for T in (Float64, Float32)
        @test_broken (optimize(T, MomentumMethod(T(0.1))); true)
        @test_broken (optimize(T, Adam(T(0.01))); true)
    end
end
