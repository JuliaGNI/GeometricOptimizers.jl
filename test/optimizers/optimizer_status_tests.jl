using GeometricOptimizers
using GeometricOptimizers: GradientCache, GradientState, OptimizerStatus, solution_scale,
                           l2norm,
                           _zero, _rmul!, isconverged
using LinearAlgebra: norm
using Test
import Random

include("../helpers/eltypes.jl")

# for the frame each `GlobalSection` completes, which draws from the global RNG; the data below are
# drawn from their own seeded generators
Random.seed!(1234)

# The two guards on `x_converged` (issue A4 in `CHANGELOG.md`). The divergence that motivated
# them -- an iterate at `1e100` taking steps of `‖δ‖ = 345` and reporting convergence -- is no longer
# reachable from a solve, because `linesearch_rejected` and `curvature_is_usable` removed its cause.
# So the state it produced is built here directly, out of the same cache and state a solve would hand
# to `OptimizerStatus`.

# An iterate far off the manifold, as a multiple of a point, in place of the `1e100` of the trace in
# issue A4, which overflows `Float32`. `1e4/eps(T)` (`8e10` in `Float32`, `5e19` in `Float64`) is far
# enough off that a step of `345` is under `x_reltol = 2eps(T)` relative to it, and near enough that
# the squares the global section of the point forms stay finite in `Float32`; `1e30` overflows there.
off_scale(::Type{T}) where {T} = T(1.0e4) / eps(T)

@testset "solution_scale is the nominal norm on a manifold and the measured one elsewhere, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1234)
    for MT in (StiefelManifold, GrassmannManifold)
        for (N, n) in ((6, 3), (5, 1), (4, 4))
            Y = rand(rng, MT{T}, N, n)
            # `YᵀY = I` makes `‖Y‖_F = √n` exactly, so the two agree while the point is on the
            # manifold -- which is what makes this change nothing for a converging solve
            @test eltype(solution_scale(Y)) == T
            @test solution_scale(Y) == √T(n)
            @test solution_scale(Y) ≈ l2norm(Y)
        end
    end

    # and they part company exactly where the iterate does
    Y = rand(rng, StiefelManifold{T}, 6, 3)
    off = StiefelManifold(off_scale(T) * Y.A)
    @test solution_scale(off) == √T(3)
    @test l2norm(off) > off_scale(T)

    # integer-valued on purpose: `3² + 4² = 5²` and `‖ones(2, 2)‖ = 2` are exact identities
    @test solution_scale(T[3, 4]) == 5
    @test solution_scale(ones(T, 2, 2)) == 2

    # a whole set combines in quadrature, using the nominal scale for its manifold blocks and the
    # measured one for the rest
    ps = NetworkParameters((w = rand(rng, StiefelManifold{T}, 6, 3), b = ones(T, 4)))
    @test eltype(solution_scale(ps)) == T
    @test solution_scale(ps) ≈ √T(3 + 4)
    @test solution_scale(NetworkParameters((w = off, b = ones(T, 4)))) ≈ √T(3 + 4)
end

# `l2norm` of a parameter set is `GeometricBase`'s method as of 0.6.1 — the quadrature fold moved
# there with the rest of issue #16's group. What moved with it is the *shape* of the fold, and this is
# the part of it that a move upstream could silently lose: the fold calls `l2norm` on each leaf, not
# `L2norm`, so a leaf that keeps its numbers behind another interface still decides what it
# contributes.
#
# A lift is the case where the two answers differ. `StiefelLieAlgHorMatrix` presents a dense `N × N`
# skew-symmetric matrix over `N(N-1)/2 - (N-n)(N-n-1)/2` free parameters, so reading the dense
# interface — which the generic `L2norm(::AbstractArray)` does — counts the off-diagonal blocks twice
# and the `A` block's own skew entries twice again. `l2norm(::AbstractLieAlgHorMatrix)` folds over the
# free parameters instead, and that is the number the stopping criteria are entitled to.
@testset "`l2norm` of a set recurses through the leaf's `l2norm`, not through `L2norm`, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(55)
    B = rand(rng, StiefelLieAlgHorMatrix{T}, 6, 3)
    # integer-valued on purpose: `b` contributes exactly `3² + 4² = 25`
    ps = NetworkParameters((w = B, b = T[3, 4]))

    # the set is the quadrature sum of the leaves' own norms ...
    @test eltype(l2norm(ps)) == T
    @test l2norm(ps) ≈ √(l2norm(B)^2 + 25)
    # ... and `b` alone accounts for 25 of it, so the `w` term is the lift's own norm and nothing else
    @test l2norm(NetworkParameters((b = T[3, 4],))) ≈ 5

    # ... which is *not* what reading the dense interface gives. If this ever stops holding, the leaf
    # has become symmetric enough not to distinguish the two and the test needs a different leaf --
    # it is not a licence to fold through `L2norm`.
    @test l2norm(B) ≉ norm(B)

    # A `VectorStorageMatrix` leaf is the same rule with the same answer by coincidence: `l2norm` of
    # one is over the stored vector either way.
    S = SymmetricMatrix(rand(rng, T, 4, 4))
    @test l2norm(NetworkParameters((w = S,))) ≈ l2norm(S)

    # And the block sum is a quadrature and not a sum of norms, which is what overestimated every
    # stopping criterion by up to `√k` before 0.6.0.
    @test l2norm(NetworkParameters((a = T[3], b = T[4]))) ≈ 5
    @test l2norm(NetworkParameters((a = T[3], b = T[4]))) < l2norm(T[3]) + l2norm(T[4])
end

"""
    manifold_status(x, δ_norm, f, f̄)

The [`OptimizerStatus`](@ref) a solve would report at the iterate `x` after a step of norm `δ_norm`,
with the objective going from `f̄` to `f`, all in the element type of `x`.
"""
function manifold_status(x::StiefelManifold{T}, δ_norm, f, f̄) where {T}
    δ = _zero(x)
    δ.B .= 1
    _rmul!(δ, T(δ_norm) / l2norm(δ))

    # `GradientCache` keeps `x` itself rather than a copy, so `solution(cache)` is already the iterate
    cache = GradientCache(x, _zero(x), δ)

    state = GradientState(x)
    state.f̄ = T(f̄)

    OptimizerStatus(state, cache, T(f); config = Options(T))
end

# The objective values in the three testsets below are integer-valued where only their order
# matters: the guards compare `f` with `f̄`, and that comparison is exact in either precision.
@testset "x_converged does not fire for a step that has left the manifold, $T" for T in REAL_ELTYPES
    Y = rand(Random.Xoshiro(100), StiefelManifold{T}, 6, 3)
    off = StiefelManifold(off_scale(T) * Y.A)

    # This is the trace recorded in issue A4: `‖δ‖ = 345` at an iterate of magnitude `1e100`. Against
    # `l2norm(x)` the relative change is `3.4e-98`, far under `x_reltol = 2eps`; against
    # `solution_scale` it is `345/√3`. At `off_scale(T)` it is `345eps(T)/(1e4√3) < 2eps(T)`.
    #
    # `f` moves here (3.38 → 9.13 is what the trace records) so that `f_converged` is out of the way
    # and this is a test of the denominator alone; the objective is the *other* guard, below.
    diverged = manifold_status(off, 345.0, 9.13, 3.38)

    @test eltype(diverged.rxₐ) == T
    @test diverged.rxₐ ≈ 345
    @test diverged.rxᵣ ≈ 345 / √T(3)
    # what the denominator used to be: under the default `x_reltol = 2eps(T)`, so it would fire
    @test T(345) / l2norm(off) < 2eps(T)
    @test !diverged.x_converged
    @test !isconverged(diverged)

    # and the same status on the manifold, with a step that really has gone to zero, still converges
    converged = manifold_status(Y, 1e-20, 1.0, 1.0)
    @test converged.rxᵣ ≈ T(1e-20) / √T(3)
    @test converged.x_converged
end

@testset "x_converged does not fire on a step that increased the objective, $T" for T in REAL_ELTYPES
    Y = rand(Random.Xoshiro(124), StiefelManifold{T}, 6, 3)

    # a vanishing step is the whole of the `x_converged` evidence, so the objective is what decides
    @test manifold_status(Y, 1e-20, 1.0, 2.0).x_converged      # f went down
    @test manifold_status(Y, 1e-20, 1.0, 1.0).x_converged      # f stood still
    @test !manifold_status(Y, 1e-20, 2.0, 1.0).x_converged     # f went up

    # `f_converged` and `g_converged` are not gated on it -- they are statements about `f` and
    # `∇f` themselves rather than about a ratio whose denominator can stop meaning anything
    increased = manifold_status(Y, 1e-20, 2.0, 1.0)
    @test eltype(increased.rxₐ) == T
    @test increased.f_increased
    @test !increased.x_converged
end

@testset "f_increased is a comparison and not a comparison of magnitudes, $T" for T in REAL_ELTYPES
    Y = rand(Random.Xoshiro(139), StiefelManifold{T}, 6, 3)
    @test eltype(manifold_status(Y, 1e-20, -6.0, -5.0).rxₐ) == T

    # `-5 → -6` is a decrease. Through `abs(f) > abs(f̄)`, which is what this used to be, it read as
    # an increase -- and with `x_converged` gated on the flag that would cost a solve its
    # convergence report on any objective that takes negative values.
    @test !manifold_status(Y, 1e-20, -6.0, -5.0).f_increased
    @test manifold_status(Y, 1e-20, -6.0, -5.0).x_converged
    @test manifold_status(Y, 1e-20, -4.0, -5.0).f_increased
end

@testset "the guards leave a Euclidean solve alone, $T" for T in REAL_ELTYPES
    # `solution_scale` is `l2norm` for an ordinary array, so the only thing that changes here is the
    # `f_increased` gate -- and a solve that ends on a decrease is unaffected by it.
    F(x) = sum(x .^ 2)
    rng = Random.Xoshiro(150)

    for method in (Newton(), BFGS(), GradientMethod())
        x = rand(rng, T, 3) .+ T(0.5)
        result = solve!(x, OptimizerState(method, x), Optimizer(x, F; algorithm = method))

        @test isconverged(GeometricOptimizers.status(result))
        @test eltype(x) == T
        # the minimizer is `0`; a minimizer is accurate to the root of the objective's precision.
        # Measured exactly `0` for all three methods over five seeds in both precisions.
        @test norm(x) ≤ √eps(T)
    end
end
