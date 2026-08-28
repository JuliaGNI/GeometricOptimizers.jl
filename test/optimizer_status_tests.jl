using GeometricOptimizers
using GeometricOptimizers: GradientCache, GradientState, OptimizerStatus, solution_scale, l2norm,
    _zero, _rmul!, isconverged
using LinearAlgebra: norm
using Test
import Random

Random.seed!(1234)

# The two guards on `x_converged` (issue A4 in `CHANGELOG.md`). The divergence that motivated
# them -- an iterate at `1e100` taking steps of `‖δ‖ = 345` and reporting convergence -- is no longer
# reachable from a solve, because `linesearch_rejected` and `curvature_is_usable` removed its cause.
# So the state it produced is built here directly, out of the same cache and state a solve would hand
# to `OptimizerStatus`.

@testset "solution_scale is the nominal norm on a manifold and the measured one elsewhere" begin
    for MT in (StiefelManifold, GrassmannManifold)
        for (N, n) in ((6, 3), (5, 1), (4, 4))
            Y = rand(MT, N, n)
            # `YᵀY = I` makes `‖Y‖_F = √n` exactly, so the two agree while the point is on the
            # manifold -- which is what makes this change nothing for a converging solve
            @test solution_scale(Y) == √n
            @test solution_scale(Y) ≈ l2norm(Y)
        end
    end

    # and they part company exactly where the iterate does
    Y = rand(StiefelManifold, 6, 3)
    off = StiefelManifold(1e100 * Y.A)
    @test solution_scale(off) == √3
    @test l2norm(off) > 1e99

    @test solution_scale([3.0, 4.0]) == 5.0
    @test solution_scale(ones(2, 2)) == 2.0

    # a `NamedTuple` combines in quadrature, using the nominal scale for its manifold blocks and the
    # measured one for the rest
    ps = (w=rand(StiefelManifold, 6, 3), b=ones(4))
    @test solution_scale(ps) ≈ √(3 + 4)
    @test solution_scale((w=off, b=ones(4))) ≈ √(3 + 4)
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
@testset "`l2norm` of a set recurses through the leaf's `l2norm`, not through `L2norm`" begin
    B = rand(StiefelLieAlgHorMatrix, 6, 3)
    ps = (w=B, b=[3.0, 4.0])

    # the set is the quadrature sum of the leaves' own norms ...
    @test l2norm(ps) ≈ √(l2norm(B)^2 + 25)
    # ... and `b` alone accounts for 25 of it, so the `w` term is the lift's own norm and nothing else
    @test l2norm((b=[3.0, 4.0],)) ≈ 5.0

    # ... which is *not* what reading the dense interface gives. If this ever stops holding, the leaf
    # has become symmetric enough not to distinguish the two and the test needs a different leaf --
    # it is not a licence to fold through `L2norm`.
    @test l2norm(B) ≉ norm(B)

    # A `VectorStorageMatrix` leaf is the same rule with the same answer by coincidence: `l2norm` of
    # one is over the stored vector either way.
    S = SymmetricMatrix(rand(4, 4))
    @test l2norm((w=S,)) ≈ l2norm(S)

    # And the block sum is a quadrature and not a sum of norms, which is what overestimated every
    # stopping criterion by up to `√k` before 0.6.0.
    @test l2norm((a=[3.0], b=[4.0])) ≈ 5.0
    @test l2norm((a=[3.0], b=[4.0])) < l2norm([3.0]) + l2norm([4.0])
end

"""
    manifold_status(x, δ_norm, f, f̄)

The [`OptimizerStatus`](@ref) a solve would report at the iterate `x` after a step of norm `δ_norm`,
with the objective going from `f̄` to `f`.
"""
function manifold_status(x::StiefelManifold, δ_norm, f, f̄)
    δ = _zero(x)
    δ.B .= 1
    _rmul!(δ, δ_norm / l2norm(δ))

    # `GradientCache` keeps `x` itself rather than a copy, so `solution(cache)` is already the iterate
    cache = GradientCache(x, _zero(x), δ)

    state = GradientState(x)
    state.f̄ = f̄

    OptimizerStatus(state, cache, f; config=Options(Float64))
end

@testset "x_converged does not fire for a step that has left the manifold" begin
    Y = rand(StiefelManifold, 6, 3)
    off = StiefelManifold(1e100 * Y.A)

    # This is the trace recorded in issue A4: `‖δ‖ = 345` at an iterate of magnitude `1e100`. Against
    # `l2norm(x)` the relative change is `3.4e-98`, far under `x_reltol = 2eps`; against
    # `solution_scale` it is `345/√3`.
    #
    # `f` moves here (3.38 → 9.13 is what the trace records) so that `f_converged` is out of the way
    # and this is a test of the denominator alone; the objective is the *other* guard, below.
    diverged = manifold_status(off, 345.0, 9.13, 3.38)

    @test diverged.rxₐ ≈ 345.0
    @test diverged.rxᵣ ≈ 345.0 / √3
    @test 345.0 / l2norm(off) < 1e-97          # what the denominator used to be
    @test !diverged.x_converged
    @test !isconverged(diverged)

    # and the same status on the manifold, with a step that really has gone to zero, still converges
    converged = manifold_status(Y, 1e-20, 1.0, 1.0)
    @test converged.rxᵣ ≈ 1e-20 / √3
    @test converged.x_converged
end

@testset "x_converged does not fire on a step that increased the objective" begin
    Y = rand(StiefelManifold, 6, 3)

    # a vanishing step is the whole of the `x_converged` evidence, so the objective is what decides
    @test manifold_status(Y, 1e-20, 1.0, 2.0).x_converged      # f went down
    @test manifold_status(Y, 1e-20, 1.0, 1.0).x_converged      # f stood still
    @test !manifold_status(Y, 1e-20, 2.0, 1.0).x_converged     # f went up

    # `f_converged` and `g_converged` are not gated on it -- they are statements about `f` and
    # `∇f` themselves rather than about a ratio whose denominator can stop meaning anything
    increased = manifold_status(Y, 1e-20, 2.0, 1.0)
    @test increased.f_increased
    @test !increased.x_converged
end

@testset "f_increased is a comparison and not a comparison of magnitudes" begin
    Y = rand(StiefelManifold, 6, 3)

    # `-5 → -6` is a decrease. Through `abs(f) > abs(f̄)`, which is what this used to be, it read as
    # an increase -- and with `x_converged` gated on the flag that would cost a solve its
    # convergence report on any objective that takes negative values.
    @test !manifold_status(Y, 1e-20, -6.0, -5.0).f_increased
    @test manifold_status(Y, 1e-20, -6.0, -5.0).x_converged
    @test manifold_status(Y, 1e-20, -4.0, -5.0).f_increased
end

@testset "the guards leave a Euclidean solve alone" begin
    # `solution_scale` is `l2norm` for an ordinary array, so the only thing that changes here is the
    # `f_increased` gate -- and a solve that ends on a decrease is unaffected by it.
    F(x) = sum(x .^ 2)

    for method in (Newton(), BFGS(), GradientMethod())
        x = ones(3)
        result = solve!(x, OptimizerState(method, x), Optimizer(x, F; algorithm=method))

        @test isconverged(GeometricOptimizers.status(result))
        @test norm(x) < 1e-7
    end
end
