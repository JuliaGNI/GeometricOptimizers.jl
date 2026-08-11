using GeometricOptimizers
using GeometricOptimizers: StiefelManifold, Cayley
using SimpleSolvers: Static, Backtracking, Bisection
using LinearAlgebra: norm, svd
using Test
import Random
Random.seed!(1234)

A = [0.06476993260924702 0.8369280855305259 0.6245358125914054 0.14072996706492302 0.3057604800441981
    0.46705795621669255 0.1112669220975867 0.4533808015358275 0.8080656034678635 0.8124722742350421;
    0.01612280707759217 0.9297364035931851 0.7748255582653033 0.18802235970624825 0.12372987461277729
    0.22056522937785483 0.2625802924956516 0.5453166264594547 0.8739293169067052 0.5623102360222683;
    0.5042709334407875 0.06723749138196022 0.9908385109048417 0.05887559832596112 0.25247620847898444
    0.26892857978090356 0.5444452518976932 0.824067874444862 0.5244383648620328 0.8596290263582982;
    0.27978796217129454 0.9577060969302862 0.639411687437416 0.6400807524147251 0.18148287150115605
    0.44375695670126103 0.3394219347742109 0.257797929549505 0.06817845936505285 0.7313859112765397;
    0.1205707103074688 0.5144924819072745 0.6995653244358568 0.7469274518396951 0.906945142161729
    0.6135243682804966 0.2873276988805561 0.7860348526516666 0.09734138426142758 0.18153213481809904;
    0.8309155499557564 0.39176753440885514 0.7125688492955281 0.6807076690603506 0.6883969854851912
    0.9551643361073993 0.5765921525201096 0.42316798328469785 0.3754036913035341 0.005086362541100731;
    0.5653842309616912 0.8824651137516092 0.586352560797524 0.8956939084804407 0.5239338220997005
    0.8944613182477159 0.4579034900412514 0.40043924031701794 0.8885718621802194 0.6942956266225304;
    0.19906872851379365 0.9054498581893393 0.9535181480911928 0.21500647871920842 0.9609481532739398
    0.5947748073096188 0.0575840223853924 0.6428951849762703 0.25586663838519186 0.13496661903454077;
    0.8828274552770472 0.7341413065751325 0.5943689939491729 0.4945456969253963 0.00504805864120339
    0.3491627076018672 0.7865142963866997 0.7478808694611998 0.8391898474716712 0.5102359749518908;
    0.838935723223811 0.5888502932130046 0.789979979782286 0.7108295494351453 0.21710960094241705
    0.7317681833003449 0.9051355184962627 0.3376918522349117 0.436545092402125 0.3462196925686055]

error(ps::NamedTuple) = norm(A - ps.w₁ * ps.w₂' * A)

# Both iterates stay on the Stiefel manifold, so this is a round-off tolerance and nothing
# else; the values actually observed are of the order of `1e-14`.
const MANIFOLD_TOLERANCE = 1e-12

# How close each algorithm gets to the best rank-`n` approximation, as a relative error, after
# `1000` iterations with `Static(0.01)` and seed `1234`. Measured on Julia 1.13:
#
#                  Geodesic   Cayley
#     GradientMethod  1.0e-2   9.8e-3
#     MomentumMethod  9.7e-3   9.3e-3
#     Adam            3.1e-5   2.3e-5
#
# The tolerances below leave roughly a factor of two on top of those (a factor of three for
# `Adam`).
#
# `MomentumMethod` used to land at `1.9e-2` / `1.7e-2` here, i.e. *worse* than plain gradient
# descent, which is what issue #18 was about: it accumulated `p ← p + α∇L` and thereby kept
# pushing after `∇L → 0`. With the classic `p ← αp + ∇L` it is slightly better than gradient
# descent, as momentum should be.
#
# This test used to apply a single blanket tolerance of `1e-1` to all three algorithms, which
# is how the two `Adam` bugs (uninitialised moments and the wrong bias-correction factors)
# survived: with them present `Adam` reached only `2.2e-4` (Geodesic) and `3.5e-3` (Cayley) at
# these 1000 steps, i.e. it was the *worst* of the three rather than the best by two orders of
# magnitude, and `1e-1` accepted that without complaint. The tolerance for `Adam` is what
# closes that hole, and it is the only one of the three that guards a known bug.
#
# On the review comment "it shouldn't be necessary to increase the iteration number": correct,
# and it is back to `main`'s 1000. An intermediate version of this branch ran 1500 steps, but
# that was never a property of the unified interface — it was only needed to satisfy a
# `gradient` tolerance of `3e-3`, which 1000 steps does not reach. Measured from an identical
# starting point (seed `1234`, Geodesic), the unified interface and the old `optimization_step!`
# code agree to every digit printed:
#
#                       old            new
#     1000 steps   0.0101613780334933   0.0101613780334933
#     1500 steps   0.00151325729788261  0.00151325729788277
#
# So there is no per-step convergence regression to paper over here; `Static(0.01)` interacts
# with the retraction exactly as it used to.
const RELATIVE_ERROR_TOLERANCE = (gradient=2e-2, momentum=2e-2, adam=1e-4)

"""
    svd_test(n; retraction)

Approximate the best rank-`n` approximation of `A` with all three algorithms and compare the
result against the one that `LinearAlgebra.svd` gives.
"""
function svd_test(n, train_steps=1000; retraction=Cayley())
    N = size(A, 1)
    U, Σ, Vt = svd(A)
    U_result = U[:, 1:n]

    err_best = norm(A - U_result * U_result' * A)
    ps = (w₁=rand(StiefelManifold, N, n), w₂=rand(StiefelManifold, N, n))

    algorithms = (gradient=GradientMethod(), momentum=MomentumMethod(), adam=GeometricOptimizers.Adam())

    relative_errors = map(algorithms) do algorithm
        # `warn_iterations = 0` silences "Optimizer took 1000 iterations", which is true and is the
        # point: this is a fixed-budget comparison of three first-order methods at one learning rate,
        # not a convergence test. None of them can converge here — with `Static(0.01)` the gradient is
        # 8.4e-2 after these 1000 steps against a gate of 1.5e-8, and it is not stuck but slow
        # (1.9e-3 / 2.1e-4 / 4.0e-5 at 5000 / 20000 / 60000 steps), so reaching the gate this way
        # would take of the order of a million. The convergence test is the `_BFGS` one below.
        optimizer = Optimizer(ps, error; retraction=retraction, algorithm=algorithm,
            linesearch=Static(0.01), max_iterations=train_steps, warn_iterations=0)
        ps_copy = deepcopy(ps)
        solve!(ps_copy, OptimizerState(algorithm, ps_copy), optimizer)

        for Y in values(ps_copy)
            @test GeometricOptimizers.check(Y) < MANIFOLD_TOLERANCE
        end
        norm((error(ps_copy) - err_best) / err_best)
    end

    for name in keys(algorithms)
        @test relative_errors[name] < RELATIVE_ERROR_TOLERANCE[name]
    end

    # The ordering is the part of this that does not depend on the exact starting point:
    # bias-corrected `Adam` beats plain gradient descent on this problem by a wide margin.
    @test relative_errors.adam < relative_errors.gradient / 10
end

for retraction in (GeometricOptimizers.Geodesic(), GeometricOptimizers.Cayley())
    svd_test(3, retraction=retraction)
end

"""
    svd_convergence_test(n; retraction)

The same problem as [`svd_test`](@ref), solved to convergence rather than to a fixed budget.

`_BFGS` needs a line search that actually searches, and until the line search learned to take its
trial step through the retraction that was impossible on manifold parameters — `Static` was the only
one that worked, because it is the only one that never evaluates the merit. So this problem had no
algorithm that converged on it at all: the three first-order methods above exhaust 1000 iterations at
a relative error of 1e-2.
"""
function svd_convergence_test(n; retraction=Cayley(), linesearch=Backtracking(Float64),
    algorithm=GeometricOptimizers._BFGS(), max_iterations=5000)
    N = size(A, 1)
    U, Σ, Vt = svd(A)
    U_result = U[:, 1:n]
    err_best = norm(A - U_result * U_result' * A)

    Random.seed!(1234)
    ps = (w₁=rand(StiefelManifold, N, n), w₂=rand(StiefelManifold, N, n))
    state = OptimizerState(algorithm, ps)
    optimizer = Optimizer(ps, error; retraction=retraction, algorithm=algorithm,
        linesearch=linesearch, max_iterations=max_iterations, warn_iterations=0)

    result = solve!(ps, state, optimizer)

    # it stops on a convergence criterion, not on the iteration cap
    @test GeometricOptimizers.iteration_number(state) < max_iterations
    @test GeometricOptimizers.status(result).rg < 1e-5

    # and it gets to the answer, which the fixed-step runs above reach to 1e-2 at best
    @test norm((error(ps) - err_best) / err_best) < 1e-10

    for Y in values(ps)
        @test GeometricOptimizers.check(Y) < MANIFOLD_TOLERANCE
    end
end

# `_DFP` needed the same lift to `OptimizerSolution` that `_BFGS` already had; before it, its cache was
# `AbstractVector`-only, so a `NamedTuple` fell through to a `NewtonOptimizerCache` and a
# `MethodError`. All eight combinations of retraction, method and line search converge on this
# problem, but the cost is wildly uneven:
#
#                              Geodesic   Cayley
#     _BFGS  Backtracking           176      172
#     _BFGS  Bisection              164      197
#     _DFP   Backtracking        35_263   21_689
#     _DFP   Bisection              116      156
#
# DFP self-corrects far less well than BFGS from a badly scaled `Q` -- the historical reason BFGS
# superseded it -- and paired with a backtracking search that is worth two orders of magnitude here.
# It does terminate on a criterion rather than a cap, but 3×10⁴ iterations is too slow to put in a
# test suite, so that pair is measured and documented rather than run.
for retraction in (GeometricOptimizers.Geodesic(), GeometricOptimizers.Cayley())
    for linesearch in (Backtracking(Float64), Bisection(Float64))
        svd_convergence_test(3; retraction=retraction, linesearch=linesearch,
            algorithm=GeometricOptimizers._BFGS())
    end
    svd_convergence_test(3; retraction=retraction, linesearch=Bisection(Float64),
        algorithm=GeometricOptimizers._DFP())
end
