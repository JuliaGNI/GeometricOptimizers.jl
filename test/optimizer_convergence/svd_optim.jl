using GeometricOptimizers
using SimpleSolvers: Static
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
# `1500` iterations with `Static(0.01)` and seed `1234`. Measured on Julia 1.13:
#
#                  Geodesic   Cayley
#     GradientMethod  1.5e-3   1.5e-3
#     MomentumMethod  1.0e-2   9.8e-3
#     Adam            1.1e-5   5.0e-5
#
# The tolerances below leave a factor of two on top of those.
#
# This test used to apply a single blanket tolerance of `1e-1` to all three algorithms, which
# is how the two `Adam` bugs (uninitialised moments and the wrong bias-correction factors)
# survived: with them present `Adam` reached only `1.1e-3` (Geodesic) and `1.7e-3` (Cayley),
# i.e. it was the *worst* of the three rather than the best by two orders of magnitude, and
# `1e-1` accepted that without complaint. The tolerance for `Adam` is an order of magnitude
# below the error of the buggy version, so that regression now fails here.
const RELATIVE_ERROR_TOLERANCE = (gradient=3e-3, momentum=2e-2, adam=1e-4)

"""
    svd_test(n; retraction)

Approximate the best rank-`n` approximation of `A` with all three algorithms and compare the
result against the one that `LinearAlgebra.svd` gives.
"""
function svd_test(n, train_steps=1500; retraction=Cayley())
    N = size(A, 1)
    U, Σ, Vt = svd(A)
    U_result = U[:, 1:n]

    err_best = norm(A - U_result * U_result' * A)
    ps = (w₁=rand(StiefelManifold, N, n), w₂=rand(StiefelManifold, N, n))

    algorithms = (gradient=GradientMethod(), momentum=MomentumMethod(), adam=GeometricOptimizers.Adam(0.01))

    relative_errors = map(algorithms) do algorithm
        optimizer = Optimizer(ps, error; retraction=retraction, algorithm=algorithm,
            linesearch=Static(0.01), max_iterations=train_steps)
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
