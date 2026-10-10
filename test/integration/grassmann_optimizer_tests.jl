using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, check, isconverged, status
using SimpleSolvers
using SimpleSolvers: Bisection
using LinearAlgebra
using Test
import Random

include("../helpers/eltypes.jl")

# A `GrassmannManifold` can be driven through an `Optimizer`, which until this file it could not:
# `GradientAutodiff(F, ::GrassmannManifold)` did not exist, so a bare one was a `MethodError` at
# construction, and a `NamedTuple` holding one died in the first step — under `BFGS` with
# `CanonicalIndexError: setindex! not defined for GrassmannManifold`, under `Adam` with `The function
# `similar` does not make sense in this context`. That was issue A11, the concrete content of
# issue #27. Every `GrassmannManifold` test in the suite exercised the manifold, its lift, its
# retraction and its `check`; none exercised a solve, because none could.
#
# `test/integration/manifold_optimizers_with_new_interface.jl` is the same file for the `StiefelManifold` and its
# problem cannot be reused. It minimizes the distance to a target point, which is not a function on
# the Grassmann manifold at all: `Y` and `-Y` are the same point of `Gr(1, 3)` and are at different
# distances from `[0, 0, 1.2]`. **Everything here is invariant under `Y ↦ YO`** — the objective by
# construction, the assertions because they are made about the projector `YYᵀ` and never about `Y`.
# That is what A11 meant by "a decision about what the Grassmann tests should then assert".
#
# The objective is the Rayleigh quotient of a fixed symmetric `M`,
#
#     F(Y) = -tr(YᵀMY),
#
# whose minimizer over `Gr(n, N)` is the invariant subspace of the `n` largest eigenvalues of `M`.
# `tr((YO)ᵀM(YO)) = tr(OᵀYᵀMYO) = tr(YᵀMY)`, so it is constant on the equivalence class, and the
# minimizer is a *subspace* rather than a matrix — which is the point of the manifold.

Random.seed!(1234)

# Distinct eigenvalues, so the dominant subspace is unique and the minimizer is isolated. Written out
# rather than drawn, so a failure is reproducible without the RNG.
const M₃ = Symmetric([3.0 0.5 0.0; 0.5 2.0 0.1; 0.0 0.1 1.0])
const M₅ = Symmetric([5.0 0.4 0.1 0.0 0.2
                      0.4 4.0 0.3 0.1 0.0
                      0.1 0.3 3.0 0.2 0.1
                      0.0 0.1 0.2 2.0 0.3
                      0.2 0.0 0.1 0.3 1.0])

# The problem matrix for `Gr(n, N)`, in element type `T`. The fallthrough is an error rather than
# `M₅`, so that a wrong `N` says so here instead of surfacing as a `DimensionMismatch` several frames
# into a solve.
function problem_matrix(::Type{T}, N::Integer) where {T}
    N == 3 && return Symmetric(T.(M₃))
    N == 5 && return Symmetric(T.(M₅))
    error("no problem matrix for N = $N; this file covers Gr(1, 3) and Gr(2, 5)")
end

objective(::Type{T}, N::Integer) where {T} =
    let M = problem_matrix(T, N)
        Y -> -tr(Y' * M * Y)
    end

"""
    dominant_projector(T, N, n)

The orthogonal projector onto the span of the `n` eigenvectors of the problem matrix belonging to its
`n` largest eigenvalues, i.e. onto the minimizer of [`objective`](@ref).

The projector and not a representative of it: `YYᵀ` is the one function of `Y` that is constant on
the equivalence class `Y ∼ YO`, so it is what a Grassmann solve can be asserted against. Comparing
`Y` itself would be asserting on the arbitrary basis the solve happened to end in.
"""
function dominant_projector(::Type{T}, N::Integer, n::Integer) where {T}
    V = eigen(Matrix(problem_matrix(T, N))).vectors[:, (N - n + 1):N]
    V * V'
end

# The start is drawn from a generator of its own; the global RNG is seeded as well, because every
# `GlobalSection` completes its frame from it (see the last testset).
function optimize(::Type{T}, N::Integer, n::Integer, algorithm; retraction = Geodesic(),
        linesearch = nothing, seed::Integer = 1234) where {T}
    Random.seed!(seed)
    f = objective(T, N)
    x = rand(Random.Xoshiro(seed), GrassmannManifold{T}, N, n)
    x₀ = copy(x)
    optimizer = isnothing(linesearch) ?
                Optimizer(x, f; algorithm = algorithm, retraction = retraction) :
                Optimizer(
        x, f; algorithm = algorithm, retraction = retraction, linesearch = linesearch)
    result = solve!(x, OptimizerState(algorithm, x), optimizer)
    x, x₀, f, result
end

# `Adam`'s direction is `-m₁/(√m₂ + δ)`, of magnitude ≈ 1 per component whatever the gradient is, so
# a fixed step never lets it settle; it gets a searching line search here for the reason
# `manifold_optimizers_with_new_interface.jl` gives it one. `Bisection` rather than the
# `Backtracking` default because Adam's direction is deliberately not required to descend.
linesearch_for(::Type{T}, algorithm) where {T} = algorithm isa Adam ? Bisection(T) : nothing

# `check` measures the deviation from the manifold, so this is a round-off tolerance. The worst
# observed over every case below and three seeds is 222eps(T) in `Float64` (the two kinds of manifold
# side by side) and 25eps(T) in `Float32`.
grassmann_manifold_tolerance(::Type{T}) where {T} = 1000 * eps(T)

# The distance between the projectors, which is the distance between *subspaces*. The objective is
# quadratic at its minimiser, so a solve stops within a multiple of `√eps(T)` of it. The worst observed
# over every case below and three seeds is 9.3√eps(T) in `Float64` and 8.4√eps(T) in `Float32`.
subspace_tolerance(::Type{T}) where {T} = 100 * sqrt(eps(T))

function algorithms(::Type{T}) where {T}
    (
        GradientMethod(), MomentumMethod(; α = T(0.1)), Adam(), BFGS(), DFP())
end

# In both precisions, with the element type threaded through: a `GrassmannManifold{Float32}` used to
# be promoted to `Float64` by `ParameterHandling.flatten`'s default, and nothing here would have caught
# it, because nothing here could run at all.
@testset "a bare GrassmannManifold can be optimized, $T" for T in REAL_ELTYPES
    for (N, n) in ((3, 1), (5, 2)), retraction in (Geodesic(), Cayley()),
        algorithm in algorithms(T)
        x, x₀, f, result = optimize(T, N, n, algorithm; retraction = retraction,
            linesearch = linesearch_for(T, algorithm))

        @test x isa GrassmannManifold{T}                                    # the type is preserved
        @test eltype(x) == T
        @test check(x) < grassmann_manifold_tolerance(T)                    # and so is the manifold
        @test isconverged(status(result))
        @test norm(x * x' - dominant_projector(T, N, n)) < subspace_tolerance(T)
        @test f(x) < f(x₀)                                                  # it improved on the start
    end
end

# The second failure mode A11 names, and the one that goes through a different set of helpers: a
# bare manifold reaches `_similar(::Manifold)` and `copyto!(::Manifold, ::Manifold)` directly, a
# `NamedTuple` reaches them one `apply_toNT` down. The ordinary `Matrix` block is there so that the
# mixed case is covered too -- `_manifold_αmax` derives the step ceiling per block and has to see a
# `GrassmannManifold` beside something that contributes no ceiling at all.
#
# The minimiser of the Euclidean block is `W = c`, with a `c` that is not a power of two.
@testset "a NamedTuple holding a GrassmannManifold, $T" for T in REAL_ELTYPES
    M = problem_matrix(T, 5)
    c = T(0.7)

    for algorithm in algorithms(T)
        Random.seed!(7)
        rng = Random.Xoshiro(7)
        ps = NetworkParameters((
            Y = rand(rng, GrassmannManifold{T}, 5, 2), W = randn(rng, T, 2, 2)))
        f = p -> -tr(p.Y' * M * p.Y) + sum(abs2, p.W .- c)
        f₀ = f(ps)
        ls = linesearch_for(T, algorithm)
        optimizer = isnothing(ls) ? Optimizer(ps, f; algorithm = algorithm) :
                    Optimizer(ps, f; algorithm = algorithm, linesearch = ls)
        result = solve!(ps, OptimizerState(algorithm, ps), optimizer)

        @test ps.Y isa GrassmannManifold{T}
        @test eltype(ps.W) == T
        @test check(ps.Y) < grassmann_manifold_tolerance(T)
        @test isconverged(status(result))
        @test norm(ps.Y * ps.Y' - dominant_projector(T, 5, 2)) < subspace_tolerance(T)
        @test norm(ps.W .- c) < subspace_tolerance(T)      # the Euclidean block converged too
        @test f(ps) < f₀
    end
end

# The two manifolds side by side in one `NamedTuple`. This is what would have caught the defect the
# 0.2.0 notes record under `ParameterHandling.flatten(T, ::Manifold)` — that it rebuilt a
# `StiefelManifold` for every kind of manifold, so a `GrassmannManifold` came back Stiefel, with a
# different `rgrad` and a different retraction and no error anywhere. That fix has had no end-to-end
# test until now, because a solve could not reach it.
#
# The two blocks are independent problems: `Y` minimises at the dominant 2-subspace of `M₅` and `S`
# at its dominant eigenvector, up to sign, so `SSᵀ` is the dominant 1-projector.
@testset "a NamedTuple holding both kinds of manifold, $T" for T in REAL_ELTYPES
    M = problem_matrix(T, 5)

    for retraction in (Geodesic(), Cayley())
        Random.seed!(3)
        rng = Random.Xoshiro(3)
        ps = NetworkParameters((Y = rand(rng, GrassmannManifold{T}, 5, 2),
            S = rand(rng, StiefelManifold{T}, 5, 1)))
        f = p -> -tr(p.Y' * M * p.Y) - tr(p.S' * M * p.S)
        f₀ = f(ps)
        optimizer = Optimizer(ps, f; algorithm = BFGS(), retraction = retraction)
        result = solve!(ps, OptimizerState(BFGS(), ps), optimizer)

        @test ps.Y isa GrassmannManifold{T}             # each block keeps its own manifold type
        @test ps.S isa StiefelManifold{T}
        @test eltype(ps.Y) == T
        @test check(ps.Y) < grassmann_manifold_tolerance(T)
        @test check(ps.S) < grassmann_manifold_tolerance(T)
        @test isconverged(status(result))
        @test norm(ps.Y * ps.Y' - dominant_projector(T, 5, 2)) < subspace_tolerance(T)
        @test norm(ps.S * ps.S' - dominant_projector(T, 5, 1)) < subspace_tolerance(T)
        @test f(ps) < f₀
    end
end

# `BFGS` on a bare manifold, run twice from the same seed, has to give the same answer.
# Not a determinism claim about the retraction — that is open issue A5, and `GlobalSection` draws a
# random complement from the *global* RNG — but a check that seeding the run is enough to reproduce
# it, which is what the rest of this file relies on.
@testset "a seeded Grassmann solve reproduces, $T" for T in REAL_ELTYPES
    x₁, _, _, _ = optimize(T, 5, 2, BFGS())
    x₂, _, _, _ = optimize(T, 5, 2, BFGS())
    @test eltype(x₁) == T
    @test x₁.A == x₂.A
end
