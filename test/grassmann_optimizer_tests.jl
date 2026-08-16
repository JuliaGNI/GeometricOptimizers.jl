using GeometricOptimizers
using GeometricOptimizers: Cayley, Geodesic, check
using SimpleSolvers
using SimpleSolvers: Bisection
using LinearAlgebra
using Test
import Random

# A `GrassmannManifold` can be driven through an `Optimizer`, which until this file it could not:
# `GradientAutodiff(F, ::GrassmannManifold)` did not exist, so a bare one was a `MethodError` at
# construction, and a `NamedTuple` holding one died in the first step — under `BFGS` with
# `CanonicalIndexError: setindex! not defined for GrassmannManifold`, under `Adam` with `The function
# `similar` does not make sense in this context`. That was issue A11, the concrete content of
# issue #27. Every `GrassmannManifold` test in the suite exercised the manifold, its lift, its
# retraction and its `check`; none exercised a solve, because none could.
#
# `test/manifold_optimizers_with_new_interface.jl` is the same file for the `StiefelManifold` and its
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

objective(::Type{T}, N::Integer) where {T} = let M = problem_matrix(T, N)
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
    V = eigen(Matrix(problem_matrix(T, N))).vectors[:, (N-n+1):N]
    V * V'
end

function optimize(::Type{T}, N::Integer, n::Integer, algorithm; retraction=Geodesic(), linesearch=nothing, seed::Integer=1234) where {T}
    Random.seed!(seed)
    f = objective(T, N)
    x = rand(GrassmannManifold{T}, N, n)
    x₀ = copy(x)
    optimizer = isnothing(linesearch) ?
                Optimizer(x, f; algorithm=algorithm, retraction=retraction) :
                Optimizer(x, f; algorithm=algorithm, retraction=retraction, linesearch=linesearch)
    solve!(x, OptimizerState(algorithm, x), optimizer)
    x, x₀, f
end

# `Adam`'s direction is `-m₁/(√m₂ + δ)`, of magnitude ≈ 1 per component whatever the gradient is, so
# a fixed step never lets it settle; it gets a searching line search here for the reason
# `manifold_optimizers_with_new_interface.jl` gives it one. `Bisection` rather than the
# `Backtracking` default because Adam's direction is deliberately not required to descend.
linesearch_for(::Type{T}, algorithm) where {T} = algorithm isa Adam ? Bisection(T) : nothing
linesearch_for(algorithm) = linesearch_for(Float64, algorithm)

# `check` measures the deviation from the manifold, so this is a round-off tolerance. The worst
# observed over every case below is 9.5e-15 (`Float64`) and 5.4e-6 (`Float32`).
manifold_tolerance(::Type{T}) where {T} = 1000 * eps(T)

# The distance between the projectors, which is the distance between *subspaces*. The worst observed
# is 4.6e-8 (`Float64`) and 2.6e-3 (`Float32`); the first-order methods are the loose ones.
subspace_tolerance(::Type{T}) where {T} = 100 * sqrt(eps(T))

const METHODS = (GradientMethod(), MomentumMethod(0.1), Adam(Float64), BFGS(), DFP())

@testset "a bare GrassmannManifold can be optimized" begin
    for (N, n) in ((3, 1), (5, 2)), retraction in (Geodesic(), Cayley()), algorithm in METHODS
        x, x₀, f = optimize(Float64, N, n, algorithm; retraction=retraction,
            linesearch=linesearch_for(algorithm))

        @test x isa GrassmannManifold{Float64}                              # the type is preserved
        @test check(x) < manifold_tolerance(Float64)                        # and so is the manifold
        @test norm(x * x' - dominant_projector(Float64, N, n)) < subspace_tolerance(Float64)
        @test f(x) < f(x₀)                                                  # it improved on the start
    end
end

# `Float32` separately, and with the element type threaded through: a `GrassmannManifold{Float32}`
# used to be promoted to `Float64` by `ParameterHandling.flatten`'s default, and nothing here would
# have caught it, because nothing here could run at all.
@testset "…in Float32 as well as Float64" begin
    for (N, n) in ((3, 1), (5, 2)), retraction in (Geodesic(), Cayley())
        for algorithm in (GradientMethod(), MomentumMethod(0.1f0), Adam(Float32), BFGS(), DFP())
            x, x₀, f = optimize(Float32, N, n, algorithm; retraction=retraction,
                linesearch=linesearch_for(Float32, algorithm))

            @test x isa GrassmannManifold{Float32}
            @test check(x) < manifold_tolerance(Float32)
            @test norm(x * x' - dominant_projector(Float32, N, n)) < subspace_tolerance(Float32)
            @test f(x) < f(x₀)
        end
    end
end

# The second failure mode A11 names, and the one that goes through a different set of helpers: a
# bare manifold reaches `_similar(::Manifold)` and `copyto!(::Manifold, ::Manifold)` directly, a
# `NamedTuple` reaches them one `apply_toNT` down. The ordinary `Matrix` block is there so that the
# mixed case is covered too -- `_manifold_αmax` derives the step ceiling per block and has to see a
# `GrassmannManifold` beside something that contributes no ceiling at all.
@testset "a NamedTuple holding a GrassmannManifold" begin
    for algorithm in METHODS
        Random.seed!(7)
        ps = (Y=rand(GrassmannManifold{Float64}, 5, 2), W=randn(2, 2))
        f = p -> -tr(p.Y' * M₅ * p.Y) + sum(abs2, p.W .- 1.0)
        f₀ = f(ps)
        ls = linesearch_for(algorithm)
        optimizer = isnothing(ls) ? Optimizer(ps, f; algorithm=algorithm) :
                    Optimizer(ps, f; algorithm=algorithm, linesearch=ls)
        solve!(ps, OptimizerState(algorithm, ps), optimizer)

        @test ps.Y isa GrassmannManifold{Float64}
        @test check(ps.Y) < manifold_tolerance(Float64)
        @test norm(ps.Y * ps.Y' - dominant_projector(Float64, 5, 2)) < subspace_tolerance(Float64)
        @test norm(ps.W .- 1.0) < subspace_tolerance(Float64)   # the Euclidean block converged too
        @test f(ps) < f₀
    end
end

# The two manifolds side by side in one `NamedTuple`. This is what would have caught the defect the
# 0.2.0 notes record under `ParameterHandling.flatten(T, ::Manifold)` — that it rebuilt a
# `StiefelManifold` for every kind of manifold, so a `GrassmannManifold` came back Stiefel, with a
# different `rgrad` and a different retraction and no error anywhere. That fix has had no end-to-end
# test until now, because a solve could not reach it.
@testset "a NamedTuple holding both kinds of manifold" begin
    for retraction in (Geodesic(), Cayley())
        Random.seed!(3)
        ps = (Y=rand(GrassmannManifold{Float64}, 5, 2), S=rand(StiefelManifold{Float64}, 5, 1))
        f = p -> -tr(p.Y' * M₅ * p.Y) - tr(p.S' * M₅ * p.S)
        f₀ = f(ps)
        optimizer = Optimizer(ps, f; algorithm=BFGS(), retraction=retraction)
        solve!(ps, OptimizerState(BFGS(), ps), optimizer)

        @test ps.Y isa GrassmannManifold{Float64}       # each block keeps its own manifold type
        @test ps.S isa StiefelManifold{Float64}
        @test check(ps.Y) < manifold_tolerance(Float64)
        @test check(ps.S) < manifold_tolerance(Float64)
        @test f(ps) < f₀
    end
end

# `BFGS` on a bare manifold, run twice from the same seed, has to give the same answer.
# Not a determinism claim about the retraction — that is open issue A5, and `GlobalSection` draws a
# random complement from the *global* RNG — but a check that seeding the run is enough to reproduce
# it, which is what the rest of this file relies on.
@testset "a seeded Grassmann solve reproduces" begin
    x₁, _, _ = optimize(Float64, 5, 2, BFGS())
    x₂, _, _ = optimize(Float64, 5, 2, BFGS())
    @test x₁.A == x₂.A
end
