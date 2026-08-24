using GeometricOptimizers
using GeometricOptimizers: ArrayNamedTuple, OptimizerCache, OptimizerSolution,
    Cayley, Geodesic, check, increase_iteration_number!, solver_step!
using NeuralNetworkParameters: flatten, unflatten
using SimpleSolvers: Static, l2norm
using Test
import Random

# The MNIST scripts of GMLDatasets.jl (https://github.com/JuliaGNI/GMLDatasets.jl) store the
# parameters of a transformer in a flat `NamedTuple` that mixes `StiefelManifold`s with ordinary
# matrices and vectors, that is made up of `Float32`s and whose gradient is computed by hand. These
# three cases are covered below, together with the property that matters for all of them: the
# `StiefelManifold` entries have to *stay* on the manifold over the course of the optimization
# (this is what `check` measures).

const N, n, m = 6, 3, 4

Random.seed!(1234)

const B₀ = randn(N, m)

# the ranges that the three parameters occupy in `flatten(ps)[1]`
const ranges = (1:(N*n), (N*n+1):(N*n+n*m), (N*n+n*m+1):(N*n+n*m+N))

"""
    test_problem(T)

Return `F` and `∇F!` for ``F(Y, W, b) = \\frac{1}{2}\\|YW + b - B\\|^2`` with
``Y\\in{}St(N, n)``, ``W\\in\\mathbb{R}^{n\\times{}m}`` and ``b\\in\\mathbb{R}^N``.

The gradients are ``\\partial_YF = RW^T``, ``\\partial_WF = Y^TR`` and
``\\partial_bF = \\sum_jR_{\\bullet{}j}`` with ``R = YW + b - B``. As required by
[`GradientFunction`](@ref), `∇F!` operates on the *flattened* parameters.
"""
function test_problem(::Type{T}) where {T}
    B = T.(B₀)

    F(ps::NamedTuple) = sum(abs2, ps.Y * ps.W .+ ps.b .- B) / 2

    function ∇F!(g::AbstractVector, v::AbstractVector)
        Y = reshape(view(v, ranges[1]), N, n)
        W = reshape(view(v, ranges[2]), n, m)
        b = view(v, ranges[3])
        R = Y * W .+ b .- B
        copyto!(view(g, ranges[1]), vec(R * W'))
        copyto!(view(g, ranges[2]), vec(Y' * R))
        copyto!(view(g, ranges[3]), vec(sum(R; dims=2)))
        g
    end

    F, ∇F!
end

initial_parameters(::Type{T}) where {T} =
    (Y=rand(Random.Xoshiro(1234), StiefelManifold{T}, N, n), W=randn(Random.Xoshiro(5678), T, n, m), b=zeros(T, N))

"""
    optimize(T, algorithm; kwargs...)

Take `steps` optimizer steps and return the parameters, the value of `check(ps.Y)` after
*every* step and the values of the objective before and after every step.

Note that the seed has to be fixed for every run: the [`GlobalSection`](@ref) is drawn at
random and the iterates depend on it.
"""
function optimize(::Type{T}, algorithm; steps=20, η=0.1, retraction=Cayley(), hand_written_gradient=false) where {T}
    Random.seed!(1234)
    F, ∇F! = test_problem(T)
    ps = initial_parameters(T)
    optimizer = if hand_written_gradient
        Optimizer(ps, F; (∇F!)=∇F!, algorithm=algorithm, linesearch=Static(T(η)), retraction=retraction)
    else
        Optimizer(ps, F; algorithm=algorithm, linesearch=Static(T(η)), retraction=retraction)
    end
    state = OptimizerState(algorithm, ps)

    checks = T[]
    losses = T[F(ps)]
    for _ in 1:steps
        increase_iteration_number!(state)
        solver_step!(ps, state, optimizer)
        update!(state, optimizer, ps)
        push!(checks, check(ps.Y))
        push!(losses, F(ps))
    end

    ps, checks, losses
end

# Note that `Adam` has to be constructed with the element type of the parameters: unlike
# `MomentumMethod`, which the `Optimizer` converts, an `Adam{Float64}` does not dispatch to
# `OptimizerCache(::Adam{T}, ::OptimizerSolution{T})` for `Float32` parameters. That mismatch
# now errors with a message that says so, see the testset at the bottom of this file.
algorithms(::Type{T}) where {T} = (GradientMethod(), MomentumMethod(T(0.1)), Adam(T))
retractions() = (Geodesic(), Cayley())

# `ArrayTuple` used to be written as `Tuple{Vararg{AT}} where {AT<:AbstractArray{T}}`, which
# Julia's diagonal rule makes *homogeneous*: a `NamedTuple` that stores a `StiefelManifold`
# and an ordinary `Matrix` at the same time was then not an `ArrayNamedTuple` and could not
# be passed to the `Optimizer`.
@testset "a heterogeneous NamedTuple is a valid set of parameters" begin
    for T in (Float64, Float32)
        ps = initial_parameters(T)
        @test length(unique(typeof.(values(ps)))) == 3      # a manifold, a matrix and a vector
        @test ps isa ArrayNamedTuple{T}
        @test ps isa OptimizerSolution{T}
    end
end

# A one-argument `flatten` that defaults to `Float64` silently promotes `Float32` parameters, so the
# flattened vector no longer matches the parameters. `NeuralNetworkParameters` takes the element type
# from the parameters; this is what pins that.
@testset "the parameters are flattened to their own element type" begin
    for T in (Float64, Float32)
        ps = initial_parameters(T)
        v, layout = flatten(ps)
        @test v isa Vector{T}
        @test length(v) == N * n + n * m + N
        ps′ = unflatten(layout, v)
        @test ps′.Y ≈ ps.Y
        @test ps′.W == ps.W
        @test ps′.b == ps.b
        @test typeof(ps′.Y) == typeof(ps.Y)
    end
end

# ... and it has to come back as the manifold it went in as. The method was written as
# `StiefelManifold(unflatten(v))` for every `Manifold`, so a `GrassmannManifold` was silently
# turned into a `StiefelManifold` — which has a different `rgrad` and a different retraction,
# so the optimization would have kept running and produced the wrong iterates.
@testset "the flattening preserves the kind of manifold" begin
    for T in (Float64, Float32), MT in (StiefelManifold, GrassmannManifold)
        Y = rand(Random.Xoshiro(1234), MT{T}, N, n)
        # `flatten` takes the element type from the parameters, for a bare manifold as much as for
        # a `NamedTuple` of them, so the explicit `T` here is belt and braces rather than required
        v, layout = flatten(T, Y)
        @test v isa Vector{T}
        @test length(v) == N * n
        Y′ = unflatten(layout, v)
        @test Y′ isa MT{T}
        @test Y′ ≈ Y

        # the same through a `NamedTuple`, which is how the optimizer sees it
        ps = (Y=Y, W=zeros(T, n, m))
        vₚ, layoutₚ = flatten(ps)
        @test unflatten(layoutₚ, vₚ).Y isa MT{T}
    end
end

# The property that the whole exercise is about: a retraction maps back onto the manifold, so
# `Y` has to satisfy `YᵀY = I` after every single step — for every algorithm, every
# retraction and both element types.
#
# The tolerance is a round-off tolerance and nothing else. The deviation grows very slowly
# with the number of iterations — measured over all the combinations below it is `5`, `10`,
# `20` and `45` times `eps(T)` for `5`, `20`, `80` and `320` steps, i.e. it behaves like a
# random walk that the retraction keeps pulling back rather than like an accumulating error.
# The worst value observed at the `20` steps taken here is `10 * eps(T)`, so the tolerance
# leaves a factor of `10`. A retraction that actually left the manifold would be off by the
# order of the step size, i.e. by `1e-1`.
const MANIFOLD_TOLERANCE_IN_EPS = 100

@testset "the manifold property is preserved during the optimization" begin
    for T in (Float64, Float32), algorithm in algorithms(T), retraction in retractions()
        tol = MANIFOLD_TOLERANCE_IN_EPS * eps(T)
        _, checks, losses = optimize(T, algorithm; retraction=retraction)
        @test check(initial_parameters(T).Y) < tol   # the starting point is on the manifold ...
        @test maximum(checks) < tol                  # ... and so is every iterate
        @test last(losses) < first(losses) / 2       # the optimization actually did something
    end
end

# Only the projections of the attention layers of the MNIST transformer are on a manifold, so
# the ordinary entries must be updated as ordinary (Euclidean) parameters.
@testset "the parameters that are not on a manifold are optimized as well" begin
    for T in (Float64, Float32)
        ps₀ = initial_parameters(T)
        ps, _, _ = optimize(T, GradientMethod())
        @test ps.W ≉ ps₀.W
        @test ps.b ≉ ps₀.b
        @test ps.Y ≉ ps₀.Y
    end
end

# `GradientFunction(F, ∇F!, nt::NamedTuple)` is what lets the GMLDatasets.jl scripts use `Zygote`
# instead of the default `ForwardDiff`; `∇F!` is called on the flattened parameters.
@testset "a hand written gradient gives the same result as the default one" begin
    for T in (Float64, Float32), algorithm in algorithms(T)
        ps₁, _, losses₁ = optimize(T, algorithm; hand_written_gradient=false)
        ps₂, checks₂, losses₂ = optimize(T, algorithm; hand_written_gradient=true)
        @test ps₁.Y ≈ ps₂.Y
        @test ps₁.W ≈ ps₂.W
        @test ps₁.b ≈ ps₂.b
        @test losses₁ ≈ losses₂
        @test maximum(checks₂) < MANIFOLD_TOLERANCE_IN_EPS * eps(T)
    end
end

# `l2norm` on a `NamedTuple` has to combine the block norms in *quadrature*: it is the ℓ² norm
# of the parameters seen as one long vector, which is what the flattening makes them. Summing
# the blocks instead overestimates it by up to `√k` for `k` blocks — and every stopping
# criterion of `solve!` is computed from it.
@testset "l2norm on a NamedTuple is the norm of the flattened parameters" begin
    # the 3-4-5 triangle, so that summing (7.0) and the quadrature (5.0) are far apart
    @test l2norm((a=[3.0, 0.0], b=[0.0, 4.0])) ≈ 5.0

    for T in (Float64, Float32)
        ps = initial_parameters(T)
        v, _ = flatten(ps)
        @test l2norm(ps) ≈ l2norm(v)
        # ... and the sum of the blocks really is a different number here
        @test !isapprox(sum(l2norm, values(ps)), l2norm(v))
    end
end

# `solve!` on `NamedTuple` parameters is the only path that builds an `OptimizerStatus` on
# them, i.e. the only one that calls `l2norm(::ArrayNamedTuple)` above and `_difference!` on
# the gradient blocks. The testsets further up drive `solver_step!` by hand and never get
# there.
@testset "solve! runs on NamedTuple parameters" begin
    for T in (Float64, Float32), algorithm in algorithms(T)
        Random.seed!(1234)
        F, _ = test_problem(T)
        ps = initial_parameters(T)
        f₀ = F(ps)
        # `max_iterations` is capped because a fixed step size does not get these all the way to
        # the convergence criteria: `GradientMethod` and `Adam` run to the iteration limit, so
        # the default of `1000` only costs time (and prints a warning) without testing more.
        optimizer = Optimizer(ps, F; algorithm=algorithm, linesearch=Static(T(0.1)), max_iterations=100)
        result = solve!(ps, OptimizerState(algorithm, ps), optimizer)

        @test result.f < f₀                                          # it made progress ...
        @test F(ps) == result.f                                      # ... and reported it
        @test check(ps.Y) < MANIFOLD_TOLERANCE_IN_EPS * eps(T)       # still on the manifold
        # the convergence measures are the ones computed from `l2norm(::ArrayNamedTuple)`
        @test !isnan(result.status.rg)
        @test result.status.rg ≥ 0
    end
end

# The `Optimizer(x, problem)` entry point is the one that `Optimizer(x, F)` delegates to, but
# it is also public on its own — and it is the one that has to build the gradient itself. For
# `NamedTuple` parameters that gradient is called on the *flattened* parameters, so sizing it
# with `length(x)` (the number of entries, `3` here) instead of constructing it from `x` used
# to make the very first step throw a `DimensionMismatch`.
@testset "Optimizer(ps, OptimizerProblem(F, ps)) supplies its own gradient" begin
    for T in (Float64, Float32)
        Random.seed!(1234)
        F, _ = test_problem(T)
        ps = initial_parameters(T)
        ps₀ = deepcopy(ps)
        algorithm = GradientMethod()
        optimizer = Optimizer(ps, OptimizerProblem(F, ps); algorithm=algorithm, linesearch=Static(T(0.1)))
        state = OptimizerState(algorithm, ps)

        increase_iteration_number!(state)
        solver_step!(ps, state, optimizer)

        @test F(ps) < F(ps₀)
        @test ps.Y ≉ ps₀.Y
        # the same step as the one taken through `Optimizer(ps, F)`, which passes its own gradient
        ps′, _, _ = optimize(T, algorithm; steps=1)
        @test ps.Y ≈ ps′.Y
        @test ps.W ≈ ps′.W
        @test ps.b ≈ ps′.b
    end
end

# The one thing about `Adam` that a user has to get right is that it carries its own
# parameters, so it has to be constructed with the element type of the parameters (see the
# comment above `algorithms`). `MomentumMethod` is converted by the `Optimizer`, `Adam` is not,
# and the mismatch used to surface as `MethodError: no method matching
# OptimizerCache(::Adam{Float64}, ::NamedTuple{...Float32...})` — which says what did not match
# but not what to do about it.
@testset "an Adam of the wrong element type says what is wrong" begin
    ps = initial_parameters(Float32)
    @test_throws "Adam(Float32)" OptimizerCache(Adam(Float64), ps)
    @test_throws ErrorException Optimizer(ps, test_problem(Float32)[1]; algorithm=Adam(Float64))
    # `Adam(Float32)` is what the message asks for, and it works
    @test OptimizerCache(Adam(Float32), ps) isa GeometricOptimizers.AdamCache{Float32}
end


# This guards a property that no other test can see, because the bug it protects against did not
# fail anything — it made the caller hang. The cache and state structs used to bound their type
# parameters by `OptimizerSolution`, `GradientArrayOrNamedTuple` and
# `GlobalSectionSingleOrNamedTuple`. Inference cannot solve those bounds down to a concrete
# `NamedTuple` (it cannot with or without them — the constructors' own signatures are already
# written in the same aliases), so the inferred cache type is a `UnionAll` either way.
# What the bounds added was *coupling*: they tied all four parameters to one shared `T` underneath
# nested `Vararg` unions, so every method-table intersection involving an inferred cache had to
# re-solve that constraint system in `subtype_unionall`. On a nine-layer network that did not
# finish in over an hour; see GeometricMachineLearning#230. Unbounded, the parameters are
# independent and the intersection is cheap.
#
# Every cache and state is checked, not only the ones that were measured to be slow: the rule is
# that the family is uniform, so that nobody has to work out per struct whether a given bound
# happens to be one that costs. `DFPState` is an alias for `BFGSState` and `AdamWithEuclideanDecay`
# reuses `AdamCache`/`AdamState`, so between them the list below is every one there is.
"""
    _all_parameters_unbounded(TT)

Whether `TT` declares all of its type parameters without an upper bound.

A parameter written `{T}` rather than `{T<:S}` has `ub === Any` once the `where`s are peeled off,
so this states the property directly. Asserting it as a *value* rather than by instantiating
`TT{Float64,Int,Int,Int}` matters: instantiating a reinstated bound throws a `TypeError`, which
would abort the loop over the remaining structs and report an error rather than naming the property
that broke.
"""
function _all_parameters_unbounded(TT::UnionAll)
    t = TT
    while t isa UnionAll
        t.var.ub === Any || return false
        t = t.body
    end
    true
end

@testset "$(nameof(TT)) leaves its type parameters unbounded" for TT in (
    GeometricOptimizers.GradientCache, GeometricOptimizers.MomentumCache,
    GeometricOptimizers.AdamCache, GeometricOptimizers.BFGSCache,
    GeometricOptimizers.DFPCache, GeometricOptimizers.NewtonOptimizerCache,
    GradientState, MomentumState, AdamState,
    BFGSState, NewtonOptimizerState)

    @test _all_parameters_unbounded(TT)
end

@testset "the cache and state type parameters are unbounded" begin
    # `OptimizerResult` is on the return path of every `solve!`, so it is one of the types an
    # inferring caller has to intersect. Only `VT` is unbounded here: `OST<:OptimizerStatus{T,YT}`
    # is a plain two-parameter struct with no union expansion behind it, and it is what ties `T` and
    # `YT` to the status the result reports.
    @test Base.unwrap_unionall(GeometricOptimizers.OptimizerResult).parameters[3].ub === Any

    # ... and the constructors still produce exactly the types they always did.
    ps = initial_parameters(Float64)
    @test OptimizerCache(Adam(Float64), ps) isa GeometricOptimizers.AdamCache{Float64}
    @test OptimizerState(Adam(Float64), ps) isa AdamState{Float64}
    @test OptimizerCache(BFGS(), ps) isa GeometricOptimizers.BFGSCache{Float64}
    @test OptimizerState(BFGS(), ps) isa BFGSState{Float64}
    @test OptimizerCache(DFP(), ps) isa GeometricOptimizers.DFPCache{Float64}
    @test OptimizerCache(Newton(), zeros(3)) isa GeometricOptimizers.NewtonOptimizerCache{Float64}
    @test OptimizerState(Newton(), zeros(3)) isa NewtonOptimizerState{Float64}
end
