# A `NeuralNetworkParameters.NetworkParameters` as the solution of an optimizer.
#
# `test/flat_parameters.jl` covers the *flat* shape — the one the MNIST scripts of GMLDatasets.jl
# keep a transformer's parameters in. This file covers the **nested** one: a container is a tree of
# layers, so its leaves are one level below what `Base.map` reaches, and that is the whole difference
# between the two. Every elementwise primitive is walked with `mapparameters` for that reason, and
# these testsets are what hold it down — `map` would hand `zero`, `copy` and `similar` a whole layer.
#
# `NetworkParameters{T}` is a member of `OptimizerSolution{T}`, so `T` binds at the eleven sites that
# take it from the *type* of the solution as well as at the primitives. A container therefore has to
# work end to end and not merely be accepted at the door, which is what the solve below is for.

using GeometricOptimizers
using GeometricOptimizers: OptimizerCache, OptimizerSolution,
                           Cayley, Geodesic, check, increase_iteration_number!,
                           solver_step!,
                           _zero, _copy, _similar, _fill!, _manifold_αmax, l2norm,
                           solution_scale
using NeuralNetworkParameters: NetworkParameters, params, flatten, unflatten, flatlength
using SimpleSolvers: Static
using Test
import Random

const N, n, m = 6, 3, 4

Random.seed!(1234)

const B₀ = randn(N, m)

"""
    initial_parameters(T)

``\\{Y\\} \\cup \\{W, b\\}`` in two layers: a `StiefelManifold` in one and ordinary parameters in the
other. Nested, which is the point — and heterogeneous, so that the container is not merely a wrapper
around something uniform.
"""
function initial_parameters(::Type{T}) where {T}
    NetworkParameters((
        attention = (Y = rand(Random.Xoshiro(1234), StiefelManifold{T}, N, n),),
        dense = (W = randn(Random.Xoshiro(5678), T, n, m), b = zeros(T, N))
    ))
end

# The same parameters as one flat `NamedTuple`, in the order the container flattens them. This is the
# reference the container is compared against: the two describe the same problem and the same flat
# vector, so an optimizer must take the same steps on both.
function flat_parameters(::Type{T}) where {T}
    let ps = initial_parameters(T)
        NetworkParameters((Y = ps.attention.Y, W = ps.dense.W, b = ps.dense.b))
    end
end

_loss(Y, W, b, B) = sum(abs2, Y * W .+ b .- B) / 2

# Both shapes are a `NetworkParameters`, so the branch is on the *keys* and not on the type: that is
# the whole point of the comparison below, which needs one objective that both can be handed.
function test_problem(::Type{T}) where {T}
    let B = T.(B₀)
        ps -> haskey(params(ps), :attention) ?
              _loss(ps.attention.Y, ps.dense.W, ps.dense.b, B) :
              _loss(ps.Y, ps.W, ps.b, B)
    end
end

"""
    optimize(ps₀, F, algorithm; kwargs...)

Take `steps` steps and return the parameters, `check` of the manifold block after every one of them,
and the objective before and after each.

The seed is fixed per run because the [`GlobalSection`](@ref) is drawn at random and the iterates
depend on it — the same reason `test/flat_parameters.jl` gives.
"""
function optimize(ps, F, algorithm; steps = 20, η = 0.1, retraction = Cayley())
    T = typeof(F(ps))
    Random.seed!(1234)
    optimizer = Optimizer(ps, F; algorithm = algorithm, linesearch = Static(T(η)),
        retraction = retraction)
    state = OptimizerState(algorithm, ps)

    checks = T[]
    losses = T[F(ps)]
    for _ in 1:steps
        increase_iteration_number!(state)
        solver_step!(ps, state, optimizer)
        update!(state, optimizer, ps)
        push!(checks, check(haskey(params(ps), :attention) ? ps.attention.Y : ps.Y))
        push!(losses, F(ps))
    end

    ps, checks, losses
end

# see `test/flat_parameters.jl` for why `Adam` is constructed with the element type
algorithms(::Type{T}) where {T} = (GradientMethod(), MomentumMethod(T(0.1)), Adam(T))
retractions() = (Geodesic(), Cayley())

# see the note on `MANIFOLD_TOLERANCE_IN_EPS` in `test/flat_parameters.jl`: a round-off
# tolerance with a factor of ten in hand, where leaving the manifold is an error of the step size
const MANIFOLD_TOLERANCE_IN_EPS = 100

@testset "a nested container is an OptimizerSolution" begin
    for T in (Float64, Float32)
        ps = initial_parameters(T)
        @test ps isa OptimizerSolution{T}
        @test ps isa NetworkParameters{T}
        # ... and the nesting is real, i.e. this is not the flat case in disguise
        @test all(v -> v isa NamedTuple, values(ps))
    end
end

# The primitives that build every cache and state. Each one used to be `map`, which on a container
# hands the function a whole *layer* — for which `zero`, `copy` and `similar` have no method that
# means anything — so these failed before the walk changed.
@testset "the out-of-place primitives return a container of the same shape" begin
    for T in (Float64, Float32), f in (_zero, _copy, _similar)

        ps = initial_parameters(T)
        out = f(ps)
        @test out isa NetworkParameters
        @test keys(out) == keys(ps)
        @test keys(out.dense) == keys(ps.dense)
        @test size(out.dense.W) == size(ps.dense.W)
    end

    # `_zero` of a manifold element is its *horizontal lift*, not a zero point, and the container must
    # not flatten that distinction away: the lift's free-parameter count is the intrinsic dimension.
    # This is what `Q` is sized by, so it is load-bearing rather than a curiosity.
    ps = initial_parameters(Float64)
    @test _zero(ps).attention.Y isa GeometricOptimizers.AbstractLieAlgHorMatrix
    @test flatlength(_zero(ps)) < flatlength(ps)
end

# `l2norm` is the norm of the flattening — every stopping criterion of `solve!` is computed from it,
# so a container that combined its *layers* instead of its leaves would silently change all of them.
@testset "l2norm and solution_scale reach the leaves of a nested container" begin
    for T in (Float64, Float32)
        ps = initial_parameters(T)
        v, _ = flatten(ps)
        @test l2norm(ps) ≈ l2norm(v)
        @test l2norm(ps) ≈ l2norm(flat_parameters(T))
        # `solution_scale` uses the *nominal* √n for the manifold block and the measured norm for the
        # others, so it is not `l2norm` — but it has to agree between the two shapes.
        @test solution_scale(ps) ≈ solution_scale(flat_parameters(T))
    end
end

# Issue A1b's ceiling, one level down. `_block_αmax` used to answer `Inf` for anything that is not a
# `Manifold`, and every top-level value of a container is a *layer* — so a container would have got no
# ceiling at all for exactly the parameter shape a network has.
@testset "a manifold block one level down still supplies a step ceiling" begin
    T = Float64
    ps = initial_parameters(T)
    δ = _fill!(_zero(ps), T(0.1))
    flat_ps = flat_parameters(T)
    flat_δ = _fill!(_zero(flat_ps), T(0.1))

    αmax = _manifold_αmax(ps, δ, one(T))
    @test isfinite(αmax)
    @test αmax > zero(T)
    # the same ceiling the flat shape gets, which is the statement that nesting changes nothing
    @test αmax ≈ _manifold_αmax(flat_ps, flat_δ, one(T))
end

# The property the whole exercise is about, for every algorithm, retraction and element type.
@testset "the manifold property is preserved during the optimization" begin
    for T in (Float64, Float32), algorithm in algorithms(T), retraction in retractions()
        tol = MANIFOLD_TOLERANCE_IN_EPS * eps(T)
        F = test_problem(T)
        _, checks, losses = optimize(initial_parameters(T), F, algorithm; retraction = retraction)
        @test check(initial_parameters(T).attention.Y) < tol
        @test maximum(checks) < tol
        @test last(losses) < first(losses) / 2
    end
end

# The statement that the swap is behaviour-preserving: the nested and the flat container are the
# same problem written two ways, so the optimizer has to take the same steps on both.
@testset "a nested container reaches the same iterates as the equivalent flat one" begin
    for T in (Float64, Float32), algorithm in algorithms(T)

        F = test_problem(T)
        psₙ, _, lossesₙ = optimize(initial_parameters(T), F, algorithm)
        psₐ, _, lossesₐ = optimize(flat_parameters(T), F, algorithm)

        @test psₙ.attention.Y ≈ psₐ.Y
        @test psₙ.dense.W ≈ psₐ.W
        @test psₙ.dense.b ≈ psₐ.b
        @test lossesₙ ≈ lossesₐ
    end
end

# `solve!` is the only path that builds an `OptimizerStatus`, i.e. the only one that calls `l2norm`
# and `solution_scale` on the parameters and `_difference!` on the gradient blocks.
@testset "solve! runs on a container" begin
    for T in (Float64, Float32), algorithm in algorithms(T)

        Random.seed!(1234)
        F = test_problem(T)
        ps = initial_parameters(T)
        f₀ = F(ps)
        optimizer = Optimizer(ps, F; algorithm = algorithm, linesearch = Static(T(0.1)),
            max_iterations = 100)
        result = solve!(ps, OptimizerState(algorithm, ps), optimizer)

        @test result.f < f₀
        @test F(ps) == result.f
        @test check(ps.attention.Y) < MANIFOLD_TOLERANCE_IN_EPS * eps(T)
        @test !isnan(result.status.rg)
        @test result.status.rg ≥ 0
    end
end

# The quasi-Newton methods are the ones that exercise `outer!`, `_mul!`, `alloc_h` and
# `flatlength(_zero(x))` — every one of which is sized by the *intrinsic* dimension of the parameters
# and not by `length`.
@testset "$(nameof(typeof(algorithm))) runs on a container" for algorithm in (BFGS(), DFP())
    T = Float64
    Random.seed!(1234)
    F = test_problem(T)
    ps = initial_parameters(T)
    f₀ = F(ps)

    # `Q` is sized by the flattening of the *direction*, i.e. of the horizontal lift
    @test size(OptimizerState(algorithm, ps).Q) ==
          (flatlength(_zero(ps)), flatlength(_zero(ps)))

    optimizer = Optimizer(ps, F; algorithm = algorithm, max_iterations = 50)
    result = solve!(ps, OptimizerState(algorithm, ps), optimizer)

    @test result.f < f₀
    @test check(ps.attention.Y) < MANIFOLD_TOLERANCE_IN_EPS * eps(T)
end
