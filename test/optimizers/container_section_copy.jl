# `_copyto!` across a container and the two section shapes, on the corner where they meet.
#
# Every pair of `_copyto!` methods in `named_tuple_wrapper.jl` is written flat-then-nested:
# `GlobalSectionNamedTuple` for a flat section tree and a bare `NamedTuple` for a nested one, because
# "a `NamedTuple` of `GlobalSection`s to any depth" is a recursive type Julia cannot express.
#
# What makes each pair *order* itself is that the parameter side is a named type: a
# `NetworkParameters` is not a `NamedTuple`, so `(::GlobalSectionNamedTuple{T}, ::NetworkParameters{T})`
# is strictly more specific than `(::NamedTuple, ::NetworkParameters)` and dispatch has somewhere to
# go. Were a parameter set allowed to be a bare `NamedTuple` as well, neither method of a pair would
# win on the overlap and each of the four calls below would be a `MethodError: … is ambiguous`. This
# file is what holds that property down.
#
# The four shapes are the middle of the overlap, which the rest of the suite steps around:
# `network_parameters_optimizer.jl` drives a **nested** container, whose section tree is a `NamedTuple`
# of `NamedTuple`s and so is not a `GlobalSectionNamedTuple`; `flat_parameters.jl` drives a flat set
# but never pairs it with a section by hand. Flat-and-wrapped beside a *flat* section tree is the
# corner between them.

using GeometricOptimizers
using GeometricOptimizers: _copyto!, GlobalSection, GlobalSectionNamedTuple, check
using NeuralNetworkParameters
using Test
import Random

include("../helpers/eltypes.jl")

# for the frame each `GlobalSection` completes, which draws from the global RNG; the data below are
# drawn from their own seeded generators
Random.seed!(1234)

# The method pairs are written on `GlobalSectionNamedTuple{T}` and `NetworkParameters{T}`, so the
# element type is part of what dispatch matches: the shapes are built, and copied, in each precision.
@testset "the four shapes on the overlap dispatch to exactly one method, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(1234)
    nt = (a = rand(rng, T, 3), b = rand(rng, T, 2, 2))
    np = NetworkParameters((a = rand(rng, T, 3), b = rand(rng, T, 2, 2)))
    sec = (a = GlobalSection(rand(rng, T, 3)), b = GlobalSection(rand(rng, T, 2, 2)))

    # `deepcopy` throughout: `_copyto!` writes into its first argument, and a shape that silently
    # aliased another would make the next case pass for the wrong reason.
    @test _copyto!(deepcopy(np), sec) isa NetworkParameters
    @test _copyto!(deepcopy(sec), np) isa GlobalSectionNamedTuple
    @test _copyto!(deepcopy(nt), np) isa NamedTuple
    @test _copyto!(deepcopy(np), nt) isa NetworkParameters

    # and the values actually move, so the method that wins is one that copies rather than one that
    # happens to return the right type
    dest = NetworkParameters((a = zeros(T, 3), b = zeros(T, 2, 2)))
    _copyto!(dest, nt)
    @test eltype(flatten(dest)[1]) == T
    @test dest.a == nt.a
    @test dest.b == nt.b

    dest_nt = (a = zeros(T, 3), b = zeros(T, 2, 2))
    _copyto!(dest_nt, np)
    @test dest_nt.a == np.a
    @test dest_nt.b == np.b

    # a section copy moves the anchors
    dest_sec = (a = GlobalSection(zeros(T, 3)), b = GlobalSection(zeros(T, 2, 2)))
    _copyto!(dest_sec, np)
    @test eltype(dest_sec.b.Y) == T
    @test dest_sec.a.Y == np.a
    @test dest_sec.b.Y == np.b
end

# The end-to-end version of the same corner: a *flat* `NetworkParameters` with a manifold leaf, driven
# the way `GMLDatasets`' MNIST scripts drive an optimizer -- `solver_step!` on a changing objective
# rather than `solve!` on a fixed one. The first step goes through
# `_copyto!(solution(cache(opt)), section(cache(opt)))`, which is exactly the pairing above, so this is
# a consumer's shape rather than a constructed one.
@testset "a flat `NetworkParameters` with a manifold leaf takes a step, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(64)
    ps = NetworkParameters((PQ = rand(rng, StiefelManifold{T}, 6, 3),
        W = rand(rng, T, 4, 4),
        b = zeros(T, 4)))
    F(p) = sum(abs2, flatten(p)[1])
    before = F(ps)

    opt = Optimizer(ps, F; algorithm = GradientMethod(),
        linesearch = GeometricOptimizers.Static(0.01))
    state = OptimizerState(GradientMethod(), ps)
    GeometricOptimizers.initialize_state!(state)
    for _ in 1:5
        GeometricOptimizers.increase_iteration_number!(state)
        GeometricOptimizers.solver_step!(ps, state, opt)
        GeometricOptimizers.update!(state, opt, ps)
    end

    @test eltype(flatten(ps)[1]) == T
    @test F(ps) < before                        # it optimized rather than merely survived
    @test ps.PQ isa StiefelManifold{T}          # and the leaf type did not drift
    # and the iterate is still on the manifold: the round-off of five Cayley steps on a 6 × 3 point,
    # measured at most `3.1eps(T)` over five seeds in both precisions
    @test check(ps.PQ) < 10eps(T)
end
