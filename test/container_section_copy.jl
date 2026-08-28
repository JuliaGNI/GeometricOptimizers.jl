# `_copyto!` across the two container halves and the two section shapes.
#
# Every pair of `_copyto!` methods in `named_tuple_wrapper.jl` is written flat-then-nested:
# `GlobalSectionNamedTuple` for a flat section tree and a bare `NamedTuple` for a nested one, because
# "a `NamedTuple` of `GlobalSection`s to any depth" is a recursive type Julia cannot express. Both of
# those aliases are `NamedTuple`, and `NetworkParameters` is also a `ParameterContainer`, so on the
# overlap neither method of a pair is more specific and the call is a run-time `MethodError: … is
# ambiguous`. Four shapes hit it, and 0.6.0 shipped all four.
#
# The suite missed them because it tested either side of the overlap and never the middle:
# `network_parameters_optimizer.jl` drives a **nested** container, whose section tree is a `NamedTuple`
# of `NamedTuple`s and so is not a `GlobalSectionNamedTuple`; `named_tuple_parameters.jl` drives a
# **bare flat** `NamedTuple`, which is not a `NetworkParameters`. Flat-and-wrapped is the corner
# between them.

using GeometricOptimizers
using GeometricOptimizers: _copyto!, GlobalSection, ArrayNamedTuple, GlobalSectionNamedTuple
using NeuralNetworkParameters
using Test
import Random

Random.seed!(1234)

const nt  = (a = rand(3), b = rand(2, 2))
const np  = NetworkParameters((a = rand(3), b = rand(2, 2)))
const sec = (a = GlobalSection(rand(3)), b = GlobalSection(rand(2, 2)))

@testset "the four shapes on the alias overlap dispatch to exactly one method" begin
    # `deepcopy` throughout: `_copyto!` writes into its first argument, and a shape that silently
    # aliased another would make the next case pass for the wrong reason.
    @test _copyto!(deepcopy(np), sec) isa NetworkParameters
    @test _copyto!(deepcopy(sec), np) isa GlobalSectionNamedTuple
    @test _copyto!(deepcopy(nt), np) isa NamedTuple
    @test _copyto!(deepcopy(np), nt) isa NetworkParameters

    # and the values actually move, so the disambiguation picked a body that copies rather than one
    # that happens to return the right type
    dest = NetworkParameters((a = zeros(3), b = zeros(2, 2)))
    _copyto!(dest, nt)
    @test dest.a == nt.a
    @test dest.b == nt.b

    dest_nt = (a = zeros(3), b = zeros(2, 2))
    _copyto!(dest_nt, np)
    @test dest_nt.a == np.a
    @test dest_nt.b == np.b

    # a section copy moves the anchors
    dest_sec = (a = GlobalSection(zeros(3)), b = GlobalSection(zeros(2, 2)))
    _copyto!(dest_sec, np)
    @test dest_sec.a.Y == np.a
    @test dest_sec.b.Y == np.b
end

# The residue is recorded in `named_tuple_wrapper.jl` rather than tested: the pairs
# `Test.detect_ambiguities` still reports in this family all intersect at shapes this package's API
# cannot build -- the empty `NamedTuple`, a mixed-element-type section, and a `Manifold`-anchored
# section with no lift. Asserting that something *is* ambiguous would pin a wart rather than a
# guarantee.

# The shape that found all of this: a *flat* `NetworkParameters` with a manifold leaf, driven the way
# `GMLDatasets`' MNIST scripts drive an optimizer -- `solver_step!` on a changing objective rather than
# `solve!` on a fixed one. Before the disambiguation this died on the first step, inside
# `_copyto!(solution(cache(opt)), section(cache(opt)))`.
@testset "a flat `NetworkParameters` with a manifold leaf takes a step" begin
    ps = NetworkParameters((PQ = rand(StiefelManifold{Float64}, 6, 3),
                            W  = rand(4, 4),
                            b  = zeros(4)))
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

    @test F(ps) < before                        # it optimized rather than merely survived
    @test ps.PQ isa StiefelManifold             # and the leaf type did not drift
    @test check(ps.PQ) < 1e-10                  # and the iterate is still on the manifold
end
