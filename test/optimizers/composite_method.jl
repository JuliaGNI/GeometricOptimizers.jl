using GeometricOptimizers
using GeometricOptimizers: check, default_step_size, iteration_number, leafmethod,
                           step_size,
                           AdamCache, ScalarMomentAdamCache, DEFAULT_LEARNING_RATE
using NeuralNetworkParameters: NetworkParameters
using Test
import Random

include("../helpers/eltypes.jl")
include("../helpers/allocations.jl")
include("../helpers/parameter_set_step.jl")

# A `CompositeMethod` steps every leaf of a parameter set with the method of that leaf. What is pinned
# here: each leaf gets the right method, each leaf moves exactly as under a `TrainingOptimizer` of its
# own, and a refused construction or step changes nothing.
Random.seed!(1234)

# A Stiefel weight beside a matrix, a vector and a `SymmetricMatrix`, nested one level as a network's
# parameters are. The seed is fixed inside, so that two calls give two copies of one point, and the
# optimizers built straight after them draw the same section completions.
function mixed_parameters(::Type{T}; N = 6, n = 2) where {T}
    Random.seed!(7)
    NetworkParameters((L1 = (weight = rand(StiefelManifold{T}, N, n),),
        L2 = (W = rand(T, 3, 2), b = rand(T, 3), S = rand(SymmetricMatrix{T}, 3))))
end

minibatch_gradient(ps, k) = random_gradient(ps, 100 + k)

# The same leaf with an optimizer of its own: a vector or a manifold point as it is, any other leaf
# as a parameter set of one leaf. Its gradient `g` is wrapped exactly when the leaf is.
alone(x::Union{AbstractVector, Manifold}, g = x) = g
alone(x, g = x) = NetworkParameters((leaf = g,))

@testset "the method of a leaf" begin
    method = CompositeMethod(; manifold = ScalarMomentAdam(), array = Adam())
    @test method == CompositeMethod(ScalarMomentAdam(), Adam())
    @test leafmethod(method, rand(StiefelManifold{Float32}, 4, 2)) === method.manifold
    @test leafmethod(method, rand(GrassmannManifold{Float32}, 4, 2)) === method.manifold
    @test leafmethod(method, rand(Float32, 3)) === method.array
    @test leafmethod(method, rand(Float32, 3, 2)) === method.array
    @test leafmethod(method, rand(SymmetricMatrix{Float32}, 3)) === method.array
    # a parameter set is not a leaf: that it needs two methods is why the composite exists
    @test_throws MethodError leafmethod(method, mixed_parameters(Float32))
end

@testset "what a composite cannot hold or be given is refused" begin
    # a method that needs a Hessian or a line search, and a composite in a composite
    inner = CompositeMethod(Adam(), Adam())
    for (manifold, array) in ((BFGS(), Adam()), (Adam(), Newton()), (inner, Adam()))
        @test_throws MethodError CompositeMethod(manifold, array)
        @test_throws MethodError CompositeMethod(; manifold, array)
    end

    method = CompositeMethod(Adam(), Adam())
    # a composite steps a parameter set, and a single leaf takes its method directly
    @test_throws MethodError TrainingOptimizer(rand(3); algorithm = method)
    @test_throws MethodError TrainingOptimizer(rand(StiefelManifold, 4, 2); algorithm = method)
    # it is not an `OptimizerMethod`, so `Optimizer` and `solve!` do not take it
    @test_throws TypeError Optimizer(rand(3), sum; algorithm = method)
end

@testset "one cache and one state per leaf, in the element type of the set, $T" for T in REAL_ELTYPES
    ps = mixed_parameters(T)
    opt = TrainingOptimizer(ps; algorithm = CompositeMethod(ScalarMomentAdam(), Adam()))
    @test opt.method.manifold isa ScalarMomentAdam{T}
    @test opt.method.array isa Adam{T}
    @test opt.cache.L1.weight isa ScalarMomentAdamCache{T}
    @test opt.state.L1.weight isa ScalarMomentAdamState{T}
    for name in (:W, :b, :S)
        @test opt.cache.L2[name] isa AdamCache{T}
        @test opt.state.L2[name] isa AdamState{T}
    end

    opt = TrainingOptimizer(ps;
        algorithm = CompositeMethod(ScalarMomentAdam(), MomentumMethod(; α = 0.5)),
        linesearch = 1.0e-2)
    @test opt.method.array isa MomentumMethod{T}
    @test opt.state.L2.b isa MomentumState{T}
end

@testset "the optimizer of a composite is inferred, $T" for T in REAL_ELTYPES
    # so that a caller that builds it and steps in one function dispatches each step statically
    build(ps, method, η) = TrainingOptimizer(ps; algorithm = method, linesearch = η)
    for method in (CompositeMethod(ScalarMomentAdam(), Adam()), CompositeMethod(Adam(), Adam()),
        CompositeMethod(ScalarMomentAdam(), GradientMethod()))
        @test (@inferred build(mixed_parameters(T), method, 1.0e-3)) isa TrainingOptimizer
    end
end

@testset "a composite of one method is that method on the whole set, exactly, $T" for T in REAL_ELTYPES
    # `Adam` pools nothing across leaves, so `Adam` on every leaf reproduces `Adam` on the set to the
    # bit, the random section completions included
    whole = mixed_parameters(T)
    whole_opt = TrainingOptimizer(whole; algorithm = Adam())
    per_leaf = mixed_parameters(T)
    per_leaf_opt = TrainingOptimizer(per_leaf; algorithm = CompositeMethod(Adam(), Adam()))
    start = copy(per_leaf.L2.b)
    for k in 1:5
        optimization_step!(whole, whole_opt, minibatch_gradient(whole, k))
        optimization_step!(per_leaf, per_leaf_opt, minibatch_gradient(per_leaf, k))
    end
    @test per_leaf.L2.b != start
    @test per_leaf.L1.weight.A == whole.L1.weight.A
    @test per_leaf.L2.W == whole.L2.W
    @test per_leaf.L2.b == whole.L2.b
    @test per_leaf.L2.S == whole.L2.S
    @test iteration_number(per_leaf_opt.state.L1.weight) == 5
    @test iteration_number(per_leaf_opt.state.L2.S) == 5
end

@testset "each leaf moves as under a TrainingOptimizer of its own, $T" for T in REAL_ELTYPES
    method = CompositeMethod(ScalarMomentAdam(), GradientMethod())
    ps = mixed_parameters(T)
    opt = TrainingOptimizer(ps; algorithm = method, linesearch = 1.0e-2)

    # the same leaves, drawn from the same seed; each optimizer is built in the order the composite
    # builds its leaves, cache and then state, so that the Stiefel leaf draws the same completions
    copies = mixed_parameters(T)
    leaves = (copies.L1.weight, copies.L2.W, copies.L2.b, copies.L2.S)
    opts = map(
        x -> TrainingOptimizer(alone(x); algorithm = leafmethod(method, x),
            linesearch = 1.0e-2), leaves)

    for k in 1:5
        dp = minibatch_gradient(ps, k)
        optimization_step!(ps, opt, dp)
        gradients = (dp.L1.weight, dp.L2.W, dp.L2.b, dp.L2.S)
        foreach((x, o, g) -> optimization_step!(alone(x), o, alone(x, g)), leaves, opts, gradients)
    end
    @test ps.L1.weight.A == copies.L1.weight.A
    @test ps.L2.W == copies.L2.W
    @test ps.L2.b == copies.L2.b
    @test ps.L2.S == copies.L2.S
    @test check(ps.L1.weight) < 100 * eps(T)
    # the scalar second moment is the Stiefel leaf's own, not one pooled over the set
    @test opt.state.L1.weight.m₂ == opts[1].state.m₂
end

@testset "the leaves share the step size of each step, $T" for T in REAL_ELTYPES
    @test default_step_size(CompositeMethod(ScalarMomentAdam(), Adam())) ==
          DEFAULT_LEARNING_RATE
    disagree = CompositeMethod(ScalarMomentAdam(), GradientMethod())
    @test_throws ArgumentError default_step_size(disagree)
    @test_throws ArgumentError TrainingOptimizer(mixed_parameters(T); algorithm = disagree)

    # the `GradientMethod` leaf moves by exactly `step_size(schedule, t)` times its gradient
    schedule = DecayingStatic(; η₁ = 0.5, η₂ = 0.05, n = 3)
    ps = mixed_parameters(T)
    opt = TrainingOptimizer(ps; algorithm = disagree, linesearch = schedule)
    for t in 1:3
        dp = minibatch_gradient(ps, t)
        b = copy(ps.L2.b)
        optimization_step!(ps, opt, dp)
        @test ps.L2.b == b - step_size(opt.linesearch, t) * dp.L2.b
        @test step_size(opt.linesearch, t) isa T
    end
    @test step_size(opt.linesearch, 1) != step_size(opt.linesearch, 3)
end

@testset "a refused step moves no leaf and no count, $T" for T in REAL_ELTYPES
    ps = mixed_parameters(T)
    before = deepcopy(ps)
    opt = TrainingOptimizer(ps; algorithm = CompositeMethod(ScalarMomentAdam(), Adam()))
    # the last leaf has the wrong size, so a step that checked leaf by leaf would move the others
    wrong_size = NetworkParameters((L1 = (weight = randn(T, 6, 2),),
        L2 = (W = randn(T, 3, 2), b = randn(T, 3), S = rand(SymmetricMatrix{T}, 4))))
    @test_throws DimensionMismatch optimization_step!(ps, opt, wrong_size)
    other = T === Float32 ? Float64 : Float32
    @test_throws ArgumentError optimization_step!(
        ps, opt, minibatch_gradient(mixed_parameters(other), 1))
    @test iteration_number(opt.state.L1.weight) == 0
    @test iteration_number(opt.state.L2.S) == 0
    @test ps.L1.weight.A == before.L1.weight.A && ps.L2.S == before.L2.S
end

@testset "a composite step allocates what its leaves' steps allocate, $T" for T in REAL_ELTYPES
    composite = CompositeMethod(Adam(), Adam())
    # dense leaves, a structured one among them: `Adam` steps these without allocating, and so does
    # the composite
    dense = NetworkParameters((L1 = (W = rand(T, 3, 2), b = rand(T, 3)),
        L2 = (S = rand(SymmetricMatrix{T}, 3),)))
    @test step_allocations(
        dense, TrainingOptimizer(dense; algorithm = composite), random_gradient(dense)) == 0

    # a Stiefel leaf: its retraction allocates, the same under the composite as under `Adam` on the
    # set. At N = 60, n = 5 one `N × 2n` temporary is 2 400 bytes in `Float32`, above the tolerance.
    whole = mixed_parameters(T; N = 60, n = 5)
    whole_bytes = step_allocations(
        whole, TrainingOptimizer(whole; algorithm = Adam()), random_gradient(whole))
    per_leaf = mixed_parameters(T; N = 60, n = 5)
    per_leaf_bytes = step_allocations(
        per_leaf, TrainingOptimizer(per_leaf; algorithm = composite), random_gradient(per_leaf))
    @test whole_bytes > 0
    @test n_independent(per_leaf_bytes, whole_bytes)
end
