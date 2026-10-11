# What one `optimization_step!` of a `TrainingOptimizer` allocates on a parameter set, and where.
#
#     julia --startup-file=no --project=. --check-bounds=auto scripts/parameter_set_step_cost.jl [profile]
#
# The parameter set is "model A": eight `Dense(64, 64)` layers and one `Dense(64, 1)`, 33 345
# parameters, built by hand so that the script needs nothing beyond this package. The second set
# replaces `L4.W` with a `64 × 8` `StiefelManifold` and `L5.W` with a `64 × 64` `SymmetricMatrix`, so
# it holds a dense, a structured and a manifold leaf. The gradient is a random set of the shape of
# the parameters, with a dense matrix where the parameter is a manifold point. Model A, the gradient
# and the barrier are the test fixtures of `test/helpers/parameter_set_step.jl`.
#
# It needs no `BenchmarkTools`, `AbstractNeuralNetworks` or `Zygote`: each figure is the minimum of
# five `@allocated` readings behind a function barrier, after two warm-up steps, and the time the
# minimum of 200 calls. BLAS runs on one thread.
#
# With the argument `profile`, `Profile.Allocs` (sample rate 1, 50 warm steps) names the innermost
# frame of this package or of `NeuralNetworkParameters` behind each allocation of one `GradientMethod`
# step on the second set.
using GeometricOptimizers
using LinearAlgebra: BLAS
using NeuralNetworkParameters: NeuralNetworkParameters, NetworkParameters
using Profile: Profile
using Random: Xoshiro

include(joinpath(@__DIR__, "..", "test", "helpers", "parameter_set_step.jl"))

BLAS.set_num_threads(1)

function model_A_structured(::Type{T}) where {T}
    rng = Xoshiro(5)
    ps = model_A(T)
    p = NeuralNetworkParameters.params(ps)
    NetworkParameters(merge(p,
        (L4 = (W = rand(rng, StiefelManifold{T}, 64, 8), b = p.L4.b),
            L5 = (W = rand(rng, SymmetricMatrix{T}, 64), b = p.L5.b))))
end

const ALGORITHMS = ("GradientMethod" => () -> (algorithm = GradientMethod(),),
    "MomentumMethod" => () -> (algorithm = MomentumMethod(),),
    "Adam" => () -> (algorithm = Adam(),),
    "AdamWithEuclideanDecay" => () -> (algorithm = AdamWithEuclideanDecay(),),
    "AdamOptimizerWithDecay(100)" => () -> AdamOptimizerWithDecay(100))

function measure()
    for T in (Float32, Float64),
        (setname, build) in (("model A", model_A),
            ("model A with Stiefel and Symmetric", model_A_structured))

        for (name, kwargs) in ALGORITHMS
            x = build(T)
            dp = random_gradient(x)
            opt = TrainingOptimizer(x; kwargs()...)
            optimization_step!(x, opt, dp)
            bytes = minimum(step_allocations(x, opt, dp) for _ in 1:5)
            time = minimum(@elapsed(optimization_step!(x, opt, dp)) for _ in 1:200)
            println(rpad(name, 29), rpad(T, 9), rpad(setname, 36), lpad(bytes, 9),
                " bytes/step   ", lpad(round(time * 1e6; digits = 1), 7), " μs/step")
        end
    end
end

function profile()
    x = model_A_structured(Float32)
    dp = random_gradient(x)
    opt = TrainingOptimizer(x; algorithm = GradientMethod())
    for _ in 1:3
        optimization_step!(x, opt, dp)
    end
    steps(x, opt, dp, n) = (for _ in 1:n
        optimization_step!(x, opt, dp)
    end)
    steps(x, opt, dp, 1)
    Profile.Allocs.clear()
    Profile.Allocs.@profile sample_rate=1 steps(x, opt, dp, 50)
    allocs = Profile.Allocs.fetch().allocs
    own(f) = occursin(r"GeometricOptimizers|NeuralNetworkParameters", string(f.file))
    count = Dict{String, Tuple{Int, Int}}()
    for a in allocs
        frames = filter(own, a.stacktrace)
        key = isempty(frames) ? "?" :
              join(
            ("$(f.func) $(basename(string(f.file))):$(f.line)"
            for f in frames[1:min(end, 3)]), " <- ") * "  [$(a.type)]"
        n, b = get(count, key, (0, 0))
        count[key] = (n + 1, b + a.size)
    end
    for (key, (n, b)) in sort(collect(count); by = c -> last(c)[2], rev = true)
        println("per step: ", n / 50, " allocations, ", b ÷ 50, " bytes   ", key)
    end
end

isempty(ARGS) ? measure() : ARGS[1] == "profile" ? profile() : error("unknown argument")
