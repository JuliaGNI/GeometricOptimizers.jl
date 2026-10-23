# The bytes of one `CompositeMethod` training step, against the same step of `Adam` on the whole set,
# on dense leaves and on a set with a Stiefel leaf, in both precisions. The `[compat]` floor of
# NeuralNetworkParameters rests on the dense row: run this once per NeuralNetworkParameters version.
#
#     julia --startup-file=no --project=<an environment with GeometricOptimizers> scripts/composite_step_cost.jl
using GeometricOptimizers
using NeuralNetworkParameters: NeuralNetworkParameters, NetworkParameters
import Random

include(joinpath(@__DIR__, "..", "test", "helpers", "parameter_set_step.jl"))

println("NeuralNetworkParameters ", pkgversion(NeuralNetworkParameters))
for T in (Float32, Float64)
    function dense()
        NetworkParameters((L1 = (W = rand(T, 3, 2), b = rand(T, 3)),
            L2 = (S = rand(SymmetricMatrix{T}, 3),)))
    end
    function mixed()
        Random.seed!(7)
        NetworkParameters((L1 = (weight = rand(StiefelManifold{T}, 60, 5),),
            L2 = (W = rand(T, 3, 2), b = rand(T, 3))))
    end
    for (name, fixture) in (("dense", dense), ("Stiefel", mixed))
        bytes = map((Adam(), CompositeMethod(Adam(), Adam()))) do algorithm
            ps = fixture()
            step_allocations(ps, TrainingOptimizer(ps; algorithm), random_gradient(ps))
        end
        println(rpad("$T $name", 18), "Adam: ", lpad(bytes[1], 6), " B   composite: ",
            lpad(bytes[2], 6), " B")
    end
end
