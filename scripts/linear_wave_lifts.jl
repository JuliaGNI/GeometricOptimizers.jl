# Records the arguments `X` that `𝔄` is handed while a symplectic autoencoder is trained on the
# linear wave equation, and writes a sample of them to `scripts/data/`. Those files are the
# realistic half of the exponential benchmark in `exponential_kernels.jl`; the other half is the
# synthetic `sweep` of `retraction_sweep.jl`.
#
# The network, the data and the optimizer are those of the linear-wave experiment of the
# symplectic-autoencoder paper, which live in `scripts/linear_wave/` of GMLDatasets.jl: a
# symplectic autoencoder from `2N = 256` to `2n`, warm-started at the PSD of its dimension, trained
# with Adam at `η = 10⁻³` on batches of 512 snapshots. The one change is the retraction, which is
# `Geodesic` around a recorder instead of `Cayley`, because a Cayley step never calls `𝔄`.
#
# It cannot run in this repository's environment, because GeometricMachineLearning requires an
# older GeometricOptimizers. Run it in an environment that holds GMLDatasets (with its
# `scripts/linear_wave/`), GeometricMachineLearning, GeometricOptimizers, GeometricProblems,
# AbstractNeuralNetworks and HDF5:
#
#     LW_HALF_REDUCED_DIMS=5 julia --project=<env> scripts/linear_wave_lifts.jl
#
# and once per `n` of `HALF_REDUCED_DIMS` (the files are per `n`, so the runs can be parallel).
# The recorded matrices do not depend on the GeometricOptimizers version beyond its Adam step, so
# the benchmark reads them under any version.
#
# For each `n` it writes `linear_wave_lifts_nXX.csv`: every `SAMPLE_EVERY`-th recorded `X`, one per
# line, as `call,m,X[1,1],X[2,1],…` in column-major order, written with `%.17g` so that it reads back
# bit for bit. It prints the range of `‖X‖₁` and of the one-norm of the lower-left block `C` over every
# call, not only the sampled ones: `‖X‖₁` is at least one from the identity block whatever the lift, so
# `‖C‖₁`, which is quadratic in the lift, is the one that says how large the steps were.

using GMLDatasets
using GeometricOptimizers
using GeometricOptimizers: AbstractExponentialAlgorithm, ScaledSquaring, opnorm₁
using Printf

include(joinpath(pkgdir(GMLDatasets), "scripts", "linear_wave", "common.jl"))

const EPOCHS = parse(Int, get(ENV, "LW_LIFT_EPOCHS", "200"))
const SAMPLE_EVERY = parse(Int, get(ENV, "LW_LIFT_SAMPLE_EVERY", "100"))
const DATA_DIR = joinpath(@__DIR__, "data")

# Records a copy of every argument and hands it on to `algorithm`. An algorithm with no `𝔄!` method
# of its own is reached through the fallback `𝔄!(ws, X, ::AbstractExponentialAlgorithm)`, which calls
# this `𝔄`, so the in-place path of an optimizer is recorded too.
struct Recording{AT <: AbstractExponentialAlgorithm} <: AbstractExponentialAlgorithm
    algorithm::AT
    lifts::Vector{Matrix{Float64}}
end

function GeometricOptimizers.𝔄(X::AbstractMatrix, r::Recording)
    push!(r.lifts, Matrix{Float64}(X))
    GeometricOptimizers.𝔄(X, r.algorithm)
end

data = training_data()
dl = DataLoader(data; autoencoder = true, suppress_info = true)
mkpath(DATA_DIR)

for n in HALF_REDUCED_DIMS
    Random.seed!(first(SEEDS))
    network = NeuralNetwork(architecture(n), CPU(), Float64)
    warm_start!(network, fit_psd(data, n))

    recorder = Recording(ScaledSquaring(), Matrix{Float64}[])
    optimizer = TrainingOptimizer(network.params;
        algorithm = Adam(; β₁ = ADAM_BETA1, β₂ = ADAM_BETA2, δ = ADAM_DELTA),
        linesearch = STEP_SIZE, retraction = Geodesic(recorder))
    losses = train!(
        network, optimizer, dl, Batch(BATCH_SIZE), EPOCHS; show_progress = false)

    lifts = recorder.lifts
    isempty(lifts) && error("no call to 𝔄 was recorded for n = $n")
    open(joinpath(DATA_DIR, @sprintf("linear_wave_lifts_n%02d.csv", n)), "w") do io
        for k in 1:SAMPLE_EVERY:length(lifts)
            X = lifts[k]
            print(io, k, ',', size(X, 1))
            foreach(x -> print(io, ',', @sprintf("%.17g", x)), X)
            println(io)
        end
    end
    C = [opnorm₁(X[(n + 1):end, 1:n]) for X in lifts]
    @printf("n = %2d: %d calls to 𝔄 in %d epochs, loss %.3e → %.3e, ‖X‖₁ in [%.3e, %.3e], ‖C‖₁ in [%.2e, %.2e]\n",
        n, length(lifts), EPOCHS, first(losses), last(losses),
        minimum(opnorm₁, lifts), maximum(opnorm₁, lifts), minimum(C), maximum(C))
end
