# Which two objective values `OptimizerStatus` subtracts, per optimizer method.
#
# Run with the repository as the active project:
#
#     julia --startup-file=no --project=. scripts/optimizer_status_delta_f.jl
#
# This is the check behind the fix of issue A24, which `CHANGELOG.md` records. It needs no warm-up
# and no cold process: every figure it prints is an exact equality between two stored numbers, not a
# timing. `test/optimizer_state_semantics.jl` asserts the same equality.
#
# ## What it compares
#
# `OptimizerStatus` computes `Δf = f - previous_value(state)`, and `Δf` feeds `rfₐ` and `rfᵣ`, which
# decide `f_converged`. So which value the state holds as the previous objective when the status is
# built is a convergence question.
#
# `store_trace = true` records one objective per iteration, so the trace is an independent record of
# what the objective did. The script asks whether the `Δf` the status reports is the one-step
# difference `f[end] - f[end-1]` or the two-step difference `f[end] - f[end-2]`.
#
# ## Why every method spans one step
#
# Every state holds a pair, and `solve!` advances it to the new iterate before it builds the status,
# so `previous_value(state)` is the objective one step back.

using GeometricOptimizers
using GeometricOptimizers: trace, status
using Printf

# A smooth convex objective with a strict minimum at the origin, so every method makes progress and
# the trace is monotone. Nothing here depends on the particular function.
objective(x) = sum(x .^ 4) + sum(x .^ 2)

const ITERATIONS = 6

@printf("%-18s  %-9s  %-13s  %-13s  %s\n",
    "method", "Δf spans", "one-step", "two-step", "status.Δf")

for method in (GradientMethod(), MomentumMethod(), Adam(), BFGS(), DFP(), Newton())
    name = string(nameof(typeof(method)))
    x = [1.0, 2.0, 3.0]

    state = OptimizerState(method, x)
    opt = Optimizer(x, objective;
        algorithm = method, store_trace = true, max_iterations = ITERATIONS)
    result = solve!(x, state, opt)

    entries = trace(result)
    if length(entries) < 3
        @printf("%-18s  only %d trace entries\n", name, length(entries))
        continue
    end

    one_step = entries[end].f - entries[end - 1].f
    two_step = entries[end].f - entries[end - 2].f
    Δf = status(result).Δf

    # exact equality, not `≈`: the status subtracts two numbers the trace also holds, so a match is
    # bit for bit or it is not the same pair of numbers
    spans = Δf == one_step ? "one step" : (Δf == two_step ? "TWO steps" : "neither")

    @printf("%-18s  %-9s  %-13.6g  %-13.6g  %.6g\n", name, spans, one_step, two_step, Δf)
end

# `ScalarMomentAdam` is absent on purpose: it accepts one `StiefelManifold` and refuses the plain
# vector every other method here takes, so it cannot join this table without a second fixture. Its
# `update!` has the same shape as `AdamState`'s at `scalar_moment_adam_optimizer.jl:227`.
