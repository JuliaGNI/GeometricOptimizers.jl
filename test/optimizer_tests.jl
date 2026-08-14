using LinearAlgebra
using NaNMath: log
using GeometricOptimizers
using GeometricOptimizers: Newton, _DFP, _BFGS
using GeometricOptimizers: gradient, hessian, linesearch, problem, initialize!, update!, solver_step!
using GeometricOptimizers: DEFAULT_LEARNING_RATE, default_linesearch
using GeometricOptimizers: iteration_number, increase_iteration_number!, status
using SimpleSolvers: Static, Backtracking, BierlaireQuadratic, Quadratic, Bisection, StrongWolfe, GradientAutodiff, GradientFunction
using Test
using Random
Random.seed!(123)

include("optimizers_problems.jl")

struct OptimizerTest{T} <: OptimizerState{T} end

test_optim = OptimizerTest{Float64}()
test_x = zeros(3)
test_obj = OptimizerProblem(F, test_x)

@test_throws MethodError gradient(test_optim)
@test_throws MethodError hessian(test_optim)
@test_throws MethodError linesearch(test_optim)
@test_throws MethodError problem(test_optim)

# test if the correct error is thrown when calling `initialize!` on an `OptimizerState`.
# @test_throws ErrorException initialize!(test_optim, test_x)
@test_throws MethodError update!(test_optim, test_x)
@test_throws MethodError solver_step!(test_x, test_optim)

for T in (Float64, Float32)
    for method in (Newton(), _DFP(), _BFGS())
        for _linesearch in (Static(T(0.1)), Backtracking(T), Backtracking(T; expand=true),
            BierlaireQuadratic(T), Quadratic(T), Bisection(T), StrongWolfe(T; c₂=T(0.1)))
            @testset "$(method) & $(_linesearch) & $(T)" begin
                n = 1
                x = ones(T, n)
                opt = Optimizer(x, F; algorithm=method, linesearch=_linesearch)
                state = OptimizerState(method, x)

                @test typeof(gradient(opt)) <: GradientAutodiff

                solve!(x, state, opt)
                @test norm(x) ≈ zero(T) atol = ∛(2000eps(T))
                @test F(x) ≈ F(zero(T)) atol = ∛(2000eps(T))

                x = ones(T, n)
                opt = Optimizer(x, F; (∇F!)=∇F!, algorithm=method, linesearch=_linesearch)

                @test typeof(gradient(opt)) <: GradientFunction

                state = OptimizerState(method, x)

                solve!(x, state, opt)
                @test norm(x) ≈ zero(T) atol = ∛(2000eps(T))
                @test F(x) ≈ F(0) atol = ∛(2000eps(T))
            end
        end
    end
end


# The first-order methods produce a *direction* of their own and nothing more, so the step
# length is entirely the `α` of the line search: for them a fixed learning rate is
# `Static(η)`, which is why it is the default. `Adam` used to carry an `η` field instead, which
# meant the step was scaled twice — once by `η` and once by the line search, whose default was
# `Backtracking` and therefore not `1`.
#
# The methods that come with a Hessian (or an approximation of it) produce a direction whose
# *length* is meaningful, so they keep `Backtracking`, which starts its trial step at `α = 1`
# and shortens it if it has to — and, with `expand = true`, lengthens it if the first trial is
# accepted and longer ones keep improving the merit.
@testset "the default line search matches the method" begin
    for T in (Float64, Float32)
        # The `AdamFamily` methods are the ones that keep a fixed step: their direction is a moving
        # average and is not required to descend on an individual step, so a sufficient-decrease
        # search has nothing to work with. For `AdamWithEuclideanDecay` a fixed step is also what
        # makes `λ` mean what it is documented to mean — the merit does not contain the penalty, so
        # a searching `α` would be picked partly in order to undo the decay, and `αλ` would be
        # whatever the search settled on. See `default_linesearch`.
        for method in (Adam(T), AdamWithEuclideanDecay(T))
            ls = default_linesearch(T, method)
            @test ls isa Static{T}
            @test ls.α == T(DEFAULT_LEARNING_RATE)

            x = ones(T, 3)
            @test linesearch(Optimizer(x, F; algorithm=method)).method isa Static{T}
        end

        # Everything else searches. `GradientMethod` and `MomentumMethod` joined this group in 0.2.0,
        # once a line search could take its trial step through a retraction; before that `Static` was
        # the only thing that worked on manifold parameters, so they had no choice. `_DFP` rejoined it
        # in SimpleSolvers 0.11: it needs a search that can *lengthen* a step, and until `expand` that
        # ruled `Backtracking` out for it entirely (47_115 iterations on the SVD problem, against 702
        # with the expansion phase). See `default_linesearch`.
        for method in (GradientMethod(), MomentumMethod(T(0.1)), Newton(), _BFGS(), _DFP())
            ls = default_linesearch(T, method)
            @test ls isa Backtracking{T}
            # the property the default turns on, and the one a future SimpleSolvers bump could drop
            # silently: without it `_DFP` does not converge on the SVD problem at all
            @test ls.expand

            x = ones(T, 3)
            opt_ls = linesearch(Optimizer(x, F; algorithm=method)).method
            @test opt_ls isa Backtracking{T}
            @test opt_ls.expand
        end
    end

    # `DEFAULT_LEARNING_RATE` is written as a `Float64` literal so that the `Float32` default is
    # `1f-3` and not `Float32(1.0e-3)` rounded through `Float64` — i.e. so that `Static`'s `α`
    # prints as `0.001` for both element types.
    @test default_linesearch(Float32, Adam(Float32)).α === 1.0f-3
    @test default_linesearch(Float32, AdamWithEuclideanDecay(Float32)).α === 1.0f-3

    # `Adam` no longer takes a learning rate, and `β₁`, `β₂` and `δ` are keyword arguments, so
    # an old positional call fails instead of quietly setting `β₁ = 0.01`.
    @test_throws MethodError Adam(0.01)
end

# The loop above covers `Newton`, `_BFGS` and `_DFP` only, and the one above that checks which line
# search the first-order methods *default* to without ever solving with it. Between them they missed
# that `GradientMethod` and `MomentumMethod` threw a `MethodError` on their own default:
# `trial_slope`'s `AbstractVector` branch calls `gradient(cache)`, which only the (quasi-)Newton
# caches defined. So the whole point of this loop is that it solves.
#
# A second, smooth, non-quadratic objective, because `F` is a quadratic and one accurate line search
# solves it exactly -- which is the case where a stale `rg` used to stop these methods one step past
# the minimiser (see `convergence_measures` and the testset below). `Fsmooth` is convex with a unique
# minimiser at `0` and is never solved in one step, so it asserts convergence and not just
# termination. (Not `H`: `optimizers_problems.jl`, included above, calls a *Hessian* `H!`.)
Fsmooth(x) = sum(sqrt.(1 .+ x .^ 2))
∇Fsmooth!(g, x) = (g .= x ./ sqrt.(1 .+ x .^ 2))

@testset "the first-order methods solve on every line search" begin
    for T in (Float64, Float32)
        linesearches = (Static(T(0.1)), Backtracking(T), Backtracking(T; expand=true),
            BierlaireQuadratic(T), Quadratic(T), Bisection(T), StrongWolfe(T; c₂=T(0.1)))
        # `Adam` is in this loop but *not* in `default_linesearch`'s searching group, and the two are
        # not in conflict: a sufficient-decrease search has nothing to work with when the direction is
        # a moving average that is deliberately allowed not to descend, so `AdamFamily` keeps
        # `Static` as its default (asserted above). It still has to *work* when one is passed
        # explicitly, which is what this covers -- and before this branch it threw as well.
        for method in (GradientMethod(), MomentumMethod(T(0.1)), Adam(T)),
            _linesearch in linesearches,
            (name, obj, ∇obj!) in (("F", F, ∇F!), ("Fsmooth", Fsmooth, ∇Fsmooth!))

            @testset "$(method) & $(_linesearch) & $(T) & $(name)" begin
                x = ones(T, 3)
                state = OptimizerState(method, x)
                opt = Optimizer(x, obj; algorithm=method, linesearch=_linesearch, max_iterations=1000)

                solve!(x, state, opt)

                # it terminated on a convergence criterion and not on the iteration cap
                @test iteration_number(state) < 1000
                # and it got there: the worst of the 168 combinations is 1.7e-3 in `Float32`,
                # against a tolerance of 6.2e-2
                @test norm(x) ≈ zero(T) atol = ∛(2000eps(T))

                # and the same with an explicit gradient rather than the autodiff one
                x = ones(T, 3)
                state = OptimizerState(method, x)
                opt = Optimizer(x, obj; (∇F!)=∇obj!, algorithm=method, linesearch=_linesearch,
                    max_iterations=1000)

                solve!(x, state, opt)

                @test iteration_number(state) < 1000
                @test norm(x) ≈ zero(T) atol = ∛(2000eps(T))
            end
        end
    end
end

@testset "a line search does not corrupt the momentum" begin
    # `trial_slope`'s `AbstractVector` branch evaluates the trial gradient *into* the cache, and
    # `update!(::MomentumState, ...)` re-runs `p ← αp + ∇f(xₖ)` from `gradient_array(cache)`
    # afterwards. Sharing one array between the two made the momentum accumulate the gradient at
    # whatever trial step the search last probed: 104% wrong under `Bisection`, `Quadratic` and
    # `StrongWolfe`, 451% under `BierlaireQuadratic`. `Backtracking` was exact by accident, because it
    # evaluates `φ'` only at `α = 0`, so it cannot stand in for the others here.
    #
    # Exact equality, not `isapprox`: both sides are the same two floating-point operations on the
    # same two arrays, so anything but a bit-identical result means a different gradient went in.
    f(x) = sum(x .^ 2 .+ 0.1 .* x .^ 4 .+ 0.3 .* sin.(3x))
    ∇f!(g, x) = (g .= 2 .* x .+ 0.4 .* x .^ 3 .+ 0.9 .* cos.(3x))
    α = 0.1

    for _linesearch in (Bisection(), Quadratic(), BierlaireQuadratic(), StrongWolfe(; c₂=0.1),
        Backtracking(; expand=true), Static(0.1))
        x = [1.5, -0.8, 0.4]
        method = MomentumMethod(α)
        state = OptimizerState(method, x)
        opt = Optimizer(x, f; (∇F!)=∇f!, algorithm=method, linesearch=_linesearch)
        g = similar(x)

        for _ in 1:8
            increase_iteration_number!(state)
            p̄ = copy(state.p)
            ∇f!(g, x)                      # ∇f at the iterate the direction is about to be built at
            solver_step!(x, state, opt)
            update!(state, opt, x)

            @test state.p == α .* p̄ .+ g
        end
    end
end

@testset "the gradient residual is measured at the iterate the solve returns" begin
    # `rg` used to be `‖∇f(xₖ)‖` at the iterate the step *started* from. Harmless under `Static`,
    # where the direction is a scaled gradient, and not harmless at all under one that carries
    # momentum: a line search accurate enough to drive `∇f(x₁) ≈ 0` made `g_converged` fire while the
    # momentum term was still moving the iterate. On `F` from `ones(3)` that left `MomentumMethod` at
    # `‖x‖ = 0.346` and `Adam` at `‖x‖ = 1.16` -- barely moved -- both reporting convergence.
    #
    # The (quasi-)Newton caches were left out of that fix and are covered here too now. For them the
    # stale `rg` was not `‖∇f(xₖ)‖` but `‖∇f‖` at whatever point the line search last probed, because
    # `trial_slope` evaluates into the same array -- on Rosenbrock from `(-1.2, 1)` with the default
    # `Backtracking` that read `5.8e4` times the true residual for `_BFGS` and `299` times it for
    # `_DFP`. That was issue A8; `Backtracking` is in the list below for exactly that reason.
    ∇F(x) = 2 .* x

    for method in (GradientMethod(), MomentumMethod(0.1), Adam(Float64), Newton(), _BFGS(), _DFP()),
        _linesearch in (Backtracking(; expand=true), Bisection(), Quadratic(), BierlaireQuadratic(),
            StrongWolfe(; c₂=0.1))

        x = ones(3)
        state = OptimizerState(method, x)
        result = solve!(x, state, Optimizer(x, F; algorithm=method, linesearch=_linesearch))

        # the residual belongs to the point the solve returns, and not to the one before it
        @test status(result).rg ≈ norm(∇F(x))
        # it stopped on a convergence criterion rather than on the iteration cap
        @test GeometricOptimizers.isconverged(status(result))
        # and it really is at the minimiser: `1.8e-8` is the worst of the thirty, against the `0.346`
        # and `1.16` this used to stop at
        @test norm(x) < 1e-7
    end
end

@testset "the gradient the direction is built from is the gradient at the iterate" begin
    # `solver_step!` refreshes `latest_gradient` at the accepted iterate, and the next
    # `update!(cache, ...)` reuses it instead of evaluating `∇f` a second time at the same point --
    # which is what keeps the refresh from doubling the gradient evaluations of a first-order step.
    # See `store_gradient!`.
    #
    # Exact equality: the reuse is only legitimate if the two are the *same* computation, so anything
    # but a bit-identical result means the reused value belongs to a different point or a different
    # frame. This is the test that catches a future reordering of `solve!` making it stale.
    f(x) = sum(x .^ 2 .+ 0.1 .* x .^ 4 .+ 0.3 .* sin.(3x))
    ∇f!(g, x) = (g .= 2 .* x .+ 0.4 .* x .^ 3 .+ 0.9 .* cos.(3x))

    for method in (GradientMethod(), MomentumMethod(0.1), Adam(Float64), _BFGS(), _DFP()),
        _linesearch in (Static(0.1), Backtracking(; expand=true), Bisection(), Quadratic(),
            BierlaireQuadratic(), StrongWolfe(; c₂=0.1))

        x = [1.5, -0.8, 0.4]
        state = OptimizerState(method, x)
        opt = Optimizer(x, f; (∇F!)=∇f!, algorithm=method, linesearch=_linesearch)
        g = similar(x)

        for k in 1:8
            increase_iteration_number!(state)
            # the reuse is available on every iteration but the first, where the scratch array has
            # never been written and the two `GlobalSection`s are independent draws
            @test GeometricOptimizers.latest_gradient_is_current(GeometricOptimizers.cache(opt), state, x) == (k > 1)

            ∇f!(g, x)
            solver_step!(x, state, opt)
            update!(state, opt, x)

            # whichever branch `store_gradient!` took, the direction was built from ∇f at the iterate
            # the step started from
            @test GeometricOptimizers.gradient_array(GeometricOptimizers.cache(opt)) == g
        end
    end

    # `Newton` is the one method that does not reliably get the reuse, and it is the state's section
    # that denies it: `update!(::NewtonOptimizerState, opt, x)` advances `state.section` by the
    # *gradient* rather than by the direction, so the two frames differ by `∇f - δ` and the guard
    # falls back to a fresh evaluation. That is the gradient evaluation per iteration the refresh
    # costs for `Newton` and for nothing else. It comes back once the solve has converged, where both
    # the gradient and the step have gone to zero and the frames agree again -- which is why what is
    # asserted here is the gradient the direction is built from and not the branch taken to get it.
    for _linesearch in (Static(0.1), Backtracking(; expand=true), Bisection())
        x = [1.5, -0.8, 0.4]
        state = OptimizerState(Newton(), x)
        opt = Optimizer(x, f; (∇F!)=∇f!, algorithm=Newton(), linesearch=_linesearch)
        g = similar(x)

        # not current on the first iteration, where nothing has written the scratch array yet
        @test !GeometricOptimizers.latest_gradient_is_current(GeometricOptimizers.cache(opt), state, x)

        for _ in 1:8
            increase_iteration_number!(state)

            ∇f!(g, x)
            solver_step!(x, state, opt)
            update!(state, opt, x)

            @test GeometricOptimizers.gradient_array(GeometricOptimizers.cache(opt)) == g
        end
    end
end

@testset "a caller that moves the iterate does not get a reused gradient" begin
    # The reuse is guarded on `solution(cache) == x` and `section(cache) == section(state)`, not on
    # the call sequence, so a loop that moves `x` behind the optimizer's back falls back to a fresh
    # evaluation rather than silently building its direction from the gradient at the old point.
    f(x) = sum(x .^ 2 .+ 0.1 .* x .^ 4)
    ∇f!(g, x) = (g .= 2 .* x .+ 0.4 .* x .^ 3)

    x = [1.5, -0.8, 0.4]
    state = OptimizerState(GradientMethod(), x)
    opt = Optimizer(x, f; (∇F!)=∇f!, algorithm=GradientMethod(), linesearch=Bisection())
    g = similar(x)

    for _ in 1:4
        increase_iteration_number!(state)
        solver_step!(x, state, opt)
        update!(state, opt, x)

        x .+= 0.25                     # the move the guard has to notice
        @test !GeometricOptimizers.latest_gradient_is_current(GeometricOptimizers.cache(opt), state, x)

        increase_iteration_number!(state)
        ∇f!(g, x)
        solver_step!(x, state, opt)
        update!(state, opt, x)

        @test GeometricOptimizers.gradient_array(GeometricOptimizers.cache(opt)) == g
    end
end

@testset "the gradient difference is the one the status prints" begin
    # `rgₐ` is `|g(x) - g(x')|`, i.e. the change over the step just taken. For the first-order caches
    # the generic `gradient_difference!` did not produce that: `state.ḡ` is two iterates behind
    # `cache.g` for them, so `rgₐ` was `‖∇f(xₖ) - ∇f(xₖ₋₂)‖` -- on the objective below, `4.976` where
    # the successive difference is `0.295` -- and on the first iteration it differenced against
    # `_similar` memory that `MomentumState` never writes. See `gradient_difference!`.
    #
    # The (quasi-)Newton caches had the same row wrong in their own way and are covered here now:
    # `BFGSCache` and `DFPCache` reported the `γ` of their secant pair, which is one step behind the
    # `rg` next to it, and for `NewtonOptimizerCache` the difference was *structurally zero* --
    # `solver_step!` advances `state.ḡ` at the very iterate the cache takes its gradient at.
    f(x) = sum(x .^ 2 .+ 0.1 .* x .^ 4)
    ∇f(x) = 2 .* x .+ 0.4 .* x .^ 3
    ∇f!(g, x) = (g .= ∇f(x))

    for method in (GradientMethod(), MomentumMethod(0.1), Adam(Float64), Newton(), _BFGS(), _DFP()),
        _linesearch in (Static(0.1), Bisection(), Quadratic())

        x = [1.5, -0.8, 0.4]
        state = OptimizerState(method, x)
        opt = Optimizer(x, f; (∇F!)=∇f!, algorithm=method, linesearch=_linesearch)

        for _ in 1:5
            increase_iteration_number!(state)
            x_before = copy(x)
            solver_step!(x, state, opt)
            _status = GeometricOptimizers.OptimizerStatus(state, GeometricOptimizers.cache(opt),
                f(x); config=GeometricOptimizers.config(opt))

            @test _status.rgₐ ≈ norm(∇f(x) .- ∇f(x_before))
            # and `rg` is the other end of that same step, so the two rows the status prints are
            # about one step and not about two different ones
            @test _status.rg ≈ norm(∇f(x))

            update!(state, opt, x)
        end
    end

    # the first iteration used to read uninitialized memory; a one-iteration solve is the smallest
    # case that reaches it
    for method in (GradientMethod(), MomentumMethod(0.1), Adam(Float64))
        x = ones(3)
        result = solve!(x, OptimizerState(method, x),
            Optimizer(x, F; algorithm=method, linesearch=Static(0.1), max_iterations=1))

        @test isfinite(status(result).rgₐ)
    end
end

@testset "Test Nan handling in optimizers" begin

    fnan(x::T) where {T} = log(x) + x^2
    Fnan(x::AbstractVector) = sum(fnan.(x))

    function test_nan_handling_for_optimizers(F, n::Integer, ::Type{T}; kwargs...) where {T}
        x = 0.2 * ones(T, n)
        opt = Optimizer(x, F; algorithm=Newton(), linesearch=Static(), verbosity=2, kwargs...)
        state = OptimizerState(Newton(), x)
        solve!(x, state, opt)
    end

    @test_warn "NaN or Inf detected in optimizer. Reducing length of direction vector." test_nan_handling_for_optimizers(Fnan, 1, Float64; max_iterations=5)

end

@testset "store_trace records one entry per iteration" begin
    # `Options(store_trace = true)` was accepted and ignored before this -- by this package and by
    # SimpleSolvers 0.11, where the field exists and nothing reads it -- so code that asked for a
    # trace got neither a trace nor an error. See `trace`.
    Fquad(x::AbstractVector) = sum(x .^ 2)
    x = [1.0, 2.0]
    state = OptimizerState(Newton(), x)
    result = solve!(x, state, Optimizer(x, Fquad; algorithm=Newton(), store_trace=true))

    @test length(GeometricOptimizers.trace(result)) == GeometricOptimizers.iteration_number(state)
    @test [entry.iteration for entry in GeometricOptimizers.trace(result)] == 1:GeometricOptimizers.iteration_number(state)

    # the last entry is the status the solve reports, so the trace and the status cannot disagree
    # about where the solve ended up
    @test last(GeometricOptimizers.trace(result)).rg == GeometricOptimizers.status(result).rg
    @test last(GeometricOptimizers.trace(result)).f == minimum(result)

    # and without the option there is no trace and no error
    y = [1.0, 2.0]
    result_untraced = solve!(y, OptimizerState(Newton(), y), Optimizer(y, Fquad; algorithm=Newton()))
    @test isempty(GeometricOptimizers.trace(result_untraced))
end
