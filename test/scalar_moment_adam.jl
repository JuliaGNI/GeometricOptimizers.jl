using GeometricOptimizers
using GeometricOptimizers: cache, direction, first_moment, second_moment, _second_moment,
    gradient, gradient_array, global_rep, increase_iteration_number!, iteration_number, l2norm,
    section, solver_step!, update!
using LinearAlgebra: norm
using Test

# `ScalarMomentAdam` is Cayley ADAM, Algorithm 2 of li2020efficient, in the global tangent space
# representation. Its docstring and the *Optimizer Methods* manual page derive the port; what is
# pinned here is (i) the identity that makes the paper's lines 8-10 *be* `global_rep`, (ii) the scalar
# second moment, which is the whole of the difference from `Adam`, (iii) which norm is squared, and
# (iv) that the accepted step still lands on the manifold. The scope the method advertises is pinned
# too, on the `Optimizer` path and not only on `OptimizerState`; see the testset for why.
#
# The objective is linear, so `∇L = C` at every iterate, which is what makes the recursion checkable
# in closed form. See `test/optimizer_step_formulas.jl`, which does the same for `Adam`.
function linear_stiefel_objective(C)
    Y -> sum(C .* Y.A)
end

# The paper's lines 8-9: `Ŵ = ZYᵀ - ½Y(YᵀZYᵀ)`, `W = Ŵ - Ŵᵀ`.
function paper_W(Y::AbstractMatrix, Z::AbstractMatrix)
    Ŵ = Z * Y' - Y * (Y' * Z * Y') / 2
    Ŵ - Ŵ'
end

@testset "ScalarMomentAdam constructor and scope" begin
    method = ScalarMomentAdam(Float32; β₁=0.8, β₂=0.95, δ=1.0f-6)
    @test method.β₁ isa Float32
    @test method.β₂ isa Float32
    @test method.δ isa Float32
    @test_throws ArgumentError ScalarMomentAdam(; β₁=1.0)
    @test_throws ArgumentError ScalarMomentAdam(; β₂=-0.1)
    @test_throws ArgumentError ScalarMomentAdam(; δ=-1.0)
    @test_throws ArgumentError ScalarMomentAdam(; β₂=1.0)
    @test method.ambient_norm == false
    @test ScalarMomentAdam(; ambient_norm=true).ambient_norm

    @test_throws ArgumentError OptimizerState(method, rand(3))
    @test_throws ArgumentError OptimizerState(method, (Y=rand(StiefelManifold, 4, 2), z=rand(3)))
    @test_throws ArgumentError OptimizerState(method, rand(GrassmannManifold{Float32}, 4, 2))

    # The state carries a gradient too, as `Adam` does through `OptimizerState(::Adam, x...)`.
    Y = rand(StiefelManifold, 4, 2)
    Ḡ = global_rep(GlobalSection(Y), rgrad(Y, randn(4, 2)))
    @test OptimizerState(ScalarMomentAdam(), Y, Ḡ) isa ScalarMomentAdamState
    @test Matrix(gradient(OptimizerState(ScalarMomentAdam(), Y, Ḡ))) ≈ Matrix(Ḡ)

    # The moments are read in the first `update!` before they are written to, so they have to start at
    # zero -- the same invariant `test/optimizer_state_initialization.jl` pins for `AdamState`.
    state = OptimizerState(ScalarMomentAdam(), Y)
    @test all(iszero, Matrix(first_moment(state)))
    @test iszero(second_moment(state))
end

# `OptimizerCache(::ScalarMomentAdam, x)` used to leave its second argument untyped, which made it
# *ambiguous* with the fallback `OptimizerCache(::OptimizerMethod, ::OptimizerSolution{T})` in
# `optimizers/newton_optimizer/newton_optimizer_cache.jl`: neither is more specific, so every one of
# the four cases below raised a `MethodError` about an ambiguity rather than the `ArgumentError` the
# docstring, the manual and the changelog promise. `OptimizerState` was unaffected -- its fallback is
# a `Vararg` method and so *is* less specific -- which is why testing only that one missed this.
@testset "ScalarMomentAdam rejects unsupported parameters through Optimizer" begin
    f(x) = sum(abs2, x)
    for x in (rand(3),
        (Y=rand(StiefelManifold, 4, 2), z=rand(3)),
        rand(GrassmannManifold, 4, 2),
        rand(StiefelManifold{Float32}, 4, 2))
        @test_throws ArgumentError Optimizer(x, f; algorithm=ScalarMomentAdam())
    end
    # ... and the element-type mismatch says so, rather than claiming the manifold is unsupported
    err = try
        Optimizer(rand(StiefelManifold{Float32}, 4, 2), f; algorithm=ScalarMomentAdam())
    catch e
        e
    end
    @test occursin("ScalarMomentAdam(Float32)", err.msg)
end

# The identity the port rests on: the paper's auxiliary matrix, skew-symmetrized, is the horizontal
# lift conjugated by the global section — `W = λ(Y)ᵀ⁻¹ global_rep(λ(Y), Z) λ(Y)ᵀ`. If this fails, the
# implementation is not computing the paper's `W_k` and lines 8-10 would have to be written out.
@testset "the paper's W is the horizontal lift" begin
    Y = rand(StiefelManifold, 7, 3)
    λY = GlobalSection(Y)
    λ = Matrix(λY)
    for Z in (rgrad(Y, randn(7, 3)), randn(7, 3))
        Ḡ = global_rep(λY, Z)
        @test λ * Matrix(Ḡ) * λ' ≈ paper_W(Y.A, Z)
        # ... and the paper's projection `π(Z) = WY` is that lift read back at `Y`: `λᵀY` is the
        # first `n` columns of the identity, so `λḠλᵀY = λḠ[:, 1:n]`.
        @test paper_W(Y.A, Z) * Y.A ≈ λ * Matrix(Ḡ)[:, 1:3]
    end
end

@testset "ScalarMomentAdam moment recursion" begin
    for T in (Float64, Float32)
        rtol = T === Float32 ? 5.0f-5 : 1.0e-12
        Y = rand(StiefelManifold{T}, 5, 2)
        C = T[1 2; -3 4; 2 -1; 1 0; -2 3]
        β₁, β₂, δ = T(0.5), T(0.25), T(0.1)
        method = ScalarMomentAdam(T; β₁, β₂, δ)
        opt = Optimizer(Y, linear_stiefel_objective(C); algorithm=method,
            linesearch=Static(T(0.01)), retraction=Cayley())
        state = OptimizerState(method, Y)

        # ---- the first update: `t = 1` makes both bias-correction factors `1` ----
        increase_iteration_number!(state)
        update!(cache(opt), state, gradient(opt), method, Y)
        Ḡ = global_rep(section(state), rgrad(Y, C))

        @test Matrix(first_moment(cache(opt))) ≈ Matrix(Ḡ) rtol = rtol
        @test first_moment(cache(opt)) isa StiefelLieAlgHorMatrix
        # the scalar second moment: a squared gradient *norm*, not a squared gradient
        @test second_moment(cache(opt)) isa T
        @test second_moment(cache(opt)) ≈ l2norm(Ḡ)^2 rtol = rtol
        # `δ` goes *inside* the root, as in the paper's line 7
        @test _second_moment(cache(opt)) ≈ √(l2norm(Ḡ)^2 + δ) rtol = rtol
        # the direction is `-m̂/√(v̂ + δ)` and carries no learning rate
        @test Matrix(direction(cache(opt))) ≈ -Matrix(Ḡ) ./ √(l2norm(Ḡ)^2 + δ) rtol = rtol

        # ---- the second update: both factors now bite ----
        m₁_prev = copy(Matrix(first_moment(cache(opt))))
        m₂_prev = second_moment(cache(opt))
        solver_step!(Y, state, opt)
        update!(state, opt, Y)
        # the moments the state carries into the next update are the cache's, unscaled by `α`
        @test second_moment(state) == m₂_prev
        @test Matrix(first_moment(state)) ≈ m₁_prev rtol = rtol

        increase_iteration_number!(state)
        @test iteration_number(state) == 2
        update!(cache(opt), state, gradient(opt), method, Y)
        Ḡ₂ = gradient_array(cache(opt))
        fac₁₁, fac₁₂ = (β₁ - β₁^2) / (1 - β₁^2), (1 - β₁) / (1 - β₁^2)
        fac₂₁, fac₂₂ = (β₂ - β₂^2) / (1 - β₂^2), (1 - β₂) / (1 - β₂^2)

        @test Matrix(first_moment(cache(opt))) ≈ fac₁₁ * m₁_prev + fac₁₂ * Matrix(Ḡ₂) rtol = rtol
        @test second_moment(cache(opt)) ≈ fac₂₁ * m₂_prev + fac₂₂ * l2norm(Ḡ₂)^2 rtol = rtol
    end
end

# The representation-sensitive assertion. `Adam` accumulates `Ḡ ⊙ Ḡ`, an element of `𝔤ʰᵒʳ`; this
# method accumulates `‖Ḡ‖²`, a number. A single scalar cannot encode the componentwise second moment,
# so the two directions are not parallel — which also rules out the port having quietly kept Adam's
# path.
@testset "the second moment is a scalar and the direction is not Adam's" begin
    Y = rand(StiefelManifold, 6, 3)
    C = randn(6, 3)

    method = ScalarMomentAdam()
    opt = Optimizer(copy(Y), linear_stiefel_objective(C); algorithm=method,
        linesearch=Static(0.01), retraction=Cayley())
    state = OptimizerState(method, Y)
    increase_iteration_number!(state)
    update!(cache(opt), state, gradient(opt), method, Y)

    adam = Adam()
    opt_adam = Optimizer(copy(Y), linear_stiefel_objective(C); algorithm=adam,
        linesearch=Static(0.01), retraction=Cayley())
    state_adam = OptimizerState(adam, Y)
    increase_iteration_number!(state_adam)
    update!(cache(opt_adam), state_adam, gradient(opt_adam), adam, Y)

    δ_scalar = Matrix(direction(cache(opt)))
    δ_adam = Matrix(direction(cache(opt_adam)))
    @test second_moment(cache(opt)) isa Real
    @test second_moment(cache(opt_adam)) isa StiefelLieAlgHorMatrix
    # not parallel: `Adam`'s direction has magnitude ≈ 1 per *component*, this one has ≈ 1 overall
    @test !isapprox(δ_scalar, δ_adam; rtol=1e-8)
    @test !isapprox(δ_scalar ./ norm(δ_scalar), δ_adam ./ norm(δ_adam); rtol=1e-8)
    # ... and that is the difference in step length the docstring warns about, in the norm that decides
    # it: `step_αmax` and `OptimizerStatus` both measure the direction with `l2norm`, which on a lift
    # is the norm of its free parameters. There are `n(n-1)/2 + (N-n)n` of them; `Adam` drives each to
    # magnitude ≈ 1, so its `l2norm` is ≈ √dim, while one scalar divisor normalizes the whole lift to
    # ≈ 1. On the same `Static(η)` that is a step √dim shorter here.
    N, n = size(Y)
    dim = n * (n - 1) ÷ 2 + (N - n) * n
    @test isapprox(l2norm(direction(cache(opt))), 1; rtol=1.0e-6)
    @test isapprox(l2norm(direction(cache(opt_adam))), √dim; rtol=1.0e-6)
end

# Which `‖·‖²` the second moment accumulates. `ambient_norm = false` -- the default -- squares the
# horizontal lift; `true` squares the paper's ambient Euclidean gradient, which on this linear
# objective is `C` exactly. The two are not interchangeable up to a constant: `rgrad` drops the normal
# component, so `∇L = YS` with `S` symmetric makes the lift vanish while `‖∇L‖_F` does not, which the
# third block below pins.
@testset "ScalarMomentAdam: which norm the second moment squares" begin
    Y = rand(StiefelManifold, 6, 3)
    C = randn(6, 3)
    objective = linear_stiefel_objective(C)

    # One `state` for both, and hence one `GlobalSection`: `B = λᵀΔ` depends on the random complement
    # `λ`, so two states would make the two first moments incomparable. `update!` writes the cache
    # only, so the same state can drive both.
    state = OptimizerState(ScalarMomentAdam(), Y)
    increase_iteration_number!(state)
    cache_lift, cache_ambient = map((false, true)) do ambient_norm
        method = ScalarMomentAdam(; ambient_norm)
        opt = Optimizer(copy(Y), objective; algorithm=method, linesearch=Static(0.01),
            retraction=Cayley())
        update!(cache(opt), state, gradient(opt), method, Y)
    end

    Ḡ = global_rep(section(state), rgrad(Y, C))
    @test second_moment(cache_lift) ≈ l2norm(Ḡ)^2
    @test second_moment(cache_ambient) ≈ sum(abs2, C)
    @test second_moment(cache_lift) != second_moment(cache_ambient)
    # the first moment is the same gradient either way -- only the divisor differs
    @test Matrix(first_moment(cache_lift)) ≈ Matrix(first_moment(cache_ambient))
    @test Matrix(direction(cache_lift)) ≈
          Matrix(direction(cache_ambient)) * √(second_moment(cache_ambient) + ScalarMomentAdam().δ) /
          √(second_moment(cache_lift) + ScalarMomentAdam().δ)

    # `ambient_norm = true` reaches the ambient gradient through the flattened closure
    # `GradientAutodiff(F, ::Manifold)` builds, which is a path nothing else in the package takes, so
    # a whole solve is run over it: it has to stay on the manifold and it has to descend.
    for ambient_norm in (false, true)
        Yₛ = rand(StiefelManifold, 6, 3)
        method = ScalarMomentAdam(; ambient_norm)
        opt = Optimizer(Yₛ, objective; algorithm=method, linesearch=Static(0.01))
        stateₛ = OptimizerState(method, Yₛ)
        f₀ = objective(Yₛ)
        for _ in 1:5
            increase_iteration_number!(stateₛ)
            solver_step!(Yₛ, stateₛ, opt)
            update!(stateₛ, opt, Yₛ)
        end
        @test check(Yₛ) < 1.0e-12
        @test objective(Yₛ) < f₀
    end

    # A gradient normal to the tangent space: the lift norm is zero where the ambient one is not, so
    # no constant relates the two.
    S = [1.0 0.5 0.0; 0.5 2.0 -1.0; 0.0 -1.0 3.0]
    ∇L = Y.A * S
    @test l2norm(global_rep(GlobalSection(Y), rgrad(Y, ∇L))) < 1.0e-12
    @test sum(abs2, ∇L) > 1
end

# The accepted step goes through the ordinary section/retraction path, so it stays on the manifold —
# with the exact Cayley transform the paper's own retraction approximates, and with the geodesic,
# which the paper has no version of.
@testset "ScalarMomentAdam steps stay on the Stiefel manifold" begin
    for T in (Float64, Float32)
        tol = T === Float32 ? 5.0f-5 : 1.0e-12
        for retraction in (Cayley(), Geodesic())
            Y = rand(StiefelManifold{T}, 5, 2)
            C = T[1 2; -3 4; 2 -1; 1 0; -2 3]
            method = ScalarMomentAdam(T)
            opt = Optimizer(Y, linear_stiefel_objective(C); algorithm=method,
                linesearch=Static(T(0.01)), retraction=retraction)
            state = OptimizerState(method, Y)

            for _ in 1:3
                increase_iteration_number!(state)
                solver_step!(Y, state, opt)
                update!(state, opt, Y)
                @test check(Y) < tol
            end
            @test iteration_number(state) == 3
        end
    end
end

@testset "ScalarMomentAdam differs from Adam over a solve" begin
    Y₁ = rand(StiefelManifold, 6, 3)
    Y₂ = copy(Y₁)
    C = randn(6, 3)
    objective = linear_stiefel_objective(C)
    opt₁ = Optimizer(Y₁, objective; algorithm=Adam(), linesearch=Static(0.01), retraction=Cayley())
    opt₂ = Optimizer(Y₂, objective; algorithm=ScalarMomentAdam(), linesearch=Static(0.01), retraction=Cayley())
    state₁ = OptimizerState(opt₁.algorithm, Y₁)
    state₂ = OptimizerState(opt₂.algorithm, Y₂)
    increase_iteration_number!(state₁)
    increase_iteration_number!(state₂)
    solver_step!(Y₁, state₁, opt₁)
    solver_step!(Y₂, state₂, opt₂)
    @test !isapprox(Y₁.A, Y₂.A; atol=1e-12, rtol=1e-12)
end
