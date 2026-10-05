# The gradient `Optimizer` builds when the caller supplies none.
#
# On a device it builds none: `GradientAutodiff` runs `ForwardDiff` on a host configuration, and
# `GradientFiniteDifferences` indexes `x[j]` and keeps host buffers, so either would raise far from
# the cause or move the iterate to the host. Every route that would build one throws an
# `ArgumentError` that names `∇F!` instead — `Optimizer(x, F)` in either `mode`,
# `Optimizer(x, problem)` and the keyword default of the lower-level constructor, both through
# `default_gradient`. On the host, `mode` selects the gradient, an unknown `mode` is refused, and a
# bare manifold gets a gradient that rebuilds the manifold before it calls `F`.

using GeometricOptimizers
using GeometricOptimizers: OptimizerCache, OptimizerProblem, RiemannianGradient,
                           default_gradient,
                           gradient, rgrad
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: qr
using NeuralNetworkParameters: NetworkParameters
using SimpleSolvers: GradientAutodiff, GradientFiniteDifferences, Hessian, Static
using Random
using Test

allowscalar(false)

const N, n = 6, 3

stiefel_point(rng, T) = Matrix{T}(qr(randn(rng, T, N, n)).Q)[:, 1:n]

function device_points(T)
    rng = Random.Xoshiro(1)
    ("vector" => JLArray(randn(rng, T, 4)),
        "Stiefel" => StiefelManifold(JLArray(stiefel_point(rng, T))),
        "parameters" => NetworkParameters((
            Y = StiefelManifold(JLArray(stiefel_point(rng, T))), W = JLArray(randn(rng, T, 2, 2)))))
end

objective(x::AbstractVector) = sum(abs2, x)
objective(Y::StiefelManifold) = sum(abs2, parent(Y))
objective(ps::NetworkParameters) = objective(ps.Y) + sum(abs2, ps.W)

function names_∇F!(f)
    err = try
        f()
        nothing
    catch e
        e
    end
    err isa ArgumentError && occursin("∇F!", sprint(showerror, err))
end

@testset "no default gradient on a device: $name, $T" for T in (Float32, Float64),
    (name, x) in device_points(T)

    for mode in (:autodiff, :finitediff)
        @test_throws ArgumentError Optimizer(x, objective; mode = mode)
        @test names_∇F!(() -> Optimizer(x, objective; mode = mode))
    end

    problem = OptimizerProblem(objective, x)
    @test names_∇F!(() -> default_gradient(problem, x))
    @test names_∇F!(() -> Optimizer(x, problem))

    method = GradientMethod()
    cache = OptimizerCache(method, x)
    @test names_∇F!(() -> Optimizer(
        method, problem, Hessian(method, problem, x), cache, Static(T(1) / 10)))
end

@testset "an unknown mode is refused, by name" begin
    err = try
        Optimizer(randn(3), objective; mode = :forwarddiff)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(":autodiff", sprint(showerror, err))
    @test occursin(":finitediff", sprint(showerror, err))
end

@testset "mode selects the gradient on the host, $T" for T in (Float32, Float64)
    rng = Random.Xoshiro(2)
    x = randn(rng, T, 4)
    Y = StiefelManifold(stiefel_point(rng, T))
    ps = NetworkParameters((
        Y = StiefelManifold(stiefel_point(rng, T)), W = randn(rng, T, 2, 2)))

    @test gradient(Optimizer(x, objective; mode = :finitediff)) isa
          GradientFiniteDifferences{T}
    @test gradient(Optimizer(Y, objective; mode = :finitediff)) isa
          GradientFiniteDifferences{T}
    finite = gradient(Optimizer(ps, objective; mode = :finitediff))
    @test finite isa RiemannianGradient
    @test finite.gradient isa GradientFiniteDifferences{T}

    @test gradient(Optimizer(x, objective)) isa GradientAutodiff{T}
    @test gradient(Optimizer(Y, objective)) isa GradientAutodiff{T}
end

# The Procrustes distance `‖Y - C‖²/2` to a matrix `C` near the manifold, written for a
# `StiefelManifold` only: a gradient that hands `F` the bare storage, or a `reshape` of it, raises a
# `MethodError` instead of differentiating. `F` is quadratic in the storage, so a central difference
# has no truncation error, and its rounding error, `eps(T) * |F| / ϵ`, is small at the minimum, where
# `|F|` is: two solves that differ only in the gradient then agree to well within `√eps(T)`. Over
# eight seeds the relative distance was below `3e-3 √eps(T)` in both precisions.
function procrustes(::Type{T}) where {T}
    rng = Random.Xoshiro(3)
    C = stiefel_point(rng, T) + randn(rng, T, N, n) / 10
    F(Y::StiefelManifold) = sum(abs2, parent(Y) - C) / 2
    F, Y -> rgrad(Y, parent(Y) - C), StiefelManifold(stiefel_point(rng, T))
end

@testset "a bare Stiefel point: the default gradient rebuilds the point, $T" for T in (Float32, Float64)
    F, riemannian_gradient, Y = procrustes(T)

    # through the lower-level constructor's `default_gradient`
    autodiff = default_gradient(OptimizerProblem(F, Y), Y)
    @test autodiff isa GradientAutodiff{T}
    @test autodiff(Y) ≈ riemannian_gradient(Y) rtol = √eps(T)

    # `mode = :finitediff` differentiates the same `F`, and solves to the autodiff answer within
    # the accuracy of a central difference, which is below `√eps(T)` at this conditioning
    finite = gradient(Optimizer(Y, F; mode = :finitediff))
    @test finite(Y) ≈ riemannian_gradient(Y) rtol = √eps(T)

    solution(mode) =
        let Y = copy(Y)
            solve!(Y, OptimizerState(BFGS(), Y), Optimizer(Y, F; mode = mode, retraction = Geodesic()))
            parent(Y)
        end
    @test solution(:finitediff) ≈ solution(:autodiff) rtol = √eps(T)
    @test eltype(solution(:finitediff)) == T
end
