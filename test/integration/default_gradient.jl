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
                           gradient, rgrad, isconverged
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: qr, svd, norm
using NeuralNetworkParameters: NetworkParameters
using SimpleSolvers: GradientAutodiff, GradientFiniteDifferences, Hessian, Static
using Random
using Test
include("../helpers/eltypes.jl")

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

@testset "no default gradient on a device: $name, $T" for T in REAL_ELTYPES,
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

@testset "mode selects the gradient on the host, $T" for T in REAL_ELTYPES
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
#
# The minimiser is known in closed form, and `C` is returned so that the testset can compute it: the
# polar factor `UVᵀ` of `C = UΣVᵀ` (the orthogonal Procrustes solution), by an SVD and not by the
# optimizer.
function procrustes(::Type{T}) where {T}
    rng = Random.Xoshiro(3)
    C = stiefel_point(rng, T) + randn(rng, T, N, n) / 10
    F(Y::StiefelManifold) = sum(abs2, parent(Y) - C) / 2
    F, Y -> rgrad(Y, parent(Y) - C), StiefelManifold(stiefel_point(rng, T)), C
end

# The distance of a converged solve to the polar factor, in `√eps(T)`: a minimiser is accurate to the
# root of the objective's precision. Measured with `procrustes`'s seed 3 replaced by each of 1 to 8,
# in both modes, the worst
# was `0.69 √eps(T)` in `Float32` and `0.72 √eps(T)` in `Float64`, so this leaves a factor of about 3.
const PROCRUSTES_TOLERANCE_IN_SQRT_EPS = 2

@testset "a bare Stiefel point: the default gradient rebuilds the point, $T" for T in REAL_ELTYPES
    F, riemannian_gradient, Y, C = procrustes(T)

    # through the lower-level constructor's `default_gradient`
    autodiff = default_gradient(OptimizerProblem(F, Y), Y)
    @test autodiff isa GradientAutodiff{T}
    @test eltype(autodiff(Y)) == T
    @test autodiff(Y) ≈ riemannian_gradient(Y) rtol = √eps(T)

    # `mode = :finitediff` differentiates the same `F`, and solves to the autodiff answer within
    # the accuracy of a central difference, which is below `√eps(T)` at this conditioning
    finite = gradient(Optimizer(Y, F; mode = :finitediff))
    @test finite(Y) ≈ riemannian_gradient(Y) rtol = √eps(T)

    solution(mode) =
        let Y = copy(Y)
            result = solve!(Y, OptimizerState(BFGS(), Y),
                Optimizer(Y, F; mode = mode, retraction = Geodesic()))
            parent(Y), result
        end
    Y_finite, result_finite = solution(:finitediff)
    Y_autodiff, result_autodiff = solution(:autodiff)
    @test Y_finite ≈ Y_autodiff rtol = √eps(T)
    @test eltype(Y_finite) == eltype(Y_autodiff) == T

    # both solves converged, and to the known minimiser
    U, _, V = svd(C)
    minimiser = U * V'
    for (Ŷ, result) in ((Y_finite, result_finite), (Y_autodiff, result_autodiff))
        @test isconverged(result.status)
        @test norm(Ŷ - minimiser) ≤ PROCRUSTES_TOLERANCE_IN_SQRT_EPS * √eps(T)
    end
end
