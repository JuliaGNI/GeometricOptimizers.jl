# Where the step ceiling `c/ρ(X)` of a biorthogonal lift sits against the poles of the Cayley
# transform, for `c = 1`. A step `α ≥ 0` forms `Cayley(αX)`, singular where `𝕀 - αX/2` is, and its
# inverse `Cayley(-αX)`, singular where `𝕀 + αX/2` is. Both are singular at `α = 2/|μ|` for each real
# eigenvalue `μ ≠ 0` of `X`, and a complex eigenvalue makes neither singular for a real `α`. So the
# nearest pole is at `2/max|μ|` over the real eigenvalues, and `1/ρ(X)` is half of it exactly when an
# eigenvalue of largest modulus is real, and less than half otherwise.
#
# The poles are found here without the eigenvalues, as the zeros of the smallest singular value of
# `𝕀 ∓ αX/2` on a fine grid of `α`, and compared with `1/ρ(X)` and with `step_αmax`.
#
#   julia --startup-file=no --project=test scripts/biorthogonal_cayley_poles.jl

using GeometricOptimizers
using GeometricOptimizers: step_αmax
using LinearAlgebra: I, eigvals, svdvals

# The first `α` of the grid at which `𝕀 - αX/2` or `𝕀 + αX/2` has a singular value below `tol`.
function nearest_pole(X; αs = range(0, 20; length = 200_001), tol = 1e-3)
    for α in αs
        σ = min(minimum(svdvals(I - α * X / 2)), minimum(svdvals(I + α * X / 2)))
        σ < tol && return α
    end
    Inf
end

function report(name, A, B, C)
    N, n = size(A, 1) + size(B, 1), size(A, 1)
    X = BiorthogonalLieAlgHorMatrix(A, B, C, N, n)
    μ = eigvals(Matrix(X))
    ρ = maximum(abs, μ)
    pole = nearest_pole(Matrix(X))
    println(
        rpad(name, 36), "ρ = ", round(ρ; digits = 4), "   1/ρ = ", round(1 / ρ; digits = 4),
        "   step_αmax = ", round(step_αmax(1.0, X); digits = 4),
        "   nearest pole = ", round(pole; digits = 4),
        "   pole/(1/ρ) = ", round(pole * ρ; digits = 4))
    (ρ = ρ, pole = pole, ceiling = step_αmax(1.0, X))
end

# real eigenvalues only: `A` symmetric, `B = C = 𝕆`, eigenvalues 1.5, -0.5 and 0
real_case = report("largest |μ| real (1.5)",
    [1.5 0.0; 0.0 -0.5], zeros(1, 2), zeros(2, 1))
@assert isapprox(real_case.pole, 2 / real_case.ρ; atol = 1e-2)
@assert real_case.ceiling ≈ 1 / real_case.ρ

# a negative real eigenvalue of largest modulus, whose pole only the inverse `Cayley(-αX)` has
negative_case = report("largest |μ| real and negative (-2)",
    [-2.0 0.0; 0.0 0.5], zeros(1, 2), zeros(2, 1))
@assert isapprox(negative_case.pole, 2 / negative_case.ρ; atol = 1e-2)

# a complex pair `±3i` of largest modulus and a real eigenvalue 1: the ceiling is `1/3`, the nearest
# pole is at `2`, six times the ceiling and not two
mixed_case = report("largest |μ| complex (±3i), real 1",
    [0.0 -3.0 0.0; 3.0 0.0 0.0; 0.0 0.0 1.0], zeros(1, 3), zeros(3, 1))
@assert isapprox(mixed_case.pole, 2.0; atol = 1e-2)
@assert mixed_case.pole > 2 / mixed_case.ρ + 1

# only a complex pair `±i` and zero: no pole on the real axis at all
complex_case = report("only complex (±i)",
    [0.0 -1.0; 1.0 0.0], zeros(1, 2), zeros(2, 1))
@assert complex_case.pole == Inf
