# How orthogonal to `Y` a global section is, for three ways of computing it on the same Gaussian
# draws, and what the second orthonormalisation costs on the host.
#
#   * one pass: project the span of `Y` out of the draw, then orthonormalise with CholeskyQR2;
#   * project twice: project twice, then orthonormalise once;
#   * orthonormalise twice: one pass, then project the orthonormal result and orthonormalise it
#     again, which is what `global_section` does.
#
# The orthonormalisation multiplies the rounding error left in the span of `Y` by the condition
# number of the projected draw. So projecting twice before one orthonormalisation leaves the worst
# draws far from orthogonal, and only the second orthonormalisation brings `‖Yᵀλ‖` to rounding
# level. The script prints, per element type and size, the largest and the median `‖Yᵀλ‖`, the
# largest `‖λᵀλ - I‖`, the number of draws CholeskyQR2 breaks down on, and the fastest of 200 calls
# of the first and the last variant. The figures in the `global_section` docstring and the host
# figures in its CHANGELOG entry come from this script; the maxima are the worst of one seed of a
# heavy-tailed value, so another seed gives a different worst case.
#
# Run in a cold process:
#
#     julia --startup-file=no --project=<env with GeometricOptimizers> \
#         scripts/global_section_orthogonality.jl

using GeometricOptimizers
using GeometricOptimizers: _cholesky_qr2
using LinearAlgebra
using Random
using Statistics

BLAS.set_num_threads(1)

project(Y, A) = A - Y * (Y' * A)
one_pass(Y, A) = _cholesky_qr2(project(Y, A))
project_twice(Y, A) = _cholesky_qr2(project(Y, project(Y, A)))
function orthonormalise_twice(Y, A)
    λ = _cholesky_qr2(project(Y, A))
    λ === nothing ? nothing : _cholesky_qr2(project(Y, λ))
end

for T in (Float32, Float64), (N, n) in ((6, 3), (50, 3), (200, 10))

    rng = Xoshiro(3)
    Y = Matrix(qr!(randn(rng, T, N, N)).Q)[:, 1:n]
    draws = [randn(rng, T, N, N - n) for _ in 1:(N ≤ 6 ? 20000 : 2000)]
    label = rpad("$T $N×$n", 16)
    for (name, f) in (("one pass", one_pass), ("project twice", project_twice),
        ("orthonormalise twice", orthonormalise_twice))
        λs = filter(!isnothing, [f(Y, A) for A in draws])
        off = [norm(Y' * λ) for λ in λs]
        orth = [norm(λ' * λ - I) for λ in λs]
        breakdowns = length(draws) - length(λs)
        sig(x) = round(x; sigdigits = 2)
        println(label, rpad(name, 22), "max ‖Yᵀλ‖ ", sig(maximum(off)), "  median ",
            sig(median(off)), "  max ‖λᵀλ-I‖ ", sig(maximum(orth)), "  breakdowns ", breakdowns)
    end
    A = first(draws)
    one_pass(Y, A)
    orthonormalise_twice(Y, A)
    t₁ = minimum(@elapsed(one_pass(Y, A)) for _ in 1:200)
    t₂ = minimum(@elapsed(orthonormalise_twice(Y, A)) for _ in 1:200)
    μs(t) = round(t * 1e6; digits = 2)
    println(label, "time: one pass ", μs(t₁), " μs, orthonormalise twice ", μs(t₂), " μs")
end
