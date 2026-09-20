# The symplectic Gram-Schmidt process. Where the ordinary process makes the columns orthonormal,
# this one makes them symplectic: the result `B` of a `2N x 2n` input satisfies `BᵀJ_{2N}B = J_{2n}`.
#
# The ordinary process is deliberately absent. `LinearAlgebra.qr` does it, faster and more stably,
# and `rand(::Manifold, …)` already calls it; a second copy here would be a second thing to keep
# right.
#
# Both names are exported and neither has a caller under `src/`: `sr!` builds its symplectic factor
# by its own reflections and does not route through here. That is intended. They are the standalone
# entry point to the process for a caller who has a matrix and a form, the doctest below is the
# documented use, and `test/decompositions/symplectic_sr.jl` is what holds them to it.

@doc raw"""
    symplectic_normalize(e, f, J)

Scale a pair of vectors so that ``e^TJf = 1``, including the sign: the factor `sign(e^TJf)` goes
into `e`, so a pair whose form is negative comes back with the form `+1` rather than `-1`.

The pair is the unit the symplectic Gram-Schmidt process works in: a single vector cannot be
normalized against a form that vanishes on it.
"""
function symplectic_normalize(e::AbstractVector, f::AbstractVector, J::AbstractMatrix)
    fac = e' * J * f
    (sign(fac) / sqrt(abs(fac)) * e, 1 / sqrt(abs(fac)) * f)
end

@doc raw"""
    symplectic_gram_schmidt!(A, J, start = 1)

Make the columns of `A` symplectic with respect to `J`, in place.

`A` is ``2N\times2n`` and is read as two stacked halves: column `i` and column `n + i` are the
pair that is normalized together. `start` skips the leading pairs, for the case where they are
symplectic already.
"""
function symplectic_gram_schmidt!(A::AbstractMatrix, J::AbstractMatrix, start = 1)
    N = size(A, 1)
    n = size(A, 2)
    @assert n ≤ N
    @assert iseven(N)
    @assert iseven(n)
    N ÷= 2
    n ÷= 2

    for i in start:n
        vec₁ = A[1:(2 * N), i]
        vec₂ = A[1:(2 * N), n + i]
        for j in 1:(i - 1)
            vec₁ = vec₁ - (A[1:(2 * N), j]' * J * vec₁) * A[1:(2 * N), n + j] -
                   (vec₁' * J * A[1:(2 * N), n + j]) * A[1:(2 * N), j]
            vec₂ = vec₂ - (A[1:(2 * N), j]' * J * vec₂) * A[1:(2 * N), n + j] -
                   (vec₂' * J * A[1:(2 * N), n + j]) * A[1:(2 * N), j]
        end
        A[1:(2 * N), i], A[1:(2 * N), n + i] = symplectic_normalize(vec₁, vec₂, J)
    end
    A
end

@doc raw"""
    symplectic_gram_schmidt(A, J, start = 1)

[`symplectic_gram_schmidt!`](@ref) on a copy of `A`.

# Examples

```jldoctest
using GeometricOptimizers
using LinearAlgebra
import Random

Random.seed!(1234)

J₆ = [zeros(3, 3) I(3); -I(3) zeros(3, 3)]
J₄ = [zeros(2, 2) I(2); -I(2) zeros(2, 2)]
B = symplectic_gram_schmidt(randn(6, 4), J₆)

norm(B' * J₆ * B - J₄) < 1e-10

# output

true
```
"""
function symplectic_gram_schmidt(A::AbstractMatrix, J::AbstractMatrix, start = 1)
    symplectic_gram_schmidt!(copy(A), J, start)
end
