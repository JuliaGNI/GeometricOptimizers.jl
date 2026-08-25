# The flat buffers, and what is left after them.
#
# Every quantity a quasi-Newton method forms lives in the *flattened* coordinates -- `Q` is sized by
# the length of the flattening, `outer!` forms its outer products there, `_dot` pairs there -- while
# the parameters themselves are a `NamedTuple`, a container, or a horizontal lift of the ambient
# shape. Until 0.6.0 every crossing between the two built a fresh flat vector: two per `_dot`, two per
# `outer!`, one plus a `ParameterLayout` per `_mul!`, and one more for the `γ` of each `update!`.
#
# Two different fixes, and this file pins both.
#
#   * `_dot` needs no buffer at all. `dot` of two flattenings is the sum of the per-leaf `dot`s, so
#     the sum can be taken without the vectors -- which is what matters most, `_dot` being the hottest
#     of the sites (once per line-search trial slope, once per `OptimizerStatus`).
#   * `outer!` and `_mul!` genuinely need the flat form, so `BFGSCache` and `DFPCache` carry buffers
#     to write into. See `_flat_scratch`.
#
# `@allocated` is measured **inside a compiled function** throughout, never at the top level: the note
# on `NeuralNetworkParameters.flatten!` records that an uninferred context costs a few tens of bytes
# per leaf, which would be measuring the harness rather than the code.

using GeometricOptimizers
using GeometricOptimizers: _dot, l2norm, solution_scale, update!, solver_step!,
    increase_iteration_number!, gradient, inverse_hessian, cache, direction, rhs,
    OptimizerCache, _flat_δ!, _flat_γ!, _flat_mul!, outer!
using NeuralNetworkParameters: NetworkParameters, flatten
using SimpleSolvers: Static
using LinearAlgebra: dot
using Test
import Random

const N, n, m = 6, 3, 4

Random.seed!(1234)
const B = randn(N, m)

lift(seed) = StiefelLieAlgHorMatrix(SkewSymMatrix(rand(Random.Xoshiro(seed), n, n)),
                                   rand(Random.Xoshiro(seed + 1), N - n, n), N, n)

flat_set(seed) = (A = lift(seed), W = rand(Random.Xoshiro(seed + 2), 3, 4),
                  b = rand(Random.Xoshiro(seed + 3), 5))

container(seed) = let p = flat_set(seed)
    NetworkParameters((L1 = (A = p.A,), L2 = (W = p.W, b = p.b)))
end

# what `_dot` and `l2norm` used to be written as, kept here as the thing to agree with
_reference_dot(a, b) = dot(flatten(Float64, a)[1], flatten(Float64, b)[1])
_reference_norm(a) = l2norm(flatten(Float64, a)[1])

# `_dot` sums per leaf and then across, where the reference sums once over the concatenation. Both are
# `Σ aᵢbᵢ`; they differ in summation order and so at round-off, which is why this is `≈` and not `==`
# for a set of more than one leaf. For a single lift the two orders coincide and it *is* `==`.
@testset "_dot is the flattened inner product" begin
    @test _dot(lift(1), lift(11)) == _reference_dot(lift(1), lift(11))
    @test _dot(flat_set(1), flat_set(11)) ≈ _reference_dot(flat_set(1), flat_set(11))
    @test _dot(container(1), container(11)) ≈ _reference_dot(container(1), container(11))
    # and the two shapes describing the same numbers agree with each other exactly
    @test _dot(container(1), container(11)) == _dot(flat_set(1), flat_set(11))
end

@testset "l2norm and solution_scale are the norm of the flattening" begin
    for a in (lift(1), flat_set(1), container(1))
        @test l2norm(a) ≈ _reference_norm(a)
    end
    ps = (Y = rand(Random.Xoshiro(7), StiefelManifold{Float64}, N, n), W = rand(3, 4))
    psc = NetworkParameters((L1 = (Y = ps.Y,), L2 = (W = ps.W,)))
    @test solution_scale(ps) ≈ solution_scale(psc)
end

# The measurement the fix exists for. `_dot` allocated two flat vectors per call and now allocates
# nothing, for every shape.
_measured_dot(a, b) = _dot(a, b)

@testset "_dot allocates nothing" begin
    for (name, a, b) in (("lift", lift(1), lift(11)),
                         ("flat NamedTuple", flat_set(1), flat_set(11)),
                         ("container", container(1), container(11)))
        _measured_dot(a, b)                       # compile
        @test (@allocated _measured_dot(a, b)) == 0
    end
end

# `l2norm` is deliberately *not* asserted to be zero, and this says why rather than leaving a gap:
# `l2norm(a::AbstractMatrix)` is `l2norm(vec(a))`, and `vec` of a `Matrix` allocates the 32-byte
# reshape wrapper. That method is one of the two pirated ones of issue #16 group 1, waiting to be
# upstreamed to `GeometricBase`; the recursion over the leaves adds nothing on top of it. The bound
# below is "one wrapper per matrix leaf and nothing else".
_measured_norm(a) = l2norm(a)

@testset "l2norm allocates only the vec wrapper of each matrix leaf" begin
    for (a, matrix_leaves) in ((lift(1), 1), (flat_set(1), 2), (container(1), 2))
        _measured_norm(a)
        @test (@allocated _measured_norm(a)) ≤ 32 * matrix_leaves
    end
    # a set of vectors has no such leaf and is free
    v = (a = rand(4), b = rand(5))
    _measured_norm(v)
    @test (@allocated _measured_norm(v)) == 0
end

# The four shapes of solution this package accepts, each with an objective. Used by the testset below
# and named here so that "for every shape" is a list rather than a claim.
manifold_problem() = (rand(Random.Xoshiro(4), StiefelManifold{Float64}, N, n),
                      Y -> sum(abs2, Y * ones(n, m) .- B) / 2)

namedtuple_problem() = ((Y = rand(Random.Xoshiro(1), StiefelManifold{Float64}, N, n),
                         W = randn(Random.Xoshiro(2), n, m), b = zeros(N)),
                        ps -> sum(abs2, ps.Y * ps.W .+ ps.b .- B) / 2)

container_problem() = let (ps, _) = namedtuple_problem()
    (NetworkParameters((L1 = (Y = ps.Y,), L2 = (W = ps.W, b = ps.b))),
     ps -> sum(abs2, ps.L1.Y * ps.L2.W .+ ps.L2.b .- B) / 2)
end

vector_problem() = (randn(Random.Xoshiro(3), 12), v -> sum(abs2, v))

# The three sites, directly. Going through `update!` instead would be measuring something else: the
# `γᵀQγ` and both `outer!`s sit inside the `curvature_is_usable` branch, and calling `update!` twice at
# one iterate -- which is what a `@allocated` needs, one call to compile and one to measure -- leaves
# `Δg` identically zero, because the cache advances `state.ḡ` itself as soon as it has used it. The
# branch is then skipped both times and the figure is the cost of not running it. The end-to-end
# figure, taken over a whole `solve!` where the branch does fire, is in the CHANGELOG.

_measured_secant!(c) = (_flat_δ!(c), _flat_γ!(c))
_measured_outer!(m, a, b) = outer!(m, a, b)
_measured_mul!(c, A, b, scratch) = _flat_mul!(c, A, b, scratch)

@testset "the flat sites of $(nameof(typeof(algorithm))) allocate nothing" for algorithm in (BFGS(), DFP())
    for (name, x) in (("Vector", vector_problem()[1]),
                      ("Manifold", manifold_problem()[1]),
                      ("NamedTuple", namedtuple_problem()[1]),
                      ("container", container_problem()[1]))
        c = OptimizerCache(algorithm, x)
        state = OptimizerState(algorithm, x)
        Q = inverse_hessian(state)

        # filling the flat mirrors of the secant pair
        _measured_secant!(c)
        @test (@allocated _measured_secant!(c)) == 0

        δ, γ = _flat_δ!(c), _flat_γ!(c)

        # `outer!`, which used to flatten both of its arguments on every call
        m = zeros(Float64, length(δ), length(γ))
        _measured_outer!(m, δ, γ)
        @test (@allocated _measured_outer!(m, δ, γ)) == 0

        # `γᵀQγ`, which used to materialise `Q * γ`
        f_quad(γ, Q) = dot(γ, Q, γ)
        f_quad(γ, Q)
        @test (@allocated f_quad(γ, Q)) == 0

        # `_mul!`, which used to allocate a flat vector for `b`, one for the result, and a
        # `ParameterLayout` besides
        _measured_mul!(direction(c), Q, rhs(c), c.flat)
        @test (@allocated _measured_mul!(direction(c), Q, rhs(c), c.flat)) == 0
    end
end
