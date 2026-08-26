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
# `@allocated` is **inside** each `_measured_*` function throughout, never in the `@testset` body, and
# each of them warms the call before measuring it. That is not fussiness. A `@testset` body is a
# closure and a `for` inside one captures its loop variables, so an `@allocated` written there boxes
# the arguments on the way in and reports the box. This file used to be written that way -- the call
# in a function, the `@allocated` in the testset -- and every one of the twelve assertions below
# passed on Julia 1.13, where the boxes are elided, and failed on 1.11, where they are not: 16 bytes
# for `_dot`, 32 for the secant pair. Nothing in `src/` was wrong; the harness was. With the compat
# floor at 1.11 the file would have failed outright, which is how it was found.

using GeometricOptimizers
using GeometricOptimizers: _dot, l2norm, solution_scale, _manifold_αmax, update!, solver_step!,
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

# The two shapes issue #70 is about, and the one this release newly admits.
#
# `wide_set` is 369 entries in one flat branch -- the parameter set of `GMLDatasets`' MNIST
# transformer, and the width at which the `Base.tail` folds this release deleted cost 26 to 71 s to
# compile on Julia 1.12 and 1.13. It is here for the *allocations* rather than the clock
# (`scripts/walk_compile_cost.jl` has the clock): it is the width at which a de-specialised `op` costs
# 3 088 bytes at arity one and 6 144 at arity two, where the three small shapes above would show 16 to
# 48 and could pass while boxing. `nested_set` is the same leaf count in narrow branches, which is the
# shape a network actually has.
#
# `nested_bare` is a nested *plain* `NamedTuple`. `_dot` turned that away until this release -- it
# dispatched on an alias that binds an element type, and a nested bare set has branches for values, so
# it is not an `ArrayNamedTuple` -- while `l2norm` and `solution_scale` have taken it since 0.6.0.
const WIDE_ENTRIES = 369

wide_set(seed) = NamedTuple{ntuple(i -> Symbol(:p, i), WIDE_ENTRIES)}(
    ntuple(i -> randn(Random.Xoshiro(seed * 1000 + i), Float32, 4, 4), WIDE_ENTRIES))

nested_bare(seed) = (L1 = (A = lift(seed),), L2 = (W = rand(Random.Xoshiro(seed + 2), 3, 4),
                                                  b = rand(Random.Xoshiro(seed + 3), 5)))

# A set whose leaves are *not* all one element type, which is what makes the accumulator's type a
# question. `NetworkParameters` derives its `T` by promotion, so this is a `NetworkParameters{Float64}`
# whose first leaf in `flatten` order is `Float32`.
mixed_container(seed) = NetworkParameters((
    L1 = (W = rand(Random.Xoshiro(seed), Float32, 3, 4),),
    L2 = (b = rand(Random.Xoshiro(seed + 1), Float64, 5),)))

# what `_dot` and `l2norm` used to be written as, kept here as the thing to agree with
_reference_dot(a, b) = dot(flatten(Float64, a)[1], flatten(Float64, b)[1])
_reference_norm(a) = l2norm(flatten(Float64, a)[1])

# `_dot` sums per leaf and then across, where the reference sums once over the concatenation. Both are
# `Σ aᵢbᵢ`; they differ in summation order and so at round-off, which is why these are `≈`.
#
# **Including the single lift**, which an earlier version of this file asserted was `==` on the grounds
# that "for a single lift the two orders coincide". They do not. A `StiefelLieAlgHorMatrix` is a
# *two*-block leaf -- `freeparameters` returns `(A, B)` -- so `_dot` takes `dot(A₁, A₂) + dot(B₁, B₂)`
# where the reference takes one `dot` over `[A; B]` concatenated, and two BLAS calls summed need not
# blocking-for-blocking match one call over twice the length. It happened to hold on the machine the
# claim was written on and does not in general: on Julia 1.11/windows, 1.12/ubuntu and 1.13 on both,
# the two come out `3.070431380702119` against `3.0704313807021184` -- one ULP, which is round-off and
# is the thing this testset is about.
@testset "_dot is the flattened inner product" begin
    @test _dot(lift(1), lift(11)) ≈ _reference_dot(lift(1), lift(11))
    @test _dot(flat_set(1), flat_set(11)) ≈ _reference_dot(flat_set(1), flat_set(11))
    @test _dot(container(1), container(11)) ≈ _reference_dot(container(1), container(11))
    # and the two shapes describing the same numbers agree with each other exactly.
    #
    # That is no longer a coincidence worth being nervous about. Upstream's fold threads its
    # accumulator through the nested branches, so a left fold over a tree equals the left fold over the
    # flat leaf list whatever the grouping -- where the `Base.tail` recursion this replaced was a right
    # fold that happened to align. The nested plain `NamedTuple` is the third spelling of the same
    # numbers and this release is the first that accepts it.
    @test _dot(container(1), container(11)) == _dot(flat_set(1), flat_set(11))
    @test _dot(nested_bare(1), nested_bare(11)) == _dot(flat_set(1), flat_set(11))
end

# The accumulator is `zero(T)` and not the strong zero `false`, and this is the assertion that says why.
# Upstream's fold is a *left* fold, so `false` would take its type from the first leaf in `flatten`
# order -- here a `Float32` -- and the `Float64` leaf's contribution would be added to a `Float32`
# running sum. `T` is the promotion over the leaves, so it is not order-dependent, and the reference
# below is the same sum taken in `Float64` throughout.
@testset "_dot accumulates in the promotion, not in the first leaf's type" begin
    a, b = mixed_container(1), mixed_container(11)
    @test _dot(a, b) isa Float64
    @test _dot(a, b) ≈ _reference_dot(a, b)
    # and the pair whose element types differ, which used to be a `MethodError`: no `T` binds on the
    # signature, so it is `promote_type` over both sets
    f32 = NetworkParameters((L1 = (W = rand(Random.Xoshiro(5), Float32, 3, 3),),))
    f64 = NetworkParameters((L1 = (W = rand(Random.Xoshiro(6), Float64, 3, 3),),))
    @test _dot(f32, f64) isa Float64
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
# nothing, for every shape. The warm-up call is the first statement of the function so that the
# `@allocated` beside it sees a compiled `_dot`; see the note at the head of this file for why it
# cannot be written in the testset.
_measured_dot(a, b) = (_dot(a, b); @allocated _dot(a, b))

@testset "_dot allocates nothing" begin
    for (name, a, b) in (("lift", lift(1), lift(11)),
                         ("flat NamedTuple", flat_set(1), flat_set(11)),
                         ("container", container(1), container(11)),
                         # 369 in one branch, bare and wrapped, which is where a boxed `op` would show
                         ("369-wide bare", wide_set(1), wide_set(2)),
                         ("369-wide container", NetworkParameters(wide_set(1)),
                          NetworkParameters(wide_set(2))),
                         # the widened method, which reaches `parameter_eltype` rather than taking its
                         # element type off the signature -- see the comment on it in
                         # `src/optimizers/named_tuple_wrapper.jl`
                         ("nested bare NamedTuple", nested_bare(1), nested_bare(11)),
                         ("mixed-precision container", mixed_container(1), mixed_container(11)))
        @test _measured_dot(a, b) == 0
    end
end

# `l2norm` is zero for every shape as of 0.6.0, and it was not before. It used to allow "one 32-byte
# `vec` wrapper per matrix leaf", because `l2norm(a::AbstractMatrix)` was `l2norm(vec(a))` here -- one
# of the two pirated methods of issue #16 group 1 -- and `vec` of a `Matrix` allocates the reshape
# wrapper. `GeometricBase` 0.14.9 takes `L2norm(x::AbstractArray)` where it had `AbstractVector`, so
# both pirated methods are deleted and no `vec` is taken. The exact zero is the point of asserting it.
_measured_norm(a) = (l2norm(a); @allocated l2norm(a))

@testset "l2norm allocates nothing, for every shape" begin
    for a in (lift(1), flat_set(1), container(1), (a = rand(4), b = rand(5)),
              wide_set(1), NetworkParameters(wide_set(1)), nested_bare(1))
        @test _measured_norm(a) == 0
    end
end

# `solution_scale` shares `_sumsq_leaves` with `l2norm` and differs only in the leaf function, and
# `_manifold_αmax` is the fourth of the folds this release replaced -- the one issue #70's count of
# three omitted, and the only one on the per-iteration path. Neither was pinned here before.
_measured_scale(a) = (solution_scale(a); @allocated solution_scale(a))
_measured_αmax(a, b) = (_manifold_αmax(a, b, 1.0f0); @allocated _manifold_αmax(a, b, 1.0f0))

@testset "the other two folds allocate nothing either" begin
    for a in (lift(1), flat_set(1), container(1), wide_set(1), nested_bare(1))
        @test _measured_scale(a) == 0
    end
    for (a, b) in ((flat_set(1), flat_set(11)), (container(1), container(11)),
                   (wide_set(1), wide_set(2)), (nested_bare(1), nested_bare(11)))
        @test _measured_αmax(a, b) == 0
    end
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

_measured_secant!(c) = (_flat_δ!(c); _flat_γ!(c);
                        @allocated begin _flat_δ!(c); _flat_γ!(c) end)
_measured_outer!(m, a, b) = (outer!(m, a, b); @allocated outer!(m, a, b))
_measured_quad(γ, Q) = (dot(γ, Q, γ); @allocated dot(γ, Q, γ))
_measured_mul!(c, A, b, scratch) = (_flat_mul!(c, A, b, scratch);
                                    @allocated _flat_mul!(c, A, b, scratch))

@testset "the flat sites of $(nameof(typeof(algorithm))) allocate nothing" for algorithm in (BFGS(), DFP())
    for (name, x) in (("Vector", vector_problem()[1]),
                      ("Manifold", manifold_problem()[1]),
                      ("NamedTuple", namedtuple_problem()[1]),
                      ("container", container_problem()[1]))
        c = OptimizerCache(algorithm, x)
        state = OptimizerState(algorithm, x)
        Q = inverse_hessian(state)

        # filling the flat mirrors of the secant pair
        @test _measured_secant!(c) == 0

        δ, γ = _flat_δ!(c), _flat_γ!(c)

        # `outer!`, which used to flatten both of its arguments on every call
        m = zeros(Float64, length(δ), length(γ))
        @test _measured_outer!(m, δ, γ) == 0

        # `γᵀQγ`, which used to materialise `Q * γ`
        @test _measured_quad(γ, Q) == 0

        # `_mul!`, which used to allocate a flat vector for `b`, one for the result, and a
        # `ParameterLayout` besides
        @test _measured_mul!(direction(c), Q, rhs(c), c.flat) == 0
    end
end
