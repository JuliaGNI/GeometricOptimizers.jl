# The flat buffers, the retraction workspace, and what is left after them.
#
# Two groups of assertions, in that order. The first is about the *flat* coordinates a quasi-Newton
# method works in; the second is about the retraction, which is where a step on a manifold spends
# about two thirds of what it allocates. See the header above the second group.
#
# Every quantity a quasi-Newton method forms lives in the *flattened* coordinates -- `Q` is sized by
# the length of the flattening, `outer!` forms its outer products there, `_dot` pairs there -- while
# the parameters themselves are a `NamedTuple`, a container, or a horizontal lift of the ambient
# shape. Until 0.6.0 every crossing between the two built a fresh flat vector: two per `_dot`, two per
# `outer!`, one plus a `ParameterLayout` per product with `Q`, and one more for the `γ` of each
# `update!`.
#
# Two different fixes, and this file pins both.
#
#   * `_dot` needs no buffer at all. `dot` of two flattenings is the sum of the per-leaf `dot`s, so
#     the sum can be taken without the vectors -- which is what matters most, `_dot` being the hottest
#     of the sites (once per line-search trial slope, once per `OptimizerStatus`).
#   * `outer!` and `_flat_mul!` genuinely need the flat form, so `QuasiNewtonCache` carries buffers
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
using GeometricOptimizers: _dot, l2norm, solution_scale, _manifold_αmax, update!,
                           _difference!, _rmul!, _add!, _rac!, _div!, _square!, _copyto!,
                           solver_step!,
                           increase_iteration_number!, gradient, inverse_hessian, cache,
                           direction, rhs,
                           OptimizerCache, _flat_mul!, _flat_secant, outer!, DottableSet,
                           GlobalSection, update_section!, lift_factors!,
                           retraction_matrix!, retraction_workspace, initialize_state!,
                           OptimizerStatus, config, problem, value,
                           _update_inverse_hessian!, curvature_is_usable, _fill!
using NeuralNetworkParameters: NetworkParameters, flatten
using SimpleSolvers: Static
using LinearAlgebra: dot
using Test
import Random
include("../helpers/eltypes.jl")

const N, n, m = 6, 3, 4

Random.seed!(1234)
const B = randn(N, m)

# Every fixture takes the element type first, so that a testset over `REAL_ELTYPES` builds its data
# in `T` rather than in `rand`'s default `Float64`.
function lift(::Type{T}, seed) where {T}
    StiefelLieAlgHorMatrix(SkewSymMatrix(rand(Random.Xoshiro(seed), T, n, n)),
        rand(Random.Xoshiro(seed + 1), T, N - n, n), N, n)
end

function flat_set(::Type{T}, seed) where {T}
    (A = lift(T, seed), W = rand(Random.Xoshiro(seed + 2), T, 3, 4),
        b = rand(Random.Xoshiro(seed + 3), T, 5))
end

function container(::Type{T}, seed) where {T}
    let p = flat_set(T, seed)
        NetworkParameters((L1 = (A = p.A,), L2 = (W = p.W, b = p.b)))
    end
end

# The same leaves, wrapped without regrouping. `container` and `flat_container` are therefore the two
# *groupings* of one leaf list, which is what the equalities below are about.
flat_container(::Type{T}, seed) where {T} = NetworkParameters(flat_set(T, seed))

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
# `nested_bare` is a nested *plain* `NamedTuple`, and it is here for `l2norm` and `solution_scale`,
# which is here as a fixture for the *shape* rather than as something this package accepts: a whole set
# of parameters reaches it as a `NetworkParameters`, so `container` is what the folds below pair, and
# it holds exactly these leaves.
const WIDE_ENTRIES = 369

function wide_set(seed, ::Type{T}) where {T}
    NamedTuple{ntuple(i -> Symbol(:p, i), WIDE_ENTRIES)}(
        ntuple(i -> randn(Random.Xoshiro(seed * 1000 + i), T, 4, 4), WIDE_ENTRIES))
end

function nested_bare(::Type{T}, seed) where {T}
    (L1 = (A = lift(T, seed),),
        L2 = (W = rand(Random.Xoshiro(seed + 2), T, 3, 4),
            b = rand(Random.Xoshiro(seed + 3), T, 5)))
end

# A set whose leaves are *not* all one element type, which is what makes the accumulator's type a
# question. `NetworkParameters` derives its `T` by promotion, so this is a `NetworkParameters{Float64}`
# whose first leaf in `flatten` order is `Float32`.
function mixed_container(seed)
    NetworkParameters((
        L1 = (W = rand(Random.Xoshiro(seed), Float32, 3, 4),),
        L2 = (b = rand(Random.Xoshiro(seed + 1), Float64, 5),)))
end

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
#
# The reference is taken in `Float64` in both passes, so in the `Float32` pass it is the more accurate
# of the two; default `≈` is `√eps(T)` relative, far above the round-off of a sum of 33 products.
@testset "_dot is the flattened inner product, $T" for T in REAL_ELTYPES
    @test eltype(_dot(lift(T, 1), lift(T, 11))) == T
    @test eltype(_dot(container(T, 1), container(T, 11))) == T
    @test _dot(lift(T, 1), lift(T, 11)) ≈ _reference_dot(lift(T, 1), lift(T, 11))
    @test _dot(flat_container(T, 1), flat_container(T, 11)) ≈
          _reference_dot(flat_set(T, 1), flat_set(T, 11))
    @test _dot(container(T, 1), container(T, 11)) ≈
          _reference_dot(container(T, 1), container(T, 11))
    # and the two shapes describing the same numbers agree with each other exactly.
    #
    # That is no longer a coincidence worth being nervous about. Upstream's fold threads its
    # accumulator through the nested branches, so a left fold over a tree equals the left fold over the
    # flat leaf list whatever the grouping -- where the `Base.tail` recursion this replaced was a right
    # fold that happened to align. The nested plain `NamedTuple` is the third spelling of the same
    # numbers, grouped one level deeper.
    @test _dot(container(T, 1), container(T, 11)) ==
          _dot(flat_container(T, 1), flat_container(T, 11))
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

# The same pair of *lifts*, which is a different assertion and the one that catches a wrong answer
# rather than a missing method.
#
# A differing-eltype pair of `NetworkParameters` raised before this release, so `isa Float64` is enough
# to pin it. A differing-eltype pair of lifts did **not** raise: an `AbstractLieAlgHorMatrix` is an
# `AbstractMatrix`, so it missed the alias that binds a `T` and took `_dot(::AbstractVecOrMat,
# ::AbstractVecOrMat)` instead -- the *ambient* Frobenius product, which counts each off-diagonal block
# of the lift twice and so comes out at exactly twice the pairing of the free parameters. `isa Float64`
# would have passed on that too. So this asserts the *value*, against the flattening, which is the only
# thing that separates the two.
#
# Every pairing of `REAL_ELTYPES`, the two equal ones included: the differing ones are the pairs that
# missed the alias, the equal ones always took the method that binds `T` and are the control.
@testset "_dot of two lifts is intrinsic, $T with $T′" for T in REAL_ELTYPES,
    T′ in REAL_ELTYPES

    a, b = lift(T, 8), lift(T′, 11)
    @test eltype(_dot(a, b)) == promote_type(T, T′)
    @test _dot(a, b) ≈ _reference_dot(a, b)
    # and not the ambient product, which is where it went before -- named as a number rather than as a
    # relation, so that the assertion above cannot be satisfied by both
    @test _dot(a, b) ≉ dot(a, b)
    @test dot(a, b) ≈ 2 * _reference_dot(a, b)
    # the same-eltype pair, which always took the method that binds `T` and is here as the control
    @test _dot(lift(T′, 1), b) ≈ _reference_dot(lift(T′, 1), b)
end

@testset "l2norm and solution_scale are the norm of the flattening, $T" for T in REAL_ELTYPES
    for a in (lift(T, 1), flat_container(T, 1), container(T, 1))
        @test eltype(l2norm(a)) == T
        @test l2norm(a) ≈ _reference_norm(a)
    end
    # the same leaves flat and nested, which is the statement that grouping changes nothing
    flat = NetworkParameters((
        Y = rand(Random.Xoshiro(7), StiefelManifold{T}, N, n),
        W = rand(Random.Xoshiro(8), T, 3, 4)))
    nested = NetworkParameters((L1 = (Y = flat.Y,), L2 = (W = flat.W,)))
    @test eltype(solution_scale(flat)) == T
    @test solution_scale(flat) ≈ solution_scale(nested)
end

# The measurement the fix exists for. `_dot` allocated two flat vectors per call and now allocates
# nothing, for every shape. The warm-up call is the first statement of the function so that the
# `@allocated` beside it sees a compiled `_dot`; see the note at the head of this file for why it
# cannot be written in the testset.
_measured_dot(a, b) = (_dot(a, b); @allocated _dot(a, b))

@testset "_dot allocates nothing, $T" for T in REAL_ELTYPES
    for (name, a, b) in (("lift", lift(T, 1), lift(T, 11)),
        ("flat container", flat_container(T, 1), flat_container(T, 11)),
        ("container", container(T, 1), container(T, 11)),
    # 369 leaves in one branch, which is where a boxed `op` would show
        ("369-wide container", NetworkParameters(wide_set(1, T)),
        NetworkParameters(wide_set(2, T))),
    # the method that reaches `parameter_eltype` rather than taking its element
    # type off the signature -- see the comment on it in
    # `src/optimizers/named_tuple_wrapper.jl`. Mixed by construction, so the
    # same pair in both passes.
        ("mixed-precision container", mixed_container(1), mixed_container(11)))
        @test eltype(_dot(a, b)) == (name == "mixed-precision container" ? Float64 : T)
        @test _measured_dot(a, b) == 0
    end
end

# The two `_dot` methods: a set whose leaves share one element type takes the method that reads `T`
# off the signature, and the method for the other pairs gives the same value on it.
@testset "a wide set takes the `_dot` that binds T, and both methods agree, $T" for T in REAL_ELTYPES
    a, b = NetworkParameters(wide_set(1, T)), NetworkParameters(wide_set(2, T))
    @test which(_dot, Tuple{typeof(a), typeof(b)}) !==
          which(_dot, Tuple{DottableSet, DottableSet})
    @test _dot(a, b) == invoke(_dot, Tuple{DottableSet, DottableSet}, a, b)
    @test _dot(a, b) isa T
    @test _measured_dot(a, b) == 0
end

# `l2norm` is zero for every shape as of 0.6.0, and it was not before. It used to allow "one 32-byte
# `vec` wrapper per matrix leaf", because `l2norm(a::AbstractMatrix)` was `l2norm(vec(a))` here -- one
# of the two pirated methods of issue #16 group 1 -- and `vec` of a `Matrix` allocates the reshape
# wrapper. `GeometricBase` 0.14.9 takes `L2norm(x::AbstractArray)` where it had `AbstractVector`, so
# both pirated methods are deleted and no `vec` is taken. The exact zero is the point of asserting it.
_measured_norm(a) = (l2norm(a); @allocated l2norm(a))

# The elementwise primitives, which every `update!` and every `OptimizerStatus` runs. Each walks the
# set down to the free parameters with `mapstorage!`; one that took the flattening instead would
# allocate two flat vectors per call. A barrier of fixed arity per call shape, and not a `Vararg`
# splat, which Julia 1.11 boxes.
_measured2(f::F, a, b) where {F} = (f(a, b); @allocated f(a, b))
_measured3(f::F, a, b, c) where {F} = (f(a, b, c); @allocated f(a, b, c))
_measured4(f::F, a, b, c, d) where {F} = (f(a, b, c, d); @allocated f(a, b, c, d))

function primitive_set(::Type{T}, seed) where {T}
    rng = Random.Xoshiro(seed)
    NetworkParameters((
        L1 = (A = rand(rng, StiefelLieAlgHorMatrix{T}, N, n),
            S = rand(rng, SymmetricMatrix{T}, n)),
        L2 = (W = rand(rng, T, 3, 4) .+ one(T), b = rand(rng, T, 5) .+ one(T))))
end

@testset "the elementwise primitives allocate nothing, $T" for T in REAL_ELTYPES
    a, b, c = primitive_set(T, 1), primitive_set(T, 2), primitive_set(T, 3)
    @test _measured2(_rmul!, c, T(2)) == 0
    @test _measured2(_add!, c, b) == 0
    @test _measured2(_add!, c, T(2)) == 0
    @test _measured2(_square!, c, a) == 0
    @test _measured2(_rac!, c, a) == 0
    @test _measured2(_copyto!, c, a) == 0
    @test _measured3(_difference!, c, a, b) == 0
    @test _measured3(_div!, c, a, b) == 0
    @test eltype(flatten(c)[1]) == T
    # the control: the barrier sees an allocation where there is one
    @test _measured2((x, y) -> flatten(x), a, b) > 0
end

@testset "l2norm allocates nothing, for every shape, $T" for T in REAL_ELTYPES
    for a in (lift(T, 1), flat_container(T, 1), container(T, 1),
        NetworkParameters((
        a = rand(Random.Xoshiro(9), T, 4), b = rand(Random.Xoshiro(10), T, 5))),
        NetworkParameters(wide_set(1, T)))
        @test eltype(l2norm(a)) == T
        @test _measured_norm(a) == 0
    end
end

# `solution_scale` shares `_sumsq_leaves` with `l2norm` and differs only in the leaf function, and
# `_manifold_αmax` is the fourth of the folds this release replaced -- the one issue #70's count of
# three omitted, and the only one on the per-iteration path. Neither was pinned here before.
_measured_scale(a) = (solution_scale(a); @allocated solution_scale(a))
_measured_αmax(a, b, c) = (_manifold_αmax(a, b, c); @allocated _manifold_αmax(a, b, c))

@testset "the other two folds allocate nothing either, $T" for T in REAL_ELTYPES
    for a in (lift(T, 1), flat_container(T, 1), container(T, 1),
        NetworkParameters(wide_set(1, T)))
        @test eltype(solution_scale(a)) == T
        @test _measured_scale(a) == 0
    end
    for (a, b) in ((flat_container(T, 1), flat_container(T, 11)),
        (container(T, 1), container(T, 11)),
        (NetworkParameters(wide_set(1, T)), NetworkParameters(wide_set(2, T))))
        @test eltype(_manifold_αmax(a, b, one(T))) == T
        @test _measured_αmax(a, b, one(T)) == 0
    end
end

# The three shapes of solution this package accepts, each with an objective. Used by the testset below
# and named here so that "for every shape" is a list rather than a claim. A whole set of parameters is
# one of the three whether it is flat or nested, because both arrive as a `NetworkParameters`.
function manifold_problem(::Type{T}) where {T}
    let B = T.(B)
        (rand(Random.Xoshiro(4), StiefelManifold{T}, N, n),
            Y -> sum(abs2, Y * ones(T, n, m) .- B) / 2)
    end
end

function flat_problem(::Type{T}) where {T}
    let B = T.(B)
        (
            NetworkParameters((Y = rand(Random.Xoshiro(1), StiefelManifold{T}, N, n),
                W = randn(Random.Xoshiro(2), T, n, m), b = zeros(T, N))),
            ps -> sum(abs2, ps.Y * ps.W .+ ps.b .- B) / 2)
    end
end

function container_problem(::Type{T}) where {T}
    let (ps, _) = flat_problem(T), B = T.(B)
        (NetworkParameters((L1 = (Y = ps.Y,), L2 = (W = ps.W, b = ps.b))),
            ps -> sum(abs2, ps.L1.Y * ps.L2.W .+ ps.L2.b .- B) / 2)
    end
end

vector_problem(::Type{T}) where {T} = (randn(Random.Xoshiro(3), T, 12), v -> sum(abs2, v))

# The three sites, directly. Going through `update!` instead would be measuring something else: the
# `γᵀQγ` and both `outer!`s sit inside the `curvature_is_usable` branch, and calling `update!` twice at
# one iterate -- which is what a `@allocated` needs, one call to compile and one to measure -- leaves
# `Δg` identically zero, because the cache advances `state.ḡ` itself as soon as it has used it. The
# branch is then skipped both times and the figure is the cost of not running it. The end-to-end
# figure, taken over a whole `solve!` where the branch does fire, is in the CHANGELOG.

_measured_secant!(c) = (_flat_secant(c); @allocated _flat_secant(c))
# the first half of the cache's `update!`, which refreshes the flat mirror of `δ` and forms `δδᵀ`
_measured_update!(c, state, x) = (update!(c, state, x); @allocated update!(c, state, x))
_measured_outer!(m, a, b) = (outer!(m, a, b); @allocated outer!(m, a, b))
_measured_quad(γ, Q) = (dot(γ, Q, γ); @allocated dot(γ, Q, γ))
function _measured_mul!(c, A, b, scratch)
    (_flat_mul!(c, A, b, scratch);
        @allocated _flat_mul!(c, A, b, scratch))
end

function quasi_newton_points(::Type{T}) where {T}
    (("Vector", vector_problem(T)[1]),
        ("Manifold", manifold_problem(T)[1]),
        ("flat container", flat_problem(T)[1]),
        ("nested container", container_problem(T)[1]))
end

@testset "the flat sites of $(nameof(typeof(algorithm))) allocate nothing, $T" for T in REAL_ELTYPES,
    algorithm in (BFGS(), DFP())

    for (name, x) in quasi_newton_points(T)
        c = OptimizerCache(algorithm, x)
        state = OptimizerState(algorithm, x)
        Q = inverse_hessian(state)
        @test eltype(Q) == T

        # filling the flat mirrors of the secant pair, in the quasi-Newton update and in the
        # cache's `update!` at an iterate
        @test _measured_secant!(c) == 0
        @test _measured_update!(c, state, x) == 0

        δ, γ = _flat_secant(c)

        # `outer!`, which used to flatten both of its arguments on every call
        m = zeros(T, length(δ), length(γ))
        @test _measured_outer!(m, δ, γ) == 0

        # `γᵀQγ`, which used to materialise `Q * γ`
        @test _measured_quad(γ, Q) == 0

        # the product with `Q`, which flattens `b` and the result into the cache's buffers
        @test _measured_mul!(direction(c), Q, rhs(c), c.flat) == 0
    end
end

# The update of the inverse Hessian itself, which calls the sites above, measured as one call. An edit
# to it that formed the secant pair without `_flat_secant`, or a product without the cache's buffers,
# would allocate here and nowhere above.
#
# The update runs only where `curvature_is_usable` holds, so the secant pair is set by hand to one for
# which it does, and `_update_inverse_hessian!` reads it without changing it: the branch runs on the
# warm-up call and on the measured one. `γ = 2δ` and not `γ = δ`, because with `Q = I` the BFGS
# correction for `γ = δ` is zero, and the assertion that `Q` changed could then not tell a skipped
# branch from a taken one. `δ` is a constant fill, which is enough for an allocation count.
@testset "the $(nameof(typeof(algorithm))) update of the inverse Hessian allocates nothing, $T" for T in REAL_ELTYPES,
    algorithm in (BFGS(), DFP())

    for (name, x) in quasi_newton_points(T)
        c = OptimizerCache(algorithm, x)
        state = OptimizerState(algorithm, x)
        _fill!(c.Δx, T(1) / 10)
        _copyto!(c.Δg, c.Δx)
        _rmul!(c.Δg, T(2))
        ΔxΔg = _dot(c.Δx, c.Δg)
        @test ΔxΔg isa T
        @test curvature_is_usable(ΔxΔg, c.Δx, c.Δg)

        Q₀ = copy(inverse_hessian(state))
        @test _measured4(_update_inverse_hessian!, algorithm, c, state, ΔxΔg) == 0
        # the branch ran: `Q` is not where it started, and the pair it was taken on is unchanged by
        # the two calls, so the guard held on both
        @test inverse_hessian(state) != Q₀
        @test _dot(c.Δx, c.Δg) == ΔxΔg
        @test eltype(inverse_hessian(state)) == T

        # the control: the barrier sees an allocation where there is one
        @test _measured4(
            (a, c, s, d) -> copy(inverse_hessian(s)), algorithm, c, state, ΔxΔg) > 0
    end
end

# The retraction, which is the other two thirds.
#
# A step on a manifold applies a retraction once per line-search trial and three times besides --
# six to fifteen times an iteration, measured -- and every one of them used to rebuild every matrix
# it needs. `RetractionWorkspace` holds them instead, and `scripts/retraction_step_allocations.jl`
# carries the end-to-end figures.
#
# **What is asserted is N-independence, not a byte count.** That is the property the workspace
# exists to establish and the one a byte count cannot state: everything a `Cayley` retraction of an
# `N × n` point allocates now comes from the `2n × 2n` `inv`, so the same lift shape at two very
# different ambient dimensions has to cost the same. A reintroduced `N × N` or `N × 2n` temporary
# makes the two diverge whatever its size, where a ceiling on either would have to be loose enough
# to hide it. This is the argument `test/quality/aqua.jl` makes for piracy and `test/ambiguities.jl`
# for ambiguities, one file over.
#
# `Geodesic` is asserted as an identity instead: its `𝔄` is evaluated in the workspace and allocates
# nothing, so the whole retraction costs what `lift_factors!` costs. See the comment at that
# assertion.
#
# No assertion is made on the bytes of a whole `solver_step!`. The figure is a function of how many
# trials the line search takes, which is a property of the problem and not of this package: the same
# step measured over 21 repeats ran from 50 112 to 95 824 bytes. The script has the medians.

const RETRACTIONS = (Cayley(), Geodesic())
const LIFT_TYPES = (StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix)

# One lift and one section per shape, at a fixed `n` and two ambient dimensions an order of
# magnitude apart.
function retraction_fixture(::Type{T}, LT, N, n) where {T}
    B = rand(Random.Xoshiro(N * 100 + n), LT{T}, N, n)
    MT = LT == StiefelLieAlgHorMatrix ? StiefelManifold : GrassmannManifold
    Y = rand(Random.Xoshiro(N + n), MT{T}, N, n)

    (B = B, Λ = GlobalSection(Y), Λ₂ = GlobalSection(Y), ws = retraction_workspace(Y))
end

# `@allocated` inside the function and the call warmed first, for the reason the head of this file
# gives.
function _measured_retraction(ws, R, B)
    retraction_matrix!(ws, R, B)
    @allocated retraction_matrix!(ws, R, B)
end

function _measured_lift_factors(ws, B)
    lift_factors!(ws, B)
    @allocated lift_factors!(ws, B)
end

function _measured_update_section(Λ₂, Λ, B, R, ws)
    update_section!(Λ₂, Λ, B, R, ws)
    @allocated update_section!(Λ₂, Λ, B, R, ws)
end

# The difference across `N` and not an equality, and the tolerance is the part to read: see
# `test/helpers/allocations.jl` for why a byte count is compared under a tolerance.
#
# **The tolerance does not weaken what is asserted**, because of the size the `large` fixture is:
# one reintroduced `N × N` `Float64` temporary at `N = 200` is 320 000 bytes, and one `N × 2n` is
# 9 600; in `Float32` they are half that, 160 000 and 4 800. The gap between those and 1 024 is what
# makes this a property and not a ceiling -- a ceiling on the absolute figure would have to sit above
# 3 792 and so could hide an `N × 2n` temporary entirely.
include("../helpers/allocations.jl")

@testset "the retraction of a $LT does not grow with N, $T" for T in REAL_ELTYPES,
    LT in LIFT_TYPES

    small, large = retraction_fixture(T, LT, 6, 3), retraction_fixture(T, LT, 200, 3)
    @test eltype(small.ws.retracted) == eltype(large.B) == T

    # `lift_factors!` writes into buffers it was handed, and on the host it densifies the lift's `A`
    # block with a broadcast rather than a kernel launch, so it costs nothing at all.
    @test n_independent(_measured_lift_factors(small.ws, small.B),
        _measured_lift_factors(large.ws, large.B))

    # Cayley, where everything left is the `2n × 2n` `inv`.
    @test n_independent(_measured_retraction(small.ws, Cayley(), small.B),
        _measured_retraction(large.ws, Cayley(), large.B))
    @test n_independent(
        _measured_update_section(small.Λ₂, small.Λ, small.B, Cayley(), small.ws),
        _measured_update_section(large.Λ₂, large.Λ, large.B, Cayley(), large.ws))

    # The geodesic evaluates its `𝔄` in the workspace too (issue #77), so it adds nothing to what
    # writing the lift's factors costs, at both ambient dimensions. The number of squarings still
    # grows with the norm of the lift, and so with `N` for a random lift; each squaring writes into
    # the same two buffers.
    for f in (small, large)
        @test n_independent(_measured_retraction(f.ws, Geodesic(), f.B),
            _measured_lift_factors(f.ws, f.B))
    end
end

# The workspace may not change the answer, and nothing else in the suite compares the two paths --
# every other retraction test goes through whichever one the `Optimizer` chose. `==` and not `≈`:
# the two take the same products into different arrays, and at these shapes, whose inner dimension
# `2n` fits in one BLAS block, they agree bit for bit; `≈` would pass on a swapped block. That is a
# property of the shapes, not of the two paths: where an inner dimension spans several blocks, as the
# `N - n` of `update_section!`'s transport does at `N = 400`, a five-argument `mul!` and a sum of two
# products round differently.
@testset "the workspace retraction is the allocating one, for a $LT, $T" for T in REAL_ELTYPES,
    LT in LIFT_TYPES

    for (N, n) in ((6, 3), (6, 1), (6, 6), (20, 4))
        f = retraction_fixture(T, LT, N, n)
        for R in RETRACTIONS
            @test eltype(retraction_matrix!(f.ws, R, f.B)) == T
            @test retraction_matrix!(f.ws, R, f.B) == retraction(R, f.B).A
        end
    end
end

# A retraction this package does not ship, which is what the generic `retraction_matrix!` arm exists
# for: `Cayley` and `Geodesic` each have their own in-place method, so nothing in the package reaches
# that arm and nothing else in the suite covers it. It is a documented extension point, so it is
# pinned here rather than left to a downstream caller to discover.
struct _DownstreamRetraction <: AbstractRetraction end
GeometricOptimizers.retraction(::_DownstreamRetraction, x::AbstractArray) = cayley(x)

# The shape assertion is what makes the fallback safe, and it cannot be left implicit: the arm
# reaches `ws.retracted` through `copyto!`, which copies linearly into an oversized destination
# instead of throwing. A workspace built for another `N` would therefore return a scrambled layout
# rather than an error, so the rejection is asserted and not only the agreement.
@testset "a retraction this package does not ship reaches the fallback, for a $LT, $T" for T in REAL_ELTYPES,
    LT in LIFT_TYPES

    f = retraction_fixture(T, LT, 6, 3)
    R = _DownstreamRetraction()
    @test eltype(retraction_matrix!(f.ws, R, f.B)) == T

    # Agreement with the allocating form is close to true by construction here -- this arm *is* the
    # allocating form plus a copy, unlike the `Cayley` and `Geodesic` arms the testset above
    # compares. What it pins that construction does not is the rest of the contract: the arm
    # dispatches at all, and the answer lands in the workspace buffer rather than in a fresh array.
    @test retraction_matrix!(f.ws, R, f.B) == retraction(R, f.B).A
    @test retraction_matrix!(f.ws, R, f.B) === f.ws.retracted
    @test_throws AssertionError retraction_matrix!(retraction_fixture(T, LT, 20, 3).ws, R, f.B)
end

# One fixture and one destination written twice, not two fixtures: `global_section` draws a random
# complement from the global RNG, so two `GlobalSection`s of the same point hold different frames and
# comparing across them would compare two different transports.
@testset "update_section! writes the same section with and without a workspace, $T" for T in REAL_ELTYPES
    for LT in LIFT_TYPES, (N, n) in ((6, 3), (20, 4))

        f = retraction_fixture(T, LT, N, n)
        for R in RETRACTIONS
            update_section!(f.Λ₂, f.Λ, f.B, R, nothing)
            reference_Y, reference_λ = copy(f.Λ₂.Y.A), copy(f.Λ₂.λ)
            fill!(f.Λ₂.Y.A, zero(T))
            fill!(f.Λ₂.λ, zero(T))

            update_section!(f.Λ₂, f.Λ, f.B, R, f.ws)
            @test eltype(f.Λ₂.Y.A) == eltype(f.Λ₂.λ) == T
            @test f.Λ₂.Y.A == reference_Y
            @test f.Λ₂.λ == reference_λ
        end
    end
end

# The body of `solve!`'s loop, and the standard the manifold path is measured against. It is
# already zero for an ordinary vector and nothing else asserts it, so this pins behaviour rather
# than reproducing a defect -- and it is the assertion that would catch a new allocation on the
# part of the step path both kinds of parameter share.
function _step!(x, state, opt)
    increase_iteration_number!(state)
    solver_step!(x, state, opt)
    f = value(problem(opt), x)
    OptimizerStatus(state, cache(opt), f; config = config(opt))
    update!(state, opt, x, f)

    f
end

# The measurement is a one-line function whose arguments are all parameters, and the optimizer is
# built by the caller below -- the same separation the head of this file insists on, for the same
# reason and with the same number. A construction inside the measuring function captures into a
# `Core.Box` that Julia 1.11 does not elide and 1.13 does, which reads as **16 bytes** on the older
# version and 0 on the newer -- a property of the measurement and not of `src/`. `solver_step!`
# itself allocates nothing on either version, and neither does `trial_iterate!` with a workspace.
_measured_step(x, state, opt) = (_step!(x, state, opt); @allocated _step!(x, state, opt))

function _euclidean_step(x, F, algorithm)
    Random.seed!(1234)
    opt = Optimizer(x, F; algorithm = algorithm, max_iterations = 10_000)
    state = OptimizerState(algorithm, x)
    initialize_state!(state)

    _measured_step(x, state, opt)
end

@testset "a Euclidean iteration of $(nameof(typeof(algorithm))) allocates nothing, $T" for T in REAL_ELTYPES,
    algorithm in (BFGS(), DFP(), GradientMethod())

    x, F = vector_problem(T)
    @test _euclidean_step(x, F, algorithm) == 0
    @test eltype(x) == eltype(F(x)) == T
end
