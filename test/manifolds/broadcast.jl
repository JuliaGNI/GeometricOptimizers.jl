# A broadcast over a manifold point returns a plain array, in both spellings.
#
# `Base.broadcast(operation, Y::Manifold)` used to rewrap the result in the manifold type, which
# claims an invariant the result does not hold: adding 1 to every entry of a point of `St(4,2)` gives
# a `StiefelManifold` whose `check` is of order 10 — `5.0` in `Float64` and `7.3` in `Float32` at
# the seed below — against a few `eps(T)` for a point. The figure depends on the draw, which is why the assertion is `> 1`. Dot syntax never
# reached that method — `Y .+ 1` lowers through `broadcasted`/`materialize` — so the two spellings
# of one operation returned different types and only the wrapped one lied.
#
# This file is what stops the method coming back. The assertions are on the *type* of the result,
# because the values were never wrong; and they cover both manifolds, because the deleted method was
# written on `Manifold` and a new one would be too.
#
# `_round` is asserted beside them for contrast. It rewraps deliberately — rounding a point's
# entries for display leaves it on the manifold, and the docstrings that print one need the type
# back — and it never reached the deleted method, because it is written in dot syntax.

using GeometricOptimizers
using GeometricOptimizers: _round, check, manifold_constructor
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using Test
import Random

include("../helpers/eltypes.jl")

const N, n = 4, 2

@testset "a broadcast over a manifold point returns a plain array, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for MT in (StiefelManifold{T}, GrassmannManifold{T})
        Y = rand(rng, MT, N, n)
        @testset "$(nameof(typeof(Y)))" begin
            # the explicit spelling, which is the one that reached the deleted method
            @test broadcast(x -> x + 1, Y) isa Matrix{T}
            @test !(broadcast(x -> x + 1, Y) isa Manifold)
            @test eltype(broadcast(x -> x + 1, Y)) == T

            # the dot spelling, which never did
            @test (Y .+ 1) isa Matrix{T}

            # so the two now agree, on the type and on the values
            @test broadcast(x -> x + 1, Y) == Y .+ 1

            # the values were never the problem; the wrapper was
            @test broadcast(x -> x + 1, Y) ≈ Y.A .+ 1

            # and this is why a wrapped result would be a lie: the shifted point is nowhere near
            # the manifold, where `Y` itself is on it, to the round-off of a CholeskyQR2 of a
            # `4 × 2` Gaussian (measured under `6eps(T)` in both precisions over 200 seeds)
            @test check(Y) < 16eps(T)
            @test check(manifold_constructor(Y)(broadcast(x -> x + 1, Y))) > 1
        end
    end
end

@testset "_round still returns the manifold type, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(124)
    for MT in (StiefelManifold{T}, GrassmannManifold{T})
        Y = rand(rng, MT, N, n)
        rounded = _round(Y; digits = 3)
        @test rounded isa MT
        @test eltype(rounded) == T
        @test rounded.A == round.(Y.A; digits = 3)
    end
end

# `_round` rounds `Y.A` and not `Y`, and on a device-backed point that is load-bearing rather than a
# matter of taste. `Manifold` declares no `Broadcast.BroadcastStyle`, so `round.(Y)` falls through to
# the manifold's scalar `getindex`, which a device array disallows. `JLArray` stands in for the
# device here, as it does in `similar_backend.jl`. This pins the distinction so that a later
# simplification of `_round` to `round.(Y)` fails here rather than on a GPU.
@testset "_round stays on the device, $T" for T in REAL_ELTYPES
    # `allowscalar(false)` is not redundant, for the reason `retractions/exponential_accuracy.jl`
    # gives: `GPUArraysCore`'s default is `ScalarDisallowed` only in a non-interactive session, so
    # from a REPL a scalar index would merely warn and three of the assertions below would pass
    # whatever the code did. The setting is task-global and left set, as it is there; every file
    # included between this one and that one is host-only, so none of them consults it.
    allowscalar(false)

    Y = StiefelManifold(JLArray(Matrix(rand(Random.Xoshiro(125), StiefelManifold{T}, N, n).A)))
    @test _round(Y; digits = 3) isa StiefelManifold{T, <:JLArray}
    @test eltype(_round(Y; digits = 3)) == T
    @test_throws "Scalar indexing is disallowed" round.(Y; digits = 3)

    # and the deletion this file is about is what makes the explicit spelling fail here too, which
    # `Y .+ 1` already did before it
    @test_throws "Scalar indexing is disallowed" broadcast(x -> x + 1, Y)
    @test_throws "Scalar indexing is disallowed" Y .+ 1
    @test broadcast(x -> x + 1, Y.A) isa JLArray
end
