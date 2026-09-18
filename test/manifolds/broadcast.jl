# A broadcast over a manifold point returns a plain array, in both spellings.
#
# `Base.broadcast(operation, Y::Manifold)` used to rewrap the result in the manifold type, which
# claims an invariant the result does not hold: adding 1 to every entry of a point of `St(4,2)` gives
# a `StiefelManifold` whose `check` is of order 10 — `11.07` at the seed below — against `~1e-16`
# for a point. The figure depends on the draw, which is why the assertion is `> 1`. Dot syntax never
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

Random.seed!(123)

const N, n = 4, 2

@testset "a broadcast over a manifold point returns a plain array" begin
    for MT in (StiefelManifold{Float64}, GrassmannManifold{Float64})
        Y = rand(MT, N, n)
        @testset "$(nameof(typeof(Y)))" begin
            # the explicit spelling, which is the one that reached the deleted method
            @test broadcast(x -> x + 1, Y) isa Matrix{Float64}
            @test !(broadcast(x -> x + 1, Y) isa Manifold)

            # the dot spelling, which never did
            @test (Y .+ 1) isa Matrix{Float64}

            # so the two now agree, on the type and on the values
            @test broadcast(x -> x + 1, Y) == Y .+ 1

            # the values were never the problem; the wrapper was
            @test broadcast(x -> x + 1, Y) ≈ Y.A .+ 1

            # and this is why a wrapped result would be a lie: the shifted point is nowhere near
            # the manifold, where `Y` itself is on it
            @test check(Y) < 1.0e-14
            @test check(manifold_constructor(Y)(broadcast(x -> x + 1, Y))) > 1
        end
    end
end

@testset "_round still returns the manifold type" begin
    for MT in (StiefelManifold{Float64}, GrassmannManifold{Float64})
        Y = rand(MT, N, n)
        rounded = _round(Y; digits = 3)
        @test rounded isa MT
        @test rounded.A == round.(Y.A; digits = 3)
    end
end

# `_round` rounds `Y.A` and not `Y`, and on a device-backed point that is load-bearing rather than a
# matter of taste. `Manifold` declares no `Broadcast.BroadcastStyle`, so `round.(Y)` falls through to
# the manifold's scalar `getindex`, which a device array disallows. `JLArray` stands in for the
# device here, as it does in `similar_backend.jl`. This pins the distinction so that a later
# simplification of `_round` to `round.(Y)` fails here rather than on a GPU.
@testset "_round stays on the device" begin
    # `allowscalar(false)` is not redundant, for the reason `retractions/exponential_accuracy.jl`
    # gives: `GPUArraysCore`'s default is `ScalarDisallowed` only in a non-interactive session, so
    # from a REPL a scalar index would merely warn and three of the assertions below would pass
    # whatever the code did. The setting is task-global and left set, as it is there; every file
    # included between this one and that one is host-only, so none of them consults it.
    allowscalar(false)

    Y = StiefelManifold(JLArray(Matrix(rand(StiefelManifold{Float32}, N, n).A)))
    @test _round(Y; digits = 3) isa StiefelManifold{Float32, <:JLArray}
    @test_throws "Scalar indexing is disallowed" round.(Y; digits = 3)

    # and the deletion this file is about is what makes the explicit spelling fail here too, which
    # `Y .+ 1` already did before it
    @test_throws "Scalar indexing is disallowed" broadcast(x -> x + 1, Y)
    @test_throws "Scalar indexing is disallowed" Y .+ 1
    @test broadcast(x -> x + 1, Y.A) isa JLArray
end
