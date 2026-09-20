# A computation whose two operands are on different backends is refused, by name.
#
# This is not only about the message. Measured on `origin/main` with `JLArrays` standing in for the
# device and `allowscalar(false)` set, **seven of twelve mixed-backend operations did not throw at
# all**: `SkewSymMatrix(host) + SkewSymMatrix(device)` returned a device matrix,
# `SkewSymMatrix(device) * host` returned a `JLArray`, and `StiefelManifold(device) * host` returned
# a **host** `Matrix` — the device operand was pulled off the device and nothing said so. Which
# backend the answer landed on depended on the argument order. The five that did throw said
# `Scalar indexing is disallowed`, which names neither operand.
#
# So the assertions below are regressions in both directions: the seven pin an answer that used to
# come back wrong, and the five pin an error that used to name nothing. On real Metal the whole set
# fails instead, inside `GPU compilation of MethodInstance for …broadcast_linear…`, which is the
# failure `CHANGELOG.md` records for §20 — that one is not reachable from here, and the host/device
# pair `JLArrays` gives is.
#
# Three operations are deliberately absent, because they must keep crossing backends: `copyto!` and
# `assign!` are transfers, which is the contract `Base` sets for `copyto!` and the one PR #85 settled
# for `assign!`, and `changebackend` is the other supported route. The last testset holds `copyto!`
# to that, so a later widening of the guard cannot quietly take it.

using GeometricOptimizers
using GeometricOptimizers: LowerTriangular, StiefelProjection, UpperTriangular, add!
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using Random
using Test

Random.seed!(2718)

const T = Float32
const N, n = 6, 3

allowscalar(false)

host_skew = SkewSymMatrix(rand(T, N, N))
dev_skew = SkewSymMatrix(JLArray(rand(T, N, N)))
host_sym = SymmetricMatrix(rand(T, N, N))
dev_sym = SymmetricMatrix(JLArray(rand(T, N, N)))
host_mat = rand(T, N, N)
dev_mat = JLArray(rand(T, N, N))

@testset "`+`, `-` and `add!` refuse a mixed-backend pair" begin
    @test_throws ArgumentError host_skew + dev_skew
    @test_throws ArgumentError dev_skew + host_skew
    @test_throws ArgumentError host_skew - dev_skew
    @test_throws ArgumentError host_skew + dev_mat
    @test_throws ArgumentError dev_skew + host_mat

    @test_throws ArgumentError host_sym + dev_sym
    @test_throws ArgumentError host_sym - dev_sym

    @test_throws ArgumentError add!(SkewSymMatrix(rand(T, N, N)), host_skew, dev_skew)

    # These three reach the guard only because their signatures were unbound from one shared type
    # variable in this change: as `(A::AT, B::AT) where {AT <: AbstractTriangular}` a host and a
    # device triangular are already different concrete types, so `AT` could not bind both and the
    # call fell through to `Base`'s generic array `+`. See the comment on `_triangular_species`.
    host_lo = LowerTriangular(rand(T, N, N))
    dev_lo = LowerTriangular(JLArray(rand(T, N, N)))
    @test_throws ArgumentError host_lo + dev_lo
    @test_throws ArgumentError host_lo - dev_lo
    @test_throws ArgumentError add!(LowerTriangular(rand(T, N, N)), host_lo, dev_lo)

    host_lift = rand(StiefelLieAlgHorMatrix{T}, N, n)
    dev_lift = StiefelLieAlgHorMatrix(
        SkewSymMatrix(JLArray(Matrix(host_lift.A))), JLArray(Matrix(host_lift.B)), N, n)
    @test_throws ArgumentError host_lift + dev_lift
    @test_throws ArgumentError host_lift - dev_lift
end

@testset "`*` refuses a mixed-backend pair" begin
    @test_throws ArgumentError host_skew * dev_mat
    @test_throws ArgumentError dev_skew * host_mat
    @test_throws ArgumentError host_sym * dev_mat
    @test_throws ArgumentError dev_sym * host_mat

    host_lo = LowerTriangular(rand(T, N, N))
    @test_throws ArgumentError host_lo * dev_mat

    host_Y = rand(StiefelManifold{T}, N, n)
    dev_Y = StiefelManifold(JLArray(Matrix(host_Y.A)))
    @test_throws ArgumentError host_Y * JLArray(rand(T, n, n))
    @test_throws ArgumentError dev_Y * rand(T, n, n)

    host_E = StiefelProjection(T, N, n)
    @test_throws ArgumentError host_E * JLArray(rand(T, n, n))
    @test_throws ArgumentError dev_mat * host_E
end

# The message is the whole point of the change, so it is asserted rather than assumed: a bare
# `ArgumentError` would satisfy every `@test_throws` above and still tell a caller nothing.
@testset "the refusal names both operands and both backends" begin
    message = try
        host_skew + dev_mat
        ""
    catch err
        sprint(showerror, err)
    end

    @test occursin("SkewSymMatrix", message)
    @test occursin("JLArray", message)
    @test occursin(string(KernelAbstractions.get_backend(host_mat)), message)
    @test occursin(string(KernelAbstractions.get_backend(dev_mat)), message)
end

@testset "a same-backend pair is untouched, on the host and on the device" begin
    @test host_skew + host_skew isa SkewSymMatrix
    @test host_skew * host_mat isa AbstractMatrix
    @test dev_skew + dev_skew isa SkewSymMatrix
    @test dev_skew * dev_mat isa AbstractMatrix
    @test host_sym + host_sym isa SymmetricMatrix
    @test dev_sym + dev_sym isa SymmetricMatrix

    # The triangular sum keeps its own species and its own storage. The device arm is what says the
    # unbinding above did not cost the type its own method: on one shared type variable this pair
    # dispatched correctly and a mixed one did not, so only a same-backend device pair distinguishes
    # "reaches the right method" from "reaches `Base`'s".
    @test LowerTriangular(rand(T, N, N)) + LowerTriangular(rand(T, N, N)) isa
          LowerTriangular
    dev_lo = LowerTriangular(JLArray(rand(T, N, N)))
    @test dev_lo + dev_lo isa LowerTriangular
    @test parent(dev_lo + dev_lo) isa JLArray
end

# Two independent arguments would otherwise let the two species be added and silently return one of
# them, so the species is compared at run time. `copyto!` on these types already did this.
@testset "a triangular refuses the other species" begin
    @test_throws ArgumentError LowerTriangular(rand(T, N, N)) +
                               UpperTriangular(rand(T, N, N))
    @test_throws ArgumentError LowerTriangular(rand(T, N, N)) -
                               UpperTriangular(rand(T, N, N))
end

# `KernelAbstractions.get_backend` *raises* for an array type it has no method for, rather than
# answering. `StiefelLieAlgHorMatrix(vec(B), N, n)` is such a case: the blocks are views into a
# `LazyArrays.Vcat`, both operands are on the host, and the operation is fine. A guard that turned
# that raise into a refusal broke 24 host-only assertions in
# `test/lie_algebras/stiefel_lie_algebra_horizontal.jl`, so the rule is that the guard refuses only
# what it can prove.
@testset "an unplaceable array is let through rather than refused" begin
    B = rand(StiefelLieAlgHorMatrix{T}, N, n)
    lazy = StiefelLieAlgHorMatrix(vec(B), N, n)

    # the premise: this really is a type `get_backend` cannot answer for
    @test_throws ArgumentError KernelAbstractions.get_backend(lazy)

    @test isapprox(lazy - lazy, B - B)
    @test isapprox(lazy + lazy, B + B)
end

# The three transfer operations keep crossing backends. `copyto!` is the one with a contract in
# `Base`, and PR #85 is what made it work for these types in the first place.
@testset "`copyto!` still crosses backends" begin
    destination = SkewSymMatrix(JLArray(zeros(T, N, N)))
    copyto!(destination, host_skew)
    # `.S` is the packed storage vector, so this compares the entries that were transferred
    @test Array(destination.S) ≈ host_skew.S
end
