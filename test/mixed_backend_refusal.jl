# A computation whose two operands are on different backends is refused, by name.
#
# This is not only about the message. Unguarded, a mixed pair does not fail cleanly on a backend
# that can fall back to the host — `JLArrays` with `allowscalar(false)` set is the one reachable from
# here. Such a pair silently picks a side: `SkewSymMatrix(host) + SkewSymMatrix(device)` gives a
# device matrix, `SkewSymMatrix(device) * host` a `JLArray`, and `StiefelManifold(device) * host` a
# **host** `Matrix` — the device operand comes off the device and nothing says so, and which backend
# the answer lands on follows the argument order. Where such a pair does raise instead, it says
# `Scalar indexing is disallowed`, which names neither operand.
#
# So the assertions below pin both halves: an answer that otherwise comes back on a backend the
# caller did not choose, and an error that otherwise names nothing. On real Metal the whole set fails
# differently, inside `GPU compilation of MethodInstance for …broadcast_linear…`, which is the
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
using LinearAlgebra: Adjoint, transpose
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

    # These three reach the guard only because their signatures bind a type variable per argument: as
    # `(A::AT, B::AT) where {AT <: AbstractTriangular}` a host and a device triangular are different
    # concrete types, so `AT` cannot bind both and the call reaches `Base`'s generic array `+`
    # instead. See the comment on `_triangular_species`.
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

# The row-vector and adjoint forms are the ones that answered on the *host* from a device operand,
# which is the quietest way to get a wrong answer here: nothing about a host `Adjoint` says the
# matrix it came from was on a device. `parent(·)` is what the guard reads for an `Adjoint`-wrapped
# point, because the owned type is inside the wrapper there.
@testset "the row-vector and adjoint products refuse a mixed-backend pair" begin
    host_Y = rand(StiefelManifold{T}, N, n)
    dev_Y = StiefelManifold(JLArray(Matrix(host_Y.A)))
    host_row = rand(T, N)'
    dev_row = JLArray(rand(T, N))'

    @test_throws ArgumentError host_row * dev_Y
    @test_throws ArgumentError transpose(rand(T, N)) * dev_Y
    @test_throws ArgumentError dev_row * host_Y
    @test_throws ArgumentError host_Y' * dev_mat
    @test_throws ArgumentError dev_Y' * host_mat
    @test_throws ArgumentError host_Y' * dev_Y

    host_E = StiefelProjection(T, N, n)
    dev_E = StiefelProjection(KernelAbstractions.get_backend(dev_mat), T, N, n)
    @test_throws ArgumentError host_row * dev_E
    @test_throws ArgumentError dev_row * host_E

    # and the same pairs on one backend still answer
    @test host_row * host_Y isa Adjoint
    @test host_Y' * host_mat isa AbstractMatrix
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
    # per-argument binding above does not cost the type its own method: on one shared type variable
    # this pair dispatches correctly and a mixed one does not, so only a same-backend device pair
    # distinguishes "reaches the right method" from "reaches `Base`'s".
    @test LowerTriangular(rand(T, N, N)) + LowerTriangular(rand(T, N, N)) isa
          LowerTriangular
    dev_lo = LowerTriangular(JLArray(rand(T, N, N)))
    @test dev_lo + dev_lo isa LowerTriangular
    @test parent(dev_lo + dev_lo) isa JLArray
end

# Two independent arguments and no species check would read one species' storage into the other's
# triangle. The sum of a lower and an upper triangular is a general matrix, though, so the pair goes
# to the dense path rather than being refused — the answer `Base`'s generic `+` gives, and the one
# `*` between the two species already gives.
@testset "the two triangular species sum to a dense matrix" begin
    lo = LowerTriangular(rand(T, N, N))
    up = UpperTriangular(rand(T, N, N))

    @test lo + up ≈ Matrix(lo) + Matrix(up)
    @test lo - up ≈ Matrix(lo) - Matrix(up)
    @test lo + up isa Matrix
    @test lo * up isa Matrix

    # `add!` is the exception, and not by choice: its destination is one species and cannot hold the
    # sum of the two
    @test_throws ArgumentError add!(
        LowerTriangular(rand(T, N, N)), lo, UpperTriangular(rand(T, N, N)))
end

# `KernelAbstractions.get_backend` *raises* for an array type it has no method for, rather than
# answering. `StiefelLieAlgHorMatrix(vec(B), N, n)` is such a case: the blocks are views into a
# `LazyArrays.Vcat`, both operands are on the host, and the operation is fine. A guard that turned
# that raise into a refusal would reject the host-only assertions in
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
# `Base`; PR #85 is the reference for how these types meet it.
@testset "`copyto!` still crosses backends" begin
    destination = SkewSymMatrix(JLArray(zeros(T, N, N)))
    copyto!(destination, host_skew)
    # `.S` is the packed storage vector, so this compares the entries that were transferred
    @test Array(destination.S) ≈ host_skew.S
end
