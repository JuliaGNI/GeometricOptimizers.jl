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
using LinearAlgebra: LinearAlgebra, Adjoint, transpose
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

# `SymplecticStiefelManifold` carries ten of the guard sites, more than any other file, and
# `scripts/mixed_backend_seam.jl` does not reach it: its enumeration stops at `StiefelManifold` and
# `StiefelProjection`. So the ten are pinned here, in both argument orders, and every one of them
# reads the point through `parent(·)` where it sits inside an `Adjoint`. A guard that read the
# wrapper instead of the point would let a mixed pair through, and the seam script would not say so.
#
# The last pair is the adjoint against another adjoint, which no point satisfies dimensionally —
# `U'` is `2n × 2N` in both slots. The guard is what it raises now, in place of the
# `DimensionMismatch` that told a caller nothing about the backends.
@testset "the symplectic products refuse a mixed-backend pair" begin
    host_U = rand(SymplecticStiefelManifold{T}, N, 4)
    dev_U = SymplecticStiefelManifold(JLArray(Matrix(host_U.A)))
    host_small = rand(T, 4, 4)
    dev_small = JLArray(rand(T, 4, 4))
    host_row = rand(T, N)'
    dev_row = JLArray(rand(T, N))'
    host_row_small = rand(T, 4)'
    dev_row_small = JLArray(rand(T, 4))'

    @test_throws ArgumentError host_U * dev_small
    @test_throws ArgumentError dev_U * host_small
    @test_throws ArgumentError host_mat * dev_U
    @test_throws ArgumentError dev_mat * host_U
    @test_throws ArgumentError host_row * dev_U
    @test_throws ArgumentError dev_row * host_U
    @test_throws ArgumentError transpose(rand(T, N)) * dev_U
    @test_throws ArgumentError host_U' * dev_mat
    @test_throws ArgumentError dev_U' * host_mat
    @test_throws ArgumentError host_small * dev_U'
    @test_throws ArgumentError dev_small * host_U'
    @test_throws ArgumentError host_row_small * dev_U'
    @test_throws ArgumentError dev_row_small * host_U'
    @test_throws ArgumentError transpose(rand(T, 4)) * dev_U'
    @test_throws ArgumentError host_U' * dev_U
    @test_throws ArgumentError host_U' * dev_U'

    # and the same pairs on one backend still answer
    @test host_U * host_small isa AbstractMatrix
    @test host_mat * host_U isa AbstractMatrix
    @test host_row * host_U isa Adjoint
    @test host_U' * host_mat isa AbstractMatrix
    @test host_small * host_U' isa AbstractMatrix
    @test host_row_small * host_U' isa Adjoint
    @test host_U' * host_U isa AbstractMatrix
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

# Without the `+` and `-` methods in `ambiguities.jl` a type with no kernel of its own falls to
# `Base`'s generic `+` or `-` at `arraymath.jl:6` and takes its backend from the argument order.
# Both orders are asserted, because the unguarded answer lands on whichever side comes first.
#
# All ten members of `OwnedMatrix` appear below, and the count is the point: a type added to that
# union and not to this testset is a widening of the guard that nothing checks.
@testset "`+` and `-` against a plain matrix refuse a mixed-backend pair" begin
    host_lo = LowerTriangular(rand(T, N, N))
    dev_lo = LowerTriangular(JLArray(rand(T, N, N)))
    host_up = UpperTriangular(rand(T, N, N))
    host_lift = rand(StiefelLieAlgHorMatrix{T}, N, n)
    host_grass = rand(GrassmannLieAlgHorMatrix{T}, N, n)
    host_Y = rand(StiefelManifold{T}, N, n)
    host_G = rand(GrassmannManifold{T}, N, n)
    host_U = rand(SymplecticStiefelManifold{T}, N, 4)
    dev_En = JLArray(rand(T, N, n))
    dev_wide = JLArray(rand(T, N, 4))
    host_E = StiefelProjection(T, N, n)

    for op in (+, -)
        for owned in (host_skew, host_sym, host_lo, host_up, host_lift, host_grass)
            @test_throws ArgumentError op(owned, dev_mat)
            @test_throws ArgumentError op(dev_mat, owned)
        end
        for owned in (host_Y, host_G, host_E)
            @test_throws ArgumentError op(owned, dev_En)
            @test_throws ArgumentError op(dev_En, owned)
        end
        @test_throws ArgumentError op(host_U, dev_wide)
        @test_throws ArgumentError op(dev_wide, host_U)

        # the device operand on the left is the arm that raises `Scalar indexing is disallowed`
        # without the guard, which names neither operand
        @test_throws ArgumentError op(dev_lo, host_mat)
        @test_throws ArgumentError op(host_mat, dev_lo)

        # and the guard fires for two owned operands of different types, which is the pair the
        # `(Owned, Owned)` method exists for
        @test_throws ArgumentError op(host_skew, SymmetricMatrix(JLArray(rand(T, N, N))))
        @test_throws ArgumentError op(host_lo, SymmetricMatrix(JLArray(rand(T, N, N))))
    end

    # A same-backend difference is untouched: same value, same type, same backend. The structured
    # results are what say the `(Owned, Owned)` method did not swallow the concrete same-type methods.
    @test host_skew - host_skew isa SkewSymMatrix
    @test host_lo - host_lo isa LowerTriangular
    @test dev_lo - dev_lo isa LowerTriangular
    @test parent(dev_lo - dev_lo) isa JLArray
    @test host_skew - host_mat ≈ Matrix(host_skew) - host_mat
    @test host_mat - host_skew ≈ host_mat - Matrix(host_skew)
    @test host_lo - host_up ≈ Matrix(host_lo) - Matrix(host_up)
end

# `mul!(::AbstractTriangular, ::AbstractTriangular, ::Real)` and its mirror carry both a species
# check and a backend check, and this testset is the only thing that asserts either. The third case
# is the one the per-argument binding exists for: a destination and a source of the same species
# whose storage arrays are different concrete types.
@testset "the triangular `mul!` checks species, backend and storage" begin
    host_lo = LowerTriangular(rand(T, N, N))
    host_up = UpperTriangular(rand(T, N, N))
    dev_lo = LowerTriangular(JLArray(rand(T, N, N)))

    @test_throws ArgumentError LinearAlgebra.mul!(host_up, host_lo, T(2))
    @test_throws ArgumentError LinearAlgebra.mul!(host_up, T(2), host_lo)
    @test_throws ArgumentError LinearAlgebra.mul!(dev_lo, host_lo, T(2))
    @test_throws ArgumentError LinearAlgebra.mul!(host_lo, dev_lo, T(2))

    source = LowerTriangular(rand(T, N, N))
    destination = LowerTriangular(view(collect(source.S), :), N)
    @test parent(destination) isa SubArray
    @test LinearAlgebra.mul!(destination, source, T(2)) isa LowerTriangular
    @test parent(destination) ≈ 2 .* parent(source)
end

# `/ᵉˡᵉ` binds a type variable per argument, as `+`, `-`, `add!` and `mul!` do, so a same-species
# pair whose storage types differ reaches it rather than a `MethodError`. It refuses a mixed species
# rather than falling back to a dense path, because an element-wise quotient of a lower by an upper
# divides by the zeros each keeps outside its own triangle.
@testset "`/ᵉˡᵉ` checks species, backend and storage" begin
    host_lo = LowerTriangular(rand(T, N, N) .+ one(T))
    host_up = UpperTriangular(rand(T, N, N) .+ one(T))
    dev_lo = LowerTriangular(JLArray(rand(T, N, N) .+ one(T)))

    @test_throws ArgumentError GeometricOptimizers.:(/ᵉˡᵉ)(host_lo, host_up)
    @test_throws ArgumentError GeometricOptimizers.:(/ᵉˡᵉ)(host_lo, dev_lo)

    other = LowerTriangular(view(collect(host_lo.S), :), N)
    quotient = GeometricOptimizers.:(/ᵉˡᵉ)(host_lo, other)
    @test quotient isa LowerTriangular
    @test parent(quotient) ≈ parent(host_lo) ./ parent(other)
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

# The three transfer operations keep crossing backends. `copyto!` is the one with a contract in
# `Base`; PR #85 is the reference for how these types meet it.
@testset "`copyto!` still crosses backends" begin
    destination = SkewSymMatrix(JLArray(zeros(T, N, N)))
    copyto!(destination, host_skew)
    # `.S` is the packed storage vector, so this compares the entries that were transferred
    @test Array(destination.S) ≈ host_skew.S
end
