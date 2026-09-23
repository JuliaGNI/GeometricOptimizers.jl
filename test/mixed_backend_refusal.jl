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
# The check sits in the entry methods of `*`, `+`, `-` and `mul!` in `src/ambiguities.jl`, in
# `add!`, and nowhere else. So the first testset calls every entry with every owned type in the slot
# the entry names, a host operand against a device one; a type that is left out of the unions there,
# or an entry that loses its check, fails it.
#
# Three operations are deliberately absent, because they must keep crossing backends: `copyto!` and
# `assign!` are transfers, which is the contract `Base` sets for `copyto!` and the one PR #85 settled
# for `assign!`, and `changebackend` is the other supported route. The last testset holds `copyto!`
# to that, so a later widening of the guard cannot quietly take it.

using GeometricOptimizers
using GeometricOptimizers: LowerTriangular, StiefelProjection, UpperTriangular, add!, sr!
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: LinearAlgebra, mul!, transpose
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

# One host instance of every member of `OwnedMatrix`, the adjoints included.
const HOST_OWNED = let
    lift = rand(StiefelLieAlgHorMatrix{T}, N, n)
    grass = rand(GrassmannLieAlgHorMatrix{T}, N, n)
    Y = rand(StiefelManifold{T}, N, n)
    G = rand(GrassmannManifold{T}, N, n)
    U = rand(SymplecticStiefelManifold{T}, N, 4)
    E = StiefelProjection(T, N, n)
    ["SkewSym" => host_skew, "SkewSym'" => host_skew', "Sym" => host_sym,
        "Lower" => LowerTriangular(rand(T, N, N)),
        "Upper" => UpperTriangular(rand(T, N, N)),
        "StiefelHor" => lift, "StiefelHor'" => lift', "GrassmannHor" => grass,
        "GrassmannHor'" => grass', "Y" => Y, "Y'" => Y', "G" => G, "G'" => G', "U" => U,
        "U'" => U', "E" => E, "E'" => E', "Sfac" => sr!(randn(T, N, 4)).S]
end

@testset "every entry method refuses a host operand against a device one" begin
    for (name, A) in HOST_OWNED
        m, k = size(A)
        right, left = JLArray(rand(T, k, 2)), JLArray(rand(T, 2, m))
        same = JLArray(rand(T, m, k))
        @testset "$name" begin
            @test_throws ArgumentError A * right
            @test_throws ArgumentError left * A
            @test_throws ArgumentError A * JLArray(rand(T, k))
            @test_throws ArgumentError JLArray(rand(T, m))' * A
            @test_throws ArgumentError transpose(JLArray(rand(T, m))) * A
            @test_throws ArgumentError A + same
            @test_throws ArgumentError same + A
            @test_throws ArgumentError A - same
            @test_throws ArgumentError same - A
            @test_throws ArgumentError mul!(zeros(T, m, 2), A, right)
            @test_throws ArgumentError mul!(JLArray(zeros(T, m, 2)), A, rand(T, k, 2))
            @test_throws ArgumentError mul!(zeros(T, 2, k), left, A)
            @test_throws ArgumentError mul!(zeros(T, m), A, JLArray(rand(T, k)))
        end
    end

    # the `(Owned, Owned)` entries, which a pair of two different owned types reaches
    Y = rand(StiefelManifold{T}, N, n)
    dev_Y = StiefelManifold(JLArray(Matrix(Y.A)))
    @test_throws ArgumentError host_skew * dev_sym
    @test_throws ArgumentError Y' * dev_Y
    @test_throws ArgumentError host_skew + dev_sym
    @test_throws ArgumentError host_skew - dev_sym
    @test_throws ArgumentError mul!(zeros(T, N, N), host_skew, dev_sym)

    # and the same-type pairs, which reach the same entries
    @test_throws ArgumentError host_skew + dev_skew
    @test_throws ArgumentError dev_skew - host_skew
    @test_throws ArgumentError host_sym + dev_sym
    @test_throws ArgumentError LowerTriangular(rand(T, N, N)) -
                               LowerTriangular(JLArray(rand(T, N, N)))

    @test_throws ArgumentError add!(SkewSymMatrix(rand(T, N, N)), host_skew, dev_skew)
    host_lo = LowerTriangular(rand(T, N, N))
    @test_throws ArgumentError add!(LowerTriangular(rand(T, N, N)), host_lo,
        LowerTriangular(JLArray(rand(T, N, N))))
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
    # per-argument binding does not cost the type its own method: on one shared type variable this
    # pair dispatches correctly and a mixed one does not, so only a same-backend device pair
    # distinguishes "reaches the right method" from "reaches `Base`'s".
    @test LowerTriangular(rand(T, N, N)) + LowerTriangular(rand(T, N, N)) isa
          LowerTriangular
    dev_lo = LowerTriangular(JLArray(rand(T, N, N)))
    @test dev_lo + dev_lo isa LowerTriangular
    @test parent(dev_lo + dev_lo) isa JLArray
    @test dev_lo - dev_lo isa LowerTriangular
    @test parent(dev_lo - dev_lo) isa JLArray
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

# `KernelAbstractions.get_backend` raises for an array type it has no method for, and the guard lets
# that `ArgumentError` through: a pair with an operand it cannot place raises rather than computing.
@testset "an unplaceable operand raises" begin
    skew = rand(SkewSymMatrix{T}, N)
    bidiag = LinearAlgebra.Bidiagonal(rand(T, N), rand(T, N - 1), :U)

    # the premise: this really is a type `get_backend` cannot answer for
    @test_throws ArgumentError KernelAbstractions.get_backend(bidiag)

    for op in (*, +, -)
        @test_throws ArgumentError op(skew, bidiag)
        @test_throws ArgumentError op(bidiag, skew)
    end
end

# The three transfer operations keep crossing backends. `copyto!` is the one with a contract in
# `Base`; PR #85 is the reference for how these types meet it.
@testset "`copyto!` still crosses backends" begin
    destination = SkewSymMatrix(JLArray(zeros(T, N, N)))
    copyto!(destination, host_skew)
    # `.S` is the packed storage vector, so this compares the entries that were transferred
    @test Array(destination.S) ≈ host_skew.S
end
