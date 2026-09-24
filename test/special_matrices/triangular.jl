using GeometricOptimizers
using GeometricOptimizers: AbstractTriangular, freeparameters
using KernelAbstractions: CPU
using LinearAlgebra: tr, transpose, tril, triu
using Test
import Random

Random.seed!(1234)

# There was no test file for the triangular types at all until the `GeometricMachineLearning` tests
# of them were folded into this suite; `test/arrays/triangular.jl` there is where most of this comes
# from. The half of that file that tested `mat_tensor_mul` and its pullback stayed behind — those are
# GML's kernels, not this package's.

@testset "the two triangles and the diagonal partition the matrix" begin
    # `A - L - U` leaves exactly the diagonal, so summing it is the trace. This is the property that
    # says the two constructors take the *strict* triangles and agree on where the split is.
    for T in (Float32, Float64), n in 2:5

        A = rand(T, n, n)
        @test tr(A) ≈ sum(A - StrictlyLowerTriangular(A) - StrictlyUpperTriangular(A))
    end
end

@testset "multiplication agrees with the dense matrix" begin
    for T in (Float32, Float64), n in 2:5

        Aₗ = rand(StrictlyLowerTriangular{T}, n)
        Aᵤ = rand(StrictlyUpperTriangular{T}, n)
        B = rand(T, n, n)
        b = rand(T, n)

        @test Aₗ * B ≈ Matrix{T}(Aₗ) * B
        @test Aᵤ * B ≈ Matrix{T}(Aᵤ) * B
        @test B * Aₗ ≈ B * Matrix{T}(Aₗ)
        @test B * Aᵤ ≈ B * Matrix{T}(Aᵤ)
        @test Aₗ * b ≈ Matrix{T}(Aₗ) * b
        @test Aᵤ * b ≈ Matrix{T}(Aᵤ) * b

        # A matrix times a vector is a vector. Both products used to go through the
        # matrix--matrix path and return its `n × 1` result, and the two assertions above pass
        # either way: `promote_shape` accepts a trailing singleton dimension, so the difference
        # against the dense product is well defined and zero.
        @test Aₗ * b isa AbstractVector
        @test Aᵤ * b isa AbstractVector
        @test size(Aₗ * b) == (n,)
        @test size(Aᵤ * b) == (n,)
    end
end

@testset "addition and scalar multiplication are linear" begin
    for T in (Float32, Float64), n in 2:5

        A = rand(T, n, n)
        B = rand(T, n, n)
        α = rand(T)

        for MT in (StrictlyLowerTriangular, StrictlyUpperTriangular)
            @test MT(A + B) ≈ MT(A) + MT(B)
            @test MT(α * A) ≈ α * MT(A)
            @test typeof(MT(A) + MT(B)) <: MT{T}
            @test typeof(α * MT(A)) <: MT{T}
        end
    end
end

@testset "adjoint aliases the storage of its argument" begin
    # `adjoint` is a type swap (`StrictlyLowerTriangular` <-> `StrictlyUpperTriangular`), built around the *same*
    # storage vector rather than a copy. This is deliberate for performance, since the package's own
    # right-multiply `*(::AbstractMatrix, ::AbstractTriangular) = (A' * B')'` is read-only and would
    # otherwise pay for an allocation it never needs. But it means a write through the adjoint is a
    # write through the original, so this pins the sharing: if `adjoint` is ever made to copy, this
    # test is the one that catches it.
    for T in (Float32, Float64), n in 2:5

        L = rand(StrictlyLowerTriangular{T}, n)
        U = rand(StrictlyUpperTriangular{T}, n)

        @test parent(L') === parent(L)
        @test parent(U') === parent(U)

        Lt = L'
        parent(Lt)[1] = zero(T)
        @test parent(L)[1] == zero(T)

        Ut = U'
        parent(Ut)[1] = zero(T)
        @test parent(U)[1] == zero(T)
    end
end

# Sharing the storage is what bounds these methods to a real element type. Reusing the vector
# transposes without conjugating, so an unbound method is `transpose` wearing the name `adjoint`,
# and `*(B, A::AbstractTriangular) = (A' * B')'` then returns a silently wrong product — measured at
# `‖B*C - B*Matrix(C)‖ = 16.2` on this 3x3 `ComplexF64` case before the bound.
#
# The bound does not reject a complex argument; it hands it to `LinearAlgebra`'s lazy `Adjoint`,
# which conjugates. So the complex path becomes correct rather than becoming an error, and the real
# path keeps the storage-sharing swap. This testset asserts both halves, because a later edit that
# widened the methods again would restore the wrong answer with nothing else complaining.
@testset "adjoint conjugates on a complex element type" begin
    for MT in (StrictlyLowerTriangular, StrictlyUpperTriangular)
        C = MT(ComplexF64[1 + 2im, 3 + 4im, 5 + 6im], 3)
        M = Matrix(C)
        B = randn(ComplexF64, 3, 3)

        @test Matrix(C') == M'
        @test Matrix(C') != transpose(M)
        @test B * C ≈ B * M

        # and the real path still takes the storage-sharing swap rather than the lazy wrapper
        R = MT(randn(3), 3)
        Br = randn(3, 3)
        @test R' isa AbstractTriangular
        @test parent(R') === parent(R)
        @test Br * R ≈ Br * Matrix(R)
    end
end

@testset "random generation" begin
    for T in (Float32, Float64), n in 2:5,
        MT in (StrictlyLowerTriangular, StrictlyUpperTriangular)
        A = rand(MT{T}, n)
        @test typeof(A) <: MT{T}
        @test eltype(A) == T
        @test size(A) == (n, n)
        @test A isa AbstractTriangular
    end
end

# `zeros` and `rand` recover the bare constructor from the type parameter without going through
# the evaluator, so both infer to a concrete type rather than `Any`.
@testset "zeros and rand infer concretely" begin
    for T in (Float32, Float64), MT in (StrictlyLowerTriangular, StrictlyUpperTriangular)

        @test (@inferred zeros(MT{T}, 4)) isa MT{T}
        @test (@inferred rand(MT{T}, 4)) isa MT{T}
    end
end

# The backendless `zeros` and `rand` place on the host, and now say so in one spelling: `zeros(T,
# m)` and `rand(rng, T, m)` rather than a route through `KernelAbstractions` with an explicit
# `CPU()`. The two give the same array at less cost, and this pins the placement and the values.
@testset "the backendless allocators place on the host" begin
    for T in (Float32, Float64), MT in (StrictlyLowerTriangular, StrictlyUpperTriangular),
        n in 2:5
        @test freeparameters(zeros(MT{T}, n)) isa Vector{T}
        @test all(iszero, freeparameters(zeros(MT{T}, n)))
        @test freeparameters(zeros(MT{T}, n)) == freeparameters(zeros(CPU(), MT{T}, n))

        # the same rng state has to give the same draw through either spelling
        @test freeparameters(rand(Random.MersenneTwister(7), MT{T}, n)) isa Vector{T}
        @test freeparameters(rand(Random.MersenneTwister(7), MT{T}, n)) ==
              freeparameters(rand(Random.MersenneTwister(7), CPU(), MT{T}, n))
    end
end

# The storage layout is public: `freeparameters` returns it and the two-argument constructor takes
# it, so a change to the index arithmetic that kept them consistent with each other would still be
# breaking. Spelling the layout out for one matrix is what pins it.
@testset "storage layout" begin
    M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]

    @test StrictlyLowerTriangular(M) == [0 0 0 0; 5 0 0 0; 9 10 0 0; 13 14 15 0]
    @test StrictlyUpperTriangular(M) == [0 2 3 4; 0 0 7 8; 0 0 0 12; 0 0 0 0]

    @test freeparameters(StrictlyLowerTriangular(M)) == [5, 9, 10, 13, 14, 15]
    @test freeparameters(StrictlyUpperTriangular(M)) == [2, 3, 7, 4, 8, 12]

    # and the round trip: the vector the second constructor takes is the one `freeparameters`
    # returns
    @test StrictlyLowerTriangular(freeparameters(StrictlyLowerTriangular(M)), 4) ==
          StrictlyLowerTriangular(M)
    @test StrictlyUpperTriangular(freeparameters(StrictlyUpperTriangular(M)), 4) ==
          StrictlyUpperTriangular(M)

    # `vec` is `Base`'s: the `n²` entries, not the storage
    @test vec(StrictlyLowerTriangular(M)) == vec(Matrix(StrictlyLowerTriangular(M)))

    @test StrictlyLowerTriangular([1, 2, 3, 4, 5, 6], 4) ==
          [0 0 0 0; 1 0 0 0; 2 3 0 0; 4 5 6 0]
    @test StrictlyUpperTriangular([1, 2, 3, 4, 5, 6], 4) ==
          [0 1 2 4; 0 0 3 5; 0 0 0 6; 0 0 0 0]
end

# A row vector on the left is the one shape `*(::AbstractMatrix, ::AbstractTriangular)` does not
# settle on its own: `LinearAlgebra` has its own method for that left operand, narrower there and
# wider on the right, so neither wins. The two row-vector methods in `src/ambiguities.jl` settle it.
# Both triangles run, because one kernel covers them and neither is symmetric, so each pins which
# triangle the body reaches for.
@testset "a row vector times a triangular matrix" begin
    for T in (Float32, Float64), N in 2:5,
        AT in (StrictlyLowerTriangular, StrictlyUpperTriangular)
        A = rand(AT{T}, N)
        v = rand(T, N)

        @test v' * A ≈ v' * Matrix(A)
        @test transpose(v) * A ≈ transpose(v) * Matrix(A)
        @test size(v' * A) == (1, N)
    end
end

# Neither constructor projects: each reads the strict triangle that is there. `map_to_low` reads it
# directly and `map_to_up` reaches it by swapping the indices, which is a transpose. Spelt `A'` the
# swap conjugates on the way, so `StrictlyUpperTriangular` stored a triangle nobody asked for — while
# `StrictlyLowerTriangular` was exact, so the two constructors disagreed with each other. The real path
# cannot see it, which is why this testset is here.
@testset "neither constructor conjugates on a complex element type" begin
    M = randn(ComplexF64, 4, 4)

    @test Matrix(StrictlyLowerTriangular(M)) == tril(M, -1)
    @test Matrix(StrictlyUpperTriangular(M)) == triu(M, 1)

    Mr = randn(4, 4)
    @test Matrix(StrictlyLowerTriangular(Mr)) == tril(Mr, -1)
    @test Matrix(StrictlyUpperTriangular(Mr)) == triu(Mr, 1)
end
