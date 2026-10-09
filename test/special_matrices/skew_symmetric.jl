using GeometricOptimizers
using GeometricOptimizers: map_to_Skew, freeparameters
using LinearAlgebra: transpose
using Test
import Random

include("../helpers/eltypes.jl")

function skew_symmetrization_operation(rng, N::Integer, T::DataType)
    A = rand(rng, T, N, N)
    A_skew = SkewSymMatrix(A)
    @test eltype(A_skew) == T

    for i in 1:N
        for j in 1:N
            # `map_to_Skew` does the same subtraction and an exact halving, so the two agree to the
            # last bit; `eps(T)` admits no more than one rounding of an entry below 1
            @test abs((A - A')[i, j] / 2 - A_skew[i, j]) < eps(T)
        end
    end
end

#check if symmetric matrix works for 1×1 matrices
function check_if_symmetric_matrix_works_for_1x1_matrices(rng, T::DataType)
    W = rand(rng, T, 1, 1)
    S = SkewSymMatrix(W)
    @test eltype(S) == T
    # a 1×1-skew-symmetric matrix is 0: `getindex` returns `zero(T)` on the diagonal, so the bound
    # admits no round-off at all
    @test abs(S[1, 1]) < eps(T)
end

#check if built-in projection, matrix addition & subtraction works
function skew_mat_add_sub(rng, N::Integer, T::DataType)
    anti_symmetrize(W) = (W - W') / 2
    W₁ = rand(rng, T, N, N)
    S₁ = SkewSymMatrix(W₁)
    W₂ = rand(rng, T, N, N)
    S₂ = SkewSymMatrix(W₂)
    S₃ = S₁ + S₂
    S₄ = S₁ - S₂
    @test typeof(S₃) <: SkewSymMatrix
    @test typeof(S₄) <: SkewSymMatrix
    @test eltype(S₃) == T
    @test eltype(S₄) == T
    # entries of `rand` are in `[0, 1)`: the reference rounds two sums below 2 and their difference
    # (at most `eps(T)/2` each) and halves exactly, so it is off by at most `3 eps(T)/4`; `S₁ ± S₂`
    # rounds two differences below 1 (`eps(T)/4` each, then halved) and one sum below 1
    # (`eps(T)/4`), so at most `eps(T)/2`. Together `5 eps(T)/4`, below `2 eps(T)`.
    @test all(abs.(anti_symmetrize(W₁ + W₂) .- S₃) .< 2 * eps(T))
    @test all(abs.(anti_symmetrize(W₁ - W₂) .- S₄) .< 2 * eps(T))
end

# this function tests if the matrix multiplication for the SkewSym Matrix is the same as the implied one.
function skew_mat_mul(rng, n::Integer, T::DataType)
    S = rand(rng, SkewSymMatrix{T}, n)
    A = rand(rng, T, n, n)
    SA1 = S * A
    SA2 = Matrix{T}(S) * A
    @test eltype(SA1) == T
    @test isapprox(SA1, SA2)
end

# A matrix times a vector is a vector. `S * b` used to go through the matrix--matrix kernel and
# return that kernel's `n × 1` result, so it came back as a `Matrix`. Comparing against the dense
# product does not catch that: `promote_shape` accepts a trailing singleton dimension, so
# `Matrix(S) * b - S * b` is well defined and zero.
function skew_mat_vec_mul(rng, n::Integer, T::DataType)
    S = rand(rng, SkewSymMatrix{T}, n)
    b = rand(rng, T, n)
    Sb = S * b
    @test Sb isa AbstractVector
    @test eltype(Sb) == T
    @test size(Sb) == (n,)
    @test isapprox(Sb, Matrix{T}(S) * b)
end

# tests if multiplication from the right also works correctly
function skew_mat_mul_from_the_right(rng, N::Integer, T::DataType)
    S = rand(rng, SkewSymMatrix{T}, N)
    A = rand(rng, T, N, N)
    AS1 = A * S
    AS2 = A * Matrix{T}(S)
    @test eltype(AS1) == T
    @test isapprox(AS1, AS2)
end

function check_map_to_Skew(rng, N::Integer, T::DataType)
    A = rand(rng, SkewSymMatrix{T}, N)
    @test eltype(map_to_Skew(A)) == T
    @test A.S ≈ map_to_Skew(A)
end

# `SkewSymMatrix(α * A) == α * SkewSymMatrix(A)`, i.e. skew-symmetrization is linear.
#
# The loop at the bottom of the file calls this: a helper nothing invokes can carry a typo in its
# own name while the suite still passes. The same property is tested in `GeometricMachineLearning`
# (`test/arrays/scalar_multiplication_for_custom_arrays.jl`).
function scalar_multiplication(rng, n::Integer, T::DataType)
    A = rand(rng, T, n, n)
    α = rand(rng, T)

    # SkewSymMatrix
    Aα_sym = SkewSymMatrix(α * A)
    Aα_sym2 = α * SkewSymMatrix(A)
    @test eltype(Aα_sym2) == T
    @test Aα_sym ≈ Aα_sym2
    @test typeof(Aα_sym) <: SkewSymMatrix{T}
    @test typeof(Aα_sym2) <: SkewSymMatrix{T}
end

# `SkewSymMatrix(A + B) == SkewSymMatrix(A) + SkewSymMatrix(B)`, the other half of linearity.
# `skew_mat_add_sub` above adds two matrices that are already `SkewSymMatrix`es; this one adds the
# dense matrices first, so it tests the constructor and not only `+`.
function addition_is_linear(rng, n::Integer, T::DataType)
    A = rand(rng, T, n, n)
    B = rand(rng, T, n, n)

    AB = SkewSymMatrix(A + B)
    AB₂ = SkewSymMatrix(A) + SkewSymMatrix(B)
    @test eltype(AB₂) == T
    @test AB ≈ AB₂
    @test typeof(AB) <: SkewSymMatrix{T}
    @test typeof(AB₂) <: SkewSymMatrix{T}
end

function test_random_array_generation(rng, n::Int, N::Int, T::DataType)
    A_sym = rand(rng, SkewSymMatrix{T}, n)
    @test typeof(A_sym) <: SkewSymMatrix{T}
    @test eltype(A_sym) == T
end

# `SkewSymMatrix` is exported, so both of these are public API. The parametric method was once
# introduced *in place of* the non-parametric one, which made `zeros(SkewSymMatrix, n)` fall
# through to `Base.zeros(::Type, ::Int)` and throw `MethodError: no method matching
# zero(::Type{SkewSymMatrix})` — and took `zeros(::Type{StiefelLieAlgHorMatrix}, N, n)`, its
# only in-repo caller, down with it.
@testset "zeros for SkewSymMatrix" begin
    for n in 2:5
        A = zeros(SkewSymMatrix, n)
        @test A isa SkewSymMatrix{Float64}
        @test size(A) == (n, n)
        @test all(iszero, A)

        for T in REAL_ELTYPES
            A_T = zeros(SkewSymMatrix{T}, n)
            @test A_T isa SkewSymMatrix{T}
            @test size(A_T) == (n, n)
            @test all(iszero, A_T)
        end
    end
end

@testset "SkewSymMatrix projection and arithmetic, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    check_if_symmetric_matrix_works_for_1x1_matrices(rng, T)
    for N in 2:5
        skew_symmetrization_operation(rng, N, T)
        skew_mat_add_sub(rng, N, T)
        skew_mat_mul(rng, N, T)
        skew_mat_vec_mul(rng, N, T)
        skew_mat_mul_from_the_right(rng, N, T)
        check_map_to_Skew(rng, N, T)
        scalar_multiplication(rng, N, T)
        addition_is_linear(rng, N, T)
        test_random_array_generation(rng, N, N + 5, T)
    end
end

# The projection of an integer matrix lands on `float(T)`: `Float64` for every fixed-width integer
# and `BigFloat` for a `BigInt`. The assertion is against `float(T)` and not against a list of
# widths, so it states the rule rather than a table of results. The loop covers every width because
# `Float32` carries 24 mantissa bits and therefore represents no wider integer type exactly. The
# expected value is computed in `float(T)`, since the integer difference `A - A'` wraps for an
# unsigned `T`.
@testset "an integer matrix is projected into float(T)" begin
    for T in (Int8, Int16, Int32, Int64, Int128, BigInt,
        UInt8, UInt16, UInt32, UInt64, UInt128)
        A = T[0 1 2; 3 0 4; 5 6 0]
        @test eltype(map_to_Skew(A)) === float(T)
        @test eltype(SkewSymMatrix(A)) === float(T)
        @test SkewSymMatrix(A) ≈ (float(T).(A) .- float(T).(A)') ./ 2
    end
    # `Bool` is an `Integer` and `float(Bool)` is `Float64`, but it holds only 0 and 1, so it needs
    # a matrix of its own.
    A = Bool[0 1 1; 0 0 1; 1 0 0]
    @test eltype(map_to_Skew(A)) === Float64
    @test eltype(SkewSymMatrix(A)) === Float64
    @test SkewSymMatrix(A) ≈ (Float64.(A) .- Float64.(A)') ./ 2
end

# The storage layout is public: `freeparameters` returns it and the two-argument constructor takes
# it. Spelling it out for one matrix pins the index arithmetic, which a change that kept the two
# consistent with *each other* would otherwise slip past. From
# `GeometricMachineLearning`'s `test/arrays/{triangular,constructor_tests_for_custom_arrays}.jl`.
@testset "storage layout" begin
    M = [1 2 3 4; 5 6 7 8; 9 10 11 12; 13 14 15 16]
    @test freeparameters(SkewSymMatrix(M)) ≈ [1.5, 3.0, 1.5, 4.5, 3.0, 1.5]

    @test SkewSymMatrix([1, 2, 3, 4, 5, 6], 4) == [0 -1 -2 -4; 1 0 -3 -5; 2 3 0 -6; 4 5 6 0]
    @test SkewSymMatrix(freeparameters(SkewSymMatrix(M)), 4) ≈ SkewSymMatrix(M)

    # `vec` is `Base`'s: the `n²` entries, not the storage
    @test vec(SkewSymMatrix(M)) == vec(Matrix(SkewSymMatrix(M)))
end

# `SkewSymMatrix` is the set `{M : Mᵀ = -M}` -- a transpose identity, which is what `getindex`
# reconstructs and what the constructor's docstring states. Both the projection and
# `*(::AbstractMatrix, ::SkewSymMatrix)` therefore spell it `transpose`. With `adjoint` the
# projection lands on neither `(A - Aᵀ)/2` nor `(A - Aᴴ)/2`, and the product returns `B·conj(A)`.
#
# The real path cannot see the difference between the two spellings, so an edit that puts `'` back
# gives a wrong answer on a complex element type with nothing else complaining. This testset is what
# complains. The triangulars keep the same testset for the same reason -- see
# `adjoint conjugates on a complex element type` in `triangular.jl`, which settles it the other way
# round.
@testset "the projection and the product are transposes on a complex element type, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    A = randn(rng, Complex{T}, 4, 4)
    S = SkewSymMatrix(A)
    B = randn(rng, Complex{T}, 3, 4)
    v = randn(rng, Complex{T}, 4)

    @test eltype(S) == Complex{T}
    @test eltype(B * S) == Complex{T}
    @test Matrix(S) ≈ (A - transpose(A)) / 2
    @test transpose(Matrix(S)) == -Matrix(S)
    @test B * S ≈ B * Matrix(S)
    @test v' * S ≈ v' * Matrix(S)
    @test transpose(v) * S ≈ transpose(v) * Matrix(S)

    # On a real element type the two spellings are one expression, so these assertions hold for
    # either one. That is why the complex block above carries the check.
    Ar = randn(rng, T, 4, 4)
    Sr = SkewSymMatrix(Ar)
    Br = randn(rng, T, 3, 4)
    @test eltype(Br * Sr) == T
    @test Matrix(Sr) ≈ (Ar - transpose(Ar)) / 2
    @test Matrix(Sr) ≈ (Ar - Ar') / 2
    @test Br * Sr ≈ Br * Matrix(Sr)
end

# A row vector on the left is the one shape `*(::AbstractMatrix, ::OwnedMatrix)` does not settle on
# its own: `LinearAlgebra` has its own method for that left operand, narrower there and wider on the
# right, so neither wins. The two row-vector methods in `src/ambiguities.jl` settle it.
@testset "a row vector times a SkewSymMatrix, $T" for T in REAL_ELTYPES
    rng = Random.Xoshiro(123)
    for N in 2:5
        A = rand(rng, SkewSymMatrix{T}, N)
        v = rand(rng, T, N)

        @test eltype(v' * A) == T
        @test v' * A ≈ v' * Matrix(A)
        @test transpose(v) * A ≈ transpose(v) * Matrix(A)
        @test size(v' * A) == (1, N)
    end
end
