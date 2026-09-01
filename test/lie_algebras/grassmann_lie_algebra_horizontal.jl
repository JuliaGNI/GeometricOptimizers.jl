using GeometricOptimizers
using GeometricOptimizers: l2norm, _add!, _difference!, _square!, _rac!, _div!, assign!
using LinearAlgebra
using NeuralNetworkParameters: flatten
using Test
import Random

Random.seed!(123)

@doc raw"""
This function tests addition for various custom arrays, i.e. if \(A + B\) is performed in the correct way. 
"""
function add_and_sub(n::Int, N::Int, T::Type)
    C = rand(T, N, N)
    D = rand(T, N, N)

    # GrassmannLieAlgHorMatrix
    CD_glahm = GrassmannLieAlgHorMatrix(C + D, n)
    CD_glahm2 = GrassmannLieAlgHorMatrix(C, n) + GrassmannLieAlgHorMatrix(D, n)
    @test CD_glahm ≈ CD_glahm2
    @test typeof(CD_glahm) <: GrassmannLieAlgHorMatrix{T}
    @test typeof(CD_glahm2) <: GrassmannLieAlgHorMatrix{T}

    CD_glahm_sub = GrassmannLieAlgHorMatrix(C - D, n)
    CD_glahm2_sub = GrassmannLieAlgHorMatrix(C, n) - GrassmannLieAlgHorMatrix(D, n)
    @test CD_glahm_sub ≈ CD_glahm2_sub
    @test typeof(CD_glahm_sub) <: GrassmannLieAlgHorMatrix{T}
    @test typeof(CD_glahm2_sub) <: GrassmannLieAlgHorMatrix{T}
end

function scalar_multiplication(n::Integer, N::Integer, T::DataType)
    C = rand(T, N, N)
    α = rand(T)

    # GrassmannLieAlgHorMatrix
    Cα_glahm = GrassmannLieAlgHorMatrix(α * C, n)
    Cα_glahm2 = α * GrassmannLieAlgHorMatrix(C, n)
    @test Cα_glahm ≈ Cα_glahm2
    @test typeof(Cα_glahm) <: GrassmannLieAlgHorMatrix{T}
    @test typeof(Cα_glahm2) <: GrassmannLieAlgHorMatrix{T}
end

function random_array_generation(n::Integer, N::Integer, T::DataType)
    A_Grassmann_hor = rand(GrassmannLieAlgHorMatrix{T}, N, n)
    @test typeof(A_Grassmann_hor) <: GrassmannLieAlgHorMatrix{T}
    @test eltype(A_Grassmann_hor) == T
end

for T in (Float32, Float64)
    for N in 3:5
        for n in 1:N
            add_and_sub(n, N, T)
            scalar_multiplication(n, N, T)
            random_array_generation(n, N, T)
        end
    end
end
# Everything below is about the *free parameters* of a horizontal lift as against the ambient `N × N`
# matrix it presents itself as, and it covers both lifts rather than only the Grassmann one, because
# the point is that they answer the same way. Each of these operations used to be written out for
# `StiefelLieAlgHorMatrix` and to have no `GrassmannLieAlgHorMatrix` method at all, so the Grassmann
# lift either raised (no `setindex!`) or silently got the ambient answer. That is issue A11; they are
# one method over `Base.parent` now.

function lifts(T, N, n)
    (rand(StiefelLieAlgHorMatrix{T}, N, n), rand(GrassmannLieAlgHorMatrix{T}, N, n))
end

# `l2norm` of a lift is the norm of its *flattening*, which is what `Q` is sized by, what `outer!`
# forms its outer product in, and what the `α` of a line search parameterizes. The ambient Frobenius
# norm is `√2` times it — each off-diagonal block is stored once and appears twice — so a lift that
# fell through to `l2norm(::AbstractMatrix)` bounded the step ceiling, the curvature condition and
# `rg` by a number that was too large by that factor.
@testset "l2norm of a horizontal lift is the norm of its flattening" begin
    for T in (Float32, Float64), N in 3:6, n in 1:(N - 1)
        for B in lifts(T, N, n)
            v, _ = flatten(T, B)
            @test l2norm(B) ≈ l2norm(v)
            @test l2norm(B) ≈ l2norm(collect(vec(B)))       # `vec` is the flattening too
            # and it is *not* the ambient norm, unless the lift is zero
            @test l2norm(B) < norm(Matrix(B))
            @test norm(Matrix(B)) ≈ √T(2) * l2norm(B)
        end
    end
end

# The free parameters of a lift, which is what every helper below is asserted against.
free(::Type{T}, B) where {T} = flatten(T, B)[1]

@testset "the elementwise helpers act on the free parameters" begin
    for T in (Float32, Float64), N in 3:6, n in 1:(N - 1)
        for B in lifts(T, N, n)
            b = free(T, B)

            # `_add!` and `_difference!`: the momentum recursion and the quasi-Newton secant pair
            @test free(T, _add!(copy(B), B)) ≈ 2b
            @test iszero(free(T, _difference!(copy(B), B, B)))

            # `_square!` / `_rac!` / `_div!`: the `Adam` moment path. `_square!` first, so that
            # `_rac!` is given something non-negative to take the root of.
            S = _square!(copy(B), B)
            @test free(T, S) ≈ b .^ 2
            @test free(T, _rac!(copy(S), S)) ≈ abs.(b)
            @test free(T, _div!(copy(B), S, S)) ≈ ones(T, length(b))

            # `assign!` and `copy` round-trip, and neither reaches `setindex!`
            E = zero(B)
            assign!(E, B)
            @test free(T, E) == b
            @test free(T, copy(B)) == b
        end
    end
end

# `Base.one` is the `N × N` identity built with a `KernelAbstractions` kernel rather than with
# `Base._one`'s scalar-indexed diagonal loop, which is what a GPU array cannot serve. `geodesic`
# reaches it on every retraction; it existed for the Stiefel lift only, so the Grassmann retraction
# was on the scalar-indexed path. See the note on issue A19.
@testset "one(::AbstractLieAlgHorMatrix) is the ambient identity" begin
    for T in (Float32, Float64), N in 3:6, n in 1:(N - 1)
        for B in lifts(T, N, n)
            @test one(B) == Matrix{T}(I, N, N)
            @test eltype(one(B)) == T
        end
    end
end
