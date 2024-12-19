using GeometricOptimizers
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

for T ∈ (Float32, Float64)
    for N ∈ 3:5
        for n ∈ 1:N
            add_and_sub(n, N, T)
        end
    end
end
