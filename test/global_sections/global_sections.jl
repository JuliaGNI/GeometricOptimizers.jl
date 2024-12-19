using GeometricOptimizers
using LinearAlgebra: norm
using Test

include("../grassmann_test_help.jl")

function grassmann_global_section(N::Integer, n::Integer, T::DataType)
    Y = rand(GrassmannManifold{T}, N, n)
    Q = Matrix(GlobalSection(Y))
    πQ = Q[1:N, 1:n]
    norm(Y - πQ * πQ' * Y) / N / n < eps(T)
end

function stiefel_global_section(N::Integer, n::Integer, T::DataType)
    Y = rand(GrassmannManifold{T}, N, n)
    Q = Matrix(GlobalSection(Y))
    πQ = Q[1:N, 1:n]
    norm(Y - πQ * πQ' * Y) / N / n < eps(T)
end

T = Float32

for N ∈ 3:5
    for n ∈ 1:N
        @test stiefel_global_section(N, n, T)
        @test grassmann_global_section(N, n, T)
    end
end