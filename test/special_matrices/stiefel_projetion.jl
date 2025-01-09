using GeometricOptimizers: StiefelProjection
using LinearAlgebra: I
using Test

# test Stiefel manifold projection test 
function stiefel_proj(N::Integer, n::Integer, T::DataType=Flaot32)
    In = I(n)
    E = StiefelProjection(N, n, T)
    @test all(abs.((E'*E) .- In) .< eps(T))
end

for T ∈ (Float32, Float64)
    for N ∈ 3:5
        for n ∈ 1:N
            stiefel_proj(N, n, T)
        end
    end
end