using GeometricOptimizers: StiefelProjection
using LinearAlgebra: I
using Test

# `Flaot32` was the default here. Harmless, because every call passes `T` — but a default nothing
# reaches is a default nothing checks, so it is gone rather than spelled correctly.
function stiefel_proj(N::Integer, n::Integer, T::DataType)
    In = I(n)
    E = StiefelProjection(N, n, T)
    @test all(abs.((E'*E) .- In) .< eps(T))
end

# `E` *is* `[I; O]`, which is the whole definition of it; the orthonormality above follows from that
# but does not imply it — `E'E = I` for any matrix with orthonormal columns. From
# `GeometricMachineLearning`'s `test/arrays/constructor_tests_for_custom_arrays.jl`.
function stiefel_proj_is_identity_over_zeros(N::Integer, n::Integer, T::DataType)
    E = StiefelProjection(T, N, n)
    @test Matrix{T}(E) ≈ vcat(I(n), zeros(T, N - n, n))
    @test size(E) == (N, n)
    @test eltype(E) == T
end

for T ∈ (Float32, Float64)
    for N ∈ 3:5
        for n ∈ 1:N
            stiefel_proj(N, n, T)
            stiefel_proj_is_identity_over_zeros(N, n, T)
        end
    end
end
