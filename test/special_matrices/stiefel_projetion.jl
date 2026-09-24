using GeometricOptimizers: StiefelProjection
using JLArrays: JLArray
using KernelAbstractions: CPU, KernelAbstractions
using LinearAlgebra: I, transpose
using Test

# `N` and `n` were declared `::Integer`, which are the package's only abstract fields. Nothing
# downstream of them lost its concrete return type -- `lift_factors`, `geodesic`, `cayley` and
# `hcat` against a `StiefelProjection` were all concrete already -- but `size` was not, so every
# caller of it paid a dynamic dispatch.
@testset "the fields are concrete" begin
    for field in (:N, :n)
        @test isconcretetype(fieldtype(StiefelProjection, field))
    end
    @test Base.return_types(size, (StiefelProjection{Float64, Matrix{Float64}},)) ==
          [Tuple{Int, Int}]
end

# The host constructor builds `[I; O]` directly rather than routing through
# `StiefelProjection(CPU(), T, N, n)`, which allocates through `KernelAbstractions.zeros` and then
# starts a kernel to write `n` ones. What is pinned here is that the two agree, entry for entry and
# in type, so that the cheaper spelling is the same matrix.
@testset "the host and the `CPU()` constructor agree" begin
    for T in (Float32, Float64), N in 3:5, n in 1:N
        E = StiefelProjection(N, n, T)
        E_backend = StiefelProjection(CPU(), T, N, n)
        @test typeof(E) === typeof(E_backend)
        @test E.A == E_backend.A
        @test E.A isa Matrix{T}
        @test KernelAbstractions.get_backend(E) == CPU()
    end
    # the element type still defaults to `Float64`, as `zeros(N, n)` does
    @test eltype(StiefelProjection(5, 3)) === Float64
end

# The backend constructor writes its ones with a kernel, one work item per diagonal entry. An
# `N × n` matrix has `min(N, n)` of them, so a wide one (`n > N`) must not launch `n` items; the host
# constructor, `Matrix{T}(I, N, n)`, is the reference for every shape.
@testset "a device StiefelProjection of any shape matches the host one" begin
    device = KernelAbstractions.get_backend(JLArray(zeros(Float32, 1)))
    for (N, n) in ((2, 4), (3, 3), (5, 2), (0, 3), (3, 0), (0, 0))
        E = StiefelProjection(device, Float32, N, n)
        @test E.A isa JLArray{Float32, 2}
        @test Array(E.A) == StiefelProjection(N, n, Float32).A
    end
end

# A row vector on the left is the one shape `*(::AbstractMatrix, ::StiefelProjection)` does not
# settle on its own: `LinearAlgebra` has its own method for that left operand, narrower there and
# wider on the right, so neither wins. The two row-vector methods in `src/ambiguities.jl` settle it.
# `E` is rectangular, so a method that swapped or dropped an operand would not conform.
@testset "a row vector times a StiefelProjection" begin
    for T in (Float32, Float64), N in 3:5, n in 1:N
        E = StiefelProjection(N, n, T)
        v = rand(T, N)

        @test v' * E ≈ v' * Matrix{T}(E)
        @test transpose(v) * E ≈ transpose(v) * Matrix{T}(E)
        @test size(v' * E) == (1, n)
    end
end

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

for T in (Float32, Float64)
    for N in 3:5
        for n in 1:N
            stiefel_proj(N, n, T)
            stiefel_proj_is_identity_over_zeros(N, n, T)
        end
    end
end
