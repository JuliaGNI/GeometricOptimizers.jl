# Every allocator that takes a backend *and* a caller-named element type returns that element type
# on that backend. A width the backend cannot hold is refused by the backend's own allocation: on
# Metal, `Metal does not support Float64 values, try using Float32 instead`, which names the width.
# No host backend refuses `Float64`, so no test here can reproduce that refusal; `test/metal.jl`
# and `scripts/metal_check.jl` are where it is seen.
#
# `_Float64GPU` allocates host arrays, which makes it a device every one of these methods actually
# runs on.

using GeometricOptimizers
using GeometricOptimizers: StiefelProjection, unit_matrix
using KernelAbstractions: KernelAbstractions, CPU, GPU
using Random
using Test

Random.seed!(2026)

struct _Float64GPU <: GPU end
function KernelAbstractions.allocate(::_Float64GPU, ::Type{T}, dims::Tuple; kwargs...) where {T}
    Array{T}(undef, dims)
end

const N, n = 6, 2

# every entry point that names a backend and an element type, as a caller writes it
function allocators(T)
    (
        "zeros SkewSymMatrix" => b -> zeros(b, SkewSymMatrix{T}, n),
        "rand SkewSymMatrix" => b -> rand(b, SkewSymMatrix{T}, n),
        "zeros SymmetricMatrix" => b -> zeros(b, SymmetricMatrix{T}, n),
        "rand SymmetricMatrix" => b -> rand(b, SymmetricMatrix{T}, n),
        "zeros StrictlyLowerTriangular" => b -> zeros(b, StrictlyLowerTriangular{T}, n),
        "rand StrictlyLowerTriangular" => b -> rand(b, StrictlyLowerTriangular{T}, n),
        "zeros StrictlyUpperTriangular" => b -> zeros(b, StrictlyUpperTriangular{T}, n),
        "rand StrictlyUpperTriangular" => b -> rand(b, StrictlyUpperTriangular{T}, n),
        "zeros StiefelLieAlgHorMatrix" => b -> zeros(b, StiefelLieAlgHorMatrix{T}, N, n),
        "rand StiefelLieAlgHorMatrix" => b -> rand(b, StiefelLieAlgHorMatrix{T}, N, n),
        "zeros GrassmannLieAlgHorMatrix" =>
            b -> zeros(b, GrassmannLieAlgHorMatrix{T}, N, n),
        "rand GrassmannLieAlgHorMatrix" => b -> rand(b, GrassmannLieAlgHorMatrix{T}, N, n),
        "StiefelProjection" => b -> StiefelProjection(b, T, N, n),
        "rand StiefelManifold" => b -> rand(b, StiefelManifold{T}, N, n),
        "rand GrassmannManifold" => b -> rand(b, GrassmannManifold{T}, N, n),
        "unit_matrix" => b -> unit_matrix(b, T, n)
    )
end

# `StiefelProjection` and `unit_matrix` are the two entry points that launch a kernel, and a
# stand-in device can allocate but not run one. Their successful paths on a device are covered on a
# real backend by `test/special_matrices/stiefel_projetion.jl` and
# `test/retractions/exponential_accuracy.jl`.
const KERNEL_LAUNCHING = ("StiefelProjection", "unit_matrix")
runnable(as) = filter(p -> first(p) ∉ KERNEL_LAUNCHING, collect(as))

@testset "a backend that carries Float64 allocates one" begin
    for T in (Float32, Float64), (name, allocate) in runnable(allocators(T))

        @testset "$name $T" begin
            A = allocate(_Float64GPU())
            @test eltype(A) === T
        end
    end
end

@testset "the host is untouched in both widths" begin
    for T in (Float32, Float64), (name, allocate) in allocators(T)

        @testset "$name $T" begin
            A = allocate(CPU())
            @test eltype(A) === T
        end
    end
end

# `SymmetricMatrix`'s backend-taking allocators mirror `SkewSymMatrix`'s, which is what these
# assert: the two types are optimizer parameters in the same way and are placed on a device alike.
@testset "SymmetricMatrix allocates on a backend as SkewSymMatrix does" begin
    for T in (Float32, Float64)
        Z = zeros(_Float64GPU(), SymmetricMatrix{T}, n)
        @test Z isa SymmetricMatrix{T}
        @test Z == zeros(SymmetricMatrix{T}, n)
        @test all(iszero, parent(Z))

        A = rand(Random.MersenneTwister(11), _Float64GPU(), SymmetricMatrix{T}, n)
        @test A isa SymmetricMatrix{T}
        @test size(A) == (n, n)
        @test A == A'
        # the storage is `n(n+1)/2`, not `n(n-1)/2`: the diagonal is carried
        @test length(parent(A)) == n * (n + 1) ÷ 2
    end
end

# `GrassmannLieAlgHorMatrix`'s backend-taking `rand` mirrors `StiefelLieAlgHorMatrix`'s, as its
# backend-taking `zeros` does.
@testset "GrassmannLieAlgHorMatrix draws on a backend as StiefelLieAlgHorMatrix does" begin
    for T in (Float32, Float64)
        B = rand(
            Random.MersenneTwister(11), _Float64GPU(), GrassmannLieAlgHorMatrix{T}, N, n)
        @test B isa GrassmannLieAlgHorMatrix{T}
        @test size(B) == (N, N)
        @test B.N == N && B.n == n
        @test size(B.B) == (N - n, n)

        # and the rng-less spelling, which is what the Stiefel one has
        @test rand(_Float64GPU(), GrassmannLieAlgHorMatrix{T}, N, n) isa
              GrassmannLieAlgHorMatrix{T}
    end
end
