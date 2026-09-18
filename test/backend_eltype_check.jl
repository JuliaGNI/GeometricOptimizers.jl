# Every allocator that takes a backend *and* a caller-named element type refuses a width the
# backend has declared it cannot hold, rather than narrowing it or failing further in.
#
# Without the check the same call still fails, but inside the backend and in its words: on Metal,
# `Metal does not support Float64 values, try using Float32 instead`, which names neither the type
# being built nor the call that asked for it. The width was never silently narrowed, so what this
# adds is the message and one place to read the rule.
#
# Three stand-in devices. `_Float64GPU` and `_NoFloat64GPU` differ in one declaration and nothing
# else, and both allocate host arrays, which makes them devices every one of these methods actually
# runs on — no backend reachable here can do that otherwise, because a real device draw needs a
# `qr` neither `Metal` nor `JLArrays` supplies.
#
# `_UnallocatableGPU` is the third and has one job: it declares no `Float64` and can allocate
# nothing at all, so an `ArgumentError` from it is proof that the check stopped the call *before*
# it reached the backend. A call that got past would be a `MethodError` on `allocate` instead.
#
# What must NOT be guarded, and is asserted below: the derived allocators. `zero`, `similar`,
# `_zero`, `_similar` and the arithmetic take an instance, so their element type comes from an
# array already on the backend and the case cannot arise. A check there would be error handling for
# something that cannot happen.

using GeometricOptimizers
using GeometricOptimizers: LowerTriangular, StiefelProjection, UpperTriangular,
                           _check_supported_eltype
using KernelAbstractions: KernelAbstractions, CPU, GPU
using Random
using Test

Random.seed!(2026)

struct _Float64GPU <: GPU end
function KernelAbstractions.allocate(::_Float64GPU, ::Type{T}, dims::Tuple; kwargs...) where {T}
    Array{T}(undef, dims)
end

struct _NoFloat64GPU <: GPU end
KernelAbstractions.supports_float64(::_NoFloat64GPU) = false
function KernelAbstractions.allocate(::_NoFloat64GPU, ::Type{T}, dims::Tuple; kwargs...) where {T}
    Array{T}(undef, dims)
end

struct _UnallocatableGPU <: GPU end
KernelAbstractions.supports_float64(::_UnallocatableGPU) = false

const N, n = 6, 2

# every entry point that names a backend and an element type, as a caller writes it
function allocators(T)
    (
        "zeros SkewSymMatrix" => b -> zeros(b, SkewSymMatrix{T}, n),
        "rand SkewSymMatrix" => b -> rand(b, SkewSymMatrix{T}, n),
        "zeros SymmetricMatrix" => b -> zeros(b, SymmetricMatrix{T}, n),
        "rand SymmetricMatrix" => b -> rand(b, SymmetricMatrix{T}, n),
        "zeros LowerTriangular" => b -> zeros(b, LowerTriangular{T}, n),
        "rand LowerTriangular" => b -> rand(b, LowerTriangular{T}, n),
        "zeros UpperTriangular" => b -> zeros(b, UpperTriangular{T}, n),
        "rand UpperTriangular" => b -> rand(b, UpperTriangular{T}, n),
        "zeros StiefelLieAlgHorMatrix" => b -> zeros(b, StiefelLieAlgHorMatrix{T}, N, n),
        "rand StiefelLieAlgHorMatrix" => b -> rand(b, StiefelLieAlgHorMatrix{T}, N, n),
        "zeros GrassmannLieAlgHorMatrix" =>
            b -> zeros(b, GrassmannLieAlgHorMatrix{T}, N, n),
        "rand GrassmannLieAlgHorMatrix" => b -> rand(b, GrassmannLieAlgHorMatrix{T}, N, n),
        "StiefelProjection" => b -> StiefelProjection(b, T, N, n),
        "rand StiefelManifold" => b -> rand(b, StiefelManifold{T}, N, n),
        "rand GrassmannManifold" => b -> rand(b, GrassmannManifold{T}, N, n)
    )
end

# `StiefelProjection` is the one entry point that launches a kernel, and a stand-in device can
# allocate but not run one. It stays in the refusal testset — the check fires before the kernel, so
# the refusal is exactly what is observable there — and drops out of the ones that have to complete
# a call. Its successful path is covered on a real backend by
# `test/special_matrices/stiefel_projetion.jl`.
runnable(as) = filter(p -> first(p) != "StiefelProjection", collect(as))

@testset "a width the backend declares it cannot hold is refused" begin
    for (name, allocate) in allocators(Float64)
        @testset "$name" begin
            @test_throws ArgumentError allocate(_NoFloat64GPU())

            err = try
                allocate(_NoFloat64GPU())
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("Float64", err.msg)
            @test occursin("Float32", err.msg)

            # and it is refused *before* the backend is asked for memory: this one can allocate
            # nothing, so anything that got past the check would be a `MethodError` instead
            @test_throws ArgumentError allocate(_UnallocatableGPU())
        end
    end
end

@testset "a width the backend declares it cannot hold is refused for nothing else" begin
    # the same calls at `Float32`, on the backend that cannot hold a `Float64`: the check is about
    # one width and must not stand in the way of any other
    for (name, allocate) in runnable(allocators(Float32))
        @testset "$name" begin
            A = allocate(_NoFloat64GPU())
            @test eltype(A) === Float32
        end
    end
end

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

# `SymmetricMatrix`'s backend-taking allocators are new: it had none where `SkewSymMatrix` had
# three, although the two mirror each other everywhere else and both are optimizer parameters.
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

# `GrassmannLieAlgHorMatrix` had the backend-taking `zeros` and not the `rand`, where
# `StiefelLieAlgHorMatrix` had both.
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

# The derived allocators take an instance, so the element type and the backend both come from the
# argument. They must keep working on a backend that has no `Float64` even for a `Float64` array,
# because nothing about them lets a caller ask for a width the holder does not already have.
@testset "an allocator that takes an instance is not checked" begin
    A = SkewSymMatrix(rand(Float64, n * (n - 1) ÷ 2), n)
    for allocate in (zero, similar, copy)
        @test eltype(allocate(A)) === Float64
    end
    @test _check_supported_eltype(_Float64GPU(), Float64) === nothing
    @test _check_supported_eltype(CPU(), Float64) === nothing
    # one width, and only on a backend that has declared it cannot hold it
    for T in (Float32, Int32, Int64, ComplexF32)
        @test _check_supported_eltype(_NoFloat64GPU(), T) === nothing
    end
    @test_throws ArgumentError _check_supported_eltype(_NoFloat64GPU(), Float64)
end
