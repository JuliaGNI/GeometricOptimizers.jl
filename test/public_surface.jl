# The public surface: one allocator chain, one type-argument convention, the renames, and the
# names that are public without being exported.
#
# The allocator convention is `rand([rng,] [backend,] X{T}, dims::Integer...)` and
# `zeros([backend,] X{T}, dims::Integer...)` for every owned array and manifold type. A bare `X`
# means `default_eltype(backend)`, and the backend defaults to `CPU()`.

using GeometricOptimizers
using GeometricOptimizers: default_eltype
using GPUArraysCore: GPUArraysCore
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions, CPU, get_backend
using Random: Random, Xoshiro
using Test

GPUArraysCore.allowscalar(false)

const jl_backend = get_backend(JLArray(zeros(Float32, 1)))

# The owned types and a size each one accepts. Manifolds have no `zeros`.
const SQUARE = (
    SkewSymMatrix, SymmetricMatrix, StrictlyLowerTriangular, StrictlyUpperTriangular)
const LIFTS = (StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix)
const MANIFOLDS = (StiefelManifold, GrassmannManifold)
_dims(X) = X in SQUARE ? (3,) : (5, 2)
const ARRAYS = (SQUARE..., LIFTS...)

@testset "the rows of the allocator table return the requested type and infer" begin
    for T in (Float32, Float64)
        for A in (@inferred(zeros(SkewSymMatrix{T}, Int32(3))),
            @inferred(rand(SkewSymMatrix{T}, Int32(3))),
            @inferred(zeros(CPU(), SkewSymMatrix{T}, Int32(3))))
            @test A isa SkewSymMatrix{T}
            @test eltype(A) === T
            @test get_backend(A) == CPU()
            @test size(A) == (3, 3)
        end
    end
    A = @inferred rand(CPU(), SkewSymMatrix, 3)
    @test eltype(A) === Float64 && get_backend(A) == CPU()
    A = @inferred rand(CPU(), StiefelLieAlgHorMatrix, 5, 2)
    @test eltype(A) === Float64 && get_backend(A) == CPU() && size(A) == (5, 5)
    A = @inferred zeros(StrictlyLowerTriangular, 3)
    @test A isa StrictlyLowerTriangular{Float64} && size(A) == (3, 3)
end

@testset "every allocator follows one convention, on the host and on a device" begin
    for backend in (CPU(), jl_backend), X in ARRAYS, T in (Float32, Float64)
        d = _dims(X)
        for A in (@inferred(zeros(backend, X{T}, d...)),
            @inferred(rand(backend, X{T}, d...)),
            @inferred(rand(Xoshiro(1), backend, X{T}, d...)))
            @test A isa X{T}
            @test eltype(A) === T
            @test get_backend(A) == backend
        end
        @test eltype(@inferred zeros(backend, X, d...)) === default_eltype(backend)
        @test eltype(@inferred rand(backend, X, d...)) === default_eltype(backend)
        @test eltype(@inferred rand(Xoshiro(1), backend, X, d...)) ===
              default_eltype(backend)
    end
    for backend in (CPU(), jl_backend), X in MANIFOLDS, T in (Float32, Float64)
        for Y in (@inferred(rand(backend, X{T}, 5, 2)),
            @inferred(rand(Xoshiro(1), backend, X{T}, 5, 2)))
            @test Y isa X{T}
            @test get_backend(Y) == backend
        end
        @test eltype(@inferred rand(backend, X, 5, 2)) === default_eltype(backend)
    end
    # the backendless forms, which place on the host
    for X in ARRAYS, T in (Float32, Float64)

        d = _dims(X)
        for A in (@inferred(zeros(X{T}, d...)), @inferred(rand(X{T}, d...)),
            @inferred(rand(Xoshiro(1), X{T}, d...)))
            @test A isa X{T}
            @test get_backend(A) == CPU()
        end
        @test eltype(@inferred zeros(X, d...)) === Float64
        @test eltype(@inferred rand(X, d...)) === Float64
    end
end

@testset "a hard-coded Float64 no longer overrides default_eltype" begin
    @test default_eltype(jl_backend) === Float32
    @test eltype(zeros(jl_backend, SkewSymMatrix, 3)) === Float32
    @test eltype(zeros(jl_backend, SymmetricMatrix, 3)) === Float32
    @test eltype(rand(jl_backend, StrictlyLowerTriangular, 3)) === Float32
    @test eltype(zeros(CPU(), SkewSymMatrix, 3)) === Float64
    @test eltype(zeros(CPU(), SymmetricMatrix, 3)) === Float64
    @test eltype(rand(CPU(), StrictlyLowerTriangular, 3)) === Float64
end

@testset "the chain supplies rng and backend consistently" begin
    for X in ARRAYS, T in (Float32, Float64)

        d = _dims(X)
        @test rand(Xoshiro(1), CPU(), X{T}, d...) == rand(Xoshiro(1), X{T}, d...)
    end
    for X in MANIFOLDS, T in (Float32, Float64)

        @test rand(Xoshiro(1), CPU(), X{T}, 5, 2) == rand(Xoshiro(1), X{T}, 5, 2)
    end
    for T in (Float32, Float64)
        @test rand(Xoshiro(1), CPU(), SymplecticStiefelManifold{T}, 6, 4) ==
              rand(Xoshiro(1), SymplecticStiefelManifold{T}, 6, 4)
    end
    # the old order, backend before rng, is gone
    @test_throws MethodError rand(CPU(), Xoshiro(1), StiefelManifold{Float64}, 5, 2)
    @test_throws MethodError rand(CPU(), Xoshiro(1), StiefelManifold, 5, 2)
end

# Against the methods of `Base` and `Random`. `GPUArrays`, which `JLArrays` loads, has
# `rand(::GPUArrays.RNG, ::Type{T}, ::Integer, ::Integer...)`, and any `rand(::AbstractRNG, ::Type{X},
# …)` of this package is ambiguous with it; that pair is not this package's to resolve.
@testset "no ambiguity between Base.rand or Base.zeros and a method of this package" begin
    isown(m) = parentmodule(m) === GeometricOptimizers
    isbase(m) = parentmodule(m) in (Base, Random, Core)
    pairs = []
    for f in (Base.rand, Base.zeros)
        ms = collect(methods(f))
        for i in eachindex(ms), j in (i + 1):lastindex(ms)

            m₁, m₂ = ms[i], ms[j]
            ((isown(m₁) && isbase(m₂)) || (isbase(m₁) && isown(m₂))) || continue
            Base.isambiguous(m₁, m₂) && push!(pairs, (m₁, m₂))
        end
    end
    for (m₁, m₂) in pairs
        @info "allocator ambiguity" m₁ m₂
    end
    @test isempty(pairs)
end

@testset "StiefelProjection takes one argument order" begin
    for T in (Float32, Float64)
        E = StiefelProjection(T, 6, 2)
        @test E == StiefelProjection(CPU(), T, 6, 2)
        @test eltype(E) === T && size(E) == (6, 2)
        @test get_backend(StiefelProjection(jl_backend, T, 6, 2)) == jl_backend
    end
    @test_throws MethodError StiefelProjection(6, 2, Float32)
end

@testset "a 1 × 1 skew-symmetric matrix stores nothing" begin
    for backend in (CPU(), jl_backend), T in (Float32, Float64)

        A = zeros(backend, SkewSymMatrix{T}, 1)
        @test size(A) == (1, 1)
        @test length(A.S) == 0
        @test eltype(A) === T
    end
end

@testset "the renames" begin
    @test isdefined(GeometricOptimizers, :NewtonState)
    @test Base.isexported(GeometricOptimizers, :NewtonState)
    @test StrictlyLowerTriangular <: AbstractTriangular
    @test StrictlyUpperTriangular <: AbstractTriangular
end

# `names` lists the `public` names as well as the exported ones, so it cannot tell the two apart.
@testset "gradient, value and check are public and not exported" begin
    for s in (:gradient, :value, :check)
        @test Base.ispublic(GeometricOptimizers, s)
        @test !Base.isexported(GeometricOptimizers, s)
    end
end

# A fresh process each, because the clash only shows in a `Main` that has `using`'d both modules.
function _in_fresh_process(code)
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$project -e $code`
    success(pipeline(cmd; stdout = devnull, stderr = stderr))
end

# Every exported name; a `public` name is not brought into `Main` by `using`.
const _EVERY_NAME = """
for s in names(GeometricOptimizers)
    Base.isexported(GeometricOptimizers, s) || continue
    Core.eval(Main, s)
end
"""

@testset "every name is usable next to LinearAlgebra" begin
    @test _in_fresh_process("""
        using GeometricOptimizers, LinearAlgebra
        LowerTriangular === LinearAlgebra.LowerTriangular || exit(1)
        UpperTriangular === LinearAlgebra.UpperTriangular || exit(1)
        $(_EVERY_NAME)
        """)
end

# `Zygote` is no test dependency; a module that exports the same three names stands in for it.
@testset "every name is usable next to a module that exports gradient, value and check" begin
    @test _in_fresh_process("""
        module Other
        export gradient, value, check
        gradient(x) = :other
        value(x) = :other
        check(x) = :other
        end
        using GeometricOptimizers, .Other
        gradient === Other.gradient || exit(1)
        value === Other.value || exit(1)
        check === Other.check || exit(1)
        $(_EVERY_NAME)
        """)
end
