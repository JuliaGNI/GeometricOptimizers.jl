# An allocation that stands in for another one is made on the backend that other one is on.
#
# This is the property an optimizer cache rests on, and nothing tested it before. `changebackend.jl`
# next door pins *transfers*; this file pins the like-for-like allocations that happen once the
# parameters are already on a device: `similar`, `zero` and `copy` on a leaf, and the
# `_similar`/`_zero`/`_copy` walk over a whole set.
#
# `OptimizerCache(Adam(), x)` calls `AdamCache(_copy(x), _zero(x), _zero(x))`, that three-argument
# method builds its fourth block as `Δg = _similar(g)`, and the four-argument method it forwards to
# constrains `g`, `δ` and `Δg` to a single `AT <: GradientStorage{T}`. So a `similar` that came back
# on the host beside a gradient that lives on a device is not a slow path or a wrong number: it is a
# `MethodError` at cache construction, with the two `NetworkParameters` types printed side by side
# and only the innermost array type differing.
#
# That is what the pendulum stage of `GMLDatasets`' revision harness hit on an RTX 4090
# (`GMLDatasets#12`, run `20260903T125418Z_smoke`): `Optimizer(Adam(), network)` on a device-resident
# network, `no method matching AdamCache(::NetworkParameters{…CuArray…}, …, ::NetworkParameters{…
# StiefelLieAlgHorMatrix{Float32, SkewSymMatrix{Float32, Vector{Float32}}, Matrix{Float32}}})`. Both
# horizontal lifts called the *host* `zeros` from `similar`. The image stages of the same run never
# reached it because their script keeps the parameters in a host container and copies to the device
# inside the objective, so the pendulum was the first device-resident optimizer in the harness.
#
# `JLArray` stands in for the device here, as it does in `retractions/exponential_accuracy.jl`: its
# backend is a `KernelAbstractions.GPU`, so it takes the same `zeros(::Backend, …)` methods a
# `CuArray` does, and it is what makes the regression visible without a GPU.
#
# `StiefelProjection` is deliberately not in the table below. It has a backend and it is an
# `AbstractMatrix`, but `similar` and `zero` on it reach the generic fallbacks and return a host
# `Array`, and `copy` fails outright on a device because the fallback indexes it. Neither is on a
# path this file is about: `E` is a fixed matrix that is only ever read, never a parameter leaf and
# never a cache block, so no `_similar` walk can reach it. Fixing it is a separate question about
# what `similar(::StiefelProjection)` should even mean.

using GeometricOptimizers
using GeometricOptimizers: AdamCache, GradientCache, MomentumCache
using GeometricOptimizers: LowerTriangular, UpperTriangular
using GeometricOptimizers: _copy, _fill!, _similar, _zero
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: qr!
using NeuralNetworkParameters: NetworkParameters
using Random
using Test

Random.seed!(1234)

const T = Float32
const N, n = 6, 3

const device = KernelAbstractions.get_backend(JLArray(zeros(T, 1)))

# An orthonormal representative drawn on the host and moved over, rather than
# `rand(device, StiefelManifold{T}, N, n)`: the device `rand` materializes its QR factor as
# `typeof(A)(qr!(A).Q)`, which `CuArray` implements and `JLArray` does not.
const host_point = Matrix(qr!(randn(T, N, n)).Q)[:, 1:n]

point(MT) = MT(JLArray(host_point))

# Every leaf shape a parameter set or a cache block can hold, on the device. The two lifts are the
# ones that regressed; the rest are here because the invariant is the same for all of them and
# because delegating `similar` to a component is easy to lose in a refactor.
leaves = (
    plain = JLArray(rand(T, N, n)),
    symmetric = SymmetricMatrix(JLArray(rand(T, n, n))),
    skew = SkewSymMatrix(JLArray(rand(T, n, n))),
    lower = LowerTriangular(JLArray(rand(T, n, n))),
    upper = UpperTriangular(JLArray(rand(T, n, n))),
    stiefhor = StiefelLieAlgHorMatrix(
        SkewSymMatrix(JLArray(rand(T, n, n))), JLArray(rand(T, N - n, n)), N, n),
    grasshor = GrassmannLieAlgHorMatrix(JLArray(rand(T, N - n, n)), N, n)
)

manifolds = (stiefel = point(StiefelManifold), grassmann = point(GrassmannManifold))

@testset "a leaf on a device allocates its stand-ins there" begin
    for (name, x) in pairs(leaves)
        @testset "$name" begin
            @test KernelAbstractions.get_backend(x) == device
            for allocate in (similar, zero, copy, _similar, _zero, _copy)
                y = allocate(x)
                @test typeof(y) === typeof(x)
                @test KernelAbstractions.get_backend(y) == device
            end
        end
    end
end

@testset "a lift keeps its backend through the `dims` method too" begin
    # `similar(A, dims...)` is the method the `AbstractArray` interface reaches; it takes the same
    # two integers and used to lose the backend in the same way
    for name in (:stiefhor, :grasshor)
        A = leaves[name]
        @testset "$name" begin
            @test typeof(similar(A, A.N, A.n)) === typeof(A)
            @test KernelAbstractions.get_backend(similar(A, A.N, A.n)) == device
        end
    end
end

@testset "a manifold point on a device" begin
    for (name, Y) in pairs(manifolds)
        @testset "$name" begin
            # `similar` of a point is an error by design: a point is not storage to be reused, and
            # the message says to use `rand`. `zero` is the horizontal lift of its tangent space,
            # which is the type the cache blocks then are.
            @test_throws ErrorException similar(Y)

            for allocate in (zero, copy, _zero, _copy)
                @test KernelAbstractions.get_backend(allocate(Y)) == device
            end
            @test zero(Y) isa GeometricOptimizers.AbstractLieAlgHorMatrix
            @test typeof(copy(Y)) === typeof(Y)
        end
    end
end

@testset "the cache blocks of a device-resident parameter set agree" begin
    # the shape a network actually has: nested, and mixing points with ordinary arrays and with
    # structured matrices
    ps = NetworkParameters((
        L1 = (Y = manifolds.stiefel, b = JLArray(rand(T, N))),
        L2 = (S = leaves.symmetric, G = manifolds.grassmann)))

    x = _copy(ps)
    g = _zero(ps)
    δ = _zero(ps)
    Δg = _similar(g)

    # the constraint the four-argument cache constructors declare: one `AT` for all three
    @test typeof(Δg) === typeof(g) === typeof(δ)

    @testset "every leaf of every block is still on the device" begin
        for block in (x, g, δ, Δg)
            @test KernelAbstractions.get_backend(block.L1.Y) == device
            @test KernelAbstractions.get_backend(block.L1.b) == device
            @test KernelAbstractions.get_backend(block.L2.S) == device
            @test KernelAbstractions.get_backend(block.L2.G) == device
        end
    end

    # the caches mark their blocks not-yet-current with a NaN fill, which is the first thing that
    # touches them after allocation
    _fill!(Δg, T(NaN))
    @test KernelAbstractions.get_backend(Δg.L1.Y) == device

    # and the dispatch a host `Δg` used to make unavailable, for every cache whose three-argument
    # method builds a fourth block this way. `applicable` rather than a constructed cache: the
    # four-argument body also builds `GlobalSection(_copy(x))`, and `global_section` materializes a
    # QR factor the way the device `rand` does, so it cannot run on a `JLArray` for a reason that
    # has nothing to do with this fix.
    @test applicable(AdamCache, x, g, δ, Δg)
    @test applicable(MomentumCache, x, g, δ, Δg)
    @test applicable(GradientCache, x, g, δ, Δg)
end

@testset "a manifold point is drawn on its own backend" begin
    # `_similar(::Manifold)` is `rand(backend, manifold_constructor(a){T}, size(a)...)`, and it is
    # the same defect on the state path: `AdamState` and `MomentumState` declare a single `OT` for
    # `x` and `x̄`, and `x̄` is `_similar(x)`, so a device `x` with a host `x̄` did not satisfy it.
    # The device half cannot be exercised on a `JLArray` for the QR reason above, so what is pinned
    # here is that routing the draw through the backend left the host path exactly as it was, point
    # on the manifold included.
    for Y in (rand(StiefelManifold{T}, N, n), rand(GrassmannManifold{T}, N, n))
        @test typeof(_similar(Y)) === typeof(Y)
        @test KernelAbstractions.get_backend(_similar(Y)) ==
              KernelAbstractions.get_backend(Y)
        @test GeometricOptimizers.check(_similar(Y)) < 10 * eps(T)
    end
end
