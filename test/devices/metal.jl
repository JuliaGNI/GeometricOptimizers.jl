# The device sweeps of `device_products.jl` and `device_solve.jl` on a real Apple GPU, with
# `MtlArray` in place of the `JLArrays` stand-in, and the direct Cayley retraction and its two
# transports against the dense host reference. `runtests.jl` runs this file in the `metal`
# group, which a default run on Apple silicon includes and `Pkg.test(test_args = ["metal"])`
# selects anywhere.
#
# Where `Metal.functional()` is `true` the file runs the sweep. Where Metal loads and
# `Metal.functional()` is `false` the file runs no test and records one visible skip: on a Mac
# without a usable device, and inside a sandbox, where `Metal.devices()` is empty although the
# hardware is present. On Apple silicon with no device and no Metal cache for the flags of the
# run, `using Metal` fails at the Metal precompile before the skip (K15 in `KNOWN_ISSUES.md`).
# The skip does not fail the run, so `.github/workflows/Metal.yml` keeps the guarantee instead: a
# step before the tests fails that job where Metal is not functional, so it cannot pass without
# having run the sweeps.
#
# Metal supplies the `lu` that `JLArrays` lacks, so the two `cayley` rows that are gaps in
# `device_products.jl` pass here and every row is asserted to pass.

using GeometricOptimizers
using GPUArraysCore: allowscalar
using LinearAlgebra: I, norm
using Metal
using Random
using Test

if Metal.functional()
    include(joinpath(@__DIR__, "..", "..", "scripts", "device_products.jl"))
    include(joinpath(@__DIR__, "..", "..", "scripts", "device_solve.jl"))

    @testset "every product and sum runs on Metal and matches the host" begin
        @info "Metal device" Metal.device()
        for (name, status) in device_products(MtlArray)
            @testset "$name" begin
                @test status === :pass
            end
        end
    end

    # Metal draws its random numbers from a generator of its own, so the `Adam` rows of (b) and (c)
    # see another global section than the host twin and are compared by a property; see
    # `UNMATCHED_ADAM_RTOL` in `device_solve.jl`. That generator is seeded with the row's seed, so
    # the section, and the row, is the same on every run. Metal supplies an `lu`, so `Cayley()` runs
    # too. Metal has no `Float64`, so the sweep runs in `Float32` only.
    @testset "solve! runs on Metal and matches the host twin, $(nameof(typeof(retraction)))" for retraction in (
        Cayley(), Geodesic())
        for (name, status) in device_solve(MtlArray, Float32; retraction = retraction,
            matched_rng = false, seed_device! = s -> Random.seed!(Metal.default_rng(), s))
            @testset "$name" begin
                @test status === :pass
            end
        end
    end

    # The direct Cayley retraction and the two direct transports on Metal, in `Float32`. The oracle
    # is the dense `M × M` Cayley transform `(𝕀 - W/2)⁻¹(𝕀 + W/2)Y` and its differential, computed
    # on the host in `Float64` from the same data, so a wrong formula fails here although a host run
    # of the same functions would agree with it. The bounds are those of the host test,
    # `test/manifolds/direct_cayley_and_metrics.jl`: the `Float32` data are exact in `Float64`, so
    # the reference adds no error of its own, and the device differs from the host only in the
    # rounding of its products and of its `2k × 2k` solve.
    @testset "the direct Cayley retraction and transports run on Metal and match the dense host reference" begin
        T = Float32
        rng = Random.Xoshiro(5)
        M, k = 20, 4
        proj(Y, Ξ) = Ξ - Y * (Y' * Ξ + Ξ' * Y) / 2
        generator(Y, Δ) = Δ * Y' - Y * Δ' + Y * Δ' * Y * Y'
        Y = rand(rng, StiefelManifold{T}, M, k)
        Δ = proj(Y.A, randn(rng, T, M, k))
        Δ = T(1 / 2) * Δ / norm(Δ)
        # a vector off the tangent space at `Y`, and a point that is not the retraction of `Δ`
        S = randn(rng, T, k, k)
        Ξ = proj(Y.A, randn(rng, T, M, k)) + Y.A * (S + S')
        Y_other = rand(rng, StiefelManifold{T}, M, k)

        Y₆₄, Δ₆₄, Ξ₆₄ = Float64.(Y.A), Float64.(Δ), Float64.(Ξ)
        Y_other₆₄ = Float64.(Y_other.A)
        L = I - generator(Y₆₄, Δ₆₄) / 2
        retracted = L \ ((I + generator(Y₆₄, Δ₆₄) / 2) * Y₆₄)
        differentiated = L \ (generator(Y₆₄, proj(Y₆₄, Ξ₆₄)) * (L \ Y₆₄))
        projected = proj(Y_other₆₄, Ξ₆₄)

        Y_d = StiefelManifold(MtlArray(Y.A))
        Δ_d, Ξ_d = MtlArray(Δ), MtlArray(Ξ)
        Y_other_d = StiefelManifold(MtlArray(Y_other.A))
        on_device(X) = X isa MtlMatrix && eltype(X) === T
        allowscalar(false)

        Y_new = cayley_direct(Y_d, Δ_d)
        @test on_device(Y_new.A)
        @test norm(Array(Y_new.A) - retracted) ≤ 50eps(T)
        @test norm(Array(Y_new.A)' * Array(Y_new.A) - I) ≤ 20eps(T)

        ws = DirectCayleyWorkspace(Y_d)
        Z = StiefelManifold(similar(Y_d.A))
        @test cayley_direct!(Z, Y_d, Δ_d, ws) === Z
        @test on_device(Z.A)
        @test norm(Array(Z.A) - retracted) ≤ 50eps(T)

        for (tr, reference, bound) in ((ProjectionTransport(), projected, 10eps(T)),
            (DifferentiatedCayleyTransport(), differentiated, 50eps(T)))
            out = transport(Y_other_d, Y_d, Δ_d, Ξ_d, tr)
            @test on_device(out)
            @test norm(Array(out) - reference) ≤ bound * norm(Ξ)
            out! = similar(Ξ_d)
            @test transport!(out!, Y_other_d, Y_d, Δ_d, Ξ_d, tr, ws) === out!
            @test on_device(out!)
            @test norm(Array(out!) - reference) ≤ bound * norm(Ξ)
        end
    end
else
    @test_skip Metal.functional()
end
