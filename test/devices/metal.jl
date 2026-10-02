# The device sweep of `device_products.jl` on a real Apple GPU, with `MtlArray` in place of the
# `JLArrays` stand-in. `runtests.jl` runs this file in the `metal` group, which a default run on
# Apple silicon includes and `Pkg.test(test_args = ["metal"])` selects anywhere.
#
# Where `Metal.functional()` is `true` the file runs the sweep. Where it is `false` the file runs
# no test and records one visible skip: on a Mac without a usable device, and inside a sandbox,
# where `Metal.device()` can be a null device although the hardware is present. The skip does not
# fail the run, so `.github/workflows/Metal.yml` keeps the guarantee instead: a step before the
# tests fails that job where Metal is not functional, so it cannot pass without having run the sweep.
#
# Metal supplies the `lu` that `JLArrays` lacks, so the two `cayley` rows that are gaps in
# `device_products.jl` pass here and every row is asserted to pass.

using Metal
using Test

if Metal.functional()
    include(joinpath(@__DIR__, "..", "..", "scripts", "device_products.jl"))

    @testset "every product and sum runs on Metal and matches the host" begin
        @info "Metal device" Metal.device()
        for (name, status) in device_products(MtlArray)
            @testset "$name" begin
                @test status === :pass
            end
        end
    end
else
    @test_skip Metal.functional()
end
