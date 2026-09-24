# The device sweep of `device_products.jl` on a real Apple GPU, with `MtlArray` in place of the
# `JLArrays` stand-in. `runtests.jl` includes this file on every Apple-silicon Mac.
#
# Where `Metal.functional()` is `false` the file skips itself: on a Mac without a usable device, and
# inside a sandbox, where `Metal.device()` can be a null device although the hardware is present.
# `Pkg.test(test_args = ["metal"])` asks for this file alone, and then a missing device is a failure
# rather than a skip. `.github/workflows/Metal.yml` runs it that way, so that job cannot pass
# without having run the sweep.
#
# Metal supplies the `lu` that `JLArrays` lacks, so the two `cayley` rows that are gaps in
# `device_products.jl` pass here and every row is asserted to pass.

using Metal
using Test

if Metal.functional()
    include(joinpath(@__DIR__, "..", "scripts", "device_products.jl"))

    @testset "every product and sum runs on Metal and matches the host" begin
        @info "Metal device" Metal.device()
        for (name, status) in device_products(MtlArray)
            @testset "$name" begin
                @test status === :pass
            end
        end
    end
elseif "metal" in ARGS
    @test Metal.functional()
else
    @info "Metal is not functional here; the Metal tests are skipped."
end
