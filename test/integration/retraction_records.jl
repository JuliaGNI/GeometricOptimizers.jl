# The schema `scripts/retraction_records.jl` writes, the CSV dialect it writes it in, the options it
# rejects, and the two row kinds a consumer distinguishes.
#
# The script is included rather than imported. Everything above the `PROGRAM_FILE` guard at its foot
# is definitions only, so including it runs nothing.
#
# The benchmark runs on the smallest sweeps there are: this pins the shape of the output, the
# warm-up structure and the lifts, not the performance figures, which are machine-dependent.

using Test

include(joinpath(@__DIR__, "..", "..", "scripts", "retraction_records.jl"))

const SOURCE_OPTIONS = ["--source-sha", "0" ^ 40, "--source-dirty", "false",
    "--source-patch-file", "go.patch", "--source-patch-sha256", "a" ^ 64]

function smoke_options(output, extra...)
    parse_options([SOURCE_OPTIONS..., "--output", output,
        "--rows", "6", "--columns", "2", "--repetitions", "1", extra...])
end

@testset "the schema is the one a consumer validates against" begin
    @test RETRACTION_SCHEMA_VERSION == 1
    @test length(RETRACTION_HEADER) == 26
    @test allunique(RETRACTION_HEADER)
    @test first(RETRACTION_HEADER) == "schema_version"
    # The four provenance columns are last, and are what the caller stamps every row with.
    @test RETRACTION_HEADER[(end - 3):end] ==
          ["go_sha", "go_dirty", "go_patch_file", "go_patch_sha256"]
end

@testset "CSV quoting is RFC 4180" begin
    @test csv_field("ScaledSquaring") == "ScaledSquaring"
    @test csv_field(12) == "12"
    @test csv_field("a,b") == "\"a,b\""
    @test csv_field("say \"hi\"") == "\"say \"\"hi\"\"\""
    @test csv_field("one\ntwo") == "\"one\ntwo\""
    # An error message is the field that can contain anything.
    @test startswith(csv_field(sanitize_error(ArgumentError("a, b"))), "\"")
    @test !occursin('\n', sanitize_error(ErrorException("one\ntwo")))
end

@testset "the command line rejects what it cannot record" begin
    @test parse_options(["--help"]) === nothing
    @test_throws ArgumentError parse_options(["--backend", "metal", SOURCE_OPTIONS...])
    @test_throws ArgumentError parse_options(["--precision", "Float16", SOURCE_OPTIONS...])
    @test_throws ArgumentError parse_options(["--rows", "0", SOURCE_OPTIONS...])
    @test_throws ArgumentError parse_options(["--columns", "30", SOURCE_OPTIONS...])
    @test_throws ArgumentError parse_options(["--repetitions", "0", SOURCE_OPTIONS...])
    @test_throws ArgumentError parse_options(["--scales", "-1", SOURCE_OPTIONS...])
    @test_throws ArgumentError parse_options(["--scales", "2.0", SOURCE_OPTIONS...])
    @test_throws ArgumentError parse_options(["--unknown", "1", SOURCE_OPTIONS...])
    @test_throws ArgumentError parse_options(["--rows"])

    # An unstamped or ill-formed provenance stamp is refused here rather than producing rows a
    # validator has to reject afterwards.
    @test_throws ArgumentError parse_options(String[])
    @test_throws ArgumentError parse_options([
        "--source-sha", "abc", "--source-dirty", "false",
        "--source-patch-file", "go.patch", "--source-patch-sha256", "a" ^ 64])
    @test_throws ArgumentError parse_options([
        "--source-sha", "0" ^ 40, "--source-dirty", "yes",
        "--source-patch-file", "go.patch", "--source-patch-sha256", "a" ^ 64])
    @test_throws ArgumentError parse_options([
        "--source-sha", "0" ^ 40, "--source-dirty", "false",
        "--source-patch-file", "", "--source-patch-sha256", "a" ^ 64])

    options = parse_options([SOURCE_OPTIONS..., "--rows", "6", "--columns", "2",
        "--scales", "0.1,1.0", "--repetitions", "3"])
    @test options.backend === :cpu
    @test options.precision === Float64
    @test options.scales == [0.1, 1.0]
    @test options.repetitions == 3
    @test options.seed == SWEEP_SEED
end

@testset "one host call is timed and measured together" begin
    calls = Ref(0)
    result, seconds, bytes = host_measure(() -> (calls[] += 1; zeros(64)))
    @test calls[] == 1          # not two: the runtime and the bytes describe the same invocation
    @test result == zeros(64)
    @test seconds ≥ 0
    @test bytes > 0
end

@testset "a smoke sweep writes the rows a consumer expects" begin
    output = joinpath(mktempdir(), "retraction-runs.csv")
    records = run_benchmark(smoke_options(output, "--scales", "0.1"))

    # three host algorithms, one warm-up plus one measured repetition each, one scale
    @test length(records) == 6
    @test all(record -> record.success, records)
    @test Set(record.algorithm for record in records) ==
          Set(("ScaledSquaring", "NativePade", "AugmentedPade"))
    @test all(record -> record.backend == "CPU", records)
    @test all(record -> record.memory_metric == "host_allocated_bytes", records)

    # Repetition 0 is the warm-up and exactly the warm-up; a consumer treats only `warmup=false`
    # rows as steady-state measurements.
    @test all(record -> record.warmup == (record.repetition == 0), records)
    @test count(record -> record.warmup, records) == 3

    # The provenance stamp is the caller's, unchanged, and identical across every row.
    @test all(record -> record.go_sha == "0" ^ 40, records)
    @test all(record -> record.go_patch_file == "go.patch", records)

    # At this lift norm every algorithm agrees with `AugmentedPade` and with the dense exponential
    # to round-off, and lands on the manifold.
    @test all(record -> record.agreement_error < 1.0e-13, records)
    @test all(record -> record.forward_error < 1.0e-13, records)
    @test all(record -> record.manifold_constraint_error < 1.0e-13, records)

    # The file on disk is the same table, header first.
    @test readlines(output) == [join(RETRACTION_HEADER, ','); csv_line.(records)]
end

@testset "a scale names the same lift whichever scales are selected" begin
    directory = mktempdir()
    full = run_benchmark(smoke_options(joinpath(directory, "full.csv")))
    alone = run_benchmark(smoke_options(joinpath(directory, "alone.csv"), "--scales", "30.0"))
    @test Set(record.scale for record in full) == Set(SCALES)
    @test only(unique(record.lift_norm for record in alone)) ==
          only(unique(record.lift_norm for record in full if record.scale == 30.0))
end

@testset "Float32 rows carry Float32 errors" begin
    records = run_benchmark(smoke_options(joinpath(mktempdir(), "single.csv"),
        "--precision", "Float32", "--scales", "0.1"))
    @test all(record -> record.success && record.precision === Float32, records)
    @test all(record -> record.forward_error < 1.0e-5, records)
    # Computed in `Float32` and compared in `Float64`, so not at `Float64` round-off.
    @test any(record -> record.forward_error > 1.0e-10, records)
end

@testset "an invocation that throws is a row and not a gap" begin
    options = smoke_options(joinpath(mktempdir(), "unused.csv"))
    failing = (name = "ScaledSquaring", backend = "CPU", device = "test",
        operation = () -> error("synthetic retraction failure"), measure = host_measure,
        memory_metric = "host_allocated_bytes")
    base = base_record(options, failing, 0.1, 1.0, 1)
    record = record_invocation(base, failing, [1.0;;], [1.0;;])

    @test keys(record) == RECORD_FIELDS
    @test !record.success
    @test record.error_type == "ErrorException"
    @test occursin("synthetic retraction failure", record.error_message)
    # `NaN`: the three errors were never computed.
    @test isnan(record.agreement_error)
    @test isnan(record.forward_error)
    @test isnan(record.manifold_constraint_error)
    @test record.memory_bytes == 0

    # A failure after the call, here in the transfer of its result, is not the algorithm's.
    returns_nothing = merge(failing, (operation = () -> nothing,))
    @test_throws MethodError record_invocation(base, returns_nothing, [1.0;;], [1.0;;])
end
