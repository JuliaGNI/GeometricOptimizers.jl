# Focused regression for `scripts/retraction_records.jl`: the schema it writes, the CSV dialect it
# writes it in, the options it rejects, and the two row kinds a consumer distinguishes.
#
# It is a *script* and not part of the package, so it is included rather than imported. Everything
# below the `PROGRAM_FILE` guard at its foot is definitions only, so including it runs nothing.
#
# The benchmark itself is run once, on the smallest sweep there is: this pins the shape of the
# output and the warm-up structure, not the performance figures, which are machine-dependent and
# belong to the tables `retraction_accuracy.jl` prints.

using Test
using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "scripts", "retraction_records.jl"))

const SOURCE_OPTIONS = ["--source-sha", "0" ^ 40, "--source-dirty", "false",
    "--source-patch-file", "go.patch", "--source-patch-sha256", "a" ^ 64]

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
    # An error message is the field that can contain anything, which is why the quoting is not
    # optional and not hand-written at the call site.
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
    @test_throws ArgumentError parse_options(["--unknown", "1", SOURCE_OPTIONS...])
    @test_throws ArgumentError parse_options(["--rows"])

    # Provenance is the caller's, so an unstamped or ill-formed stamp is refused here rather than
    # producing rows a validator has to reject afterwards.
    @test_throws ArgumentError parse_options(String[])
    @test_throws ArgumentError parse_options(["--source-sha", "abc", "--source-dirty", "false",
        "--source-patch-file", "go.patch", "--source-patch-sha256", "a" ^ 64])
    @test_throws ArgumentError parse_options(["--source-sha", "0" ^ 40, "--source-dirty", "yes",
        "--source-patch-file", "go.patch", "--source-patch-sha256", "a" ^ 64])
    @test_throws ArgumentError parse_options(["--source-sha", "0" ^ 40, "--source-dirty", "false",
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
    options = parse_options([SOURCE_OPTIONS..., "--output", output, "--rows", "6",
        "--columns", "2", "--scales", "0.1", "--repetitions", "1"])
    records = run_benchmark(options)

    # three host algorithms, one warm-up plus one measured repetition each, one scale
    @test length(records) == 6
    @test all(record -> Set(keys(record)) == Set(RETRACTION_HEADER), records)
    @test all(record -> record["success"] == "true", records)
    @test Set(record["algorithm"] for record in records) ==
          Set(("ScaledSquaring", "NativePade", "AugmentedPade"))
    @test all(record -> record["backend"] == "CPU", records)
    @test all(record -> record["memory_metric"] == "host_allocated_bytes", records)

    # Repetition 0 is the warm-up and exactly the warm-up; a consumer treats only `warmup=false`
    # rows as steady-state measurements.
    @test all(record -> (record["warmup"] == "true") == (record["repetition"] == "0"), records)
    @test count(record -> record["warmup"] == "true", records) == 3

    # The provenance stamp is the caller's, unchanged, and identical across every row.
    @test all(record -> record["go_sha"] == "0" ^ 40, records)
    @test all(record -> record["go_patch_file"] == "go.patch", records)

    # `AugmentedPade` is the agreement reference, so it agrees with itself exactly, and every
    # algorithm still lands on the manifold at this lift norm.
    augmented = first(filter(record -> record["algorithm"] == "AugmentedPade", records))
    @test parse(Float64, augmented["agreement_error"]) < 1.0e-14
    @test all(records) do record
        parse(Float64, record["manifold_constraint_error"]) < 1.0e-10
    end
    @test all(record -> parse(Float64, record["forward_error"]) < 1.0e-8, records)

    # and the file on disk is the same table, header first
    lines = readlines(output)
    @test first(lines) == join(RETRACTION_HEADER, ',')
    @test length(lines) == length(records) + 1
end

@testset "an invocation that throws is a row and not a gap" begin
    failing = (name = "ScaledSquaring", backend = "CPU", device = "test",
        operation = () -> error("synthetic retraction failure"),
        synchronize = () -> nothing, measure = host_measure,
        to_host = identity, memory_metric = "host_allocated_bytes")
    base = base_record(; algorithm = failing.name, backend = failing.backend,
        device = failing.device, precision = Float64, rows = 6, stiefel_columns = 2,
        lift_norm = 1.0, memory_metric = failing.memory_metric, repetition = 1,
        warmup = false, seed = SWEEP_SEED, scale = 0.1,
        source = (sha = "0" ^ 40, dirty = "false", patch_file = "go.patch",
            patch_sha256 = "a" ^ 64))
    record = record_invocation(base, failing, [1.0;;], [1.0;;])

    @test Set(keys(record)) == Set(RETRACTION_HEADER)
    @test record["success"] == "false"
    @test record["error_type"] == "ErrorException"
    @test occursin("synthetic retraction failure", record["error_message"])
    # `NaN` and not a measured zero: the three errors were never computed.
    @test all(isnan(parse(Float64, record[field]))
    for field in ("agreement_error", "forward_error", "manifold_constraint_error"))
    @test record["memory_bytes"] == "0"
end
