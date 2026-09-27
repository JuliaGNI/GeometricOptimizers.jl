#!/usr/bin/env julia

# The retraction benchmark of `retraction_accuracy.jl`, as machine-readable records instead of
# tables.
#
# `retraction_accuracy.jl` answers "how do these algorithms compare" for a reader, and prints a
# `minimum` over repetitions. An external experiment needs a different shape of the same
# measurement: one row per *invocation* rather than an aggregate, a CUDA path beside the host one,
# host and device allocation kept apart, and a row for an invocation that threw rather than a gap in
# a table. All of that is about the recording; the algorithms, the reference and the seeded lift
# sweep are this package's and are shared with that script through `retraction_sweep.jl`.
#
# Run it with this repository as the active project; CUDA mode additionally needs `CUDA` on the load
# path. The four `--source-*` values stamp every row with the identity of the checkout being
# measured, and are the caller's to supply:
#
#     julia --project=. scripts/retraction_records.jl --output retraction-runs.csv \
#         --source-sha $(git rev-parse HEAD) --source-dirty false \
#         --source-patch-file go.patch --source-patch-sha256 $(shasum -a 256 go.patch | cut -d" " -f1)
#
# ## Provenance is the caller's
#
# The four `--source-*` options are written into every row and are not captured here. A benchmark of
# *this* checkout run from an experiment harness has to be stamped with the identity that harness
# archives -- the commit, whether the tree was dirty, and the patch file it keeps beside the CSV --
# and a second capture here would be a second implementation of that identity, which is the one kind
# of duplication that undermines the thing it exists to provide. The validator on the other side
# checks the four against the patch, so they have to come from whoever wrote it.
#
# ## The schema
#
# `RETRACTION_HEADER` below is the column order, and schema version 1. A consumer validates against
# its own copy of that list and rejects a header it does not recognise; the version is what a change
# here has to bump.

using GeometricOptimizers
using GeometricOptimizers: AugmentedPade, NativePade, ScaledSquaring, SkewSymMatrix,
                           StiefelLieAlgHorMatrix, geodesic
using LinearAlgebra: I, norm

include(joinpath(@__DIR__, "retraction_sweep.jl"))

const RETRACTION_SCHEMA_VERSION = 1

const RETRACTION_HEADER = [
    "schema_version", "algorithm", "backend", "device", "precision",
    "matrix_rows", "matrix_columns", "stiefel_columns", "lift_norm",
    "agreement_error", "forward_error", "manifold_constraint_error",
    "runtime_seconds", "memory_bytes", "memory_metric",
    "repetition", "warmup", "success", "error_type", "error_message",
    "seed", "scale", "go_sha", "go_dirty", "go_patch_file", "go_patch_sha256"
]

const USAGE = """usage: retraction_records.jl [options]

Write schema-$(RETRACTION_SCHEMA_VERSION) CSV records for the retraction benchmark.

  --output FILE            CSV destination (default: retraction-records.csv)
  --backend cpu|cuda       execution backend (default: cpu)
  --precision TYPE         Float32 or Float64 (default: Float64)
  --rows N                 full square lift size (default: 20)
  --columns N              Stiefel columns (default: 3)
  --scales LIST            comma-separated lift scales (default: this package's sweep)
  --repetitions N          measured calls after one recorded warm-up (default: 20)
  --seed N                 deterministic lift seed (default: $(SWEEP_SEED))

  --source-sha SHA         40-hex commit the rows are stamped with   (required)
  --source-dirty BOOL      whether that checkout was dirty           (required)
  --source-patch-file NAME patch file name, relative to the CSV      (required)
  --source-patch-sha256 H  64-hex SHA-256 of that patch              (required)

CPU mode records ScaledSquaring, NativePade and AugmentedPade on the host. CUDA mode records
ScaledSquaring and NativePade on the GPU and AugmentedPade on the host -- it is dense LAPACK --
and transfers the GPU results before comparing them with the host AugmentedPade result.

The four --source-* values are written into every row unchanged; see the header of this file for
why they are arguments rather than something this script captures.
"""

const DEFAULTS = (output = "retraction-records.csv", backend = "cpu", precision = "Float64",
    rows = 20, columns = 3, scales = join(SCALES, ','), repetitions = 20, seed = SWEEP_SEED,
    source_sha = "", source_dirty = "", source_patch_file = "", source_patch_sha256 = "")

# ------------------------------------------------------------------ the command line ---

"""
    parse_arguments(args, defaults; usage)

Parse `args` against `defaults` and return a `NamedTuple` with the same keys. The option spelling of
a `:snake_case` key is the same word in `--kebab-case`, and a value is parsed to the type of its
default. `--help` prints `usage` and returns `nothing`.
"""
function parse_arguments(args, defaults::NamedTuple; usage::AbstractString)
    options = Dict(String(key) => key for key in keys(defaults))
    values = Dict{Symbol, Any}(key => getfield(defaults, key) for key in keys(defaults))
    index = 1
    while index <= length(args)
        argument = args[index]
        argument in ("-h", "--help") && (println(usage); return nothing)
        startswith(argument, "--") || throw(ArgumentError("unexpected argument: $argument"))
        name = replace(argument[3:end], '-' => '_')
        haskey(options, name) || throw(ArgumentError("unknown argument: $argument"))
        index == length(args) && throw(ArgumentError("missing value for $argument"))
        key = options[name]
        values[key] = parse_argument(getfield(defaults, key), args[index + 1], argument)
        index += 2
    end
    NamedTuple{keys(defaults)}(Tuple(values[key] for key in keys(defaults)))
end

parse_argument(::AbstractString, value, _) = value
function parse_argument(default::Number, value, option)
    parsed = tryparse(typeof(default), value)
    parsed === nothing &&
        throw(ArgumentError("$option requires a $(typeof(default)), got $value"))
    parsed
end

is_hex(value::AbstractString, digits::Integer) =
    ncodeunits(value) == digits && all(c -> isdigit(c) || c in 'a':'f', value)

function parse_options(args)
    parsed = parse_arguments(args, DEFAULTS; usage = USAGE)
    parsed === nothing && return nothing

    parsed.backend in ("cpu", "cuda") || throw(ArgumentError("--backend must be cpu or cuda"))
    parsed.precision in ("Float32", "Float64") ||
        throw(ArgumentError("--precision must be Float32 or Float64"))
    parsed.rows > 0 || throw(ArgumentError("--rows must be positive"))
    0 < parsed.columns <= parsed.rows ||
        throw(ArgumentError("--columns must be positive and no greater than --rows"))
    parsed.repetitions > 0 || throw(ArgumentError("--repetitions must be positive"))
    parsed.seed >= 0 || throw(ArgumentError("--seed must be nonnegative"))
    scales = parse.(Float64, split(parsed.scales, ','; keepempty = false))
    !isempty(scales) && all(scale -> isfinite(scale) && scale >= 0, scales) ||
        throw(ArgumentError("--scales must contain finite nonnegative values"))

    is_hex(parsed.source_sha, 40) ||
        throw(ArgumentError("--source-sha must be a 40-character hexadecimal commit"))
    is_hex(parsed.source_patch_sha256, 64) ||
        throw(ArgumentError("--source-patch-sha256 must be a 64-character hexadecimal digest"))
    parsed.source_dirty in ("true", "false") ||
        throw(ArgumentError("--source-dirty must be true or false"))
    isempty(parsed.source_patch_file) &&
        throw(ArgumentError("--source-patch-file is required"))

    merge(parsed, (backend = Symbol(parsed.backend), scales = scales,
        precision = parsed.precision == "Float32" ? Float32 : Float64))
end

# ------------------------------------------------------------------------------ CSV ---

"""Quote `value` for CSV output, but only where a bare field would be ambiguous."""
function csv_field(value)
    text = string(value)
    occursin(r"[\",\r\n]", text) ? "\"" * replace(text, '"' => "\"\"") * "\"" : text
end

"""Write `records` -- dictionaries keyed by `RETRACTION_HEADER` -- to `path`, in header order."""
function write_records(path::AbstractString, records)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, join(RETRACTION_HEADER, ','))
        for record in records
            Set(keys(record)) == Set(RETRACTION_HEADER) ||
                throw(ArgumentError("record fields do not match schema version $(RETRACTION_SCHEMA_VERSION)"))
            println(io, join((csv_field(record[field]) for field in RETRACTION_HEADER), ','))
        end
    end
    path
end

# ------------------------------------------------------------------------- the CUDA path ---

# Julia 1.12 forbids calling methods that were added after the running frame started, so the import
# has to finish before the benchmark frame is entered through `invokelatest`. Loading CUDA also
# brings in package extensions, whose methods are subject to the same rule.
function import_cuda()
    @eval import CUDA
    # One call, timed and measured together, with the device synchronized at both ends of it. See
    # `record_invocation` for why that matters and what it replaces.
    @eval function device_timed_allocated(operation, synchronize)
        result = nothing
        synchronize()
        started = time_ns()
        bytes = CUDA.@allocated(result = operation())
        synchronize()
        (result, (time_ns() - started) / 1.0e9, bytes)
    end
    nothing
end

function loaded_cuda()
    cuda = getglobal(@__MODULE__, :CUDA)
    cuda.functional(true) || throw(ArgumentError("CUDA is not functional"))
    cuda
end

function cuda_lift(host, cuda)
    A = SkewSymMatrix(cuda.CuArray(copy(parent(host.A))), host.n)
    B = cuda.CuArray(copy(host.B))
    StiefelLieAlgHorMatrix(A, B, host.N, host.n)
end

# --------------------------------------------------------------------------- one row ---

sanitize_error(error) = replace(sprint(showerror, error), r"\s+" => " ")
float_string(value) = string(Float64(value))

function base_record(; algorithm, backend, device, precision, rows, stiefel_columns,
        lift_norm, memory_metric, repetition, warmup, seed, scale, source)
    Dict{String, Any}(
        "schema_version" => string(RETRACTION_SCHEMA_VERSION),
        "algorithm" => algorithm,
        "backend" => backend,
        "device" => string(device),
        "precision" => string(precision),
        "matrix_rows" => string(rows),
        "matrix_columns" => string(rows),
        "stiefel_columns" => string(stiefel_columns),
        "lift_norm" => float_string(lift_norm),
        "memory_metric" => memory_metric,
        "repetition" => string(repetition),
        "warmup" => string(warmup),
        "seed" => string(seed),
        "scale" => float_string(scale),
        "go_sha" => source.sha,
        "go_dirty" => source.dirty,
        "go_patch_file" => source.patch_file,
        "go_patch_sha256" => source.patch_sha256
    )
end

"""
    record_invocation(base, path, agreement_reference, forward_reference)

One benchmark invocation, as a record: the timing, the allocation and the three errors of a single
call to `path.operation`, or an explicit failure row if it threw.

# One call, not two

The runtime and the memory figure describe **the same invocation**. The obvious way to write this
measures them separately -- a timed call, and then a second call inside `@allocated` -- which
doubles the benchmark's cost and, on a device backend, synchronizes the two separately, so the
timing and the memory figure end up describing different invocations of the same operation.
`path.measure` wraps `@allocated` around the *timed* region instead and returns both from one
execution, with the device synchronized at both ends of it.
"""
function record_invocation(base, path, agreement_reference, forward_reference)
    started = time_ns()
    try
        result, elapsed, memory = path.measure(path.operation)
        path.synchronize()
        output = Matrix{Float64}(path.to_host(result))
        merge(base,
            Dict{String, Any}(
                "agreement_error" => float_string(
                    norm(output - agreement_reference) / norm(agreement_reference)),
                "forward_error" => float_string(
                    norm(output - forward_reference) / norm(forward_reference)),
                "manifold_constraint_error" => float_string(norm(output' * output - I)),
                "runtime_seconds" => float_string(elapsed),
                "memory_bytes" => string(memory),
                "success" => "true",
                "error_type" => "",
                "error_message" => ""
            ))
    catch error
        try
            path.synchronize()
        catch
        end
        merge(base,
            Dict{String, Any}(
                "agreement_error" => "NaN",
                "forward_error" => "NaN",
                "manifold_constraint_error" => "NaN",
                "runtime_seconds" => float_string(max(0.0, (time_ns() - started) / 1.0e9)),
                "memory_bytes" => "0",
                "success" => "false",
                "error_type" => string(typeof(error)),
                "error_message" => sanitize_error(error)
            ))
    end
end

# --------------------------------------------------------------------------- the paths ---

"""One timed, measured call on the host: `(result, seconds, bytes)`, from a single execution."""
function host_measure(operation)
    result = nothing
    started = time_ns()
    bytes = @allocated(result = operation())
    (result, (time_ns() - started) / 1.0e9, bytes)
end

# The host path: allocation is Julia's own, nothing needs synchronizing, and every algorithm runs on
# it. `AugmentedPade` uses dense LAPACK and stays here even in CUDA mode.
function host_path(name, algorithm, lift)
    (name = name, backend = "CPU", device = Sys.CPU_NAME,
        operation = () -> geodesic(lift, algorithm),
        synchronize = () -> nothing,
        measure = host_measure,
        to_host = result -> Matrix(result), memory_metric = "host_allocated_bytes")
end

function algorithm_paths(options, host_lift, cuda)
    if options.backend == :cpu
        return [host_path("ScaledSquaring", ScaledSquaring(), host_lift),
            host_path("NativePade", NativePade(), host_lift),
            host_path("AugmentedPade", AugmentedPade(), host_lift)]
    end

    gpu_lift = cuda_lift(host_lift, cuda)
    device_path(name, algorithm) = (name = name, backend = "CUDA",
        device = string(cuda.name(cuda.device())),
        operation = () -> geodesic(gpu_lift, algorithm),
        synchronize = () -> cuda.synchronize(),
        measure = operation -> Base.invokelatest(device_timed_allocated, operation,
            () -> cuda.synchronize()),
        to_host = result -> Array(parent(result)),
        memory_metric = "device_allocated_bytes")
    [device_path("ScaledSquaring", ScaledSquaring()),
        device_path("NativePade", NativePade()),
        host_path("AugmentedPade", AugmentedPade(), host_lift)]
end

# ------------------------------------------------------------------------ the benchmark ---

function run_benchmark(options)
    source = (sha = options.source_sha, dirty = options.source_dirty,
        patch_file = options.source_patch_file, patch_sha256 = options.source_patch_sha256)
    cuda = options.backend == :cuda ? loaded_cuda() : nothing
    records = Dict{String, Any}[]

    lifts = sweep(options.precision, options.rows, options.columns;
        scales = options.scales, seed = options.seed)
    for (scale, host_lift) in zip(options.scales, lifts)
        dense_lift = Matrix(host_lift)
        lift_norm = norm(dense_lift)
        agreement_reference = Matrix{Float64}(Matrix(geodesic(host_lift, AugmentedPade())))
        forward_reference = exp(Matrix{Float64}(dense_lift))

        for path in algorithm_paths(options, host_lift, cuda)
            for repetition in 0:(options.repetitions)
                base = base_record(; algorithm = path.name, backend = path.backend,
                    device = path.device, precision = options.precision,
                    rows = options.rows, stiefel_columns = options.columns, lift_norm,
                    memory_metric = path.memory_metric, repetition,
                    warmup = repetition == 0, seed = options.seed, scale, source)
                push!(records,
                    record_invocation(base, path, agreement_reference, forward_reference))
            end
        end
    end
    write_records(options.output, records)
    records
end

function main(args = ARGS)
    options = parse_options(args)
    options === nothing && return 0
    options.backend == :cuda && import_cuda()
    records = Base.invokelatest(run_benchmark, options)
    failures = count(record -> record["success"] == "false", records)
    warmups = count(record -> record["warmup"] == "true", records)
    println("wrote $(length(records)) retraction records ($(warmups) warm-up, " *
            "$(length(records) - warmups) steady-state, $(failures) failure) to " *
            "$(abspath(options.output))")
    failures == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(main())
    catch error
        println(stderr, "retraction record benchmark failed: ", sprint(showerror, error))
        exit(1)
    end
end
