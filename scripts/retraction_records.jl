#!/usr/bin/env julia

# The retraction benchmark of `retraction_accuracy.jl` as machine-readable records: one CSV row per
# invocation rather than a `minimum` over repetitions, a CUDA path beside the host one, host and
# device allocation in separate metrics, and a row for an invocation that threw. The lifts are the
# seeded sweep of `retraction_sweep.jl`, which `retraction_accuracy.jl` draws from too.
#
# Run it with this repository as the active project; CUDA mode also needs `CUDA` on the load path.
# The four `--source-*` values identify the checkout being measured. The caller archives that
# identity beside the CSV and supplies it, and every row carries it unchanged:
#
#     julia --project=. scripts/retraction_records.jl --output retraction-runs.csv \
#         --source-sha $(git rev-parse HEAD) --source-dirty false \
#         --source-patch-file go.patch --source-patch-sha256 $(shasum -a 256 go.patch | cut -d" " -f1)
#
# `RETRACTION_HEADER` is the column order of schema version `RETRACTION_SCHEMA_VERSION`. A consumer
# validates a file against its own copy of the header, so a change to the columns bumps the version.

using GeometricOptimizers
using GeometricOptimizers: AugmentedPade, NativePade, ScaledSquaring, SkewSymMatrix,
                           StiefelLieAlgHorMatrix, geodesic
using LinearAlgebra: BLAS, I, norm

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

# A record is a `NamedTuple` with these fields, in header order.
const RECORD_FIELDS = Tuple(Symbol.(RETRACTION_HEADER))

const USAGE = """usage: retraction_records.jl [options]

Write schema-$(RETRACTION_SCHEMA_VERSION) CSV records for the retraction benchmark.

  --output FILE            CSV destination (default: retraction-records.csv)
  --backend cpu|cuda       execution backend (default: cpu)
  --precision TYPE         Float32 or Float64 (default: Float64)
  --rows N                 full square lift size (default: 20)
  --columns N              Stiefel columns (default: 3)
  --scales LIST            comma-separated subset of $(join(SCALES, ',')) (default: all)
  --repetitions N          measured calls after one recorded warm-up (default: 20)
  --seed N                 deterministic lift seed (default: $(SWEEP_SEED))

  --source-sha SHA         40-hex commit the rows are stamped with   (required)
  --source-dirty BOOL      whether that checkout was dirty           (required)
  --source-patch-file NAME patch file name, relative to the CSV      (required)
  --source-patch-sha256 H  64-hex SHA-256 of that patch              (required)

CPU mode records ScaledSquaring, NativePade and AugmentedPade on the host. CUDA mode records
ScaledSquaring and NativePade on the GPU and AugmentedPade on the host -- it is dense LAPACK --
and transfers the GPU results before comparing them with the host AugmentedPade result.

A lift is the one the full sweep draws for its scale, whichever subset --scales selects. The host
runs with one BLAS thread, as for the tables of retraction_accuracy.jl.
"""

const DEFAULTS = (
    output = "retraction-records.csv", backend = "cpu", precision = "Float64",
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

function is_hex(value::AbstractString, digits::Integer)
    ncodeunits(value) == digits && all(c -> isdigit(c) || c in 'a':'f', value)
end

function parse_options(args)
    parsed = parse_arguments(args, DEFAULTS; usage = USAGE)
    parsed === nothing && return nothing

    parsed.backend in ("cpu", "cuda") ||
        throw(ArgumentError("--backend must be cpu or cuda"))
    parsed.precision in ("Float32", "Float64") ||
        throw(ArgumentError("--precision must be Float32 or Float64"))
    parsed.rows > 0 || throw(ArgumentError("--rows must be positive"))
    0 < parsed.columns <= parsed.rows ||
        throw(ArgumentError("--columns must be positive and no greater than --rows"))
    parsed.repetitions > 0 || throw(ArgumentError("--repetitions must be positive"))
    parsed.seed >= 0 || throw(ArgumentError("--seed must be nonnegative"))
    scales = parse.(Float64, split(parsed.scales, ','; keepempty = false))
    !isempty(scales) && scales ⊆ SCALES ||
        throw(ArgumentError("--scales must be a nonempty subset of $(join(SCALES, ','))"))

    is_hex(parsed.source_sha, 40) ||
        throw(ArgumentError("--source-sha must be a 40-character hexadecimal commit"))
    is_hex(parsed.source_patch_sha256, 64) ||
        throw(ArgumentError("--source-patch-sha256 must be a 64-character hexadecimal digest"))
    parsed.source_dirty in ("true", "false") ||
        throw(ArgumentError("--source-dirty must be true or false"))
    isempty(parsed.source_patch_file) &&
        throw(ArgumentError("--source-patch-file is required"))

    merge(parsed,
        (backend = Symbol(parsed.backend), scales = scales,
            precision = parsed.precision == "Float32" ? Float32 : Float64))
end

# ------------------------------------------------------------------------------ CSV ---

"""Quote `value` for CSV output, but only where a bare field would be ambiguous."""
function csv_field(value)
    text = string(value)
    occursin(r"[\",\r\n]", text) ? "\"" * replace(text, '"' => "\"\"") * "\"" : text
end

csv_line(record) = join((csv_field(value) for value in record), ',')

# --------------------------------------------------------------------------- the paths ---

"""One timed, measured call on the host: `(result, seconds, bytes)`, from a single execution."""
function host_measure(operation)
    stats = @timed operation()
    (stats.value, stats.time, stats.bytes)
end

# `AugmentedPade` uses dense LAPACK, so it runs on the host in CUDA mode too.
function host_path(name, algorithm, lift)
    (name = name, backend = "CPU", device = Sys.CPU_NAME,
        operation = () -> geodesic(lift, algorithm), measure = host_measure,
        memory_metric = "host_allocated_bytes")
end

# `CUDA.@allocated` needs `CUDA` when this file is parsed, so the device measurement is defined
# after the import. `main` enters the benchmark through `invokelatest`, in a world that has both.
function import_cuda()
    @eval import CUDA
    @eval function device_measure(operation)
        result = nothing
        CUDA.synchronize()
        started = time_ns()
        bytes = CUDA.@allocated(result = operation())
        CUDA.synchronize()
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
    A = SkewSymMatrix(cuda.CuArray(parent(host.A)), host.n)
    StiefelLieAlgHorMatrix(A, cuda.CuArray(host.B), host.N, host.n)
end

function algorithm_paths(options, host_lift, cuda)
    if options.backend == :cpu
        return [host_path("ScaledSquaring", ScaledSquaring(), host_lift),
            host_path("NativePade", NativePade(), host_lift),
            host_path("AugmentedPade", AugmentedPade(), host_lift)]
    end

    gpu_lift = cuda_lift(host_lift, cuda)
    device = string(cuda.name(cuda.device()))
    device_path(name, algorithm) = (name = name, backend = "CUDA", device = device,
        operation = () -> geodesic(gpu_lift, algorithm), measure = device_measure,
        memory_metric = "device_allocated_bytes")
    [device_path("ScaledSquaring", ScaledSquaring()),
        device_path("NativePade", NativePade()),
        host_path("AugmentedPade", AugmentedPade(), host_lift)]
end

# --------------------------------------------------------------------------- one row ---

sanitize_error(error) = replace(sprint(showerror, error), r"\s+" => " ")

"""The fields of a record that do not depend on the outcome of the invocation."""
function base_record(options, path, scale, lift_norm, repetition)
    (schema_version = RETRACTION_SCHEMA_VERSION,
        algorithm = path.name, backend = path.backend,
        device = path.device, precision = options.precision, matrix_rows = options.rows,
        matrix_columns = options.rows, stiefel_columns = options.columns,
        lift_norm = Float64(lift_norm), memory_metric = path.memory_metric,
        repetition = repetition, warmup = repetition == 0, seed = options.seed, scale = scale,
        go_sha = options.source_sha, go_dirty = options.source_dirty,
        go_patch_file = options.source_patch_file, go_patch_sha256 = options.source_patch_sha256)
end

"""
    record_invocation(base, path, agreement_reference, forward_reference)

One call to `path.operation` as a record: its runtime and allocation, measured by `path.measure` in
the same execution, and the three errors of its result. If the call throws, the record is a failure
row with `NaN` errors. An error after the call, in the transfer or the comparison, is not the
algorithm's and propagates.
"""
function record_invocation(base, path, agreement_reference, forward_reference)
    started = time_ns()
    result, runtime_seconds, memory_bytes = try
        path.measure(path.operation)
    catch error
        return NamedTuple{RECORD_FIELDS}(merge(base,
            (agreement_error = NaN, forward_error = NaN, manifold_constraint_error = NaN,
                runtime_seconds = (time_ns() - started) / 1.0e9, memory_bytes = 0,
                success = false, error_type = string(typeof(error)),
                error_message = sanitize_error(error))))
    end
    output = Matrix{Float64}(Array(parent(result)))
    relative_error(reference) = norm(output - reference) / norm(reference)
    NamedTuple{RECORD_FIELDS}(merge(base,
        (agreement_error = relative_error(agreement_reference),
            forward_error = relative_error(forward_reference),
            manifold_constraint_error = norm(output' * output - I), runtime_seconds,
            memory_bytes, success = true, error_type = "", error_message = "")))
end

# ------------------------------------------------------------------------ the benchmark ---

"""
    run_benchmark(options)

Write a record per invocation to `options.output`, each row as soon as it is measured, and return
the records.
"""
function run_benchmark(options)
    cuda = options.backend == :cuda ? loaded_cuda() : nothing
    lifts = sweep(options.precision, options.rows, options.columns; seed = options.seed)
    records = NamedTuple{RECORD_FIELDS}[]
    mkpath(dirname(abspath(options.output)))
    open(options.output, "w") do io
        println(io, join(RETRACTION_HEADER, ','))
        for scale in options.scales
            host_lift = lifts[findfirst(==(scale), SCALES)]
            dense_lift = Matrix(host_lift)
            lift_norm = norm(dense_lift)
            agreement_reference = Matrix{Float64}(Matrix(geodesic(host_lift, AugmentedPade())))
            forward_reference = exp(Matrix{Float64}(dense_lift))

            for path in algorithm_paths(options, host_lift, cuda),
                repetition in 0:(options.repetitions)

                base = base_record(options, path, scale, lift_norm, repetition)
                record = record_invocation(base, path, agreement_reference, forward_reference)
                println(io, csv_line(record))
                flush(io)
                push!(records, record)
            end
        end
    end
    records
end

function main(args = ARGS)
    options = parse_options(args)
    options === nothing && return 0
    options.backend == :cuda && import_cuda()
    BLAS.set_num_threads(1)
    records = Base.invokelatest(run_benchmark, options)
    failures = count(record -> !record.success, records)
    warmups = count(record -> record.warmup, records)
    println("wrote $(length(records)) retraction records ($(warmups) warm-up, " *
            "$(length(records) - warmups) steady-state, $(failures) failure) to " *
            "$(abspath(options.output))")
    failures == 0 ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
