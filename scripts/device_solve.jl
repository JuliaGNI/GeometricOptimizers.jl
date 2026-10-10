# `solve!` on a device backend, each run against a host twin built from the same numbers.
#
# `device_solve(todevice, T)` takes the function that moves a host array onto the device and returns
# one row per run: its name, then `:pass`, a symbol naming the first check that failed, or the first
# line of the error the run raised. Scalar indexing has to be off for the rows to mean anything, and
# the function turns it off.
#
#     using JLArrays; device_solve(JLArray, Float64)                # what `test/integration/device_solve.jl` asserts
#     using Metal;    device_solve(MtlArray, Float32; retraction = Cayley(), matched_rng = false)
#
# The runs are `GradientMethod`, `MomentumMethod`, `Adam`, `BFGS` and `DFP`, each with a supplied
# `∇F!`, on three iterates:
#
# - (a) a plain device vector;
# - (b) a bare `StiefelManifold` whose storage is on the device;
# - (c) a `NetworkParameters` with a `StiefelManifold` leaf, a `GrassmannManifold` leaf, one leaf of
#   each `VectorStorageMatrix` type, one leaf of each horizontal lift and one plain matrix, every
#   leaf on the device.
#
# Each run takes `STEPS` iterations, no more and no fewer, and a row checks, in this order: every
# leaf of the result is on the device and of its starting array type (`:off_device`), every value is
# finite (`:nonfinite`), the solve took `STEPS` iterations (`:iterations`), and the result matches
# the host twin (`:mismatch`).
#
# The match is `isapprox(device, host; rtol = √eps(T))` on the flattened result: the two runs take
# the same steps, and a device reduction adds in a different order, so they agree to round-off and
# not to the bit. A manifold run draws its global section at random. On a `JLArray`, `randn!` draws
# on the host's global stream, so `Random.seed!` before each twin gives both the same section. A
# device with a generator of its own draws another one; `matched_rng = false` says so. A section is a
# frame of the complement, and the gradient, momentum and quasi-Newton steps do not depend on which
# frame up to round-off, but `Adam`'s componentwise moments do. So with `matched_rng = false` the
# `Adam` rows of (b) and (c) are compared by a property instead: the objective decreases, and the
# final objective is within `adam_rtol` of the host twin's. `seed_device!(seed)` seeds the device's
# own generator before the device twin runs, so that its section, and the row, is the same on every
# run: `seed_device! = s -> Random.seed!(Metal.default_rng(), s)` on Metal.

using GeometricOptimizers
using GeometricOptimizers: StrictlyLowerTriangular, StrictlyUpperTriangular, Manifold,
                           iteration_number
using GPUArraysCore: allowscalar
using KernelAbstractions: get_backend
using LinearAlgebra: Diagonal, qr!
using NeuralNetworkParameters: NetworkParameters, flatten, foldstorage, mapstorage,
                               unflatten
using Random

const STEPS = 10

# the dense numbers of a result, on the host, in the order `flatten` writes them
hostflat(x::AbstractVector) = Vector(x)
hostflat(x::Manifold) = vec(Matrix(parent(x)))
hostflat(x::NetworkParameters) = first(flatten(mapstorage(Array, x)))

# every storage array of a result, for the backend and the array-type checks
storages(x::AbstractVector) = (x,)
storages(x::Manifold) = (parent(x),)
storages(x::NetworkParameters) = foldstorage((acc, s) -> (acc..., s), (), x)

# (a): a least-squares objective. `M` and `b` are on the iterate's backend.
function vector_problem(rng, ::Type{T}, todevice) where {T}
    m, k = 8, 5
    M, b = randn(rng, T, m, k), randn(rng, T, m)
    x = randn(rng, T, k)
    make(dev) =
        let M = dev(M), b = dev(b)
            F(x) = sum(abs2, M * x - b) / 2
            ∇F!(g, x) = (g .= M' * (M * x - b); g)
            F, ∇F!
        end
    (x, identity, make(identity)), (todevice(x), todevice, make(todevice))
end

# (b): the Brockett function `-tr(YᵀAYD)/2` on `St(N, n)`. `∇F!` gets the dense storage flattened,
# column-major, and writes the Euclidean gradient `-AYD` there.
function stiefel_problem(rng, ::Type{T}, todevice) where {T}
    N, n = 6, 3
    Q = Matrix(qr!(randn(rng, T, N, N)).Q)
    A = Q * Diagonal(T.(1:N)) * Q'
    A = (A + A') / 2
    D = Matrix(Diagonal(T[1, 2, 3]))
    Y = Matrix(qr!(randn(rng, T, N, n)).Q)[:, 1:n]
    make(dev) =
        let A = dev(A), D = dev(D)
            F(Y) = -sum((A * parent(Y) * D) .* parent(Y)) / 2
            ∇F!(g, v) = (g .= vec(-(A * reshape(v, N, n) * D)); g)
            F, ∇F!
        end
    (StiefelManifold(copy(Y)), identity, make(identity)),
    (StiefelManifold(todevice(Y)), todevice, make(todevice))
end

# (c): a weighted distance to a target, taken on the storage of every leaf, so `∇F!` is
# `2w(v - c)` on the flat vector. On the `StiefelManifold` leaf that is a Procrustes problem.
function parameters_problem(rng, ::Type{T}, todevice) where {T}
    N, n = 6, 3
    host = NetworkParameters((
        Y = StiefelManifold(Matrix(qr!(randn(rng, T, N, n)).Q)[:, 1:n]),
        G = GrassmannManifold(Matrix(qr!(randn(rng, T, N, n)).Q)[:, 1:n]),
        S = rand(rng, SymmetricMatrix{T}, n),
        K = rand(rng, SkewSymMatrix{T}, n),
        L = rand(rng, StrictlyLowerTriangular{T}, n),
        U = rand(rng, StrictlyUpperTriangular{T}, n),
        H = rand(rng, StiefelLieAlgHorMatrix{T}, N, n),
        R = rand(rng, GrassmannLieAlgHorMatrix{T}, N, n),
        W = randn(rng, T, n, 2)))
    v, layout = flatten(host)
    c, w = randn(rng, T, length(v)), T(1) .+ rand(rng, T, length(v))
    make(dev) =
        let c = dev(c), w = dev(w)
            targets = unflatten(layout, c)
            weights = unflatten(layout, w)
            term(acc, x, c, w) = acc + sum(w .* (x .- c) .^ 2)
            F(ps) = foldstorage(term, zero(T), ps, targets, weights)
            ∇F!(g, v) = (g .= 2 .* w .* (v .- c); g)
            F, ∇F!
        end
    (host, identity, make(identity)), (mapstorage(todevice, host), todevice, make(todevice))
end

# A fixed step for the first-order methods. A searching line search decides by comparing objective
# values, and a device sum that differs from the host's in the last bit can flip one such decision:
# measured on (a) with `GradientMethod` and the default `Backtracking`, the twins agree to 1e-15 for
# four steps and differ by 1e-2 after the fifth. `BFGS` and `DFP` keep their default line search.
step_rule(::Type{T}, ::Union{GradientMethod, MomentumMethod, Adam}) where {T} = T(1) / 50
function step_rule(::Type{T}, method::Union{BFGS, DFP}) where {T}
    GeometricOptimizers.default_linesearch(T, method)
end

function run_solve(x, F, ∇F!, method, retraction, seed, ::Type{T};
        seed_device! = nothing) where {T}
    Random.seed!(seed)
    seed_device! === nothing || seed_device!(seed)
    optimizer = Optimizer(x, F; (∇F!) = ∇F!, algorithm = method, retraction = retraction,
        linesearch = step_rule(T, method), min_iterations = STEPS, max_iterations = STEPS)
    state = OptimizerState(method, x)
    solve!(x, state, optimizer)
    x, iteration_number(state)
end

function solve_row(name, problem, method, retraction, seed, backend; property = false,
        adam_rtol = 0, seed_device! = nothing)
    status = try
        (xh, _, (Fh, ∇Fh!)), (xd, _, (Fd, ∇Fd!)) = problem
        types = map(typeof, storages(xd))
        f₀ = Fd(xd)
        T = typeof(f₀)
        run_solve(xh, Fh, ∇Fh!, method, retraction, seed, T)
        _, iterations = run_solve(
            xd, Fd, ∇Fd!, method, retraction, seed, T; seed_device! = seed_device!)
        v = hostflat(xd)
        if !all(s -> get_backend(s) == backend, storages(xd)) ||
           map(typeof, storages(xd)) != types
            :off_device
        elseif !all(isfinite, v)
            :nonfinite
        elseif iterations != STEPS
            :iterations
        elseif property
            Fd(xd) < f₀ && isapprox(Fd(xd), Fh(xh); rtol = adam_rtol) ? :pass : :mismatch
        else
            isapprox(v, hostflat(xh); rtol = sqrt(eps(T))) ? :pass : :mismatch
        end
    catch err
        first(split(sprint(showerror, err), '\n'))
    end
    name => status
end

# How near the final objective of an `Adam` run of (b) or (c) has to be to the host twin's when the
# two draw different global sections, relative to the host twin's. Measured in `Float32`, ten steps:
# over 40 seeds, two host runs that differ only in the section reach final objectives up to 3.2 %
# apart on (b) and 1.8 % on (c), under `Geodesic()` and `Cayley()` alike, and on Metal (an M4 Max)
# the device run of (b) is up to 1.5 % from its host twin over 6 seeds. The objective falls by about
# 15 % over the ten steps, so 5 % separates a run that steps from one that does not.
const UNMATCHED_ADAM_RTOL = 0.05

function device_solve(todevice, ::Type{T}; retraction = Geodesic(), matched_rng = true,
        adam_rtol = UNMATCHED_ADAM_RTOL, seed = 1234, seed_device! = nothing) where {T}
    allowscalar(false)
    backend = get_backend(todevice(zeros(T, 1)))
    rows = Pair{String, Any}[]
    cases = (("(a) vector", vector_problem), ("(b) Stiefel", stiefel_problem),
        ("(c) parameters", parameters_problem))
    for (case, make_problem) in cases
        for method in (GradientMethod(), MomentumMethod(), Adam(), BFGS(), DFP())
            problem = make_problem(Random.Xoshiro(seed), T, todevice)
            property = !matched_rng && method isa Adam && case != "(a) vector"
            push!(rows,
                solve_row("$case, $(nameof(typeof(method)))", problem, method,
                    retraction, seed, backend; property = property, adam_rtol = adam_rtol,
                    seed_device! = seed_device!))
        end
    end
    rows
end

failures(rows) = filter(r -> last(r) !== :pass, rows)
