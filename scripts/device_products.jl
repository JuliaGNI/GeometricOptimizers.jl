# Every product and sum among this package's own matrix types, and between one of them and a plain
# array, on a device backend, each compared with a host twin built from the same numbers.
#
# `device_products(todevice)` takes the function that moves a host array onto the device and returns
# one row per call: its name, `:pass`, `:wrong` (it ran and disagrees with the host twin, or its
# result left the device) or the first line of the error it raised. Scalar indexing has to be off
# for the rows to mean anything, and the function turns it off.
#
#     using JLArrays; device_products(JLArray)     # what `test/device_products.jl` asserts
#     using Metal;    device_products(MtlArray)    # the hardware check, through Kaimon only
#
# `Float32` throughout, because Metal has no `Float64`. The symplectic SR factor `Sfac` is absent:
# the decomposition that builds one runs on the host only.

using AbstractNeuralNetworks: changebackend
using GeometricOptimizers
using GeometricOptimizers: LowerTriangular, UpperTriangular, StiefelProjection, Manifold,
                           VectorStorageMatrix, AbstractLieAlgHorMatrix, GlobalSection,
                           apply_section, Ω
using GPUArraysCore: AbstractGPUArray, allowscalar
using KernelAbstractions: CPU, get_backend
using LinearAlgebra: Adjoint, Transpose, mul!, qr!
using NeuralNetworkParameters: mapstorage
using Random

const T = Float32

tohost(x::AbstractGPUArray) = Array(x)
tohost(x::Adjoint) = adjoint(tohost(parent(x)))
tohost(x::Transpose) = transpose(tohost(parent(x)))
tohost(x::StiefelProjection) = Array(x.A)
tohost(x::Union{VectorStorageMatrix, AbstractLieAlgHorMatrix}) = mapstorage(tohost, x)
# a point by its constructor: `SymplecticStiefelManifold` has no `rebuild`, so `mapstorage` refuses it
tohost(x::Manifold) = Base.typename(typeof(x)).wrapper(tohost(x.A))
tohost(x) = x

todev(todevice, x::Adjoint) = adjoint(todev(todevice, parent(x)))
function todev(todevice, x::StiefelProjection)
    StiefelProjection(get_backend(todevice(zeros(T, 1))), T, size(x)...)
end
function todev(todevice, x::Union{VectorStorageMatrix, AbstractLieAlgHorMatrix})
    mapstorage(todevice, x)
end
todev(todevice, x::Manifold) = Base.typename(typeof(x)).wrapper(todevice(x.A))
todev(todevice, x::AbstractArray) = todevice(x)

# the entries a host `Matrix` of a result holds, whatever wraps it
dense(x) = Matrix{T}(tohost(x))
dense(x::AbstractVector) = Vector{T}(tohost(x))

function fixtures(rng)
    N, n = 6, 3
    Q = Matrix(qr!(randn(rng, T, N, N)).Q)
    U = rand(rng, SymplecticStiefelManifold{T}, N, 4)
    # a horizontal lift with a nonzero `A` block, drawn in its packed form
    lift = StiefelLieAlgHorMatrix(rand(rng, SkewSymMatrix{T}, n), randn(rng, T, N - n, n), N, n)
    skew = rand(rng, SkewSymMatrix{T}, N)
    grass = rand(rng, GrassmannLieAlgHorMatrix{T}, N, n)
    # the adjoints of the two triangulars are the other triangular, and a symmetric matrix is its own
    ["SkewSym" => skew, "SkewSym'" => skew', "Sym" => rand(rng, SymmetricMatrix{T}, N),
        "Lower" => rand(rng, LowerTriangular{T}, N),
        "Upper" => rand(rng, UpperTriangular{T}, N),
        "StiefelHor" => lift, "StiefelHor'" => lift', "GrassmannHor" => grass,
        "GrassmannHor'" => grass',
        "Y" => StiefelManifold(Q[:, 1:n]), "Y'" => StiefelManifold(Q[:, 1:n])',
        "G" => GrassmannManifold(Q[:, 1:n]), "G'" => GrassmannManifold(Q[:, 1:n])',
        "U" => U, "U'" => U', "E" => StiefelProjection(N, n, T),
        "E'" => StiefelProjection(N, n, T)']
end

function row(name, f, host_args, dev_args, backend)
    status = try
        r = f(dev_args...)
        expected = f(host_args...)
        on_device = r isa Number ||
                    get_backend(r isa Union{Adjoint, Transpose} ? parent(r) : r) == backend
        on_device && dense(r) ≈ dense(expected) ? :pass : :wrong
    catch err
        first(split(sprint(showerror, err), '\n'))
    end
    name => status
end

function device_products(todevice; seed = 1234)
    allowscalar(false)
    rng = Random.Xoshiro(seed)
    backend = get_backend(todevice(zeros(T, 1)))
    rows = Pair{String, Any}[]
    ops = fixtures(rng)
    dev(x) = todev(todevice, x)

    for (name, h) in ops
        m, k = size(h)
        P, M, L = randn(rng, T, k, 2), randn(rng, T, m, k), randn(rng, T, 2, m)
        v, w = randn(rng, T, k), randn(rng, T, m)
        C, Cᵣ = zeros(T, m, 2), zeros(T, 2, k)
        for (label, f, args) in (
            ("$name * B", *, (h, P)), ("B * $name", *, (L, h)), ("$name * v", *, (h, v)),
            ("w' * $name", (a, b) -> b' * a, (h, w)),
            ("transpose(w) * $name", (a, b) -> transpose(b) * a, (h, w)),
            ("$name + B", +, (h, M)), ("B + $name", +, (M, h)),
            ("$name - B", -, (h, M)), ("B - $name", -, (M, h)),
            ("mul!(C, $name, B)", (c, a, b) -> mul!(c, a, b), (C, h, P)),
            ("mul!(C, B, $name)", (c, a, b) -> mul!(c, a, b), (Cᵣ, L, h)),
            ("2f0 * $name", a -> 2.0f0 * a, (h,)), ("$name * 2f0", a -> a * 2.0f0, (h,)),
            ("-$name", -, (h,)))
            push!(rows, row(label, f, copy.(args), map(dev, args), backend))
        end
    end

    for (lname, L) in ops, (rname, R) in ops

        dL, dR = dev(L), dev(R)
        if size(L, 2) == size(R, 1)
            push!(rows, row("$lname * $rname", *, (L, R), (dL, dR), backend))
            C = zeros(T, size(L, 1), size(R, 2))
            push!(rows,
                row("mul!(C, $lname, $rname)", (c, a, b) -> mul!(c, a, b),
                    (copy(C), L, R), (dev(C), dL, dR), backend))
        end
        if size(L) == size(R)
            push!(rows, row("$lname + $rname", +, (L, R), (dL, dR), backend))
            push!(rows, row("$lname - $rname", -, (L, R), (dL, dR), backend))
        end
    end

    # Back to the host. `SymplecticStiefelManifold` has no `rebuild`, so the parameter protocol that
    # `changebackend` walks does not cover it; the adjoints are not parameters.
    for (name, h) in ops
        (h isa Adjoint || h isa SymplecticStiefelManifold) && continue
        push!(rows,
            "changebackend(CPU(), $name)" => try
                r = changebackend(CPU(), dev(h))
                get_backend(r) == CPU() && typeof(r) == typeof(h) && dense(r) ≈ dense(h) ? :pass :
                :wrong
            catch err
                first(split(sprint(showerror, err), '\n'))
            end)
    end

    # The manifold operations `[precision-metal]` found failing, and the retractions they sit on.
    for M in (StiefelManifold, GrassmannManifold)
        Y = M(Matrix(qr!(randn(rng, T, 6, 6)).Q)[:, 1:3])
        Δ = rgrad(Y, randn(rng, T, 6, 3)) / 10
        dY, dΔ = dev(Y), dev(Δ)
        name = string(nameof(M))
        push!(rows, row("Ω($name, Δ)", Ω, (Y, Δ), (dY, dΔ), backend))
        push!(rows, row("$name * $name'", (a) -> a * a', (Y,), (dY,), backend))
        # the section is random, so the retractions are compared through a property rather than
        # against the host: a point on the manifold, on the device
        for (label, f) in (("geodesic", geodesic), ("cayley", cayley))
            push!(rows,
                "$label($name, Δ)" => try
                    Y₂ = f(dY, dΔ)
                    get_backend(Y₂) == backend && GeometricOptimizers.check(Y₂) < 1.0f-4 ? :pass : :wrong
                catch err
                    first(split(sprint(showerror, err), '\n'))
                end)
        end
        push!(rows, "apply_section($name)" => try
            λY = GlobalSection(dY)
            Y₂ = apply_section(λY, dY)
            get_backend(Y₂) == backend ? :pass : :wrong
        catch err
            first(split(sprint(showerror, err), '\n'))
        end)
    end

    rows
end

failures(rows) = filter(r -> last(r) !== :pass, rows)
