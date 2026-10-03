# The storage gradient of this package's structured leaves: `NeuralNetworkParameters.storage_gradient`.
#
# AD differentiates a leaf's dense interface, and its cotangent `G` is paired with that interface. A
# parameter gradient is `∂L/∂S`, with respect to the storage. An off-diagonal entry of a
# `SymmetricMatrix` appears twice in the interface, so its storage gradient is `G_ij + G_ji`, and a
# diagonal entry `G_ii`; a `SkewSymMatrix` stores `S_ij` at `(i, j)` and `-S_ij` at `(j, i)`, so its
# storage gradient is `G_ij - G_ji`. Read as storage without this, `G` is half the gradient off the
# diagonal: the ratio of a central difference to the Zygote cotangent was `[1, 2, 1, 2, 2, 1]` for a
# 3 × 3 symmetric leaf and `[2, 2, 2]` for a skew one.
#
# The reference is a central difference on the flat storage, in `Float64`, so it shares no formula
# with the conversion. One loss uses every leaf once; the other two use every leaf twice, or mix in
# a loss that reads the leaf as a dense array, because that is where Zygote adds two cotangents as
# dense matrices and where `ProjectTo` converts the sum back to the leaf's type.

using GeometricOptimizers
using GeometricOptimizers: StrictlyLowerTriangular, StrictlyUpperTriangular
using GeometricOptimizers.ChainRulesCore: Tangent
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: qr
using NeuralNetworkParameters: NetworkParameters, flatten, mapstorage, storage_gradient,
                               unflatten
using Random
using Test
using Zygote: Zygote

allowscalar(false)

const N, n = 5, 3

# Every number is drawn in `Float64` and rounded to `T`, so that the `Float64` reference can be taken
# at exactly the numbers the `T` run sees: `structured_set(T)` and `losses(T)` hold them in `T`, and
# `structured_set(T, Float64)` and `losses(T, Float64)` hold the same numbers in `Float64`.
function structured_set(::Type{T}, ::Type{S} = T) where {T, S}
    rng = Random.Xoshiro(7)
    point() = Matrix(qr(randn(rng, N, n)).Q)[:, 1:n]
    set = NetworkParameters((
        S = rand(rng, SymmetricMatrix{Float64}, n), K = rand(rng, SkewSymMatrix{Float64}, n),
        Lo = rand(rng, StrictlyLowerTriangular{Float64}, n),
        Up = rand(rng, StrictlyUpperTriangular{Float64}, n),
        H = rand(rng, StiefelLieAlgHorMatrix{Float64}, N, n),
        R = rand(rng, GrassmannLieAlgHorMatrix{Float64}, N, n),
        Y = StiefelManifold(point()), G = GrassmannManifold(point())))
    mapstorage(s -> S.(T.(s)), set)
end

# a nonlinear use of every leaf through a product, with weights that make `G` neither symmetric nor
# skew; `L₂` is a second use of every leaf, and `dense` reads every leaf as a plain array
function losses(::Type{T}, ::Type{S} = T) where {T, S}
    rng = Random.Xoshiro(8)
    round(X) = S.(T.(X))
    Xs = map(round,
        (S = randn(rng, n, n), K = randn(rng, n, n),
            Lo = randn(rng, n, n), Up = randn(rng, n, n),
            H = randn(rng, N, N), R = randn(rng, N, N), Y = randn(rng, n, n), G = randn(rng, n, n)))
    # the size of each product `leaf * X`
    Ws = map((X, m) -> round(randn(rng, m, size(X, 2))), Xs,
        (S = n, K = n, Lo = n, Up = n, H = N, R = N, Y = N, G = N))
    # every leaf, written out so that Zygote sees a type-stable sum
    each(f, ps) = f(ps.S, Xs.S, Ws.S) + f(ps.K, Xs.K, Ws.K) + f(ps.Lo, Xs.Lo, Ws.Lo) +
                  f(ps.Up, Xs.Up, Ws.Up) + f(ps.H, Xs.H, Ws.H) + f(ps.R, Xs.R, Ws.R) +
                  f(ps.Y, Xs.Y, Ws.Y) + f(ps.G, Xs.G, Ws.G)
    L₁(ps) = each((A, X, W) -> sum(sin.(W .* (A * X))), ps)
    L₂(ps) = each((A, X, _) -> sum(abs2, A * X), ps)
    dense(ps) = each((A, _, _) -> sum(abs2, A), ps)
    ("one use" => L₁, "two uses" => ps -> L₁(ps) + L₂(ps),
        "mixed" => ps -> L₁(ps) + dense(ps))
end

function central_difference(L, ps; h = 1e-6)
    v, layout = flatten(ps)
    map(eachindex(v)) do k
        p, m = copy(v), copy(v)
        p[k] += h
        m[k] -= h
        (L(unflatten(layout, p)) - L(unflatten(layout, m))) / (2h)
    end
end

@testset "Zygote's gradient of a parameter set is the storage gradient: $name, $T" for T in (Float32, Float64),
    (name, L) in losses(T)

    ps = structured_set(T)
    g = Zygote.gradient(L, ps)[1]
    reference = central_difference(Dict(losses(T, Float64))[name], structured_set(T, Float64))

    for k in keys(ps)
        @test typeof(getproperty(g, k)) == typeof(getproperty(ps, k)) ||
              getproperty(ps, k) isa Union{StiefelManifold, GrassmannManifold}
    end
    flat = first(flatten(g))
    @test eltype(flat) == T
    @test flat ≈ reference rtol = 1e-6
end

# the storage gradient written out entry by entry, from the dense `G`
function expected_storage(::Type{SymmetricMatrix}, G)
    [i == j ? G[i, i] : G[i, j] + G[j, i] for i in axes(G, 1) for j in 1:i]
end
function expected_storage(::Type{SkewSymMatrix}, G)
    [G[i, j] - G[j, i] for i in axes(G, 1) for j in 1:(i - 1)]
end

function leaf(::Type{SymmetricMatrix}, ::Type{T}, n) where {T}
    SymmetricMatrix(zeros(T, n * (n + 1) ÷ 2), n)
end
function leaf(::Type{SkewSymMatrix}, ::Type{T}, n) where {T}
    SkewSymMatrix(zeros(T, n * (n - 1) ÷ 2), n)
end

# Integer-valued entries, so that `G_ij ± G_ji` is exact and the comparison can be `==`.
@testset "the storage gradient of a $X cotangent, exactly: $T" for X in (SymmetricMatrix, SkewSymMatrix),
    T in (Float32, Float64)

    rng = Random.Xoshiro(9)
    G = T.(rand(rng, -9:9, n, n))
    expected = expected_storage(X, G)
    A = leaf(X, T, n)

    host = storage_gradient(A, G)
    @test host isa X{T}
    @test parent(host) == expected

    # an `Adjoint` cotangent: the transpose of the transpose
    @test parent(storage_gradient(A, copy(G')')) == expected

    # on a device, from a device leaf and a device cotangent
    device = storage_gradient(X(JLArray(parent(A)), n), JLArray(G))
    @test device isa X{T}
    @test parent(device) isa JLArray{T, 1}
    @test Array(parent(device)) == expected
    adjoint_device = storage_gradient(X(JLArray(parent(A)), n), JLArray(copy(G'))')
    @test Array(parent(adjoint_device)) == expected

    # a cotangent of the leaf's own type is a dense matrix with that structure, so its storage
    # gradient doubles the off-diagonal storage
    C = X(T.(rand(rng, -9:9, length(parent(A)))), n)
    @test parent(storage_gradient(A, C)) == expected_storage(X, Matrix(C))

    # a structural tangent holds `∂L/∂S` already and passes through
    tangent = Tangent{typeof(A)}(S = T.(1:length(parent(A))))
    @test storage_gradient(A, tangent) === tangent

    # `NaN` propagates as arithmetic does, into the one entry that reads it, `(n, 1)`
    G_nan = copy(G)
    G_nan[n, 1] = T(NaN)
    S_nan = parent(storage_gradient(A, G_nan))
    @test count(isnan, S_nan) == 1
    @test isnan(S_nan[X === SymmetricMatrix ? n * (n - 1) ÷ 2 + 1 :
                      (n - 2) * (n - 1) ÷ 2 + 1])
end

# `NeuralNetworkParameters.storage_gradient` asks for a leaf of the parameter leaf's type, so a
# cotangent of another precision gives a leaf of the parameter's element type. Integer-valued entries
# make the result exact in both precisions, so the reference is the same-precision call.
function mixed_leaves(::Type{T}, todevice) where {T}
    rng = Random.Xoshiro(10)
    draw(dims...) = todevice(T.(rand(rng, -9:9, dims...)))
    (SymmetricMatrix(draw(n * (n + 1) ÷ 2), n), SkewSymMatrix(draw(n * (n - 1) ÷ 2), n),
        StiefelLieAlgHorMatrix(SkewSymMatrix(draw(n * (n - 1) ÷ 2), n), draw(N - n, n), N, n),
        GrassmannLieAlgHorMatrix(draw(N - n, n), N, n))
end

# the storage of a leaf, on the host, in the order `flatten` writes it
hostflat(A) = first(flatten(mapstorage(Array, NetworkParameters((A = A,)))))

@testset "the storage gradient of a $T leaf and a $S cotangent is a $T leaf, $(nameof(todevice))" for (
        T, S) in (
        (Float32, Float64), (Float64, Float32)),
    todevice in (identity, JLArray)

    rng = Random.Xoshiro(11)
    for A in mixed_leaves(T, todevice)
        m = size(A, 1)
        G = T.(rand(rng, -9:9, m, m))
        expected = storage_gradient(A, todevice(G))
        for cotangent in (todevice(S.(G)), todevice(copy(S.(G)'))')
            g = storage_gradient(A, cotangent)
            @test typeof(g) == typeof(A)
            @test hostflat(g) == hostflat(expected)
        end
    end
    # a cotangent of the leaf's own type, in the other precision
    for X in (SymmetricMatrix, SkewSymMatrix)
        A = X(todevice(parent(leaf(X, T, n))), n)
        C = rand(rng, -9:9, length(parent(A)))
        g = storage_gradient(A, X(todevice(S.(C)), n))
        @test typeof(g) == typeof(A)
        @test Array(parent(g)) == T.(expected_storage(X, Matrix(X(C, n))))
    end
end

@testset "the smallest leaves: $T" for T in (Float32, Float64)
    # a 1 × 1 skew-symmetric matrix stores nothing, and a 1 × 1 symmetric one stores its entry
    skew = storage_gradient(SkewSymMatrix(T[], 1), fill(T(3), 1, 1))
    @test skew isa SkewSymMatrix{T}
    @test isempty(parent(skew))
    sym = storage_gradient(SymmetricMatrix(T[0], 1), fill(T(3), 1, 1))
    @test parent(sym) == T[3]
end
