# The elementwise optimizer primitives: `_difference!`, `_rmul!`, `_add!`, `_rac!`, `_div!`,
# `_square!` and `_copyto!`.
#
# Each acts on the free parameters of a leaf, at any depth of a `NetworkParameters`, and the tests
# below state that as a value: the flattening of the result is the elementwise formula applied to
# the flattenings of the operands, compared with `==`. The flattening is the storage of each leaf one
# after another, so an elementwise formula commutes with it, and `==` is what catches a primitive that
# computes the same quantity in another order (`x * inv(y)` for `x / y`).
#
# The arithmetic primitives act on gradients, directions and moments, which are lifts, arrays and the
# structured matrices. On a manifold point they compute on its storage and do not guard it; only
# `_copyto!` is meant to write a point. A testset at the end pins that.
using GeometricOptimizers
using GeometricOptimizers: _difference!, _rmul!, _add!, _rac!, _div!, _square!, _copyto!,
                           _zero,
                           _copy, OptimizerCache, direction, update!
using NeuralNetworkParameters: NetworkParameters, flatten
using Test
import Random

Random.seed!(1234)

flat(x) = flatten(x)[1]

# One leaf of each kind the arithmetic primitives see, with positive storage so that `_rac!` and
# `_div!` are defined everywhere.
function tangent_leaves(::Type{T}) where {T}
    (vector = rand(T, 4) .+ one(T), matrix = rand(T, 2, 3) .+ one(T),
        sym = rand(SymmetricMatrix{T}, 3), skew = rand(SkewSymMatrix{T}, 3),
        lower = rand(StrictlyLowerTriangular{T}, 3), upper = rand(StrictlyUpperTriangular{T}, 3),
        stiefel_lift = rand(StiefelLieAlgHorMatrix{T}, 5, 2),
        grassmann_lift = rand(GrassmannLieAlgHorMatrix{T}, 5, 2))
end

# Every structured leaf once in one parameter set, nested one level, which is the shape a network has.
function tangent_set(::Type{T}) where {T}
    l = tangent_leaves(T)
    NetworkParameters((L1 = (S = l.sym, K = l.skew, W = l.matrix),
        L2 = (lo = l.lower, up = l.upper, b = l.vector),
        L3 = (A = l.stiefel_lift, B = l.grassmann_lift)))
end

# the same with the two manifold points, for `_copyto!`
function point_set(::Type{T}) where {T}
    NetworkParameters((L1 = (Y = rand(StiefelManifold{T}, 5, 2), W = rand(T, 2, 3)),
        L2 = (Z = rand(GrassmannManifold{T}, 5, 2), S = rand(SymmetricMatrix{T}, 3))))
end

function operands(T, kind)
    kind === :set ? (tangent_set(T), tangent_set(T), tangent_set(T)) :
    (getfield(tangent_leaves(T), kind), getfield(tangent_leaves(T), kind),
        getfield(tangent_leaves(T), kind))
end

const KINDS = (
    :vector, :matrix, :sym, :skew, :lower, :upper, :stiefel_lift, :grassmann_lift, :set)

@testset "each primitive is the elementwise formula on the free parameters, $T" for T in (
    Float32, Float64)
    for kind in KINDS
        a, b, c = operands(T, kind)
        A, B = flat(a), flat(b)

        @test _difference!(c, a, b) === c
        @test flat(c) == A .- B

        for s in (T(2.5), -one(T))
            x = _copy(a)
            @test _rmul!(x, s) === x
            @test flat(x) == A .* s
        end

        x = _copy(a)
        @test _add!(x, b) === x
        @test flat(x) == A .+ B
        x = _copy(a)
        @test _add!(x, T(0.5)) === x
        @test flat(x) == A .+ T(0.5)

        @test _square!(c, a) === c
        @test flat(c) == A .^ 2
        @test _rac!(c, a) === c
        @test flat(c) == sqrt.(A)
        # in place, as `AdamCache` calls it
        x = _copy(a)
        @test _rac!(x, x) === x
        @test flat(x) == sqrt.(A)

        @test _div!(c, a, b) === c
        @test flat(c) == A ./ B
        x = _copy(a)
        @test _div!(x, x, b) === x
        @test flat(x) == A ./ B

        @test _copyto!(c, a) === c
        @test flat(c) == A
        @test eltype(flat(c)) == T
    end
end

@testset "_copyto! copies a parameter set with manifold points, $T" for T in (Float32, Float64)
    a, b = point_set(T), point_set(T)
    @test _copyto!(a, b) === a
    @test flat(a) == flat(b)
    @test a.L1.Y isa StiefelManifold{T}
    @test a.L2.Z isa GrassmannManifold{T}
    Y, Z = rand(StiefelManifold{T}, 5, 2), rand(StiefelManifold{T}, 5, 2)
    @test _copyto!(Y, Z) === Y
    @test Y == Z
end

# `_copyto!` calls each leaf's own `copyto!`, so it keeps that method's refusals: a point or a
# structured matrix of another kind or size, bare and inside a parameter set.
@testset "_copyto! refuses a leaf of another kind or size, $T" for T in (Float32, Float64)
    @test_throws ArgumentError _copyto!(rand(StrictlyLowerTriangular{T}, 3),
        rand(StrictlyUpperTriangular{T}, 3))
    @test_throws ArgumentError _copyto!(rand(StiefelManifold{T}, 6, 2),
        rand(GrassmannManifold{T}, 6, 2))
    @test_throws ArgumentError _copyto!(rand(GrassmannManifold{T}, 5, 2),
        rand(StiefelManifold{T}, 5, 2))
    @test_throws AssertionError _copyto!(rand(StiefelManifold{T}, 6, 2),
        rand(StiefelManifold{T}, 5, 2))
    @test_throws AssertionError _copyto!(rand(StiefelManifold{T}, 6, 2),
        rand(StiefelManifold{T}, 4, 3))
    @test_throws AssertionError _copyto!(rand(SymmetricMatrix{T}, 3), rand(SymmetricMatrix{T}, 2))
    @test_throws AssertionError _copyto!(rand(SkewSymMatrix{T}, 4), rand(SkewSymMatrix{T}, 3))
    @test_throws Exception _copyto!(rand(SkewSymMatrix{T}, 4), rand(SymmetricMatrix{T}, 3))
    @test_throws ArgumentError _copyto!(
        NetworkParameters((L = rand(StrictlyLowerTriangular{T}, 3),)),
        NetworkParameters((L = rand(StrictlyUpperTriangular{T}, 3),)))
    @test_throws ArgumentError _copyto!(
        NetworkParameters((Y = rand(StiefelManifold{T}, 6, 2),)),
        NetworkParameters((Y = rand(GrassmannManifold{T}, 6, 2),)))
    @test_throws AssertionError _copyto!(
        NetworkParameters((Y = rand(StiefelManifold{T}, 6, 2),)),
        NetworkParameters((Y = rand(StiefelManifold{T}, 4, 3),)))
end

# The arithmetic primitives refuse a structured leaf of another kind or size, also where the two
# storage vectors have the same length; and `_difference!` and `_div!` assert the axes of arrays.
@testset "the arithmetic primitives refuse a leaf of another kind or size, $T" for T in (
    Float32, Float64)
    K, S = rand(SkewSymMatrix{T}, 4), rand(SymmetricMatrix{T}, 3)   # 6 numbers each
    L, U = rand(StrictlyLowerTriangular{T}, 3), rand(StrictlyUpperTriangular{T}, 3)
    Y, Z = rand(StiefelManifold{T}, 5, 2), rand(GrassmannManifold{T}, 5, 2)
    small, large = rand(StiefelLieAlgHorMatrix{T}, 5, 2),
    rand(StiefelLieAlgHorMatrix{T}, 6, 2)
    # the same `N` and another `n`: both report `size == (N, N)`
    narrow, wide = rand(StiefelLieAlgHorMatrix{T}, 5, 2),
    rand(StiefelLieAlgHorMatrix{T}, 5, 3)
    gnarrow, gwide = rand(GrassmannLieAlgHorMatrix{T}, 5, 2),
    rand(GrassmannLieAlgHorMatrix{T}, 5, 3)
    for (x, y) in ((K, S), (L, U), (Y, Z), (small, large), (narrow, wide), (gnarrow, gwide))
        @test_throws ArgumentError _difference!(_copy(x), x, y)
        @test_throws ArgumentError _add!(_copy(x), y)
        @test_throws ArgumentError _rac!(_copy(x), y)
        @test_throws ArgumentError _div!(_copy(x), x, y)
        @test_throws ArgumentError _square!(_copy(x), y)
    end
    @test_throws ArgumentError _add!(NetworkParameters((A = _copy(L),)),
        NetworkParameters((A = U,)))

    a, b = rand(T, 3), rand(T, 4)
    @test_throws AssertionError _difference!(_copy(a), a, b)
    @test_throws AssertionError _div!(_copy(a), a, b)
end

# A hot path: every `update!` and every `OptimizerStatus` runs these, so they infer, in both
# precisions. That they allocate nothing is asserted in `test/flat_buffer_allocations.jl`.
@testset "the primitives infer, $T" for T in (Float32, Float64)
    for kind in (:vector, :stiefel_lift, :set)
        a, b, c = operands(T, kind)
        @test (@inferred _difference!(c, a, b)) === c
        @test (@inferred _rmul!(c, T(2))) === c
        @test (@inferred _add!(c, b)) === c
        @test (@inferred _add!(c, T(2))) === c
        @test (@inferred _square!(c, a)) === c
        @test (@inferred _rac!(c, a)) === c
        @test (@inferred _div!(c, a, b)) === c
        @test (@inferred _copyto!(c, a)) === c
    end
end

# A gap in a source set is skipped, as `mapparameters!` and `mapstorage!` both skip it, so the leaf it
# would have written keeps its value. A bare `nothing` in place of a whole operand has no method: a
# silent skip there would turn a missing gradient into an iterate that never moves.
@testset "a `nothing` source, $T" for T in (Float32, Float64)
    dest() = NetworkParameters((
        W = rand(Random.Xoshiro(1), T, 3), b = rand(Random.Xoshiro(2), T, 2)))
    gap = NetworkParameters((W = rand(Random.Xoshiro(3), T, 3), b = nothing))
    full = dest()

    for (name, f!) in (("_difference!", c -> _difference!(c, full, gap)),
        ("_add!", c -> _add!(c, gap)), ("_rac!", c -> _rac!(c, gap)),
        ("_div!", c -> _div!(c, full, gap)), ("_square!", c -> _square!(c, gap)),
        ("_copyto!", c -> _copyto!(c, gap)))
        c = dest()
        b = _copy(c.b)
        @test f!(c) === c
        @test c.b == b
    end

    x = rand(T, 3)
    @test_throws MethodError _difference!(x, x, nothing)
    @test_throws MethodError _rmul!(x, nothing)
    @test_throws MethodError _add!(x, nothing)
    @test_throws MethodError _rac!(x, nothing)
    @test_throws MethodError _div!(x, x, nothing)
    @test_throws MethodError _square!(x, nothing)
    @test_throws MethodError _copyto!(x, nothing)
    @test_throws MethodError _copyto!(nothing, x)
end

# A leaf with no free parameters, and a layer with none: nothing to do, and nothing raised.
@testset "zero-length storage, $T" for T in (Float32, Float64)
    for a in (rand(SkewSymMatrix{T}, 1),
        NetworkParameters((L1 = (K = rand(SkewSymMatrix{T}, 1),), L2 = (W = zeros(T, 0),))))
        b, c = _copy(a), _copy(a)
        @test isempty(flat(a))
        @test _difference!(c, a, b) === c
        @test _rmul!(c, T(2)) === c
        @test _add!(c, b) === c
        @test _add!(c, T(2)) === c
        @test _square!(c, a) === c
        @test _rac!(c, a) === c
        @test _div!(c, a, b) === c
        @test _copyto!(c, a) === c
    end
end

# Two operands of different element types, on an array and on a parameter set of arrays: the
# primitives that bind one element type on every operand refuse the pair, and the others compute in
# the promoted type and store in the destination's.
@testset "operands of different element types" begin
    for (narrow, wide) in ((rand(Float32, 3) .+ 1, rand(Float64, 3) .+ 1),
        (NetworkParameters((W = rand(Float32, 2, 2) .+ 1,)),
        NetworkParameters((W = rand(Float64, 2, 2) .+ 1,))))
        @test_throws MethodError _difference!(_copy(narrow), narrow, wide)
        @test_throws MethodError _add!(_copy(narrow), wide)
        @test_throws MethodError _add!(_copy(narrow), 2.0)
        @test_throws MethodError _copyto!(_copy(narrow), wide)

        A, B = flat(narrow), flat(wide)
        x = _copy(narrow)
        @test _rmul!(x, 2.5) === x
        @test flat(x) == Float32.(A .* 2.5)
        x = _copy(narrow)
        @test _rac!(x, wide) === x
        @test flat(x) == Float32.(sqrt.(B))
        x = _copy(narrow)
        @test _div!(x, narrow, wide) === x
        @test flat(x) == Float32.(A ./ B)
        x = _copy(narrow)
        @test _square!(x, wide) === x
        @test flat(x) == Float32.(B .^ 2)
        @test eltype(flat(x)) == Float32
    end
end

# `NaN` and `Inf` go through as IEEE arithmetic takes them; nothing checks for them.
@testset "NaN and Inf propagate, $T" for T in (Float32, Float64)
    a, b, c = operands(T, :stiefel_lift)
    a.B[1] = T(NaN)
    b.B[2] = T(Inf)
    A, B = flat(a), flat(b)
    _difference!(c, a, b)
    @test isequal(flat(c), A .- B)
    _div!(c, a, b)
    @test isequal(flat(c), A ./ B)
    _square!(c, b)
    @test isequal(flat(c), B .^ 2)
    x = _copy(a)
    _add!(x, b)
    @test isequal(flat(x), A .+ B)
    _copyto!(c, a)
    @test isequal(flat(c), A)
end

# The `NaN` poison of issue #22: an uninitialised tangent of a quasi-Newton state reads `NaN`, and the
# copy a step takes of it reads `NaN` too, on a bare point and on a parameter set that holds one.
@testset "the NaN poison of a quasi-Newton state survives _copyto!, $T" for T in (Float32, Float64)
    for x in (rand(StiefelManifold{T}, 6, 3),
        NetworkParameters((Y = rand(StiefelManifold{T}, 6, 3), W = rand(T, 2, 2))))
        state = OptimizerState(BFGS(), x)
        @test all(isnan, flat(state.s))

        d = _zero(x)
        @test _copyto!(d, state.s) === d
        @test all(isnan, flat(d))

        cache = OptimizerCache(BFGS(), x)
        _copyto!(direction(cache), _zero(x))
        update!(cache, state, x)
        @test all(isnan, flat(direction(cache)))
    end
end

# The internal primitives do not guard a manifold point: they compute on its storage, so the result
# is in general not a point. No path in the package applies them to one. A structured leaf of
# another element type is computed on its storage too, as a `Vector` is.
@testset "the internal primitives do not guard a manifold point, $T" for T in (Float32, Float64)
    for P in (StiefelManifold, GrassmannManifold)
        a, b = rand(P{T}, 5, 2), rand(P{T}, 5, 2)
        A, B = parent(a), parent(b)
        x = _copy(a)
        @test _difference!(x, a, b) === x
        @test parent(x) == A .- B
        x = _copy(a)
        @test _rmul!(x, T(2)) === x
        @test parent(x) == A .* T(2)
        x = _copy(a)
        @test _add!(x, b) === x
        @test parent(x) == A .+ B
        x = _copy(a)
        @test _add!(x, T(0.5)) === x
        @test parent(x) == A .+ T(0.5)
        x = _copy(a)
        @test _square!(x, a) === x
        @test parent(x) == A .^ 2
        @test _rac!(x, x) === x
        @test parent(x) == sqrt.(A .^ 2)
        x = _copy(a)
        @test _div!(x, a, b) === x
        @test parent(x) == A ./ B
    end
end

@testset "a structured leaf of another element type is computed on its storage" begin
    for (narrow, wide) in ((
        rand(SkewSymMatrix{Float32}, 3), rand(SkewSymMatrix{Float64}, 3)),
        (rand(SymmetricMatrix{Float32}, 3), rand(SymmetricMatrix{Float64}, 3)),
        (rand(StrictlyLowerTriangular{Float32}, 3),
        rand(StrictlyLowerTriangular{Float64}, 3)),
        (rand(StiefelLieAlgHorMatrix{Float32}, 5, 2),
        rand(StiefelLieAlgHorMatrix{Float64}, 5, 2)))
        A, B = flat(narrow), flat(wide)
        x = _copy(narrow)
        @test _rac!(x, wide) === x
        @test flat(x) == Float32.(sqrt.(B))
        x = _copy(narrow)
        @test _div!(x, narrow, wide) === x
        @test flat(x) == Float32.(A ./ B)
        x = _copy(narrow)
        @test _square!(x, wide) === x
        @test flat(x) == Float32.(B .^ 2)
    end
end
