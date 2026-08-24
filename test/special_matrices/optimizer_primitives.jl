using GeometricOptimizers
using GeometricOptimizers: VectorStorageMatrix, _add!, _rac!, _square!, _div!, _rmul!,
    _difference!, increase_iteration_number!, solver_step!
using NeuralNetworkParameters: flatten, unflatten
using SimpleSolvers: Static, l2norm
using Test
import Random

Random.seed!(1234)

# The four types that store their free parameters in one vector. Everything below is written on
# `parent`, which is that vector, because that is the whole point of `VectorStorageMatrix`: the
# generic `AbstractArray` methods broadcast, and only `SymmetricMatrix` has a `setindex!` for them to
# broadcast through. Before these methods existed, a `SymmetricMatrix` or a triangular matrix could
# not be an optimizer parameter at all -- `GeometricMachineLearning` carried its own copies of them
# to work around it (GeometricMachineLearning#234).
const TYPES = (SkewSymMatrix, SymmetricMatrix, LowerTriangular, UpperTriangular)

const N = 5

@testset "$(nameof(MT))" for MT in TYPES
    @testset "similar preserves the type" begin
        # Not a detail: the optimizer caches allocate their scratch with `similar` and then require
        # every one of them to have the same type as the parameter. The `AbstractArray` fallback
        # returns a dense `Matrix`, which makes the cache constructors inapplicable.
        for T in (Float32, Float64)
            A = rand(MT{T}, N)
            S = similar(A)
            @test typeof(S) == typeof(A)
            @test size(S) == size(A)
            @test length(parent(S)) == length(parent(A))
        end
    end

    @testset "fill! writes the storage" begin
        A = fill!(similar(rand(MT{Float64}, N)), 3.0)
        @test all(parent(A) .== 3.0)
        # `NaN` is what the caches actually poison scratch arrays with, and it has to survive the
        # round trip rather than being swallowed by a comparison somewhere.
        @test all(isnan, parent(fill!(similar(A), NaN)))
    end

    @testset "the elementwise primitives act on the free parameters" begin
        A = rand(MT{Float64}, N)
        B = rand(MT{Float64}, N)

        a = copy(A)
        @test _add!(a, B) === a
        @test parent(a) ≈ parent(A) .+ parent(B)

        a = copy(A)
        @test _add!(a, 2.0) === a
        @test parent(a) ≈ parent(A) .+ 2.0

        # squares of a known matrix, so that `_rac!` inverts `_square!` exactly rather than to a
        # tolerance that a sign error could hide in
        sq = similar(A)
        @test _square!(sq, A) === sq
        @test parent(sq) ≈ parent(A) .^ 2

        rt = similar(A)
        @test _rac!(rt, sq) === rt
        @test parent(rt) ≈ abs.(parent(A))

        d = similar(A)
        @test _div!(d, A, B) === d
        @test parent(d) ≈ parent(A) ./ parent(B)

        m = copy(A)
        @test _rmul!(m, 2.0) === m
        @test parent(m) ≈ 2 .* parent(A)

        # `_difference!` belongs to the same family and is reached from a different place: not from
        # `update!` but from `gradient_difference!`, on *every* `OptimizerStatus`. It was the one
        # member left on `SkewSymMatrix` alone, which made the two triangular types unusable as
        # parameters -- they are the two of the four with neither a `setindex!` for the generic
        # broadcast nor a method of their own.
        c = similar(A)
        @test _difference!(c, A, B) === c
        @test parent(c) ≈ parent(A) .- parent(B)
        @test Matrix(c) ≈ Matrix(A) .- Matrix(B)
    end

    @testset "l2norm is taken over the free parameters" begin
        A = rand(MT{Float64}, N)
        @test l2norm(A) ≈ sqrt(sum(abs2, parent(A)))
    end

    @testset "flatten round trips through the free parameters" begin
        # The `AbstractMatrix` method reshapes the flattened vector to `n × n`, which for `n(n±1)/2`
        # numbers is a `DimensionMismatch`. That is where a `SymmetricMatrix` or a triangular
        # parameter used to die -- inside `Optimizer`, before a single primitive was reached.
        for T in (Float32, Float64)
            A = rand(MT{T}, N)
            v, layout = flatten(T, A)
            @test v isa Vector{T}
            @test length(v) == length(parent(A))
            A′ = unflatten(layout, v)
            @test typeof(A′) == typeof(A)
            @test A′ ≈ A
        end
    end

    @testset "the primitives leave the represented matrix consistent" begin
        # `_rmul!` writes the storage, but what it has to *mean* is scaling the matrix. For a
        # skew-symmetric matrix the two differ by a sign in the upper triangle, so this checks the
        # full dense matrix and not only `parent`.
        A = rand(MT{Float64}, N)
        dense = Matrix(A)
        @test Matrix(_rmul!(copy(A), 2.0)) ≈ 2 .* dense

        B = rand(MT{Float64}, N)
        @test Matrix(_add!(copy(A), B)) ≈ dense .+ Matrix(B)
    end

    @testset "update_section! is addition on the free parameters" begin
        # These are ordinary vector-space parameters, so the extended retraction is addition and the
        # section carries no `λ`. The retraction argument is ignored, which is what the two calls
        # below assert by giving different ones and expecting the same answer.
        A = rand(MT{Float64}, N)
        B = rand(MT{Float64}, N)

        Λᵗ = GlobalSection(copy(A))
        Λ⁽ᵗ⁻¹⁾ = GlobalSection(copy(A))
        @test update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B, Cayley()) === Λᵗ
        @test parent(Λᵗ.Y) ≈ parent(A) .+ parent(B)
        @test Λᵗ.λ === nothing

        Λ₂ = GlobalSection(copy(A))
        update_section!(Λ₂, GlobalSection(copy(A)), B, Geodesic())
        @test parent(Λ₂.Y) ≈ parent(Λᵗ.Y)

        # the in-place three-argument form, which is what `update_section!(Λ, B, retraction)` is
        Λ₃ = GlobalSection(copy(A))
        update_section!(Λ₃, B, Cayley())
        @test parent(Λ₃.Y) ≈ parent(A) .+ parent(B)
    end

    @testset "every one of them is a VectorStorageMatrix" begin
        @test rand(MT{Float64}, N) isa VectorStorageMatrix
        @test rand(MT{Float32}, N) isa VectorStorageMatrix{Float32}
    end
end

# Addition against a dense matrix has to commute. `SkewSymMatrix` had
# `+(B::AbstractMatrix, A::SkewSymMatrix) = B + A` in `GeometricMachineLearning`, which calls itself
# and dies with a `StackOverflowError`; the method here is `A + B` and always was. The loop closes
# the class rather than that one instance.
@testset "addition with a dense matrix commutes -- $(nameof(MT))" for MT in TYPES
    A = rand(MT{Float64}, N)
    B = rand(N, N)
    @test A + B ≈ B + A
    @test A + B ≈ Matrix(A) + B
end

# The claim the methods above are here for, stated once and end to end: an optimizer *runs* over one
# of these types. Nothing else in the suite asserts it, and testing the primitives one at a time does
# not -- both `_difference!` and the flattening were missing while every direct test of
# `_add!` and friends passed, because neither is called from `update!`. `flatten` fails inside the
# `Optimizer` constructor and `_difference!` on the first `OptimizerStatus`, so a run of a few steps
# is what covers the whole path.
"""
    optimize(MT, algorithm; steps, η)

Minimize ``\\frac{1}{2}\\|A - A_\\star\\|^2`` over the free parameters of an `MT`, and return the
objective before and after `steps` steps.
"""
function optimize(MT, algorithm; steps=5, η=0.1)
    Random.seed!(1234)
    A_target = rand(MT{Float64}, N)
    ps = (A=rand(MT{Float64}, N),)
    # on `parent`, i.e. on the free parameters: the ambient Frobenius norm would count the strict
    # triangle of a `SkewSymMatrix` twice and the objective is not the point here
    F(p) = sum(abs2, parent(p.A) .- parent(A_target)) / 2

    optimizer = Optimizer(ps, F; algorithm=algorithm, linesearch=Static(η))
    state = OptimizerState(algorithm, ps)

    before = F(ps)
    for _ in 1:steps
        increase_iteration_number!(state)
        solver_step!(ps, state, optimizer)
        update!(state, optimizer, ps)
    end

    before, F(ps), ps
end

@testset "an optimizer runs over a $(nameof(MT)) parameter -- $(nameof(typeof(algorithm)))" for
MT in TYPES, algorithm in (GradientMethod(), MomentumMethod(0.1), Adam(Float64))

    before, after, ps = optimize(MT, algorithm)
    @test after < before
    # and it is still the type it went in as, rather than a dense matrix the round trip lost the
    # structure of
    @test typeof(ps.A) <: MT{Float64}
end
