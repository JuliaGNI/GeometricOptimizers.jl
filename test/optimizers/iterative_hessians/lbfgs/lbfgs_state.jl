# `LBFGS` is `BFGS` with `Q` never formed. With `initial = :bfgs` its initial matrix is the initial
# `Q` of `BFGS`, `M⁻¹`, and with a memory that holds every accepted pair the two-loop recursion gives
# `Q rhs` for the `Q` that `BFGS` holds, so the two take the same iterates up to round-off. The
# comparison is over 15 steps of a line-search solve, with at least 10 accepted pairs, on a vector,
# a Stiefel, a symplectic Stiefel point and a parameter set holding all three; the control is the
# same solve with a memory of 2, which drops pairs and leaves the path of `BFGS`.

using GeometricOptimizers
using GeometricOptimizers: solver_step!, increase_iteration_number!, initialize_state!,
                           update!,
                           restart!, flatlength, _zero, OptimizerCache, direction
using SimpleSolvers: Backtracking
using NeuralNetworkParameters: NetworkParameters
using LinearAlgebra: Diagonal, I, dot, norm, tr
using Test
import Random

include("../../../helpers/symplectic_points.jl")

const T = Float64

spd(rng, n) = (A = randn(rng, T, n, n); A * A' + I)

const A = spd(Random.Xoshiro(1), 8)
const S = spd(Random.Xoshiro(2), 10)
const M = spd(Random.Xoshiro(3), 10)
const D = Diagonal(T[3, 2, 1])

f_vector(x) = dot(x, A * x) / 2 + sum(x .^ 4) / 4
f_stiefel(Y) = -tr(Y' * S * Y * D) / 2 + sum(abs2, Y.A .- T(0.1)) / 10
f_symplectic(U) = tr(U' * M * U)
f_set(p) = f_vector(p.x) + f_stiefel(p.Y) + f_symplectic(p.U)

start_vector() = ones(T, 8)
start_stiefel() = rand(Random.Xoshiro(4), StiefelManifold{T}, 10, 3)
start_symplectic() = conditioned_point(Random.Xoshiro(5), T, 10, 4)
function start_set()
    NetworkParameters((x = start_vector(), Y = start_stiefel(), U = start_symplectic()))
end

points(x::AbstractVector) = copy(x)
points(x::Manifold) = copy(parent(x))
points(p::NetworkParameters) = map(points, (p.x, p.Y, p.U))
distance(a, b) = norm(a .- b)
distance(a::Tuple, b::Tuple) = sqrt(sum(distance.(a, b) .^ 2))
magnitude(a) = norm(a)
magnitude(a::Tuple) = sqrt(sum(magnitude.(a) .^ 2))

# the iterates of 15 steps of a solve, and the state after them
function path(x, F, algorithm; steps = 15)
    mode = x isa NetworkParameters ? :autodiff : nothing
    opt = Optimizer(x, F; algorithm = algorithm, linesearch = Backtracking(T), mode = mode)
    state = OptimizerState(algorithm, x)
    initialize_state!(state, opt, x, F(x))
    iterates = []
    for _ in 1:steps
        increase_iteration_number!(state)
        solver_step!(x, state, opt)
        update!(state, opt, x, F(x))
        push!(iterates, points(x))
    end
    iterates, state
end

relative_distance(a, b) = maximum(distance.(a, b) ./ magnitude.(a))

const PROBLEMS = (
    vector = (start_vector, f_vector),
    stiefel = (start_stiefel, f_stiefel),
    symplectic = (start_symplectic, f_symplectic),
    set = (start_set, f_set))

@testset "LBFGS with a full memory and H₀ = M⁻¹ takes the iterates of BFGS, $name" for (name, (
    start, F)) in pairs(PROBLEMS)
    dense, _ = path(start(), F, BFGS())
    limited, state = path(start(), F, LBFGS(; memory = 20, initial = :bfgs))
    @test state.npairs ≥ 10
    @test distance(dense[end], points(start())) > 0.1 * magnitude(points(start()))
    @test relative_distance(dense, limited) < 1e-10

    # the control: two pairs are not the `Q` of `BFGS`
    short, state = path(start(), F, LBFGS(; memory = 2, initial = :bfgs))
    @test state.npairs == 2
    @test relative_distance(dense, short) > 1e-6
end

# `curvature_is_usable` decides whether a pair is pushed, as it decides whether `BFGS` updates `Q`.
@testset "a pair with δᵀγ ≤ 0 is skipped, and the memory is a ring" begin
    x = zeros(T, 3)
    cache = OptimizerCache(LBFGS(; memory = 2), x)
    state = OptimizerState(LBFGS(; memory = 2), x)
    function push(δ, γ)
        state.s .= δ
        state.ḡ .= 0
        update!(cache, state, x, γ)
        state.npairs
    end
    @test push(T[1, 0, 0], T[-1, 0, 0]) == 0
    @test push(T[1, 0, 0], T[0, 1, 0]) == 0
    @test push(T[1, 0, 0], T[2, 0, 0]) == 1
    @test push(T[0, 1, 0], T[0, 3, 0]) == 2
    @test push(T[0, 0, 1], T[0, 0, 4]) == 2
    # the oldest pair, `(e₁, 2e₁)`, was dropped: the newest two are `(e₂, 3e₂)` and `(e₃, 4e₃)`
    @test sort([s[2] + 2s[3] for s in state.S]) == T[1, 2]
    # with `θ = δᵀγ/γᵀγ = 1/4` of the newest pair, `H(-g)` for `g = (1, 1, 1)` is
    # `-(θ, 1/3, 1/4)`: `θ` on the direction no pair spans. A `NaN` step pushes no pair.
    state.s .= NaN
    state.ḡ .= 0
    update!(cache, state, x, T[1, 1, 1])
    @test state.npairs == 2
    @test direction(cache) ≈ -T[1 / 4, 1 / 3, 1 / 4]

    restart!(state)
    @test state.npairs == 0
    @test state.θ == 1
end

@testset "LBFGS refuses an empty memory and an unknown initial matrix" begin
    @test_throws ArgumentError LBFGS(; memory = 0)
    @test_throws ArgumentError LBFGS(; initial = :identity)
    @test_throws ArgumentError LBFGSState(zeros(T, 3); memory = 0)
end

# The state of `LBFGS` on a `d`-dimensional problem holds `(2 memory + 2)d` numbers in its vectors
# besides the iterates, the gradients and the section, `4d` more on a symplectic point; dense `BFGS`
# holds `Q`, `d²`, and five more `d × d` matrices in its cache.
@testset "LBFGS holds O(memory·d) numbers, BFGS O(d²)" begin
    d, memory = 1000, 5
    x = randn(Random.Xoshiro(6), T, d)
    bytes(object) = Base.summarysize(object) / sizeof(T)

    limited = OptimizerState(LBFGS(; memory = memory), x)
    @test length(limited.S) == length(limited.Y) == memory
    @test all(length.(limited.S) .== d)
    # `x`, `x̄`, `s`, `ḡ`, `q`, `z` and the section's copies of `x`, with room for the headers
    @test bytes(limited) < (2memory + 10) * d
    @test bytes(OptimizerCache(LBFGS(; memory = memory), x)) < 10d
    @test bytes(OptimizerState(BFGS(), x)) > d^2
    @test bytes(OptimizerCache(BFGS(), x)) > 5d^2

    # on a symplectic point the vectors and the conjugate-gradient buffers are of length `d`
    U = conditioned_point(Random.Xoshiro(7), T, 40, 4)
    d = flatlength(_zero(U))
    limited = OptimizerState(LBFGS(; memory = memory), U)
    @test all(length.(limited.S) .== d)
    @test length(limited.metric_workspace.p) == d
    @test bytes(limited.S) + bytes(limited.Y) + bytes(limited.metric_workspace) <
          (2memory + 10) * d
end
