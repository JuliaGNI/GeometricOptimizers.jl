"""
    OptimizerMethod <: SolverMethod

The `OptimizerMethod` is used in [`Optimizer`](@ref) and determines the algorithm that is used.
"""
abstract type OptimizerMethod <: SolverMethod end

"""
    QuasiNewtonOptimizerMethod <: OptimizerMethod

Includes [`_BFGS`](@ref) and [`_DFP`](@ref).
"""
abstract type QuasiNewtonOptimizerMethod <: OptimizerMethod end

@doc raw"""
    Newton

Newton's method: the direction solves ``\nabla^2f(x)\delta = -\nabla{}f(x)`` with the exact Hessian,
which [`SimpleSolvers.HessianAutodiff`](@extref) supplies.

Unlike [`_BFGS`](@ref) and [`_DFP`](@ref) this needs no approximation to build up, so it converges in
few iterations, but it also inherits the Hessian's indefiniteness: where ``\nabla^2f`` is not positive
definite the direction ascends, and [`ensure_descent!`](@ref) substitutes the steepest-descent
direction for it.
"""
struct Newton <: OptimizerMethod end

Hessian(::Newton, ForOBJ::Union{Callable,OptimizerProblem}, x::AbstractVector) = HessianAutodiff(ForOBJ, x)
HessianAutodiff(F::OptimizerProblem, x) = HessianAutodiff(F.F, x)

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct _DFP <: QuasiNewtonOptimizerMethod end

"""
Algorithm taken from [nocedal2006numerical](@cite).
"""
struct _BFGS <: QuasiNewtonOptimizerMethod end

"""
The gradient descent algorithm.
"""
struct GradientMethod <: OptimizerMethod end

@doc raw"""
    MomentumMethod(α)

The gradient descent algorithm with momentum, i.e. the *heavy ball* method.

Stores the *momentum coefficient* `α`. The momentum is accumulated as
```math
    p \gets \alpha{}p + \nabla{}L
```
and the direction is ``-p``.

!!! info "The step size is not stored here"
    Like every [`OptimizerMethod`](@ref), `MomentumMethod` only produces a direction; how far
    the optimizer goes along it is the line search's business. A fixed learning rate ``\eta``
    is therefore expressed as `linesearch = Static(η)`, which is also the default (see
    [`default_linesearch`](@ref)).
"""
struct MomentumMethod{T} <: OptimizerMethod
    α::T

    MomentumMethod(α::T=DEFAULT_MOMENTUM_α) where {T} = new{T}(α)
end

@doc raw"""
    Adam(T; β₁, β₂, δ)

The Adam optimizer, with the defaults suggested in [goodfellow2016deep; page 301](@cite).

The cache consists of a first and a second moment, stored in *bias-corrected* form, i.e.
updated in the ``t``-th iteration (counted from ``t = 1``) as
```math
m_1 \gets \frac{\beta_1 - \beta_1^t}{1 - \beta_1^t}m_1 + \frac{1 - \beta_1}{1 - \beta_1^t}\nabla{}L,
```
```math
m_2 \gets \frac{\beta_2 - \beta_2^t}{1 - \beta_2^t}m_2 + \frac{1 - \beta_2}{1 - \beta_2^t}\nabla{}L\odot\nabla{}L,
```
from which the direction is computed as ``-m_1/(\sqrt{m_2} + \delta)``.

`T` is the element type of the parameters that are to be optimized; unlike
[`MomentumMethod`](@ref), `Adam` is *not* converted by [`Optimizer`](@ref), so
`Adam(Float32)` is needed for `Float32` parameters.

!!! info "The learning rate is not stored here"
    `Adam` only produces a direction, of magnitude ``\approx{}1`` per component; the learning
    rate ``\eta`` is the line search's `α`, i.e. it is passed as `linesearch = Static(η)`,
    which is also the default (see [`default_linesearch`](@ref)). `Adam` used to carry an `η`
    field that was never applied to the direction, so `Adam(1e-3)` and `Adam(1e2)` gave
    identical results; it is gone, and because it used to be the *first positional* argument,
    `β₁`, `β₂` and `δ` are keyword arguments now so that an old `Adam(1e-3)` call fails
    instead of silently setting `β₁ = 1e-3`.
"""
struct Adam{T} <: OptimizerMethod
    β₁::T
    β₂::T
    δ::T

    # the defaults are written as `Float64` literals so that `T(9.0e-1)` is `0.9` and not
    # `Float64(9.0f-1) = 0.8999999761581421`; for `T = Float32` they are the same numbers
    function Adam(::Type{T}=Float64; β₁=9.0e-1, β₂=9.9e-1, δ=1.0e-8) where {T}
        new{T}(T(β₁), T(β₂), T(δ))
    end
end

const DEFAULT_MOMENTUM_α = 0.01

const DEFAULT_LEARNING_RATE = 1.0e-3

@doc raw"""
    default_linesearch(T, method)

Return the line search that [`Optimizer`](@ref) uses for `method` if none is supplied.

Everything except [`Adam`](@ref) defaults to
[`SimpleSolvers.Backtracking`](@extref)`(T; expand = true)`: the (quasi-)Newton methods because they
build a direction with a scale of its own, and [`GradientMethod`](@ref) and [`MomentumMethod`](@ref)
because a searching line search is what makes them *converge* rather than merely descend. All produce
genuine descent directions, so a backtracking search always has an `α` to find.

`expand = true` is what lets that search *lengthen* a step as well as shorten it, and it is not the
SimpleSolvers default — see the tip below for why it is the default here.

`Adam` is the exception and keeps a fixed `Static(DEFAULT_LEARNING_RATE)`. Its direction is
``-m_1/(\sqrt{m_2} + \delta)``, a moving average that is deliberately *not* required to descend on any
individual step, so a sufficient-decrease search has nothing to work with and would spend every such
step reporting that it found no descent direction.

!!! info "This changed in 0.2.0"
    `GradientMethod` and `MomentumMethod` used to default to `Static(DEFAULT_LEARNING_RATE)` as well.
    They could not do anything else: until the line search learned to take its trial step through the
    retraction (see [`trial_iterate!`](@ref)), `Static` was the only line search that worked on
    manifold parameters at all. Pass `linesearch = Static(η)` to get the old fixed learning rate back.

!!! tip "Why a backtracking search, and why `expand`"
    `Backtracking` returns the *first* `α` that decreases `f` enough, while `Bisection`,
    [`SimpleSolvers.Quadratic`](@extref) and [`SimpleSolvers.BierlaireQuadratic`](@extref) bracket and
    then refine a line *minimum*, which costs an order of magnitude more merit evaluations per
    iteration. Iterations are therefore the wrong unit to compare them in. Counting objective
    evaluations instead, on the SVD problem of `test/optimizer_convergence/svd_optim.jl` (`Geodesic`;
    `Static` needs ≈4 evaluations per iteration, so subtract that for the search's own cost):

    | search | evals/iteration | `_BFGS`: iters / evals | `_DFP`: iters / evals |
    |---|---|---|---|
    | `Backtracking(expand = true)` | **26** | **93** / **2 374** | 830 / 21 540 |
    | `Backtracking` (shrink only) | 25 | 113 / 2 857 | 49 679 / 1 241 987 |
    | `StrongWolfe` (`c₂ = 0.1`) | 57 | 118 / 6 738 | 201 / **16 466** |
    | `StrongWolfe` (`c₂ = 0.9`) | 36 | 159 / 5 708 | no convergence |
    | `BierlaireQuadratic` | 102 | 170 / 17 340 | 322 / 27 484 |
    | `Quadratic` | 129 | 173 / 22 267 | 189 / 18 313 |
    | `Bisection` | 583 | 143 / 83 353 | 134 / 78 698 |

    A shrink-only backtracking search starts at `α = 1` and can never exceed it, which is right for a
    direction already scaled like a Newton step — `_BFGS` accepts `α = 1` on 74% of its iterations —
    but wrong for one that is systematically *under*-scaled. `_DFP` wants a median `α` of 8, so it
    accepted the ceiling on **100%** of its iterations and crawled to the gate in 49 679 of them.
    `expand = true` lets an accepted *first* trial step be lengthened while each longer trial still
    satisfies sufficient decrease and strictly improves the merit, at most `nexpand = 3` rounds of at
    most `q = 10` each.

    That fixes `_DFP` outright and makes `_BFGS` slightly better as well, at a cost of under 4% per
    iteration — and of exactly nothing on a well-scaled problem, since the extrapolation reuses
    ``\varphi(0)``, ``\varphi'(0)`` and ``\varphi(\alpha)``, all known once the trial step is accepted,
    so declining to expand costs no evaluation at all. On the sphere problem the evaluation counts are
    identical with and without it.

    Reach for one of the bracketing methods when iteration count rather than evaluation count is what
    you are paying for — a very expensive objective, or an outer loop bounded in iterations. For the
    first-order methods that trade is poor: `Bisection` burns 1.8M evaluations against
    `Backtracking`'s 79 500 for the same 3 000 iterations.

!!! note "[`_DFP`](@ref) converges under the default, but [`SimpleSolvers.StrongWolfe`](@extref) suits it better"
    DFP's direction stays under-scaled — the expansion phase makes that harmless rather than absent, so
    `_DFP` needs 830 iterations on `Geodesic` and 1 237 on `Cayley` where `_BFGS` needs 93 and 118, on
    the starting point the test suite uses. **Those two numbers are not representative.** Over eight
    starting points on the same problem `_DFP` + the default ranges over 512–77 890 iterations
    (`Geodesic`) and 465–3 834 (`Cayley`): `Q` becomes badly conditioned (κ ≈ 1e9) and how quickly the
    expansion phase digs it out is close to arbitrary.

    `StrongWolfe(T; c₂ = 0.1)` is both faster and far steadier, and is the choice to pass explicitly on
    a DFP-heavy workload: 201 and 274 iterations on that starting point, 201–624 and 192–483 across the
    eight, 16 466 and 23 312 evaluations, about 1.7× faster in wall clock (0.12 s against 0.20 s on
    `Geodesic`, 0.18 s against 0.33 s on `Cayley`). `Bisection` is steadier still (103–143 / 96–141)
    at four to five times the work.

    `c₂ = 0.1` and not `StrongWolfe`'s own default of `0.9`: at `0.9` the strong Wolfe conditions are
    already satisfied at `α = 1` on 99.4% of iterations, so its bracketing phase never fires and it
    crawls just as a shrink-only `Backtracking` does. `0.1` is the value [nocedal2006numerical](@cite)
    recommends where a more accurate line search is needed, and it makes the expansion fire on 94.5% of
    iterations.

    `Quadratic` is competitive on `Geodesic` (189 iterations) and falls apart on `Cayley` (555) —
    probably because [`trial_slope`](@ref) is only first-order correct there and `Quadratic` uses
    ``\varphi'`` *quantitatively* in its polynomial fit, where `Bisection` uses only its sign and
    `StrongWolfe` only compares it against ``\varphi'(0)``.

    None of this is a property of DFP as such: given a search that can exceed `α = 1` it is competitive
    with `_BFGS`. The expansion phase exists because of this package — see
    JuliaGNI/SimpleSolvers.jl#174, which was filed from these measurements and released in
    SimpleSolvers 0.11.
"""
default_linesearch(::Type{T}, ::OptimizerMethod) where {T} = Backtracking(T; expand=true)
default_linesearch(::Type{T}, ::Adam) where {T} = Static(T(DEFAULT_LEARNING_RATE))

Base.show(io::IO, alg::Newton) = print(io, "Newton")
Base.show(io::IO, alg::_DFP) = print(io, "DFP")
Base.show(io::IO, alg::_BFGS) = print(io, "BFGS")
