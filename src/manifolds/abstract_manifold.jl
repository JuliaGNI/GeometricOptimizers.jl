@doc raw"""
    Manifold <: AbstractMatrix

A manifold in `GeometricOptimizers` is a sutype of `AbstractMatrix`. All manifolds are matrix manifolds and therefore stored as matrices. More details can be found in the docstrings for the [`StiefelManifold`](@ref), the [`GrassmannManifold`](@ref) and the [`SymplecticStiefelManifold`](@ref).
"""
abstract type Manifold{T} <: AbstractMatrix{T} end

# TEMPORARY. A shim for a defect that is not in this package: the ambient gradient is an *input* to
# `rgrad`, so a caller holding its parameters on a device and its gradients on the host is broken
# wherever those gradients are allocated. Matching them here hides that, and pays a host-to-device
# transfer per manifold leaf per step, inside the region `PhaseTimer` attributes to the step. Remove
# this function, its two call sites and `test/gradient_backend.jl` once
# JuliaGNI/GeometricMachineLearning.jl#258 and JuliaGNI/AbstractNeuralNetworks.jl#39 are closed.
#
# The point's backend and not the gradient's, because the point is the parameter: it is what the
# caller chose to put on a device and what the retraction has to write back to. A point that is not
# on a device returns `∇L` untouched, without asking it for a backend — which is what keeps a
# gradient `KernelAbstractions` cannot place, a `ForwardDiff.Dual` matrix among them, on the host
# path it was always on.
function _match_backend(Y::Manifold, ∇L::AbstractMatrix)
    backend = KernelAbstractions.get_backend(Y)
    backend isa GPU || return ∇L
    KernelAbstractions.get_backend(∇L) == backend && return ∇L

    copyto!(KernelAbstractions.allocate(backend, eltype(∇L), size(∇L)...), ∇L)
end

@doc raw"""
    _cholesky_qr2(A)

An ``N\times{}m`` matrix whose columns are an orthonormal basis of the columns of the
``N\times{}m`` matrix `A`, or `nothing` when `A` is too ill-conditioned for its element type.

CholeskyQR2: take ``R`` from `cholesky(AᵀA)`, set ``Q = AR^{-1}``, and do it again. Every step is a
matrix product, a reduction or a triangular solve, so the whole factorization runs wherever `A`
already is. `LinearAlgebra.qr!` does not: `Metal` implements no `qr` for an `MtlArray`, and a `qr`
on a `JLArray` cannot rebuild its `Q`. Measured on an M4 Max with scalar indexing disallowed, this
runs on Metal and its ``\|Q^TQ - \mathbb{I}\|`` is *smaller* than the host Householder QR's at every
size tried.

**The second pass is not a refinement.** Measured on Metal in `Float32` at the shape
[`global_section`](@ref) uses, ``\|Q^TQ - \mathbb{I}\|`` after one pass is 9.7e-6 at ``N = 20`` and
1.9e-3 at ``N = 400``, against 7.1e-7 and 7.7e-6 after the second — and the manifold tests assert
`check(Y) < 1e-14` in `Float64`.

**Forming ``A^TA`` squares the condition number**, which is what the `nothing` is for. `cholesky`
with `check = false` reports a Gram matrix that is no longer positive definite rather than throwing,
and at this shape in `Float32` an ordinary Gaussian draw reaches that about once in two hundred.
[`_orthonormal_columns`](@ref GeometricOptimizers._orthonormal_columns) is what answers it.

**Squaring the entries is a second hazard, and the scaling below is what answers that one.** A
Householder QR scales internally and this does not, so an argument whose entries are large enough
gives an infinite Gram matrix and no answer at all. It is reachable: `optimizer_status_tests.jl`
builds a point ``10^{100}`` off the manifold to test a convergence guard, the projection carries that
magnitude into `A`, and ``A^TA`` is then `Inf` in `Float64`. Dividing `A` by
``\max_{ij}|A_{ij}|`` — a reduction that cannot overflow, unlike a column norm — leaves ``Q``
unchanged, since ``(A/s)^T(A/s) = A^TA/s^2`` has Cholesky factor ``R/s``.
"""
function _cholesky_qr2(A::AbstractMatrix)
    # `n = N` is a legitimate shape whose complement is empty, so [`global_section`](@ref) asks for
    # an `N × 0` factor. There is nothing to orthonormalize and it is already its own answer. The
    # early return is not tidiness: `maximum` over no entries is `typemin` rather than an error, so
    # the scaling test below would reject such a matrix instead of accepting it.
    isempty(A) && return A

    scale = maximum(abs, A)
    (scale > 0 && isfinite(scale)) || return nothing
    B = A / scale

    F₁ = cholesky(Symmetric(B' * B); check = false)
    issuccess(F₁) || return nothing
    Q = B / F₁.U
    F₂ = cholesky(Symmetric(Q' * Q); check = false)
    issuccess(F₂) || return nothing

    Q / F₂.U
end

# How many Gaussian draws `_orthonormal_columns` takes before it gives up. At the rate measured in
# its docstring -- about one draw in two hundred -- eight independent draws put the chance of
# exhausting them below 1e-18, and a caller that does exhaust them is not looking at bad luck.
const ORTHONORMALIZATION_ATTEMPTS = 8

@doc raw"""
    _orthonormal_columns(draw)

Orthonormalize `draw()` with [`_cholesky_qr2`](@ref GeometricOptimizers._cholesky_qr2), drawing
again while the draw is too ill-conditioned for it.

Both callers — `rand(backend, manifold_type, N, n)` and [`global_section`](@ref) — draw their own
Gaussian matrix, which is what makes redrawing the right answer rather than a retry. The draw
carries no information, so one `CholeskyQR2` cannot orthonormalize is *replaced*; nothing is
repaired and no result is kept that the algorithm did not produce cleanly.

**The rate is not negligible.** [`global_section`](@ref) factorizes ``N\times(N-n)`` Gaussian
columns with the span of `Y` projected out, so the matrix is square inside that complement and its
condition number has the heavy tail a square Gaussian's does. Measured 2026-09-18 in `Float32`, 600
draws over ``N = 50, 100, 200`` at ``n = 3``: three broke `CholeskyQR2` down and none broke the
redraw. Shifted CholeskyQR3 was measured on the same draws and does not close it — one failure in
300 even with the exact ``\|A\|_2`` in the shift, because forming ``A^TA`` in `Float32` loses a
singular value that small whatever the shift is.
"""
function _orthonormal_columns(draw)
    for _ in 1:ORTHONORMALIZATION_ATTEMPTS
        Q = _cholesky_qr2(draw())
        Q === nothing || return Q
    end

    throw(ErrorException("orthonormalization failed on $(ORTHONORMALIZATION_ATTEMPTS) independent Gaussian draws, which at the measured rate is not bad luck; the element type is probably too narrow for this size"))
end

# TODO: check the distribution this is coming from - related to the Haar measure ???
function Base.rand(::CPU, rng::Random.AbstractRNG, ::Type{MT},
        N::Integer, n::Integer) where {T, MT <: Manifold{T}}
    @assert N ≥ n
    Q = _orthonormal_columns(() -> randn(rng, T, N, n))
    # `MT` may name the storage array type as well as the element type --
    # `StiefelManifold{Float64, Matrix{Float64}}` -- and is then already concrete, so applying a
    # further parameter to it is an error. `StiefelManifold{Float64}` still needs one. The branch
    # is on a type parameter and folds away. This used to be a second method in
    # `stiefel_manifold.jl`, written against `StiefelManifold{T, AT}` and so available to that one
    # manifold only; `Manifold{T}` has a single parameter and cannot name the storage type, which
    # is why this is a branch rather than a signature.
    (isconcretetype(MT) ? MT : MT{typeof(Q)})(Q)
end

function Base.rand(backend::GPU, rng::Random.AbstractRNG, ::Type{MT},
        N::Integer, n::Integer) where {T, MT <: Manifold{T}}
    @assert N ≥ n
    _check_supported_eltype(backend, T)
    Q = _orthonormal_columns() do
        A = KernelAbstractions.allocate(backend, T, N, n)
        Random.randn!(rng, A)
        A
    end
    # the branch on the host method above, for the same reason and with the same comment
    (isconcretetype(MT) ? MT : MT{typeof(Q)})(Q)
end

@doc raw"""
    default_eltype(backend)

The element type a `rand` that names a backend but no element type draws in: `Float64` on the host
and `Float32` on a device.

Neither value is arbitrary, which is the whole reason this is a function rather than a literal in
each of two methods. `Float64` on the host is what `zeros(n)` and `rand(n)` already give, so a
manifold drawn without an element type matches every other array drawn without one. `Float32` on a
device is the width an accelerator is built for, and a device that carries `Float64` at all
normally carries it at a fraction of the `Float32` rate.

**The rule deliberately does not ask `KernelAbstractions.supports_float64`.** A backend being
*able* to hold a `Float64` is not a reason to hand it one: a caller who has not said which width it
wants is better served by the width the device is fast at. That trait answers the other half of the
question instead — an element type the caller *does* name and the backend cannot hold is rejected
rather than narrowed, which the `rand(backend, manifold_type, N, n)` docstring states.

A caller who wants a fixed width names it, in the parametric form
`rand(backend, StiefelManifold{Float64}, N, n)`.
"""
function default_eltype end

default_eltype(::CPU) = Float64
default_eltype(::GPU) = Float32

function Base.rand(
        backend::KernelAbstractions.Backend, rng::Random.AbstractRNG, ::Type{MT},
        N::Integer, n::Integer) where {MT <: Manifold}
    rand(backend, rng, MT{default_eltype(backend)}, N, n)
end

function Base.rand(rng::Random.AbstractRNG, manifold_type::Type{MT},
        N::Integer, n::Integer) where {MT <: Manifold}
    rand(CPU(), rng, manifold_type, N, n)
end

# `_round` rewraps, and is the only thing here that may: rounding a point's entries to a few decimals
# is a *display* operation on a point that is already on the manifold, and the docstrings that print
# one need the type back. Nothing else is entitled to that — a general `broadcast(f, Y::Manifold)`
# rewrapping its result claims an invariant the result does not hold, which is why this package
# defines no `broadcast` method for `Manifold` at all and why `broadcast(f, Y)` returns a plain
# array. `CHANGELOG.md` has the measurement.
#
# `Y.A` and not `Y` is load-bearing on a device-backed point: `Manifold` declares no
# `Broadcast.BroadcastStyle`, so `round.(Y)` reaches the manifold's scalar `getindex`, which a
# device array disallows, while `round.(Y.A)` runs on the device and keeps the result there. On the
# host the two agree.
function _round(Y::Manifold; kwargs...)
    typeof(Y)(round.(Y.A; kwargs...))
end

@doc raw"""
    rand(backend, manifold_type, N, n)

Draw random elements for a specific device.

# Examples

Random elements of the manifold can be allocated on GPU.  Call ...

```julia
rand(CUDABackend(), StiefelManifold{Float32}, N, n)
```

... for drawing elements on a `CUDA` device.

# The element type

Naming it, as above, is what fixes it, and a named element type is honoured or refused — never
narrowed. `rand(MetalBackend(), StiefelManifold{Float64}, N, n)` throws an `ArgumentError` saying
the backend has no `Float64`, rather than quietly returning a `Float32` point of a type the caller
did not ask for.

A call that leaves the element type open — `rand(CUDABackend(), StiefelManifold, N, n)` — gets
[`default_eltype`](@ref GeometricOptimizers.default_eltype) of the backend: `Float64` on the host
and `Float32` on a device, for the reasons given there.

# What the backend has to supply

A matrix product, a `cholesky` and a triangular solve, all for an array of its own type. The draw
orthonormalizes a Gaussian matrix with
[`_cholesky_qr2`](@ref GeometricOptimizers._cholesky_qr2) and [`global_section`](@ref) does the same
for the complement, so those three are what a backend needs to carry a point, draw one and carry an
[`Optimizer`](@ref).

**`qr` is deliberately not among them.** It was, and that made this whole path host-only in
practice: `Metal` implements no `qr` for an `MtlArray`, so every call above failed there with
`Cannot access the contents of a private buffer`, measured on real hardware, and so did
`GlobalSection(Y)` and `Optimizer(Y, F)`. `Metal` does implement `cholesky`. CholeskyQR2 therefore
runs where Householder QR cannot, at *better* orthogonality than the host QR at every size measured
— and a `CUDA` backend, which does supply `qr` through CUSOLVER, takes the same route as everything
else rather than a second one.

# The manifolds this draws

[`StiefelManifold`](@ref) and [`GrassmannManifold`](@ref). [`SymplecticStiefelManifold`](@ref) is
drawn on the host alone: its draw is the symplectic SR decomposition [`sr!`](@ref), which is a host
factorization, so a device spelling would be a host draw followed by a transfer rather than the
device-native draw the other two get. Naming a device for it throws an `ArgumentError` saying so.
"""
function Base.rand(backend::KernelAbstractions.Backend, manifold_type::Type{MT},
        N::Integer, n::Integer) where {MT <: Manifold}
    rand(backend, Random.default_rng(), manifold_type, N, n)
end

@doc raw"""
    rand(manifold_type, N, n)

Draw random elements from the Stiefel and the Grassmann manifold.

Because both of these manifolds are compact spaces we can sample them uniformly [mezzadri2006generate](@cite).

# Examples
When we call ...

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: _round # hide
import Random
Random.seed!(123)

N, n = 5, 3
Y = rand(StiefelManifold{Float32}, N, n)
_round(Y; digits = 5) # hide

# output

5×3 StiefelManifold{Float32, Matrix{Float32}}:
 -0.27575   0.32991   0.77275
 -0.62485  -0.33224  -0.0686
 -0.69333   0.36724  -0.18988
 -0.09295  -0.73145   0.46064
  0.2102    0.33301   0.38717
```

... the sampling is done by first allocating a random matrix of size ``N\times{}n`` via `Y = randn(Float32, N, n)`.

We then orthonormalize its columns with
[`_cholesky_qr2`](@ref GeometricOptimizers._cholesky_qr2) and return those. CholeskyQR2 rather than
`LinearAlgebra.qr` — which uses Householder reflections internally — because `qr` is a host
factorization for several of the backends this package supports, and CholeskyQR2 is expressible in
products, reductions and triangular solves alone. The two answers differ by the sign of each column,
since CholeskyQR2's ``R`` has a positive diagonal and Householder's need not.
"""
function Base.rand(manifold_type::Type{MT}, N::Integer, n::Integer) where {MT <: Manifold}
    rand(Random.default_rng(), manifold_type, N, n)
end

@doc raw"""
    check(Y::Manifold)

Measure how far `Y` is from the manifold, as ``\|Y^TY - \mathbb{I}\|``.

Two of the three manifolds this package provides store a representative whose columns are
orthonormal — for [`StiefelManifold`](@ref) that is the point itself, for
[`GrassmannManifold`](@ref) it is the representative of the equivalence class — so the same
expression measures both. A retraction maps onto the manifold by construction, so in exact
arithmetic this is zero and what it actually returns is accumulated round-off.

[`SymplecticStiefelManifold`](@ref) is the third, and its constraint is a different bilinear form,
so it carries its own method. Anything defining a further manifold has to decide which of the two
it is rather than inherit this one by default.

This is the assertion the manifold tests rest on. It used to exist for [`StiefelManifold`](@ref)
only, which is why the accuracy loss in [`GeometricOptimizers.𝔄`](@ref) went unnoticed for so long:
half the retraction paths had nothing that could have caught it.

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: check
import Random
Random.seed!(123)

check(rand(GrassmannManifold, 5, 3)) < 1e-14

# output

true
```
"""
check(Y::Manifold) = norm(Y.A' * Y.A - I)

Base.size(A::Manifold) = size(A.A)
Base.parent(A::Manifold) = A.A
Base.getindex(A::Manifold, i::Int, j::Int) = A.A[i, j]
Base.copy(A::MT) where {MT <: Manifold} = MT(copy(A.A))

@doc raw"""
    manifold_constructor(x::Manifold)

The one-argument constructor of `x`'s manifold: `GrassmannManifold` for a
`GrassmannManifold{Float32, Matrix{Float32}}`.

The type *name* and not `typeof(x)`, because the array the result is applied to may have a different
element type from `x`'s: the closure `GradientAutodiff` differentiates is handed a vector of
`ForwardDiff.Dual`s. And `x`'s manifold rather than a hardcoded `StiefelManifold`, which is what used
to make a bare one a `MethodError` at [`Optimizer`](@ref) construction (issue A11).

Its two callers are `GeometricOptimizers._similar(::Manifold)` and
`GradientAutodiff(F, ::Manifold)`. It used to have a third: the flattening reconstructed a manifold
through this, and hardcoding `StiefelManifold` there turned a [`GrassmannManifold`](@ref) into a
[`StiefelManifold`](@ref) on every round trip. `NeuralNetworkParameters.rebuild` takes a *prototype*
rather than a type, so that bug class is gone from the flat path rather than guarded against.
"""
manifold_constructor(x::Manifold) = Base.typename(typeof(x)).wrapper

# No `Manifold` defines `setindex!`, so the generic `AbstractArray` `copyto!` — which routes through
# it — is not available to any of them. This method existed for `StiefelManifold` alone, which is why
# a `NamedTuple` holding a `GrassmannManifold` died with a `CanonicalIndexError` in
# `update!(::BFGSCache, …)`; see issue A11. It returns `A` and not `nothing`: that is the `copyto!`
# contract, and it is what `copyto!(::GrassmannLieAlgHorMatrix, …)` and
# `copyto!(::GlobalSection, …)` next to it already do.
#
# The species check is a runtime one because a type parameter shared by both arguments cannot
# express it: the parameter binds the whole type, storage array included, so a host `A` and a device
# `B` are different concrete types and a shared parameter excludes exactly the host-to-device
# transfer this method exists for. `manifold_constructor` right above carries the same "compare the
# type name, not the type" idiom.
function Base.copyto!(A::Manifold, B::Manifold)
    manifold_constructor(A) === manifold_constructor(B) ||
        throw(ArgumentError("cannot copyto! a $(manifold_constructor(B)) into a $(manifold_constructor(A))"))
    @assert size(A) == size(B)
    copyto!(A.A, B.A)
    A
end

function Base.similar(::Manifold)
    error("The function `similar` does not make sense in this context. Consider using rand.")
end

function Base.fill!(::Manifold, b)
    error("The function `fill!` does not make sense in this context.")
end
