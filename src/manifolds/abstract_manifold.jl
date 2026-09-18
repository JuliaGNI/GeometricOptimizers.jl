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

@kernel function assign_columns_kernel!(Y::AbstractMatrix{T}, A::AbstractMatrix{T}) where {T}
    i, j = @index(Global, NTuple)
    Y[i, j] = A[i, j]
end

function assign_columns(Q::AbstractMatrix{T}, N::Integer, n::Integer) where {T}
    backend = KernelAbstractions.get_backend(Q)
    Y = KernelAbstractions.allocate(backend, T, N, n)
    assign_columns! = assign_columns_kernel!(backend)
    assign_columns!(Y, Q, ndrange = size(Y))
    Y
end

# TODO: check the distribution this is coming from - related to the Haar measure ???
function Base.rand(::CPU, rng::Random.AbstractRNG, ::Type{MT},
        N::Integer, n::Integer) where {T, MT <: Manifold{T}}
    @assert N ≥ n
    A = randn(rng, T, N, n)
    Q = assign_columns(typeof(A)(qr!(A).Q), N, n)
    # `MT` may name the storage array type as well as the element type --
    # `StiefelManifold{Float64, Matrix{Float64}}` -- and is then already concrete, so applying a
    # further parameter to it is an error. `StiefelManifold{Float64}` still needs one. The branch
    # is on a type parameter and folds away. This used to be a second method in
    # `stiefel_manifold.jl`, written against `StiefelManifold{T, AT}` and so available to that one
    # manifold only; `Manifold{T}` has a single parameter and cannot name the storage type, which
    # is why this is a branch rather than a signature.
    (isconcretetype(MT) ? MT : MT{typeof(A)})(Q)
end

# A named element type a backend cannot hold is rejected and not narrowed. Narrowing would return a
# point of a different type from the one the caller asked for, which is the one thing a call that
# names its element type has ruled out; the caller would then carry `Float32` results through code
# written for `Float64` with nothing to say so.
#
# `KernelAbstractions.supports_float64` is the backend's own declaration. It answers `true` for
# every backend that does not override it, so this can only fire where a backend author has stated
# that the width is unavailable -- `Metal.jl` sets it `false`, and `CUDA` is unaffected. Without
# this the same call still fails, but further in and in the backend's words: allocating a `Float64`
# `MtlArray` raises `Metal does not support Float64 values, try using Float32 instead`, which names
# neither the manifold nor the call that asked for it.
function _check_supported_eltype(backend::KernelAbstractions.Backend, ::Type{T}) where {T}
    if T === Float64 && !KernelAbstractions.supports_float64(backend)
        throw(ArgumentError("$(backend) does not support Float64; ask for Float32 explicitly, as in rand(backend, StiefelManifold{Float32}, N, n)"))
    end
end

function Base.rand(backend::GPU, rng::Random.AbstractRNG, ::Type{MT},
        N::Integer, n::Integer) where {T, MT <: Manifold{T}}
    @assert N ≥ n
    _check_supported_eltype(backend, T)
    A = KernelAbstractions.allocate(backend, T, N, n)
    Random.randn!(rng, A)
    MT{typeof(A)}(assign_columns(typeof(A)(qr!(A).Q), N, n))
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

`qr`, for an array of its own type. The draw orthonormalises a Gaussian matrix with it, and
[`global_section`](@ref) does the same for the complement, so a backend without `qr` can hold a
point and take a step on one but cannot draw one and cannot carry an [`Optimizer`](@ref).

`CUDA` supplies `qr` through CUSOLVER. `Metal` does not: every call above fails there with
`Cannot access the contents of a private buffer`, measured on real hardware, and so do
`GlobalSection(Y)` and `Optimizer(Y, F)`. That is `Metal.jl`'s gap and not this package's, but
nothing stated the requirement, which left a reader to find it by hitting it.
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

We then perform a QR decomposition `Q, R = qr(Y)` with the `qr` function from the `LinearAlgebra` package (this is using Householder reflections internally).

The final output are then the first `n` columns of the `Q` matrix.
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
