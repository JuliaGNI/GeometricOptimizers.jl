# A computation needs both of its operands on one backend. This says so, and names both operand
# types and both backends. Its return value is the shared backend; no call site reads it, because a
# site launches on the backend of one named operand and that operand is not always this function's
# first argument.
#
# Without the guard a mismatch is not a clean failure. On a backend that can fall back to the host —
# `JLArrays` under `allowscalar(false)` is the one reachable from the test suite — a host/device pair
# silently picks a side: `SkewSymMatrix(host) + SkewSymMatrix(device)` gives a device matrix,
# `SkewSymMatrix(device) * host` a `JLArray`, and `StiefelManifold(device) * host` a **host**
# `Matrix` — the device operand comes off the device and nothing says so. Which backend the answer
# lands on follows the argument order. Where such a pair does raise instead, it says
# `Scalar indexing is disallowed`, which names neither operand; on Metal it fails inside
# `GPU compilation of MethodInstance for …broadcast_linear…`, which names neither backend nor the
# mismatch.
#
# So this is not only a better message: on a backend that can fall back to the host it is the
# difference between an answer computed somewhere the caller did not choose and no answer.
#
# `copyto!`, `assign!` and `changebackend` are exempt and must stay exempt — a transfer's whole
# purpose is to cross backends, which is the rule `Base` already sets for `copyto!`. `_match_backend`
# in `abstract_manifold.jl` is the third exemption and is deliberate: `rgrad` moves the gradient onto
# the point's backend rather than refusing, because the point is the parameter.
#
# Both reads fold away for concrete array types, so a same-backend call pays nothing. An array that
# `KernelAbstractions.get_backend` cannot place raises that function's own `ArgumentError`.
function _check_same_backend(A, B)
    backend_a = KernelAbstractions.get_backend(A)
    backend_b = KernelAbstractions.get_backend(B)
    backend_a == backend_b && return backend_a

    throw(ArgumentError("mixed backends: $(nameof(typeof(A))) is on $(backend_a) and $(nameof(typeof(B))) is on $(backend_b). A computation needs both operands on one backend; move one with `copyto!` or `changebackend` first."))
end

# The destination of a `mul!` or an `add!` is the third array on the backend.
_check_same_backend(A, B, C) = (_check_same_backend(A, B); _check_same_backend(A, C))

# Writes the diagonal of an identity matrix. `unit_matrix` below is the only caller; a kernel is what
# it takes to write a diagonal without scalar indexing, and that docstring says why that matters.
@kernel function write_ones_kernel!(matrix::AbstractMatrix{T}) where {T}
    i = @index(Global)
    matrix[i, i] = one(T)
end

@doc raw"""
    unit_matrix(backend, T, n)
    unit_matrix(A::AbstractMatrix)

The ``n\times{}n`` identity of element type `T` on `backend`, with the diagonal written by
`write_ones_kernel!` above.

`Base.one(::AbstractMatrix)` is the natural spelling and is *not* used: `Base._one` allocates and
then writes the diagonal in a **scalar-indexed loop**, which is precisely what an array on a
`KernelAbstractions` backend cannot serve. It is the same hazard
[`GeometricOptimizers.opnorm₁`](@ref) exists to avoid one level up, and every identity this package
builds goes through here rather than through `Base.one` — `Base.one` for
[`SymmetricMatrix`](@ref), [`SkewSymMatrix`](@ref), the two [`AbstractTriangular`](@ref)s and
[`AbstractLieAlgHorMatrix`](@ref), and the ``2n\times{}2n`` identities that
[`GeometricOptimizers.𝔄`](@ref) and [`NativePade`](@ref) need.

`LinearAlgebra.I` covers some of those uses and is not enough either. `GPUArrays` supplies a
kernel-based `+(::AbstractGPUMatrix, ::UniformScaling)`, so `X + I` is portable on the array types
that package covers — but `KernelAbstractions` is the interface this package is written against, and
a backend of its own is under no obligation to be one of them.

The matrix form takes the backend and the element type from `A` and its size from
`LinearAlgebra.checksquare`, so it throws on a non-square argument exactly as `Base.one` does.
"""
function unit_matrix(backend::KernelAbstractions.Backend, ::Type{T}, n::Integer) where {T}
    matrix = KernelAbstractions.zeros(backend, T, n, n)
    write_ones! = write_ones_kernel!(backend)
    write_ones!(matrix; ndrange = n)

    matrix
end

# The host arm allocates and fills in one step, with no kernel launch, and returns the same
# `Matrix{T}` the generic arm above returns on a `CPU`. This is the argument
# `StiefelProjection(T, N, n)` already makes in `src/special_matrices/stiefel_projection.jl`, and the
# retraction path is where it is worth making: `KernelAbstractions.zeros` on a `CPU` writes the zeros
# itself rather than taking a page the operating system has already zeroed
# (`scripts/host_allocation_cost.jl`), and the launch then costs three allocations of
# `KernelAbstractions.CompilerMetadata` per call regardless of `n`. A `cayley` of a horizontal lift
# builds two of these identities and is called once per line-search trial.
function unit_matrix(::KernelAbstractions.CPU, ::Type{T}, n::Integer) where {T}
    Matrix{T}(I, n, n)
end

function unit_matrix(A::AbstractMatrix{T}) where {T}
    unit_matrix(KernelAbstractions.get_backend(A), T, LinearAlgebra.checksquare(A))
end

# `A .+ B` and not `A + B`: the latter is a *call*, so it materialises a whole temporary array and
# then copies it into `C`, which is one allocation of `C`'s size per call for a function whose entire
# purpose is to write into a destination the caller already owns. Fused into the broadcast on the
# left, nothing is allocated at all.
#
# `axes` and not `size`, for the same reason `GeometricBase.L2norm(x, y)` compares `axes`: two arrays
# of equal `size` can have different indices, and it is the indices this broadcast pairs.
function add!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::AbstractVecOrMat)
    @assert axes(A) == axes(B) == axes(C)
    # every structured `add!` unwraps to this one, so the three-way check is written once here
    _check_same_backend(A, B, C)
    C .= A .+ B
end

# There used to be a parameter-set arm of `add!` here, recursing over the leaves. Nothing in `src/`,
# `test/`, `docs/` or `scripts/` ever called it, and `GeometricMachineLearning` deleted its own
# `NamedTuple` arm of `AbstractNeuralNetworks.add!` for the same reason in 0.7 -- so this had the
# distinction of being the only signature in 0.6.0 that got *narrower* rather than wider. It was
# `::NamedTuple` and became `::NetworkParameters`, which drops a nested plain `NamedTuple` and any
# layer whose weights do not share one element type: exactly the two cases every other primitive in
# this release went out of its way to keep. Deleted rather than widened, since widening dead code only
# makes it harder to notice.

(grad::Gradient{T})(x::Manifold{T}) where {T} = rgrad(x, reshape(grad(vec(x)), size(x)...))

# `Manifold` and not `StiefelManifold`: hardcoding the latter is what made a bare
# `GrassmannManifold` a `MethodError` at `Optimizer` construction (issue A11). The manifold is
# rebuilt with `manifold_constructor` and not with `typeof(x)`, for the reason that function gives:
# the argument this closure is called on is a vector of `ForwardDiff.Dual`s, whose element type is
# not `x`'s.
#
# Both of these dispatch on `Manifold`, which is this package's type, so neither is type piracy; the
# `Matrix` pair that used to stand below them was. `GradientAutodiff(F, ::AbstractMatrix)` is
# `SimpleSolvers`' own method as of 0.13.2 -- the same body with the `Manifold` reconstruction taken
# out -- and this one takes precedence over it for a manifold. The functor is [`RiemannianGradient`](@ref)
# below.
function GradientAutodiff(F, x::Manifold)
    GradientAutodiff(_x -> F(manifold_constructor(x)(reshape(_x, size(x)...))), vec(x))
end

@doc raw"""
    RiemannianGradient(gradient) <: SimpleSolvers.Gradient

A [`SimpleSolvers.Gradient`](@extref) whose value is projected onto the tangent space of the point it
is evaluated at.

`gradient` is the Euclidean gradient, over the *flattened* iterate; the functor evaluates it there and
then applies [`rgrad`](@ref) — leaf by leaf for a parameter set, whole for a matrix. Vector calls are
forwarded unchanged, so a line search that works in coordinates sees the inner gradient and nothing
else.

# Implementation

This type exists to own a signature. The projection used to be written as methods on
`SimpleSolvers.Gradient` itself, which for a `Matrix` and for a parameter set was type piracy: the
function is `SimpleSolvers`' and neither `Base.Matrix` nor
[`NeuralNetworkParameters.NetworkParameters`](@extref) is this package's, so any package that loaded
this one changed what calling *any* gradient on either of those meant. Unlike the two `Gradient`
constructors that went upstream in 0.6.1, these bodies cannot follow them: `rgrad` is this package's
Riemannian projection, and `SimpleSolvers` neither has it nor should. Wrapping is the other way to
de-pirate a method, and it is the shape `GeometricMachineLearning` already uses for its
`_GMLGradient`. See issue #16.

[`Optimizer`](@ref) wraps for you, and only where it changes something — see `_riemannian_gradient`.
A `Manifold` iterate needs no wrapper, because `Manifold` is this package's type and the method above
is therefore already owned; wrapping one is harmless all the same, since the functor below leaves that
method to handle it.
"""
struct RiemannianGradient{T, GT <: Gradient{T}} <: Gradient{T}
    gradient::GT
end

# Julia's own outer constructor solves `T` out of `GT <: Gradient{T}`, so `RiemannianGradient(grad)`
# needs no method of its own. This one makes wrapping idempotent, so that `_riemannian_gradient` may
# be applied to a gradient a caller has already wrapped.
RiemannianGradient(gradient::RiemannianGradient) = gradient

# Keep the observation wrapper around the flat Euclidean gradient, inside the Riemannian wrapper.
# A whole parameter set dispatches specifically on `RiemannianGradient`, so wrapping outside it
# would hide that method; placing it here also measures differentiation without charging the
# leaf-wise tangent projection to the gradient/AD phase.
_observed_gradient(grad::RiemannianGradient, ::NoStepObserver) = grad
function _observed_gradient(grad::RiemannianGradient, observer)
    RiemannianGradient(_observed_gradient(grad.gradient, observer))
end

# The coordinate interface forwards. Only the two-argument form is needed: `SimpleSolvers`'
# `(grad::Gradient)(x::AbstractVector)` allocates a gradient and calls this.
function (grad::RiemannianGradient{T})(g::AbstractVector{T}, x::AbstractVector{T}) where {T}
    grad.gradient(g, x)
end

# `SimpleSolvers.Gradient` guarantees a functor and no field, so `grad.F` reaches a convention its
# three concrete subtypes happen to share rather than an interface. Every wrapper in this package
# therefore has to be asked for the objective instead of read for it -- `NewtonState`'s
# `update!` is the one consumer that needs the objective and holds only a gradient.
_objective(grad::Gradient) = grad.F
_objective(grad::RiemannianGradient) = _objective(grad.gradient)

# `Matrix` and not `AbstractMatrix`: widening it would make a `RiemannianGradient` called on a
# `Manifold` ambiguous against `(::Gradient)(::Manifold)` above, which is the method that rebuilds the
# manifold and so the one that has to win.
function (grad::RiemannianGradient{T})(x::Matrix{T}) where {T}
    rgrad(x, reshape(grad.gradient(vec(x)), size(x)...))
end

# Wrap only where the wrapper changes something. A plain vector iterate has no projection to apply,
# and a `Manifold` reaches the owned method above either way; a parameter set is the case that needs
# it, because its leaves are projected one at a time and `SimpleSolvers` has no method that reaches
# them.
_riemannian_gradient(grad::Gradient, ::Union{AbstractVector, Manifold}) = grad
_riemannian_gradient(grad::Gradient, ::NetworkParameters) = RiemannianGradient(grad)

function compute_new_iterate!(
        xₖ₁::Manifold{T}, xₖ::Manifold{T}, α::T, pₖ::AbstractLieAlgHorMatrix{T},
        cache::OptimizerCache{T}, retraction_type::AbstractRetraction) where {T}
    _retraction(x) = retraction(retraction_type, x)
    update_section!(section(cache), α * pₖ, _retraction)
    apply_section!(xₖ₁, section(cache), xₖ)
end

function compute_new_iterate!(xₖ::Manifold{T}, α::T, pₖ::AbstractLieAlgHorMatrix{T},
        cache::OptimizerCache{T}, retraction_type::AbstractRetraction) where {T}
    compute_new_iterate!(xₖ, xₖ, α, pₖ, cache, retraction_type)
end

function compute_new_iterate!(
        xₖ::AbstractVector{T}, x::AbstractVector{T}, α::T, pₖ::AbstractVector{T},
        cache::OptimizerCache{T}, retraction_type::AbstractRetraction) where {T}
    _retraction(x) = retraction(retraction_type, x)
    update_section!(section(cache), α * pₖ, _retraction)
    apply_section!(xₖ, section(cache), x)
end

global_section(::AbstractVecOrMat) = nothing
