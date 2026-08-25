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
function unit_matrix(backend, ::Type{T}, n::Integer) where {T}
    matrix = KernelAbstractions.zeros(backend, T, n, n)
    write_ones! = write_ones_kernel!(backend)
    write_ones!(matrix; ndrange=n)

    matrix
end

function unit_matrix(A::AbstractMatrix{T}) where {T}
    unit_matrix(KernelAbstractions.get_backend(A), T, LinearAlgebra.checksquare(A))
end

function add!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::AbstractVecOrMat)
    @assert size(A) == size(B) == size(C)
    C .= A + B
end

function add!(dx₁::ParameterContainer, dx₂::ParameterContainer, dx₃::ParameterContainer)
    _mapleaves!(add!, dx₁, dx₂, dx₃)
end

(grad::Gradient{T})(x::Manifold{T}) where {T} = rgrad(x, reshape(grad(vec(x)), size(x)...))

# `Manifold` and not `StiefelManifold`: hardcoding the latter is what made a bare
# `GrassmannManifold` a `MethodError` at `Optimizer` construction (issue A11). The manifold is
# rebuilt with `manifold_constructor` and not with `typeof(x)`, for the reason that function gives:
# the argument this closure is called on is a vector of `ForwardDiff.Dual`s, whose element type is
# not `x`'s.
GradientAutodiff(F, x::Manifold) = GradientAutodiff(_x -> F(manifold_constructor(x)(reshape(_x, size(x)...))), vec(x))

(grad::Gradient{T})(x::Matrix{T}) where {T} = rgrad(x, reshape(grad(vec(x)), size(x)...))
GradientAutodiff(F, x::Matrix{T}) where {T} = GradientAutodiff(_x -> F(reshape(_x, size(x)...)), vec(x))

function compute_new_iterate!(xₖ₁::Manifold{T}, xₖ::Manifold{T}, α::T, pₖ::AbstractLieAlgHorMatrix{T}, cache::OptimizerCache{T}, retraction_type::AbstractRetraction) where {T}
    _retraction(x) = retraction(retraction_type, x)
    update_section!(section(cache), α * pₖ, _retraction)
    apply_section!(xₖ₁, section(cache), xₖ)
end

compute_new_iterate!(xₖ::Manifold{T}, α::T, pₖ::AbstractLieAlgHorMatrix{T}, cache::OptimizerCache{T}, retraction_type::AbstractRetraction) where {T} = compute_new_iterate!(xₖ, xₖ, α, pₖ, cache, retraction_type)

function compute_new_iterate!(xₖ::AbstractVector{T}, x::AbstractVector{T}, α::T, pₖ::AbstractVector{T}, cache::OptimizerCache{T}, retraction_type::AbstractRetraction) where {T}
    _retraction(x) = retraction(retraction_type, x)
    update_section!(section(cache), α * pₖ, _retraction)
    apply_section!(xₖ, section(cache), x)
end

global_section(::AbstractVecOrMat) = nothing
