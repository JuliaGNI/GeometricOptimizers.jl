# Kernel that is needed for functions relating to `SymmetricMatrix` and `SkewSymMatrix`
@kernel function write_ones_kernel!(unit_matrix::AbstractMatrix{T}) where {T}
    i = @index(Global)
    unit_matrix[i, i] = one(T)
end

function apply_toNT(fun, ps::NamedTuple...)
    for p in ps
        @assert keys(ps[1]) == keys(p)
    end
    NamedTuple{keys(ps[1])}(fun(p...) for p in zip(ps...))
end

function add!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::AbstractVecOrMat)
    @assert size(A) == size(B) == size(C)
    C .= A + B
end

function add!(dx₁::NamedTuple, dx₂::NamedTuple, dx₃::NamedTuple)
    apply_toNT(add!, dx₁, dx₂, dx₃)
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
