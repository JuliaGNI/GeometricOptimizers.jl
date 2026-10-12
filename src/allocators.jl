# One allocator chain for every owned array and manifold type:
#
#     zeros([backend,] X{T}, dims::Integer...)
#     rand([rng,] [backend,] X{T}, dims::Integer...)
#
# Each owned type has two methods, `zeros(backend, X{T}, dims...)` and
# `rand(rng, backend, X{T}, dims...)`, next to its definition; a manifold has the `rand` alone. The
# methods below supply what a call leaves out: `rng = Random.default_rng()`, `backend = CPU()`, and
# `T = default_eltype(backend)` for a bare `X`.
#
# The per-type methods take exactly `Type{X{T}}`, so a type that names its storage as well --
# `SkewSymMatrix{Float32, Vector{Float32}}`, or `SkewSymMatrix{Float32, JLArray{Float32, 1}}` on
# the host -- matches none of them and is a `MethodError` rather than a result of another type, even
# where the backend gives that storage. A manifold's `rand` is the exception: it takes
# `MT <: Manifold{T}` and keeps a concrete `MT`. The methods that fill in `T` take the bare types
# only, so every other `X` that has no per-type method is a `MethodError` from
# dispatch.
#
# Each method takes a leading `Integer` before the `Integer...`. Without it the zero-dimension call
# `rand(X)` is a method of these as well, and `Random`'s `rand(::Type{X})` is ambiguous with it.
const _OwnedArrayType = Union{SkewSymMatrix, SymmetricMatrix, AbstractTriangular,
    AbstractLieAlgHorMatrix}
const _OwnedAllocType = Union{_OwnedArrayType, Manifold}

function Base.zeros(::Type{X}, d::Integer, dims::Integer...) where {X <: _OwnedArrayType}
    zeros(CPU(), X, d, dims...)
end

# One method per bare type, generated, and not one method on a `Union` of `Type`s: Julia does not
# specialize on an argument of such a `Union`, and the result would infer as `Any`.
for X in (SkewSymMatrix, SymmetricMatrix, StrictlyLowerTriangular, StrictlyUpperTriangular,
    StiefelLieAlgHorMatrix, GrassmannLieAlgHorMatrix,
    StiefelManifold, GrassmannManifold, SymplecticStiefelManifold)
    X <: Manifold || @eval function Base.zeros(backend::KernelAbstractions.Backend,
            ::Type{$X}, d::Integer, dims::Integer...)
        zeros(backend, $X{default_eltype(backend)}, d, dims...)
    end
    @eval function Base.rand(rng::AbstractRNG, backend::KernelAbstractions.Backend,
            ::Type{$X}, d::Integer, dims::Integer...)
        rand(rng, backend, $X{default_eltype(backend)}, d, dims...)
    end
end

function Base.rand(::Type{X}, d::Integer, dims::Integer...) where {X <: _OwnedAllocType}
    rand(Random.default_rng(), CPU(), X, d, dims...)
end

function Base.rand(rng::AbstractRNG, ::Type{X}, d::Integer,
        dims::Integer...) where {X <: _OwnedAllocType}
    rand(rng, CPU(), X, d, dims...)
end

function Base.rand(backend::KernelAbstractions.Backend, ::Type{X}, d::Integer,
        dims::Integer...) where {X <: _OwnedAllocType}
    rand(Random.default_rng(), backend, X, d, dims...)
end

# The storage every per-type method allocates.
#
# The host spelling is `zeros(T, dims...)` and not `KernelAbstractions.zeros(CPU(), T, dims...)`:
# `KernelAbstractions.zeros` on a `CPU` returns the same `Array{T}` with the same values and costs
# more at every length, in bytes and in time. It fills rather than reaching `calloc`, so it loses the
# zero page, and the gap grows with the length. `scripts/host_allocation_cost.jl` measures it. The
# host path is the common one here and must not pay for the device machinery.
_zeros(::CPU, ::Type{T}, dims::Integer...) where {T} = zeros(T, dims...)
function _zeros(backend::KernelAbstractions.Backend, ::Type{T}, dims::Integer...) where {T}
    KernelAbstractions.zeros(backend, T, dims...)
end

# `allocate` and `rand!` on every backend: on the host that is what `rand(rng, T, dims...)` does, so
# the host draw is the same numbers whether or not the call names `CPU()`.
function _rand(rng::AbstractRNG, backend::KernelAbstractions.Backend, ::Type{T},
        dims::Integer...) where {T}
    rand!(rng, KernelAbstractions.allocate(backend, T, dims...))
end
