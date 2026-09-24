# One allocator chain for every owned array and manifold type:
#
#     zeros([backend,] X{T}, dims::Integer...)
#     rand([rng,] [backend,] X{T}, dims::Integer...)
#
# Each owned type has two methods, `zeros(backend, X{T}, dims...)` and
# `rand(rng, backend, X{T}, dims...)`, next to its definition; a manifold has the `rand` alone. The
# methods below supply what a call leaves out: `rng = Random.default_rng()`, `backend = CPU()`, and
# `T = default_eltype(backend)` for a bare `X`. The per-type methods take `Type{<:X{T}}`, which a
# bare `X` does not match, so a bare `X` reaches the last method of each function here and comes back
# parametrized.
#
# Each method takes a leading `Integer` before the `Integer...`. Without it the zero-dimension call
# `rand(X)` is a method of these as well, and `Random`'s `rand(::Type{X})` is ambiguous with it.
const _OwnedAllocType = Union{SkewSymMatrix, SymmetricMatrix, AbstractTriangular,
    AbstractLieAlgHorMatrix, Manifold}

function Base.zeros(::Type{X}, d::Integer, dims::Integer...) where {X <: _OwnedAllocType}
    zeros(CPU(), X, d, dims...)
end

function Base.zeros(backend::KernelAbstractions.Backend, ::Type{X}, d::Integer,
        dims::Integer...) where {X <: _OwnedAllocType}
    zeros(backend, X{default_eltype(backend)}, d, dims...)
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

function Base.rand(
        rng::AbstractRNG, backend::KernelAbstractions.Backend, ::Type{X}, d::Integer,
        dims::Integer...) where {X <: _OwnedAllocType}
    rand(rng, backend, X{default_eltype(backend)}, d, dims...)
end

# The storage every per-type method allocates.
#
# The host spelling is `zeros(T, dims...)` and not `KernelAbstractions.zeros(CPU(), T, dims...)`:
# `KernelAbstractions.zeros` on a `CPU` returns the same `Array{T}` with the same values and charges
# for it. It fills rather than reaching `calloc`, so it loses the zero page, and the gap therefore
# grows with the length rather than being a constant. Measured by `scripts/host_allocation_cost.jl`
# on Julia 1.13 at `--check-bounds=auto`, one cold process per run: 128 B of overhead up to 500
# elements, 144 B at 1024 and 12 384 B at 2^18.
#
# The time ratio is not monotone, and the worst case is the *small* matrix rather than the large
# one. `KernelAbstractions.zeros` has a floor of about 120 ns whatever the length, so the ratio
# starts near 25x at one element, falls to about 1.9x at 1024 to 2048 elements as the host path
# grows into that floor, then rises again to about 5x at 2^18 as the fill outgrows `calloc`.
#
# Treat the figures as this machine's -- what does not move is that an overhead is paid at every
# length, that the bytes grow with the length, and that the device spelling was never the faster of
# the two at any length measured. The host path is the common one here and must not pay for the
# device machinery.
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
