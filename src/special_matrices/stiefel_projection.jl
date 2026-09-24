@doc raw"""
    StiefelProjection(backend, T, N, n)

Make a matrix of the form ``\begin{bmatrix} \mathbb{I} & \mathbb{O} \end{bmatrix}^T`` for a specific backend and data type.

An array that essentially does `vcat(I(n), zeros(N-n, n))` with GPU support. 

# Extended help

An instance of `StiefelProjection` should technically also belong to [`StiefelManifold`](@ref). 
"""
struct StiefelProjection{T, AT} <: AbstractMatrix{T}
    N::Int
    n::Int
    A::AT
    # `backend` is annotated rather than left open, and both of this type's own callers pass a
    # `KernelAbstractions.Backend` through `get_backend`. Naming the type is what lets the element
    # type be checked against it; anything else reaches `KernelAbstractions.zeros` on the next line
    # and fails there regardless.
    function StiefelProjection(
            backend::KernelAbstractions.Backend, T::Type, N::Integer, n::Integer)
        _check_supported_eltype(backend, T)
        A = KernelAbstractions.zeros(backend, T, N, n)
        assign_ones_for_stiefel_projection! = assign_ones_for_stiefel_projection_kernel!(backend)
        # one work item per diagonal entry, and an `N × n` matrix has `min(N, n)` of them
        assign_ones_for_stiefel_projection!(A, ndrange = min(N, n))
        new{T, typeof(A)}(N, n, A)
    end

    # The host constructor allocates and fills in one step, with no backend and no kernel launch:
    # `Matrix{T}(I, N, n)` is exactly the matrix the docstring above describes. Routing through
    # `StiefelProjection(CPU(), T, N, n)` instead allocates through `KernelAbstractions.zeros` and
    # then starts a kernel to write `n` ones. A host placement is
    # the common case here and must not pay for the device machinery: `KernelAbstractions.zeros` on
    # a `CPU` costs an overhead at every length, growing with it, and between about 1.9x and 25x
    # the time of `zeros` -- worst at the smallest lengths, where its fixed floor dominates. The
    # comment on `zeros(::Type{AT}, n)` in `triangular.jl` has the measurement, and
    # `scripts/host_allocation_cost.jl` is the script. All of that is before the kernel launch.
    function StiefelProjection(N::Integer, n::Integer, T::Type = Float64)
        A = Matrix{T}(I, N, n)
        new{T, typeof(A)}(N, n, A)
    end
end

@doc raw"""
    StiefelProjection(A::AbstractMatrix)

Extract necessary information from `A` and build an instance of `StiefelProjection`. 

Necessary information here refers to the backend, the data type and the size of the matrix.
"""
function StiefelProjection(A::AbstractMatrix{T}) where {T}
    StiefelProjection(KernelAbstractions.get_backend(A), T, size(A)...)
end

@kernel function assign_ones_for_stiefel_projection_kernel!(A::AbstractArray{T}) where {T}
    i = @index(Global)
    A[i, i] = one(T)
end

StiefelProjection(T::Type, N::Integer, n::Integer) = StiefelProjection(N, n, T)

# The host constructor is what a `CPU` backend reaches. Every caller in the package names a backend,
# through `get_backend` on an array or on a horizontal lift, so without this method the argument the
# inner constructor's own comment makes for the host form never applies on the host — the backend
# arm answers every call. `Matrix{T}(I, N, n)` is the same `Matrix{T}` the backend arm returns
# there, so nothing about the returned object changes -- only that it is built in one allocation
# rather than in a `KernelAbstractions.zeros` plus a kernel launch.
#
# `_check_supported_eltype` is called here too, although `supports_float64(CPU())` is `true` and so
# it can never fire, for the reason the host arm of `unit_matrix` gives in `src/utils.jl`: the
# invariant that file's header states is that *every* allocator a caller reaches by naming a backend
# and an element type calls it, and `StiefelProjection` is on the list it names. Leaving it out here
# would take this constructor off that list while the header still claimed it.
function StiefelProjection(
        backend::KernelAbstractions.CPU, T::Type, N::Integer, n::Integer)
    _check_supported_eltype(backend, T)

    StiefelProjection(N, n, T)
end

Base.size(E::StiefelProjection) = (E.N, E.n)
Base.getindex(E::StiefelProjection, i, j) = getindex(E.A, i, j)

@doc raw"""
    *(E::StiefelProjection, A::AbstractMatrix)
    *(A::AbstractMatrix, E::StiefelProjection)
    *(E::StiefelProjection, b::AbstractVector)

The product, taken on the wrapped array.

`StiefelProjection` holds its entries in an ordinary array, so unwrapping is all these do — for `E`
and `E'` alike, and for `+`, `-`, `mul!` and a scalar product as well, with the same methods that a
manifold point uses. Without them the product falls through to the generic `AbstractMatrix` path,
which reaches `getindex` one entry at a time. **That is scalar indexing, and it is what stops a
retraction on a device**: [`geodesic`](@ref) and [`cayley`](@ref) each take one
product against the projection — `expB * E` and `cayleyB * E` — with `E` built from the horizontal
lift and so carrying the point's own backend. Both operands are on the device, and only the wrapper
puts the product on the host path.

Both operands have to be on one backend: a pair on two backends raises an `ArgumentError` naming
both, rather than answering on whichever backend the argument order picks.
"""
Base.:*(::StiefelProjection, ::AbstractMatrix)

function Base.vcat(A::AbstractVecOrMat{T}, E::StiefelProjection{T}) where {T <: Number}
    vcat(A, E.A)
end
function Base.vcat(E::StiefelProjection{T}, A::AbstractVecOrMat{T}) where {T <: Number}
    vcat(E.A, A)
end
function Base.hcat(A::AbstractVecOrMat{T}, E::StiefelProjection{T}) where {T <: Number}
    hcat(A, E.A)
end
function Base.hcat(E::StiefelProjection{T}, A::AbstractVecOrMat{T}) where {T <: Number}
    hcat(E.A, A)
end

function KernelAbstractions.get_backend(E::StiefelProjection)
    KernelAbstractions.get_backend(E.A)
end
