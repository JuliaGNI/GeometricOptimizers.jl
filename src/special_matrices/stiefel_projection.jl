@doc raw"""
    StiefelProjection(backend, T, N, n)

Make a matrix of the form ``\begin{bmatrix} \mathbb{I} & \mathbb{O} \end{bmatrix}^T`` for a specific backend and data type.

An array that essentially does `vcat(I(n), zeros(N-n, n))` with GPU support.

# Extended help

For ``N \geq n`` an instance of `StiefelProjection` should technically also belong to
[`StiefelManifold`](@ref): its columns are orthonormal.

Any other shape is the same matrix `Matrix{T}(I, N, n)`: ones on the diagonal and zeros elsewhere.
For ``n > N`` that is ``\begin{bmatrix} \mathbb{I} & \mathbb{O} \end{bmatrix}``, whose columns are not
orthonormal, and a projection with no rows or no columns is the empty matrix of its size. The host
and the backend constructors agree on every shape.
"""
struct StiefelProjection{T, AT} <: AbstractMatrix{T}
    N::Int
    n::Int
    A::AT
    function StiefelProjection(backend::KernelAbstractions.Backend, ::Type{T}, N::Integer,
            n::Integer) where {T}
        A = KernelAbstractions.zeros(backend, T, N, n)
        assign_ones_for_stiefel_projection! = assign_ones_for_stiefel_projection_kernel!(backend)
        # one work item per diagonal entry, and an `N × n` matrix has `min(N, n)` of them. With none
        # there is nothing to write, and Metal's launch raises a `DivideError` on an empty range.
        k = min(N, n)
        if k > 0
            assign_ones_for_stiefel_projection!(A, ndrange = k)
        end
        new{T, typeof(A)}(N, n, A)
    end

    # The host constructor allocates and fills in one step, with no backend and no kernel launch:
    # `Matrix{T}(I, N, n)` is exactly the matrix the docstring above describes. The backend
    # constructor above instead allocates through `KernelAbstractions.zeros` and then starts a
    # kernel to write `min(N, n)` ones; the `CPU` method below routes around it. A host placement is
    # the common case here and must not pay for the device machinery: `KernelAbstractions.zeros` on
    # a `CPU` costs an overhead at every length, growing with it, and between about 1.9x and 25x
    # the time of `zeros` -- worst at the smallest lengths, where its fixed floor dominates. The
    # comment on `_zeros` in `allocators.jl` has the measurement, and
    # `scripts/host_allocation_cost.jl` is the script. All of that is before the kernel launch.
    function StiefelProjection(::Type{T}, N::Integer, n::Integer) where {T}
        A = Matrix{T}(I, N, n)
        new{T, typeof(A)}(N, n, A)
    end
end

StiefelProjection(N::Integer, n::Integer) = StiefelProjection(Float64, N, n)

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

# The host constructor is what a `CPU` backend reaches. Every caller in the package names a backend,
# through `get_backend` on an array or on a horizontal lift, so without this method the argument the
# inner constructor's own comment makes for the host form never applies on the host — the backend
# arm answers every call. `Matrix{T}(I, N, n)` is the same `Matrix{T}` the backend arm returns
# there, so nothing about the returned object changes -- only that it is built in one allocation
# rather than in a `KernelAbstractions.zeros` plus a kernel launch.
function StiefelProjection(::CPU, ::Type{T}, N::Integer, n::Integer) where {T}
    StiefelProjection(T, N, n)
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
