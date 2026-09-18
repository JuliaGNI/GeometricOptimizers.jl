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
        assign_ones_for_stiefel_projection!(A, ndrange = n)
        new{T, typeof(A)}(N, n, A)
    end

    # The host constructor allocates and fills in one step, with no backend and no kernel launch:
    # `Matrix{T}(I, N, n)` is exactly the matrix the docstring above describes. It used to route
    # through `StiefelProjection(CPU(), T, N, n)`, which allocates through
    # `KernelAbstractions.zeros` and then starts a kernel to write `n` ones. A host placement is
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

Necessary information here referes to the backend, the data type and the size of the matrix.
"""
function StiefelProjection(A::AbstractMatrix{T}) where {T}
    StiefelProjection(KernelAbstractions.get_backend(A), T, size(A)...)
end

@kernel function assign_ones_for_stiefel_projection_kernel!(A::AbstractArray{T}) where {T}
    i = @index(Global)
    A[i, i] = one(T)
end

StiefelProjection(T::Type, N::Integer, n::Integer) = StiefelProjection(N, n, T)

Base.size(E::StiefelProjection) = (E.N, E.n)
Base.getindex(E::StiefelProjection, i, j) = getindex(E.A, i, j)
Base.:+(E::StiefelProjection, A::AbstractMatrix) = E.A + A
Base.:+(A::AbstractMatrix, E::StiefelProjection) = +(E, A)
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
