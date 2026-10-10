# The `NeuralNetworkParameters` leaf protocol for this package's structured matrices.
#
# `NeuralNetworkParameters` walks a parameter set over two methods per leaf type — `freeparameters`,
# saying where the differentiable numbers live, and `rebuild`, putting a leaf back together around
# them. Everything written against that protocol (flattening, the elementwise optimizer primitives,
# the HDF5 traversal) then works for these types without knowing they exist.
#
# The methods belong here and not in a package that *uses* both. `freeparameters(::SymmetricMatrix)`
# written anywhere else is piracy twice over — on `NeuralNetworkParameters`' generic and on this
# package's type — and two such packages can silently disagree. This is the arrangement
# `NeuralNetworkParameters`' own `freeparameters` docstring points at.

# One method covers all three families: this package already exposes exactly this relation as
# `Base.parent` — `A.S` for a `VectorStorageMatrix`, `A.A` for a manifold element, and the tuple of
# blocks `(A, B)` / `(B,)` for a horizontal lift. `VectorStorageMatrix` is this package's alias for
# the four types that keep their ``n(n\pm1)/2`` free parameters in one vector; its docstring says why
# those numbers and not the entries of the dense interface, which do not even have the right length
# and, for three of the four, cannot be broadcast through at all.
freeparameters(x::Union{Manifold, VectorStorageMatrix, AbstractLieAlgHorMatrix}) = parent(x)

# `freeparameters` is defined on the abstract types, `rebuild` on the concrete ones below, and all of
# these are `AbstractMatrix`es — so `NeuralNetworkParameters`' `rebuild(::AbstractArray, data) = data`
# would catch a subtype added later and hand back the bare storage, flattening and unflattening it to
# a dense matrix with no error anywhere. Say so instead. The methods below are strictly more specific,
# so they win wherever they exist.
function rebuild(x::Union{Manifold, VectorStorageMatrix, AbstractLieAlgHorMatrix}, data)
    throw(ArgumentError(string("no `rebuild` for `", typeof(x), "`. This package's ",
        "`NeuralNetworkParameters` protocol covers it with `freeparameters` but not with `rebuild`; ",
        "add the missing method next to the others in `src/parameter_protocol.jl`.")))
end

rebuild(::StiefelManifold, data) = StiefelManifold(data)
rebuild(::GrassmannManifold, data) = GrassmannManifold(data)

rebuild(A::SymmetricMatrix, data) = SymmetricMatrix(data, A.n)
rebuild(A::SkewSymMatrix, data) = SkewSymMatrix(data, A.n)
rebuild(A::StrictlyLowerTriangular, data) = StrictlyLowerTriangular(data, A.n)
rebuild(A::StrictlyUpperTriangular, data) = StrictlyUpperTriangular(data, A.n)

# The blocks arrive in the order `parent` returned them. `A` is itself a `SkewSymMatrix`, so it has
# already been rebuilt by the time this sees it.
function rebuild(A::StiefelLieAlgHorMatrix, data)
    StiefelLieAlgHorMatrix(data[1], data[2], A.N, A.n)
end
rebuild(A::GrassmannLieAlgHorMatrix, data) = GrassmannLieAlgHorMatrix(data[1], A.N, A.n)

# The storage gradient `∂L/∂S` from a cotangent `G` of the dense interface; see
# `NeuralNetworkParameters.storage_gradient`. AD pairs `G` with the interface, and a stored number
# can appear at two places in it. A `SymmetricMatrix` holds `S_ij` at `(i, j)` and `(j, i)`, so
# `∂L/∂S_ij = G_ij + G_ji` off the diagonal and `G_ii` on it; a `SkewSymMatrix` holds `S_ij` at
# `(i, j)` and `-S_ij` at `(j, i)`, so `∂L/∂S_ij = G_ij - G_ji`. Without this a flat gradient read off
# the cotangent's own storage is half the gradient off the diagonal. Both are one
# addition per entry, so exact wherever that addition is.
#
# One kernel each, so a dense `G`, an `Adjoint` and a device array take one path, and the result is
# on `G`'s backend. A cotangent of the leaf's own type is the dense matrix it represents, so its
# storage gradient doubles the off-diagonal storage. A structural tangent, which a loss that reads the
# storage field directly produces, holds `∂L/∂S` already and takes the identity that
# `NeuralNetworkParameters` defines. The triangular types store each number once and need no method.
@kernel function symmetric_storage_gradient_kernel!(S, G)
    i, j = @index(Global, NTuple)
    if i ≥ j
        S[i * (i - 1) ÷ 2 + j] = i == j ? G[i, i] : G[i, j] + G[j, i]
    end
end

@kernel function skew_storage_gradient_kernel!(S, G)
    i, j = @index(Global, NTuple)
    if i > j
        S[(i - 2) * (i - 1) ÷ 2 + j] = G[i, j] - G[j, i]
    end
end

@kernel function symmetric_storage_doubling_kernel!(S, S_G)
    i, j = @index(Global, NTuple)
    if i ≥ j
        k = i * (i - 1) ÷ 2 + j
        S[k] = i == j ? S_G[k] : S_G[k] + S_G[k]
    end
end

# `kernel!` writes each of the `len` storage entries once over an `n × n` index range, from `G`: the
# dense cotangent, or the storage of a cotangent of the leaf's own type. The result has the element
# type of the leaf `A`, which `NeuralNetworkParameters.storage_gradient` asks for, whatever the
# precision of `G`.
function _launch_storage_gradient(kernel!, A, len::Integer, n::Integer, G)
    backend = KernelAbstractions.get_backend(G)
    S = KernelAbstractions.allocate(backend, eltype(A), len)
    kernel!(backend)(S, G; ndrange = (n, n))
    S
end

function storage_gradient(A::SymmetricMatrix, G::AbstractMatrix)
    @assert size(G) == (A.n, A.n)
    S = _launch_storage_gradient(
        symmetric_storage_gradient_kernel!, A, A.n * (A.n + 1) ÷ 2, A.n, G)
    SymmetricMatrix(S, A.n)
end

function storage_gradient(A::SymmetricMatrix, G::SymmetricMatrix)
    @assert G.n == A.n
    S = _launch_storage_gradient(
        symmetric_storage_doubling_kernel!, A, length(G.S), A.n, G.S)
    SymmetricMatrix(S, A.n)
end

function storage_gradient(A::SkewSymMatrix, G::AbstractMatrix)
    @assert size(G) == (A.n, A.n)
    SkewSymMatrix(
        _launch_storage_gradient(
            skew_storage_gradient_kernel!, A, A.n * (A.n - 1) ÷ 2, A.n, G),
        A.n)
end

function storage_gradient(A::SkewSymMatrix, G::SkewSymMatrix)
    @assert G.n == A.n
    S = similar(G.S, eltype(A))
    S .= G.S .+ G.S
    SkewSymMatrix(S, A.n)
end

# A horizontal lift stores its blocks once each in the dense `[A -Bᵀ; B 0]` (`[0 -Bᵀ; B 0]` for the
# Grassmann lift), so `∂L/∂B` is the `B` block of `G` less the transpose of its `-Bᵀ` block, and the
# `A` block is a `SkewSymMatrix` cotangent. Zygote gives a lift a dense cotangent.
function storage_gradient(A::StiefelLieAlgHorMatrix, G::AbstractMatrix)
    N, n = A.N, A.n
    @assert size(G) == (N, N)
    StiefelLieAlgHorMatrix(
        storage_gradient(A.A, G[1:n, 1:n]), _lift_block_gradient(A, G, N, n),
        N, n)
end

# A cotangent of the lift's own type is the dense `[G.A -G.Bᵀ; G.B 0]` it represents, so the `A`
# block is a `SkewSymMatrix` cotangent and `∂L/∂B` is `G.B + G.B`.
function storage_gradient(A::StiefelLieAlgHorMatrix, G::StiefelLieAlgHorMatrix)
    @assert (G.N, G.n) == (A.N, A.n)
    B = similar(G.B, eltype(A))
    B .= G.B .+ G.B
    StiefelLieAlgHorMatrix(storage_gradient(A.A, G.A), B, A.N, A.n)
end

function storage_gradient(A::GrassmannLieAlgHorMatrix, G::AbstractMatrix)
    @assert size(G) == (A.N, A.N)
    GrassmannLieAlgHorMatrix(_lift_block_gradient(A, G, A.N, A.n), A.N, A.n)
end

# in the element type of the lift `A`, as `_launch_storage_gradient`
function _lift_block_gradient(A, G, N, n)
    B = similar(G, eltype(A), N - n, n)
    B .= G[(n + 1):N, 1:n] .- transpose(G[1:n, (n + 1):N])
end

# What `rebuild` takes from its prototype and a file has no prototype to take it from. `n` does
# follow from `length(S)` for the storage matrices, but only by solving a quadratic that differs per
# family, so it is cheaper and less brittle to write it down. A manifold element needs nothing: its
# storage is the dense matrix.
parameter_metadata(A::VectorStorageMatrix) = (n = A.n,)
parameter_metadata(A::AbstractLieAlgHorMatrix) = (N = A.N, n = A.n)

# Reading back a file that has no prototype to rebuild against.
#
# `load` hands a registered reconstructor `(storage, metadata)`. For a file this protocol wrote,
# `storage` is what `freeparameters` produced and `metadata` is `parameter_metadata`. There is also
# an older shape to cope with: `GeometricMachineLearning` used to write these matrices itself as a
# group tagged `gml_type`, holding the fields under their own names. `NeuralNetworkParameters` has no
# way to tell storage from metadata in such a file, so it passes the group's fields as *both* — a
# `NamedTuple` in each position. Normalising here is what keeps those files loading, and this is the
# only place that can do it, since this is where the types are.
_dense(storage) = storage isa NamedTuple ? storage.A : storage
function _vector(storage, metadata)
    storage isa NamedTuple ? (storage.S, storage.n) :
    (storage, metadata.n)
end

# The registrations live in the module's `__init__`; see the bottom of `GeometricOptimizers.jl`.
