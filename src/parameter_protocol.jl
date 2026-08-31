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
rebuild(A::LowerTriangular, data) = LowerTriangular(data, A.n)
rebuild(A::UpperTriangular, data) = UpperTriangular(data, A.n)

# The blocks arrive in the order `parent` returned them. `A` is itself a `SkewSymMatrix`, so it has
# already been rebuilt by the time this sees it.
function rebuild(A::StiefelLieAlgHorMatrix, data)
    StiefelLieAlgHorMatrix(data[1], data[2], A.N, A.n)
end
rebuild(A::GrassmannLieAlgHorMatrix, data) = GrassmannLieAlgHorMatrix(data[1], A.N, A.n)

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
