module NeuralNetworkParametersExt

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

using GeometricOptimizers: Manifold, StiefelManifold, GrassmannManifold,
                           SymmetricMatrix, SkewSymMatrix, AbstractTriangular,
                           LowerTriangular, UpperTriangular,
                           AbstractLieAlgHorMatrix, StiefelLieAlgHorMatrix,
                           GrassmannLieAlgHorMatrix

import NeuralNetworkParameters: freeparameters, rebuild, parameter_metadata,
                                register_parameter_type!

# The matrices that keep their degrees of freedom in a flat vector: a symmetric or skew-symmetric
# ``n \times n`` matrix stores ``n(n+1)/2`` or ``n(n-1)/2`` numbers, a triangular one likewise. It is
# those numbers that belong in a flat parameter vector, not the entries of the dense interface —
# which do not even have the right length, and for the skew-symmetric and triangular types cannot be
# broadcast through at all.
const StorageMatrix = Union{SymmetricMatrix, SkewSymMatrix, AbstractTriangular}

# One method covers all three families: this package already exposes exactly this relation as
# `Base.parent` — `A.S` for the storage matrices, `A.A` for a manifold element, and the tuple of
# blocks `(A, B)` / `(B,)` for a horizontal lift.
freeparameters(x::Union{Manifold, StorageMatrix, AbstractLieAlgHorMatrix}) = parent(x)

rebuild(::StiefelManifold, data) = StiefelManifold(data)
rebuild(::GrassmannManifold, data) = GrassmannManifold(data)

rebuild(A::SymmetricMatrix, data) = SymmetricMatrix(data, A.n)
rebuild(A::SkewSymMatrix, data) = SkewSymMatrix(data, A.n)
rebuild(A::LowerTriangular, data) = LowerTriangular(data, A.n)
rebuild(A::UpperTriangular, data) = UpperTriangular(data, A.n)

# The blocks arrive in the order `parent` returned them. `A` is itself a `SkewSymMatrix`, so it has
# already been rebuilt by the time this sees it.
rebuild(A::StiefelLieAlgHorMatrix, data) = StiefelLieAlgHorMatrix(data[1], data[2], A.N, A.n)
rebuild(A::GrassmannLieAlgHorMatrix, data) = GrassmannLieAlgHorMatrix(data[1], A.N, A.n)

# What `rebuild` takes from its prototype and a file has no prototype to take it from. `n` does
# follow from `length(S)` for the storage matrices, but only by solving a quadratic that differs per
# family, so it is cheaper and less brittle to write it down. A manifold element needs nothing: its
# storage is the dense matrix.
parameter_metadata(A::StorageMatrix) = (n = A.n,)
parameter_metadata(A::AbstractLieAlgHorMatrix) = (N = A.N, n = A.n)

# Registering the types is what lets `load(NetworkParameters, h5)` rebuild them with no prototype to
# work from. Nothing in `NeuralNetworkParameters` has to know these types exist.
function __init__()
    register_parameter_type!("StiefelManifold", (S, md) -> StiefelManifold(S))
    register_parameter_type!("GrassmannManifold", (S, md) -> GrassmannManifold(S))
    register_parameter_type!("SymmetricMatrix", (S, md) -> SymmetricMatrix(S, md.n))
    register_parameter_type!("SkewSymMatrix", (S, md) -> SkewSymMatrix(S, md.n))
    register_parameter_type!("LowerTriangular", (S, md) -> LowerTriangular(S, md.n))
    register_parameter_type!("UpperTriangular", (S, md) -> UpperTriangular(S, md.n))
    register_parameter_type!("StiefelLieAlgHorMatrix",
        (S, md) -> StiefelLieAlgHorMatrix(S[1], S[2], md.N, md.n))
    register_parameter_type!("GrassmannLieAlgHorMatrix",
        (S, md) -> GrassmannLieAlgHorMatrix(S[1], md.N, md.n))
end

end
