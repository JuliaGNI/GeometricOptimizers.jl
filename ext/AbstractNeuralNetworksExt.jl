module AbstractNeuralNetworksExt

# `changebackend` for this package's structured matrices.
#
# `AbstractNeuralNetworks` moves a network between devices by walking its parameters and calling
# `changebackend` on each leaf. Its own methods cover an `AbstractArray` and the containers; the
# structured types need one of their own, because allocating on a backend and copying has to happen to
# the *storage* and the leaf has to be rebuilt around it.
#
# The methods belong here for the same reason `src/parameter_protocol.jl` does, and it is the reason
# `GeometricMachineLearning` gave for wanting to be rid of them: `changebackend` is
# `AbstractNeuralNetworks`' generic and these types are this package's, so a method on the pair has
# one owned argument on each side and is legitimate in either of those two packages -- but in a third
# package that merely uses both, it is piracy, and two such packages can silently disagree. They lived
# in `GeometricMachineLearning`'s HDF5 extension until now, which also meant that
# `changebackend(GPU(), nn)` on a network with a manifold weight was a `MethodError` unless HDF5
# happened to be loaded.
#
# One method covers every family. `mapstorage` hands the function the `freeparameters` of a leaf and
# `rebuild`s the leaf around the result, so the five hand-written methods this replaces -- and the
# horizontal lifts, which never had one -- reduce to a delegation. A type added to
# `src/parameter_protocol.jl` later is covered without a change here.

using GeometricOptimizers: Manifold, VectorStorageMatrix, AbstractLieAlgHorMatrix

import AbstractNeuralNetworks: changebackend
using AbstractNeuralNetworks: NeuralNetworkBackend

using NeuralNetworkParameters: mapstorage

function changebackend(backend::NeuralNetworkBackend,
        x::Union{Manifold, VectorStorageMatrix, AbstractLieAlgHorMatrix})
    mapstorage(y -> changebackend(backend, y), x)
end

end
