# The walk the elementwise optimizer primitives take over a set of parameters.
#
# There are two shapes to cover and they want different things. A bare `NamedTuple` of arrays is
# *flat*: its entries are its leaves, and it can be very wide — the MNIST transformer of GMLDatasets.jl
# keeps 369 parameters in one. A `NetworkParameters` is a *tree* of layers, so its leaves are one or
# more levels down and `Base.map` alone would hand a primitive a whole layer.
#
# So: `map` at each level, which is cheap at any width because Base drops to a loop past 32 fields, and
# recursion on whatever turns out to be a branch. It reaches leaves at any depth, keeps the shape it was
# given — a container comes back a container — and costs the flat case exactly what `map` costs, which
# is what these primitives used to be written as.
#
# `rest` is normalised with `_as_walkable`, so a container may be walked in lockstep with the plain
# `NamedTuple` tree that `GlobalSection(::NetworkParameters)` deliberately returns. Keys still have to
# agree: `map` over `NamedTuple`s raises `ArgumentError: Named tuple names do not match.` on mismatched
# or reordered keys, unconditionally, which is the check this package relies on.
#
# ## Why this is not `NeuralNetworkParameters.mapparameters`, and why that is now only half a reason
#
# It was written here because `mapparameters` could not be compiled on the shape above. Its
# across-children walk was an `@inline`d `Base.tail` chain, which cost one specialisation per child over
# argument types each as long as the branch, so inference on it grew as the cube of the width: on the
# flat 369-entry set `map(zero, ps)` compiled in 0.01 s and `mapparameters(zero, ps)` did not finish.
#
# **`NeuralNetworkParameters` 0.2.2 fixed that**, from this package's report — the walks are written out
# now and `mapparameters` on the same set compiles in 0.00 s. So the reason this file exists is gone,
# and it should be retired in favour of upstream. Recorded under *Known issues*.
#
# The one apparent obstacle turns out to be an argument *for* retiring it. `_as_walkable` has a
# catch-all where `NeuralNetworkParameters._as_namedtuple` has three exhaustive methods, so a tree
# zipped against a *leaf* at that level raises upstream and does not raise here. That is not a
# capability: `map` over a `NamedTuple` and a bare `Vector` matches neither Base's `NamedTuple` method
# nor its `AbstractArray` one, so it falls through to the generic iterator `map`, which `zip`s and
# `collect`s -- pairing the branch's *entries* against the leaf's *elements* and handing back an
# `Array`. Measured: `map(f, (a = [1.0], b = [2.0]), [9.0, 9.0])` gives
# `[([1.0], 9.0), ([2.0], 9.0)]`. A leaf shorter than the branch is silently truncated instead.
#
# So the catch-all converts a caller's bug into a wrong answer, where upstream's exhaustive methods
# raise a `MethodError` naming the type. Nothing in this package pairs a leaf with a branch, so nothing
# depends on either behaviour -- but if something ever does, it wants explicit broadcast semantics (the
# shape `GeometricMachineLearning`'s `_tree_optim_step!` hand-writes, where one `GlobalSection` stands
# in for a whole subtree), not a fallthrough in a zip normaliser.

# `NeuralNetworkParameters._as_namedtuple` is the same function, minus the catch-all -- see above for
# why that is the better of the two.
_as_walkable(x::NetworkParameters) = params(x)
_as_walkable(x) = x

"""
    _mapleaves(f, a, rest...)

Apply `f` to corresponding leaves of `a` and `rest`, returning a set of `a`'s shape.

See the comment at the head of `src/parameter_walks.jl` for why this exists rather than
`NeuralNetworkParameters.mapparameters`.
"""
_mapleaves(f, a::NetworkParameters, rest...) =
    NetworkParameters(_mapleaves(f, params(a), map(_as_walkable, rest)...))

_mapleaves(f, a::NamedTuple, rest...) =
    map((xs...) -> _mapleaves(f, xs...), a, map(_as_walkable, rest)...)

_mapleaves(f, a, rest...) = f(a, rest...)

"""
    _mapleaves!(f, a, rest...)

[`_mapleaves`](@ref) for an `f` that is called for its effect on the leaves of `a`, returning `a`
itself rather than the tree of results.

# Implementation

`NeuralNetworkParameters.foreachparameters` and not [`_mapleaves`](@ref) with the answer thrown away,
which is what this used to be: the tree of results is one `NamedTuple` per branch, allocated and
immediately discarded on every call of every in-place primitive in
`src/optimizers/named_tuple_wrapper.jl`. On the flat `NamedTuple` problem of
`scripts/optimizer_allocations.jl` that was 992 bytes per `update!`.

Upstream and not `foreach` written out here, for the reason the head of this file gives about keys.
`Base.foreach` over `NamedTuple`s goes through `zip`, which iterates values and so neither checks that
the keys agree nor notices that one tree is shorter — where `foreachparameters` carries
`_check_keys` and is the walk `map`'s `ArgumentError` was standing in for. It is also the half of
upstream this file can already use: it takes whole leaves, it is allocation free, and since
`NeuralNetworkParameters` 0.2.2 it compiles on a wide-flat branch.
"""
_mapleaves!(f, a, rest...) = (foreachparameters(f, a, map(_as_walkable, rest)...); a)
