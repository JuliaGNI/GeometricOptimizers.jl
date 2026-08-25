# The walk the elementwise optimizer primitives take over a set of parameters.
#
# There are two shapes to cover and they want different things. A bare `NamedTuple` of arrays is
# *flat*: its entries are its leaves, and it can be very wide — the MNIST transformer of GMLDatasets.jl
# keeps 369 parameters in one. A `NetworkParameters` is a *tree* of layers, so its leaves are one or
# more levels down and `Base.map` alone would hand a primitive a whole layer.
#
# `NeuralNetworkParameters` supplies `mapparameters`, which walks the tree — but it is an `@inline`d
# `Base.tail` recursion, which is what makes it type stable and allocation free and also what makes it
# superlinear in the number of children *at one level*: a chain of `k` inlined frames goes into one
# body. On the flat 369-entry set `map(zero, ps)` compiles in 0.01 s and `mapparameters(zero, ps)` had
# not finished after twenty minutes of CPU. `scripts/walk_compile_cost.jl` is the harness; see open
# issue D9, which is where that belongs, since the recursion is upstream's.
#
# So the recursion is written here instead, over `map`: `map` at each level, which is cheap at any
# width because Base drops to a loop past 32 fields, and recursion on whatever turns out to be a
# branch. It reaches leaves at any depth, keeps the shape it was given — a container comes back a
# container — and costs the flat case exactly what `map` costs, which is what these primitives used to
# be written as.
#
# `rest` is normalised with `_as_walkable`, so a container may be walked in lockstep with the plain
# `NamedTuple` tree that `GlobalSection(::NetworkParameters)` deliberately returns. Keys still have to
# agree: `map` over `NamedTuple`s raises `ArgumentError: Named tuple names do not match.` on mismatched
# or reordered keys, unconditionally, which is the check this package relies on.

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
"""
function _mapleaves!(f, a::Union{NamedTuple,NetworkParameters}, rest...)
    _mapleaves(f, a, rest...)
    a
end
