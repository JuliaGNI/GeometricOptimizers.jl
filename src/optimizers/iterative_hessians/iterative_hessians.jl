"""
    IterativeHessian <: Hessian

An abstract type derived from [`SimpleSolvers.Hessian`](@extref).
Its main purpose is defining a supertype that encompasses [`HessianBFGS`](@ref) and [`HessianDFP`](@ref) for dispatch.
"""
abstract type IterativeHessian{T} <: Hessian{T} end

@doc raw"""
    _flat_scratch(T, g)

The flat buffers a quasi-Newton cache keeps, or `nothing` where it needs none.

``Q`` lives in the *flattened* coordinates — it is sized by the length of the flattening, it is where
the outer products that build it are formed and where [`_dot`](@ref) pairs — while the secant pair, the
right-hand side and the direction are handed around in the parameters' own representation: a
`NamedTuple`, a container, or a horizontal lift of the ambient shape. Crossing between the two used to
allocate a fresh vector every time, four times per `update!`. These are the buffers to write into
instead, through `NeuralNetworkParameters`' allocation-free `flatten!` and `unflatten!`.

A `NeuralNetworkParameters.FlatParameters` rather than a bare `Vector`, because it carries its own
`ParameterLayout` and keeps it through `similar` — so δ is built once and the other three buffers are
one `similar` each, with no `parameterlayout` call written anywhere below this line. One is still
*made*: `FlatParameters(T, g)` is `flatten(T, g)`, whose first act is to build the layout. What the
`similar`s buy is that it happens once per cache rather than once per `_mul!`.

That layout then goes into the cache's own type, as `FlatParameters`' third type parameter and so as
`BFGSCache`/`DFPCache`'s `FT`. Until `NeuralNetworkParameters` 0.2.3 that meant every leaf's *concrete
array type* came with it, `LeafLayout` having carried a `prototype` field nothing read — and a live
reference to every leaf array besides, so a cache retained the set its buffers were sized from.
`LeafLayout{N}` is the shape alone now, and neither is true. See the 0.6.0 changelog.

(Those four in plain code and not `@extref`s, for the reason `descent_direction.jl` gives about
`solve_with_status`: `docs/make.jl` carries `DocumenterInterLinks` inventories for `SimpleSolvers` and
`GeometricMachineLearning`, not for `NeuralNetworkParameters`, and `docs/src/api.md` renders every
docstring in this package — so an unresolvable `@extref` is a build error rather than a dead link.)

# The ambient and the intrinsic

This is where the distinction the quasi-Newton methods turn on is written down, having moved here from
the `outer!` methods 0.6.0 deleted. ``Q`` is sized by the *intrinsic* dimension of the parameters — the
length of their flattening — while the direction and the gradient are handed around in the *ambient*
representation. For a bare `StiefelManifold` of size ``(3, 1)`` those are 2 and ``3 \times 3``
respectively, so `SimpleSolvers.outer!`, which indexes its arguments linearly against `axes(m)`, would
assert on the mismatch. Flattening first is what makes `BFGS` and `DFP` run on a bare `Manifold` at
all; before this release each of `outer!` and `_mul!` did it per call, and now the buffers below hold
the flat form once.

Built from `g`, which callers pass as `_zero(x)` and not `x`, for the reason the `flatlength(_zero(x))`
beside it gives: on a manifold the flattening of the *lift* is the intrinsic dimension, 12 against 18
for a `StiefelManifold(6, 3)`, and that is the length `Q` multiplies.

`nothing` for an `AbstractVector` solution. There the parameters *are* the flat coordinates, `outer!`
and `_mul!` reach `SimpleSolvers`' own methods on them, and nothing is allocated to begin with — so
buffering would add a copy per iteration and buy nothing. The accessors below make that case take the
identical path it always did.
"""
_flat_scratch(::Type, ::AbstractVector) = nothing

function _flat_scratch(::Type{T}, g) where {T}
    δ = FlatParameters(T, g)
    (δ = δ, γ = similar(δ), rhs = similar(δ), direction = similar(δ))
end

"""
    _flat_δ!(cache)
    _flat_γ!(cache)

The secant pair ``(\\delta, \\gamma)`` in the coordinates ``Q`` lives in: the flat mirrors, refreshed
from `cache.Δx` and `cache.Δg`, or those two unchanged where the solution is already flat.

Refreshed rather than assumed current: both are written by `_copyto!`/`_difference!` on the parameter
side during `update!`, and these are called after that.
"""
_flat_δ!(cache) = _flat_side!(cache.flat === nothing ? nothing : cache.flat.δ, cache.Δx)
_flat_γ!(cache) = _flat_side!(cache.flat === nothing ? nothing : cache.flat.γ, cache.Δg)

_flat_side!(::Nothing, x) = x
_flat_side!(buffer, x) = flatten!(buffer, x)

@doc raw"""
    _flat_mul!(c, A, b, scratch)

``c \gets Ab`` where `A` is in the flattened coordinates and `c`, `b` are in the parameters'.

The `nothing` method is `_mul!` unchanged, for a solution that is already flat — plain code and not an
`@ref` because `_mul!` carries no docstring, as `docs/src/linesearch_on_manifolds.md` already writes it.
The other flattens `b` into scratch, multiplies into scratch, and writes the result back through
`unflatten!`, where `_mul!` allocated one flat vector for `b`, one for the result and a
`ParameterLayout` besides.
"""
_flat_mul!(c, A, b, ::Nothing) = _mul!(c, A, b)

function _flat_mul!(c, A, b, scratch)
    flatten!(scratch.rhs, b)
    mul!(parent(scratch.direction), A, parent(scratch.rhs))
    unflatten!(c, scratch.direction)
end
