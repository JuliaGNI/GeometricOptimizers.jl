@doc raw"""
    _flat_scratch(T, g)

The flat buffers a quasi-Newton cache keeps, or `nothing` where it needs none.

``Q`` lives in the *flattened* coordinates — it is sized by the length of the flattening, it is where
the outer products that build it are formed and where [`_dot`](@ref) pairs — while the secant pair, the
right-hand side and the direction are handed around in the parameters' own representation: a
`NamedTuple`, a container, or a horizontal lift of the ambient shape. These are the buffers that the
crossing between the two writes into, through `NeuralNetworkParameters`' allocation-free `flatten!`
and `unflatten!`, so that no `update!` allocates a fresh vector.

A `NeuralNetworkParameters.FlatParameters` rather than a bare `Vector`, because it carries its own
`ParameterLayout` and keeps it through `similar` — so δ is built once and the other three buffers are
one `similar` each, with no `parameterlayout` call written anywhere below this line. One is still
*made*: δ is the flattening of `g` on its backend, whose first act is to build the layout. What the
`similar`s buy is that it happens once per cache rather than once per product with ``Q``.

That layout then goes into the cache's own type, as `FlatParameters`' third type parameter and so as
`QuasiNewtonCache`'s `FT`. A `LeafLayout{N}` holds the shape alone, so the cache type does not carry
the array types of the leaves, and the cache keeps no reference to the leaves.

# The ambient and the intrinsic

This is where the distinction the quasi-Newton methods turn on is written down. ``Q`` is sized by the
*intrinsic* dimension of the parameters — the
length of their flattening — while the direction and the gradient are handed around in the *ambient*
representation. For a bare `StiefelManifold` of size ``(3, 1)`` those are 2 and ``3 \times 3``
respectively, so `SimpleSolvers.outer!`, which checks the axes of its arguments against those of its
destination, would throw a `DimensionMismatch`. Flattening first is what makes `BFGS` and `DFP` run
on a bare `Manifold` at all, and the buffers below hold the flat form once per cache rather than
once per call.

Built from `g`, which callers pass as `_zero(x)` and not `x`, for the reason the `flatlength(_zero(x))`
beside it gives: on a manifold the flattening of the *lift* is the intrinsic dimension, 12 against 18
for a `StiefelManifold(6, 3)`, and that is the length `Q` multiplies.

`nothing` for an `AbstractVector` solution. There the parameters *are* the flat coordinates, `outer!`
and `mul!` reach their own methods on them, and nothing is allocated to begin with — so buffering
would add a copy per iteration and buy nothing. `_flat_mul!` and the secant pair of
`QuasiNewtonCache`'s `update!` take the parameters unchanged in that case.
"""
_flat_scratch(::Type{T}, ::AbstractVector) where {T} = nothing

# On the backend of `g`, as `Q` is, so that the products with `Q` stay there; see
# `_flatten_on_backend`.
function _flat_scratch(::Type{T}, g) where {T}
    δ = FlatParameters(_flatten_on_backend(T, g)...)
    (δ = δ, γ = similar(δ), rhs = similar(δ), direction = similar(δ))
end

@doc raw"""
    _flat_mul!(c, A, b, scratch)

``c \gets Ab`` where `A` is in the flattened coordinates and `c`, `b` are in the parameters'.

The `nothing` method is `mul!`, for a solution that is already flat. The other flattens `b` into
scratch, multiplies into scratch, and writes the result back through `unflatten!`, so that no flat
vector and no `ParameterLayout` is allocated.
"""
_flat_mul!(c, A, b, ::Nothing) = mul!(c, A, b)

function _flat_mul!(c, A, b, scratch)
    flatten!(scratch.rhs, b)
    mul!(parent(scratch.direction), A, parent(scratch.rhs))
    unflatten!(c, scratch.direction)
end
