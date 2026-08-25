"""
    IterativeHessian <: Hessian

An abstract type derived from [`SimpleSolvers.Hessian`](@extref).
Its main purpose is defining a supertype that encompasses [`HessianBFGS`](@ref) and [`HessianDFP`](@ref) for dispatch.
"""
abstract type IterativeHessian{T} <: Hessian{T} end
@doc raw"""
    _flat_scratch(T, g)

The flat buffers a quasi-Newton cache keeps, or `nothing` where it needs none.

``Q`` lives in the *flattened* coordinates — it is sized by the length of the flattening, [`outer!`](@ref)
forms its outer products there and [`_dot`](@ref) pairs there — while the secant pair, the right-hand
side and the direction are handed around in the parameters' own representation: a `NamedTuple`, a
container, or a horizontal lift of the ambient shape. Crossing between the two used to allocate a fresh
vector every time, four times per `update!`. These are the buffers to write into instead, through
`NeuralNetworkParameters`' allocation-free [`flatten!`](@extref) and [`unflatten!`](@extref).

A [`FlatParameters`](@extref) rather than a bare `Vector`, because it carries its own
[`ParameterLayout`](@extref) and keeps it through `similar` — so the three scratch buffers are one
`similar` each and no `parameterlayout` call survives in `_mul!`.

Built from `g`, which callers pass as `_zero(x)` and not `x`, for the reason the `flatlength(_zero(x))`
beside it gives: on a manifold the flattening of the *lift* is the intrinsic dimension, 12 against 18
for a `StiefelManifold(6, 3)`, and that is the length `Q` multiplies.

`nothing` for an `AbstractVector` solution. There the parameters *are* the flat coordinates, `outer!`
and `_mul!` reach `SimpleSolvers`' own methods on them, and nothing is allocated to begin with — so
buffering would add a copy per iteration and buy nothing. The accessors below make that case take the
identical path it always did.
"""
_flat_scratch(::Type{T}, ::AbstractVector) where {T} = nothing

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

The `nothing` method is [`_mul!`](@ref) unchanged, for a solution that is already flat. The other
flattens `b` into scratch, multiplies into scratch, and writes the result back through `unflatten!` —
where `_mul!` allocated one flat vector for `b`, one for the result and a `ParameterLayout` besides.
"""
_flat_mul!(c, A, b, ::Nothing) = _mul!(c, A, b)

function _flat_mul!(c, A, b, scratch)
    flatten!(scratch.rhs, b)
    mul!(parent(scratch.direction), A, parent(scratch.rhs))
    unflatten!(c, scratch.direction)
end
