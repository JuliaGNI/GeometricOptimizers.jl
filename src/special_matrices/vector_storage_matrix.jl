@doc raw"""
    VectorStorageMatrix

The structured matrix types that keep their free parameters in one vector, reachable as `parent(A)`:
[`SkewSymMatrix`](@ref), [`SymmetricMatrix`](@ref) and the two [`AbstractTriangular`](@ref)s.

Each of them stores ``n(n\pm1)/2`` numbers behind an ``n \times n`` interface, so none of the generic
`AbstractArray` methods, which either broadcast or reshape, is usable for them:

- three of the four have no `setindex!` for a broadcast to write through. `SkewSymMatrix` and the two
  triangulars cannot represent an arbitrary matrix at all, so there is no entry to assign.
  `SymmetricMatrix` is the exception — it *does* have one (writing ``[A]_{ij}`` writes ``[A]_{ji}``
  too, which is what a symmetric matrix means), so the broadcast would give the right answer for that
  one type, at twice the work and only as long as it stays the exception.
- `similar` has to preserve the type, because the optimizer caches allocate their scratch with it and
  then require every array to have the same type as the parameter. The `AbstractArray` fallback
  returns a dense `Matrix`.
- flattening has to round trip through the free parameters. A generic `AbstractMatrix` treatment
  reshapes the flattened vector back to ``n \times n``, and ``n(n\pm1)/2`` numbers do not reshape to
  that. `NeuralNetworkParameters` asks [`Base.parent`](@ref) instead, through `freeparameters`.

For every one of these types the free parameters *are* the coordinates the optimizer should work in,
so each primitive is the corresponding operation on `parent`. `_add!`, `_rac!`, `_square!`, `_div!`,
`_rmul!` and `_difference!` live next to the rest of their family in
`optimizers/named_tuple_wrapper.jl`, `l2norm` in `optimizers/optimizer_status.jl`; `update_section!`
in `global_sections/global_sections.jl` splits on this alias for the same reason.

This is what lets a `SymmetricMatrix` or a triangular matrix be an optimizer parameter — which is
what `GeometricMachineLearning`'s SympNet, symplectic-attention and volume-preserving layers need,
and what this package could not do before.

The alias lives in a file of its own because it is needed by `global_sections.jl`, which is included
long before the primitives are.
"""
const VectorStorageMatrix{T} = Union{SkewSymMatrix{T},SymmetricMatrix{T},AbstractTriangular{T}}
