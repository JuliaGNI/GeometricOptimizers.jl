@doc raw"""
    VectorStorageMatrix

The structured matrix types that keep their free parameters in one vector, reachable as `parent(A)`:
[`SkewSymMatrix`](@ref), [`SymmetricMatrix`](@ref) and the two [`AbstractTriangular`](@ref)s.

Each of them stores ``n(n\pm1)/2`` numbers behind an ``n \times n`` interface, and none of them has a
`setindex!` that an elementwise operation could write through — writing to a `SymmetricMatrix` entry
would silently symmetrise, and a `SkewSymMatrix` cannot represent an arbitrary matrix at all. The
optimizer primitives `_add!`, `_rac!`, `_square!`, `_div!` and `_rmul!` therefore cannot use the
generic `AbstractArray` methods, which broadcast.

For every one of these types the free parameters *are* the coordinates the optimizer should work in,
so each primitive is the corresponding operation on `parent`. The methods live next to the rest of
their family in `optimizers/named_tuple_wrapper.jl`; `update_section!` in
`global_sections/global_sections.jl` splits on this alias for the same reason.

This is what lets a `SymmetricMatrix` or a triangular matrix be an optimizer parameter — which is
what `GeometricMachineLearning`'s SympNet, symplectic-attention and volume-preserving layers need,
and what this package could not do before.

The alias lives in a file of its own because it is needed by `global_sections.jl`, which is included
long before the primitives are.
"""
const VectorStorageMatrix{T} = Union{SkewSymMatrix{T},SymmetricMatrix{T},AbstractTriangular{T}}
