```@raw latex
\texttt{GeometricOptimizers} has custom versions of matrices such as the symmetric and the skew-symmetric matrix implemented. These are important ingredients in e.g. SympNets and volume-preserving transformers and it is therefore important that those implementations also run efficiently on GPU. We also show how their storage layout is what an optimizer has to update them through.
```

# Symmetric, Skew-Symmetric and Triangular Matrices

Among the special arrays implemented in `GeometricOptimizers` [`SymmetricMatrix`](@ref), [`SkewSymMatrix`](@ref), [`StrictlyUpperTriangular`](@ref) and [`StrictlyLowerTriangular`](@ref) are the most common ones and similar implementations can also be found in other libraries; `LinearAlgebra.jl` has an implementation of a symmetric matrix called [`Symmetric`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Symmetric) for example. The versions of these matrices in `GeometricOptimizers` are however more memory efficient as they only store as many parameters as are necessary, i.e. ``n(n+1)/2`` for the symmetric matrix and ``n(n-1)/2`` for the other three. In addition, `GeometricMachineLearning` implements matrix and tensor multiplication for these matrices so that they work in parallel on GPU; see [Tensors](@extref GeometricMachineLearning Tensors-in-GeometricMachineLearning) there. We here give an overview of *elementary* custom matrices that are implemented in `GeometricOptimizers`. More *involved* matrices are the so-called [global tangent spaces](@ref "Global Tangent Spaces").

## Custom Matrices

`GeometricOptimizers` has two types of *triangular matrices*. The first one is [`StrictlyUpperTriangular`](@ref):

```math 
U = \begin{pmatrix}
     0 & a_{12} & \cdots & a_{1n}      \\
     0 & \ddots &        & a_{2n} \\
     \vdots & \ddots & \ddots & \vdots \\
     0 & \cdots & 0      & 0 
\end{pmatrix}.
```

And the second one is [`StrictlyLowerTriangular`](@ref):

```math 
L = \begin{pmatrix}
     0 & 0 & \cdots & 0      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
```

`adjoint` swaps between the two: `L'` is a `StrictlyUpperTriangular` and `U'` is a `StrictlyLowerTriangular`.
That swap is built around the *same* storage vector rather than a copy, so `parent(L') === parent(L)`
holds and writing into `L'` also writes into `L`. Reusing the storage transposes without
conjugating, so that swap is bound to a real element type; a complex one falls through to
`LinearAlgebra`'s lazy `Adjoint`, which conjugates and does not alias.

An instance of [`SkewSymMatrix`](@ref) can be written as ``A = L - L^T`` or ``A = U^T - U``:

```math 
A = \begin{pmatrix}
     0 & - a_{21} & \cdots & - a_{n1}     \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
```

And lastly a [`SymmetricMatrix`](@ref):

```math 
B = \begin{pmatrix}
     a_{11} & a_{21} & \cdots & a_{n1}      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & a_{nn}
\end{pmatrix}.
```

Note that any matrix ``M\in\mathbb{R}^{n\times{}n}`` can be written

```math
M = \frac{1}{2}(M - M^T) + \frac{1}{2}(M + M^T),
```
where the first part of this matrix is skew-symmetric and the second part is symmetric. This is also how the constructors for [`SkewSymMatrix`](@ref) and [`SymmetricMatrix`](@ref) are designed. Consider an arbitrary matrix:

```@example sym_skew_sym_example
using GeometricOptimizers  # hide

M = [1; 2; 3;; 4; 5; 6;; 7; 8; 9]
```

Calling [`SkewSymMatrix`](@ref) on ``M`` is equivalent to doing ``M \to \frac{1}{2}(M - M^T)``:

```@example sym_skew_sym_example
A = SkewSymMatrix(M)
```

And calling [`SymmetricMatrix`](@ref) on ``M`` is equivalent to doing ``M \to \frac{1}{2}(M + M^T)``:

```@example sym_skew_sym_example
B = SymmetricMatrix(M)
```

We can further confirm the identity above:

```@example sym_skew_sym_example
@assert M  ≈ A + B # hide
M  ≈ A + B
```

Note that for [`StrictlyLowerTriangular`](@ref) and [`StrictlyUpperTriangular`](@ref) no projection step is involved, which means that if we start with a matrix of type `AbstractMatrix{Int64}` we will end up with a matrix that is also of type `AbstractMatrix{Int64}`. The type changes however when we call [`SkewSymMatrix`](@ref) and [`SymmetricMatrix`](@ref):

```@example sym_skew_sym_example
@assert (typeof(A) <: AbstractMatrix{Int64}) == false # hide
@assert (typeof(B) <: AbstractMatrix{Int64}) == false # hide
(typeof(A) <: AbstractMatrix{Int64}, typeof(B) <: AbstractMatrix{Int64})
```

For the triangular matrices:

```@example sym_skew_sym_example
U = StrictlyUpperTriangular(M)
L = StrictlyLowerTriangular(M)
@assert (typeof(U) <: AbstractMatrix{Int64}) == true # hide
@assert (typeof(L) <: AbstractMatrix{Int64}) == true # hide
(typeof(U) <: AbstractMatrix{Int64}, typeof(L) <: AbstractMatrix{Int64})
```

## How are Special Matrices Stored?

The following image demonstrates how a skew-symmetric matrix is stored in `GeometricOptimizers`:

![The elements of a skew-symmetric matrix (and other special matrices) are stored as a vector. The elements of the big vector are the entries on the lower left of the matrix, stored row-wise.](tikz/skew_sym_visualization_light.png)
![The elements of a skew-symmetric matrix (and other special matrices) are stored as a vector. The elements of the big vector are the entries on the lower left of the matrix, stored row-wise.](tikz/skew_sym_visualization_dark.png)

So what is stored internally is a vector of size ``n(n-1)/2`` for the skew-symmetric matrix and the triangular matrices, and a vector of size ``n(n+1)/2`` for the symmetric matrix. 

## Sample Random Matrices

We can sample a random skew-symmetric matrix: 

```@example skew_sym
using GeometricOptimizers # hide
import Random # hide
Random.seed!(123) # hide

A = rand(SkewSymMatrix, 3)
```

and then access the vector:

```@example skew_sym
A.S 
```

This is equivalent to sampling a vector and then assigning a matrix[^1]:

[^1]: We fixed the seed to the same value in both these examples.

```@example skew_sym
using GeometricOptimizers # hide
import Random # hide
Random.seed!(123) # hide

S = rand(3 * (3 - 1) ÷ 2)
@assert A == SkewSymMatrix(S, 3) # hide
SkewSymMatrix(S, 3)
```

These special matrices are what the layers of
[`GeometricMachineLearning`](@extref GeometricMachineLearning :doc:`index`) are parametrized by:
[SympNets](@extref GeometricMachineLearning SympNet-Architecture), the
[volume-preserving transformer](@extref GeometricMachineLearning Volume-Preserving-Transformer)
and the
[linear symplectic transformer](@extref GeometricMachineLearning Linear-Symplectic-Transformer) all
use one or more of them. That package also batches them over the third axis of a tensor, with
`mat_tensor_mul` and `tensor_mat_mul`; see
[Tensors](@extref GeometricMachineLearning Tensors-in-GeometricMachineLearning) there.

## Where a sampled array lands, and what it holds

Every owned type has one allocator convention, `zeros([backend,] X{T}, dims...)` and
`rand([rng,] [backend,] X{T}, dims...)`: the backend defaults to `CPU()`, a bare `X` means
[`default_eltype`](@ref GeometricOptimizers.default_eltype) of the backend, and `rng` defaults to
`Random.default_rng()`. A manifold has `rand` only. So a call comes in four shapes, by whether it
names a backend and whether it names an element type. The shape decides both answers, and it
decides them the same way for the structured matrices above, for the
[manifolds](@ref "The Stiefel Manifold") and for the horizontal lifts:

| a call of this shape | backend | element type | gives |
|:--|:--|:--|:--|
| `rand(backend, SkewSymMatrix{Float32}, n)` | named | named | exactly what was asked for |
| `rand(backend, SkewSymMatrix, n)` | named | **chosen** | the backend's array, element type from [`default_eltype`](@ref GeometricOptimizers.default_eltype) |
| `rand(SkewSymMatrix{Float32}, n)` | — | named | the named element type, **on the host** |
| `rand(SkewSymMatrix, n)` | — | — | the host, and `Float64` |

The third and fourth shapes place on the host without saying so, and that is deliberate. They
mirror `Base`, where `zeros(Float32, 3)` is a host array and nothing about the call suggests
otherwise; these types present as `AbstractMatrix`, so `zeros(SkewSymMatrix{Float32}, n)` should
read as the `Array` case does. A host placement also cannot quietly corrupt a device computation:
mixing one with a device array throws at the first arithmetic — `*`, `+`, `-`, `mul!` and `add!`
refuse a pair on two backends with an `ArgumentError` that names both — so the loud failure already
gives the guarantee that making these shapes take a backend would buy. `copyto!` and `assign!` are the deliberate exception, because they
*are* the transfer: moving a host-built structured matrix onto a device is what they exist for.

The second shape is the only one where the package decides something the caller did not, which is
why the choice is a stated rule rather than a literal: `Float64` on the host, `Float32` on a
device. See [`default_eltype`](@ref GeometricOptimizers.default_eltype) for why each value is what
it is, and note that a backend being *able* to hold a `Float64` is not one of the reasons.

The first shape has a rule of its own, in the other direction: an element type the caller names and
the backend cannot hold is **refused**, not narrowed. The backend's own allocation refuses it, so
`rand(MetalBackend(), SkewSymMatrix{Float64}, n)` and
`rand(MetalBackend(), StiefelManifold{Float64}, N, n)` both raise Metal's `ErrorException`, whose
message names `Float64`. A narrowed result would have a different type from the one asked for,
which is exactly what naming the element type rules out.

None of this reaches an allocation the package makes for itself. `zero`, `similar`, `_zero` and
`_similar` all take an *instance*, so the backend and the element type both come from the argument
and there is nothing to default — which is what every optimizer cache and every state allocates
through. A parameter set on a device stays there.

## Arithmetic and broadcasting on a device

A product, a sum, a difference and a `mul!` between two of these matrices, or between one of them and
a plain array, run on the backend of their operands: the structured matrices, the horizontal lifts,
the manifold points, `StiefelProjection` and the adjoints of each. None of them reads an entry at a
time, which a device does not serve.

A **broadcast** does. These matrices define no broadcast style, so `A .+ 1` or `f.(A)` reads `A`
through `getindex`, and on a device that raises `Scalar indexing is disallowed`. Broadcast over the
storage instead — `parent(A)` for the structured matrices, `Y.A` for a manifold point — and rebuild
the matrix around the result where its structure still holds. A manifold point is left without a
broadcast style on purpose: a broadcast over a point returns a plain array, because its result is in
general not on the manifold.

## Why the storage matters here

Because these types keep only their free parameters, they are also what an optimizer has to be able
to *update* — and the generic array methods cannot do it: three of the four have no `setindex!` for an
elementwise operation to broadcast through, `similar` has to preserve the type rather than widen to a
dense `Matrix`, and ``n(n\pm1)/2`` numbers do not reshape back to ``n \times n``. See
[`VectorStorageMatrix`](@ref GeometricOptimizers.VectorStorageMatrix) for the methods that make them usable as optimizer parameters.

The same storage is what a *flat* parameter vector and a saved file have to hold, for the same reason:
``n(n\pm1)/2`` numbers are the whole content of one of these matrices, and the ``n^2`` entries of the
dense interface are neither the right length nor, for three of the four types, writable at all.
[`NeuralNetworkParameters`](https://github.com/JuliaGNI/NeuralNetworkParameters.jl) asks a leaf type
for exactly that relation, through its `freeparameters`/`rebuild` pair, and loading it alongside this
package brings in an extension that answers for all three families here — these matrices, the
manifolds, and the horizontal lifts. Flattening, differentiating and saving a parameter set that
contains them therefore needs no case per type in the package doing the training.

A gradient of one of these matrices has two forms, and they are not equal. The *natural cotangent* is
a matrix of the same structure: `ChainRulesCore.ProjectTo` gives it for a dense cotangent
``\bar{A} = \partial L/\partial A``, as the Frobenius projection ``\frac{1}{2}(\bar{A} \pm \bar{A}^T)``.
Automatic differentiation can add two natural cotangents and project the sum again, because the
projection is linear and idempotent. The *storage gradient* ``\partial L/\partial S`` is what the flat
parameter vector and forward-mode differentiation give. An off-diagonal entry of a
[`SymmetricMatrix`](@ref) appears twice in the matrix, so its storage gradient is
``\bar{A}_{ij} + \bar{A}_{ji}``, twice the natural cotangent; the diagonal entries agree. For a
[`SkewSymMatrix`](@ref) every storage entry is ``\bar{A}_{ij} - \bar{A}_{ji}``, again twice the
natural cotangent. The triangular types store each entry once, so the two forms agree.

## Element types

These matrices, and the package as a whole, support real element types only. The storage of a
[`SymmetricMatrix`](@ref) and a [`SkewSymMatrix`](@ref) describes ``A^T = \pm A``, which is not a
Hermitian structure for a complex ``A``. A complex element type is not rejected, and some operations
give a wrong answer for it.

## Library functions

[`AbstractTriangular`](@ref), [`StrictlyUpperTriangular`](@ref), [`StrictlyLowerTriangular`](@ref),
[`SkewSymMatrix`](@ref), [`SymmetricMatrix`](@ref) and
[`VectorStorageMatrix`](@ref GeometricOptimizers.VectorStorageMatrix). Their docstrings are on the
[reference page](@ref GeometricOptimizers), where every docstring in the package is rendered once;
the names above link to them.
