@doc raw"""
    GlobalSection(Y)

Construct a global section for `Y`.

A global section ``\lambda`` is a mapping from a homogeneous space ``\mathcal{M}`` to the corresponding Lie group ``G`` such that

```math
\lambda(Y)E = Y,
```

Also see [`apply_section`](@ref) and [`global_rep`](@ref).

# Implementation

For an implementation of `GlobalSection` for a custom array (especially manifolds), the function [`global_section`](@ref) has to be generalized.
"""
struct GlobalSection{T, AT <: AbstractArray{T}, λT <: Union{AbstractArray{T}, Nothing}}
    Y::AT
    # for now the only lift that is implemented is the Stiefel one - these types will have to be expanded!
    λ::λT

    function GlobalSection(Y::AbstractVecOrMat)
        λ = global_section(Y)
        new{eltype(Y), typeof(Y), typeof(λ)}(copy(Y), λ)
    end
end

# `mapparameters` and not `map`: a container is a tree of layers, so its leaves are more than one
# level down and `map` would hand `GlobalSection` a whole layer. The plain `NamedTuple` method
# delegates here too, so both shapes take the same walk.
GlobalSection(ps::NamedTuple) = mapparameters(GlobalSection, ps)

# The container `NeuralNetworkParameters` holds a network's parameters in. `GlobalSection` is this
# package's own function, so a method on it is this package's to write -- and it has to be, because a
# package that merely uses both owns neither name and its method would be piracy.
# `GeometricMachineLearning` carried exactly that method until now, with a comment saying as much.
#
# The result is deliberately the same `NamedTuple` tree the method above returns rather than a
# `NetworkParameters` of sections: a section is not a parameter, and everything downstream --
# `update_section!`, `apply_section`, `global_rep`, the four `copyto!` methods -- walks it as a plain
# container.
GlobalSection(ps::NetworkParameters) = GlobalSection(params(ps))

# The `NamedTuple` method above returns whatever shape `mapparameters` rebuilt, which for a section
# tree is a plain `NamedTuple` — so `apply_section`, `global_rep` and `update_section!` below take a
# `NamedTuple` on the section side and a [`NeuralNetworkParameters.NetworkParameters`](@extref) on the parameter side. Each is
# written with the section first, since `mapparameters` dispatches on its first argument and
# normalises the rest.

Base.size(λY::GlobalSection) = (size(λY.Y, 1), size(λY.Y, 2) + size(λY.λ, 2))

@doc raw"""
    Matrix(λY::GlobalSection)

Put `λY` into matrix form.

This is not recommended if speed is important!

Use [`apply_section`](@ref) and [`global_rep`](@ref) instead!
"""
function Base.Matrix(λY::GlobalSection)
    hcat(Matrix(λY.Y), Matrix(λY.λ))
end

@doc raw"""
    λY * Y

Apply the element `λY` onto `Y`.

Here `λY` is an element of a Lie group and `Y` is an element of a homogeneous space.
"""
Base.:*(λY::GlobalSection, Y::Manifold) = apply_section(λY, Y)

@doc raw"""
    apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT <: StiefelManifold{T}}

Apply `λY` to `Y₂`.

Mathematically this is the group action of the element ``\lambda{}Y\in{}G`` on the element ``Y_2`` of the homogeneous space ``\mathcal{M}``.

Internally it calls [`apply_section!`](@ref).
"""
function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT <: StiefelManifold{T}}
    Y = StiefelManifold(zero(Y₂.A))
    apply_section!(Y, λY, Y₂)

    Y
end

@doc raw"""
    apply_section!(Y::AT, λY::GlobalSection{T, AT}, Y₂::AT) where {T, AT<:StiefelManifold{T}}

Apply `λY` to `Y₂` and store the result in `Y`.

This is the inplace version of [`apply_section`](@ref).
"""
function apply_section!(Y::AT, λY::GlobalSection{T, AT},
        Y₂::MT) where {T, AT <: StiefelManifold{T}, MT <: StiefelManifold{T}}
    N, n = size(λY.Y)

    @views Y.A .= λY.Y * Y₂.A[1:n, :] .+ λY.λ * Y₂.A[(n + 1):N, :]

    Y
end

# This one is `StiefelManifold`-only and stays so: it has no live caller — the only `apply_section!`
# call sites in the package are the two in `src/utils.jl` — and the commented-out `update_section!`
# below is what it was written for. Widening it to `Manifold` would be widening dead code.
function apply_section!(Λᵗ::GlobalSection{T, MT}, λY::GlobalSection{T, MT},
        Y₂::MT) where {T, MT <: StiefelManifold{T}}
    N, n = size(Λᵗ.Y)
    @assert size(Y₂) == size(Λᵗ) == size(λY)

    @views apply_section!(Λᵗ.Y, λY, StiefelManifold(Y₂.A[:, 1:n]))
    @views Λᵗ.λ .= λY.Y * Y₂.A[1:n, (n + 1):N] .+ λY.λ * Y₂.A[(n + 1):N, (n + 1):N]

    Λᵗ
end

function apply_section(λY::GlobalSection{T, AT}, Y₂::AT) where {
        T, AT <: GrassmannManifold{T}}
    Y = GrassmannManifold(zero(Y₂.A))
    apply_section!(Y, λY, Y₂)

    Y
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT},
        Y₂::MT) where {T, AT <: GrassmannManifold{T}, MT <: GrassmannManifold{T}}
    N, n = size(λY.Y)

    # `.=` and not `=`, as in the Stiefel method above: assigning the field replaced `Y`'s array on
    # every solver step rather than writing into it, and returned that array instead of `Y`. Safe
    # where `Y === Y₂`, which `update_section!` relies on -- the two products are materialised before
    # the broadcast assignment.
    @views Y.A .= λY.Y * Y₂.A[1:n, :] .+ λY.λ * Y₂.A[(n + 1):N, :]

    Y
end

function apply_section(λY::GlobalSection{T}, Y₂::AbstractVecOrMat{T}) where {T}
    Y = copy(Y₂)
    apply_section!(Y, λY, Y₂)
end

function apply_section!(Y::AT, λY::GlobalSection{T, AT, Nothing},
        Y₂::AbstractVecOrMat{T}) where {T, AT <: AbstractVecOrMat{T}}
    Y .= Y₂ .+ λY.Y
end

function apply_section(λY::NamedTuple, Y₂::NamedTuple)
    mapparameters(apply_section, λY, Y₂)
end

function apply_section!(Y::NetworkParameters, λY::NamedTuple, Y₂)
    mapparameters!(apply_section!, Y, λY, Y₂)
end

function global_rep(λY::NamedTuple, gx::NamedTuple)
    mapparameters(global_rep, λY, gx)
end

# The container versions of the two that *build* a tree, and the reason they are written with their
# arguments the other way round.
#
# `mapparameters` rebuilds in the shape of its **first** argument, so `mapparameters(f, λY, gx)` would
# hand back a plain `NamedTuple` — the section tree's shape — for a container `gx`. What these produce
# is a point and a gradient, i.e. *parameters*, and a parameter set has the parameters' shape. Getting
# this backwards is not a cosmetic matter: the gradient would come out a plain nested `NamedTuple`,
# which is not a container, so no alias in this package covers it and
# `_copyto!(gradient_array(cache), ·)` has no method for it.
#
# The section stays the plain tree `GlobalSection(::NetworkParameters)` returns; it is passed through
# as the second argument and `mapparameters` normalises it.
function apply_section(λY::NamedTuple, Y₂::NetworkParameters)
    mapparameters((y, λ) -> apply_section(λ, y), Y₂, λY)
end

function global_rep(λY::NamedTuple, gx::NetworkParameters)
    mapparameters((g, λ) -> global_rep(λ, g), gx, λY)
end

##auxiliary function
function global_rep(::GlobalSection{T}, gx::AbstractVecOrMat{T}) where {T}
    gx
end

@doc raw"""
    global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:StiefelManifold{T}}

Express `Δ` (an the tangent space of `Y`) as an instance of `StiefelLieAlgHorMatrix`.

This maps an element from ``T_Y\mathcal{M}`` to an element of ``\mathfrak{g}^\mathrm{hor}``.

These two spaces are isomorphic where the isomorphism where the isomorphism is established through ``\lambda(Y)\in{}G`` via:

```math
T_Y\mathcal{M} \to \mathfrak{g}^{\mathrm{hor}}, \Delta \mapsto \lambda(Y)^{-1}\Omega(Y, \Delta)\lambda(Y).
```

Also see [`GeometricOptimizers.Ω`](@ref).

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: _round
import Random

Random.seed!(123)

Y = rand(StiefelManifold, 6, 3)
Δ = rgrad(Y, randn(6, 3))
λY = GlobalSection(Y)

_round(global_rep(λY, Δ); digits = 3)

# output

6×6 StiefelLieAlgHorMatrix{Float64, SkewSymMatrix{Float64, Vector{Float64}}, Matrix{Float64}}:
  0.0     0.679   1.925   0.981  -2.058   0.4
 -0.679   0.0     0.298  -0.424   0.733  -0.919
 -1.925  -0.298   0.0    -1.815   1.409   1.085
 -0.981   0.424   1.815   0.0     0.0     0.0
  2.058  -0.733  -1.409   0.0     0.0     0.0
 -0.4     0.919  -1.085   0.0     0.0     0.0
```

# Implementation

The function `global_rep` does in fact not perform the entire map ``\lambda(Y)^{-1}\Omega(Y, \Delta)\lambda(Y)`` but only

```math
\Delta \mapsto \mathrm{skew}(Y^T\Delta),
```

to get the small skew-symmetric matrix ``A\in\mathcal{S}_\mathrm{skew}(n)`` and

```math
\Delta \mapsto (\lambda(Y)_{[1:N, n:N]}^T \Delta)_{[1:(N-n), 1:n]},
```

to get the arbitrary matrix ``B\in\mathbb{R}^{(N-n)\times{}n}``.
"""
function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {
        T, AT <: StiefelManifold{T}}
    N, n = size(λY.Y)
    StiefelLieAlgHorMatrix(
        SkewSymMatrix(λY.Y.A' * Δ),
        λY.λ' * Δ,
        N,
        n
    )
end

@doc raw"""
    global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {T, AT<:GrassmannManifold{T}}

Express `Δ` (an element of the tangent space of `Y`) as an instance of [`GrassmannLieAlgHorMatrix`](@ref).

The method `global_rep` for [`GrassmannManifold`](@ref) is similar to that for [`StiefelManifold`](@ref).

# Examples

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: _round
import Random

Random.seed!(123)

Y = rand(GrassmannManifold, 6, 3)
Δ = rgrad(Y, randn(6, 3))
λY = GlobalSection(Y)

_round(global_rep(λY, Δ); digits = 3)

# output

6×6 GrassmannLieAlgHorMatrix{Float64, Matrix{Float64}}:
  0.0     0.0     0.0     0.981  -2.058   0.4
  0.0     0.0     0.0    -0.424   0.733  -0.919
  0.0     0.0     0.0    -1.815   1.409   1.085
 -0.981   0.424   1.815   0.0     0.0     0.0
  2.058  -0.733  -1.409   0.0     0.0     0.0
 -0.4     0.919  -1.085   0.0     0.0     0.0
```
"""
function global_rep(λY::GlobalSection{T, AT}, Δ::AbstractMatrix{T}) where {
        T, AT <: GrassmannManifold{T}}
    N, n = size(λY.Y)
    GrassmannLieAlgHorMatrix(
        λY.λ' * Δ,
        N,
        n
    )
end

# function update_section!(Λᵗ::GlobalSection{T, MT}, Λ⁽ᵗ⁻¹⁾::GlobalSection{T, MT}, B⁽ᵗ⁻¹⁾::AbstractLieAlgHorMatrix{T}, retraction) where {T, MT <: Manifold}
#     N, n = B⁽ᵗ⁻¹⁾.N, B⁽ᵗ⁻¹⁾.n
#     expB = retraction(B⁽ᵗ⁻¹⁾)
#     apply_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, expB)
#
#     Λᵗ
# end

@doc raw"""
    update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction)
    update_section!(Λ, B, retraction)

Transport the [`GlobalSection`](@ref) `Λ⁽ᵗ⁻¹⁾` along the step `B⁽ᵗ⁻¹⁾` and write the result into `Λᵗ`:

```math
\Lambda^{(t)} \leftarrow \Lambda^{(t-1)}\mathrm{Retraction}(B^{(t-1)}).
```

This is the fourth of the five steps an optimizer step consists of on a homogeneous space — see
[Optimization on Homogeneous Spaces](@ref) — and the one that makes the cache independent of the
iterate: the new point is read back out of the section afterwards with [`apply_section!`](@ref),
rather than the section being recomputed at the new point.

`B⁽ᵗ⁻¹⁾` is the final velocity the optimizer method produced, already scaled by the step length. It is
an [`AbstractLieAlgHorMatrix`](@ref) when the parameter is on a manifold, and of the parameter's own
type when it is not — a vector-space parameter carries no ``\lambda``, and the extended retraction on
a vector space is addition, so `retraction` is then ignored.

The three-argument form is the two-argument section written in place, `Λᵗ === Λ⁽ᵗ⁻¹⁾`. A `NamedTuple`
of parameters is walked leaf by leaf.
"""
function update_section!(Λ⁽ᵗ⁻¹⁾::GlobalSection{T, MT}, B⁽ᵗ⁻¹⁾::AbstractLieAlgHorMatrix{T},
        retraction) where {T, MT <: Manifold{T}}
    N, n = B⁽ᵗ⁻¹⁾.N, B⁽ᵗ⁻¹⁾.n
    expB = retraction(B⁽ᵗ⁻¹⁾)
    apply_section!(expB, Λ⁽ᵗ⁻¹⁾, expB)
    Λ⁽ᵗ⁻¹⁾.Y.A .= @view expB.A[:, 1:n]
    Λ⁽ᵗ⁻¹⁾.λ .= @view expB.A[:, (n + 1):N]

    nothing
end

function update_section!(Λᵗ::GlobalSection{T, MT}, Λ⁽ᵗ⁻¹⁾::GlobalSection{T, MT},
        B⁽ᵗ⁻¹⁾::AbstractLieAlgHorMatrix{T}, retraction) where {T, MT <: Manifold{T}}
    N, n = B⁽ᵗ⁻¹⁾.N, B⁽ᵗ⁻¹⁾.n
    expB = retraction(B⁽ᵗ⁻¹⁾)
    apply_section!(expB, Λ⁽ᵗ⁻¹⁾, expB)
    Λᵗ.Y.A .= @view expB.A[:, 1:n]
    Λᵗ.λ .= @view expB.A[:, (n + 1):N]

    nothing
end

function update_section!(Λᵗ::GlobalSection{T, AT, Nothing}, Λ⁽ᵗ⁻¹⁾::GlobalSection{T, AT},
        B⁽ᵗ⁻¹⁾::AT, retraction) where {T, AT <: AbstractVecOrMat{T}}
    Λᵗ.Y .= Λ⁽ᵗ⁻¹⁾.Y .+ B⁽ᵗ⁻¹⁾

    Λᵗ
end

# Same update as the one above -- these are ordinary vector-space parameters and the extended
# retraction on a vector space is addition -- but written on the free parameters, which is where a
# [`VectorStorageMatrix`](@ref) keeps them. Three of the four have no `setindex!` for the broadcast
# above to write through, and for the fourth (`SymmetricMatrix`) it would visit each entry twice.
function update_section!(Λᵗ::GlobalSection{T, AT, Nothing}, Λ⁽ᵗ⁻¹⁾::GlobalSection{T, AT},
        B⁽ᵗ⁻¹⁾::AT, retraction) where {T, AT <: VectorStorageMatrix{T}}
    parent(Λᵗ.Y) .= parent(Λ⁽ᵗ⁻¹⁾.Y) .+ parent(B⁽ᵗ⁻¹⁾)

    Λᵗ
end

# The direction `B⁽ᵗ⁻¹⁾` is of the parameters' shape and the two sections are plain `NamedTuple`s, so
# this is the mixed-shape walk: `mapparameters!` takes the section as its first argument and
# normalises the direction, whichever of the two shapes it arrived in.
function update_section!(Λᵗ::NamedTuple, Λ⁽ᵗ⁻¹⁾::NamedTuple, B⁽ᵗ⁻¹⁾::NetworkParameters, retraction)
    update_section_closure!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾) = update_section!(Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction)
    mapparameters!(update_section_closure!, Λᵗ, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾)

    Λᵗ
end

function update_section!(Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction)
    update_section!(Λ⁽ᵗ⁻¹⁾, Λ⁽ᵗ⁻¹⁾, B⁽ᵗ⁻¹⁾, retraction)
end

# The default for a `struct` is `===`, which is `false` for two sections that hold equal frames in
# different arrays -- and comparing the frames is exactly what `latest_gradient_is_current` needs in
# order to tell "the cache is at the iterate the state is at" from "it is at a line-search trial
# point". `λ === nothing` on Euclidean parameters, and `nothing == nothing` is `true`.
Base.:(==)(Λ₁::GlobalSection, Λ₂::GlobalSection) = Λ₁.Y == Λ₂.Y && Λ₁.λ == Λ₂.λ

function Base.copyto!(dest::GlobalSection{T, MT}, src::GlobalSection{
        T, MT}) where {T, MT <: Manifold}
    copyto!(dest.Y, src.Y)
    copyto!(dest.λ, src.λ)
    dest
end

function Base.copyto!(dest::GlobalSection{T, AT, Nothing},
        src::GlobalSection{T, AT, Nothing}) where {T, AT <: AbstractVecOrMat{T}}
    copyto!(dest.Y, src.Y)
    dest
end

function Base.copyto!(
        dest::GlobalSection{
            T, AT, Nothing}, src::AT) where {T, AT <: AbstractVecOrMat{T}}
    copyto!(dest.Y, src)
    dest
end

# auxiliary function
function global_rep(::GlobalSection{T, AT, Nothing},
        gx::AbstractVecOrMat{T}) where {T, AT <: AbstractVecOrMat{T}}
    gx
end
