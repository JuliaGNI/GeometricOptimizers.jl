```@meta
CurrentModule = GeometricOptimizers
```

# Retractions

A retraction is how every step in this package is taken. An [`OptimizerMethod`](@ref) produces a
direction in the horizontal component ``\mathfrak{g}^\mathrm{hor}`` of the Lie algebra, and the
retraction turns that direction back into a point of the manifold. Two of them ship with the
package — [`Cayley`](@ref), the Cayley transform, and [`Geodesic`](@ref), the exponential map — and
since 0.2.0 [`Geodesic`](@ref) additionally carries an *algorithm* that says how the exponential is
evaluated. There are five of those, and the choice between them is a numerical one: they compute the
same map and differ in accuracy at a large step, in cost, and in which backends they run on.

This page collects the theory of all of it: what the retractions are, why the exponential needs an
algorithm at all, what the five algorithms do, and which one to reach for. The optimizer that uses
them is described on the [Optimization on Homogeneous Spaces](@ref) page.

## What a retraction is

In practice we usually do not solve the geodesic equation exactly in each optimization step (even though this is possible and computationally feasible), but prefer approximations that are called "retractions" [absil2008optimization](@cite) for numerical stability. The definition of a retraction in `GeometricOptimizers` is slightly different from how it is usually defined in textbooks [absil2008optimization, hairer2006geometric](@cite). We discuss these differences here.

## Classical Retractions

By "classical retraction" we here mean the textbook definition. 

```@eval
Main.definition(raw"A **classical retraction** is a smooth map
" * Main.indentation * raw"```math 
" * Main.indentation * raw"R: T\mathcal{M}\to\mathcal{M}:(x,v)\mapsto{}R_x(v),
" * Main.indentation * raw"```
" * Main.indentation * raw"such that each curve ``c(t) := R_x(tv)`` is a local approximation of a geodesic, i.e. the following two conditions hold:
" * Main.indentation * raw"1. ``c(0) = x`` and 
" * Main.indentation * raw"2. ``c'(0) = v.``
")
```

Perhaps the most common example for matrix manifolds is the *Cayley retraction*. It is a retraction for many matrix Lie groups [hairer2006geometric, bendokat2021real, gao2021riemannian](@cite).

```@eval
Main.example(raw"The **Cayley retraction** for ``V\in{}T_\mathbb{I}G\equiv\mathfrak{g}`` is defined as
" * Main.indentation * raw"```math
" * Main.indentation * raw"\mathrm{Cayley}(V) = \left(\mathbb{I} - \frac{1}{2}V\right)^{-1}\left(\mathbb{I} +\frac{1}{2}V\right).
" * Main.indentation * raw"```")
```

We show that the Cayley transform is a retraction for ``G = SO(N)`` at ``\mathbb{I}\in{}SO(N)``:
```@eval
Main.proof(raw"The Cayley transform trivially satisfies ``\mathrm{Cayley}(\mathbb{O}) = \mathbb{I}``. So what we have to show is the second condition for a retraction and that ``\mathrm{Cayley}(V)\in{}SO(N)``. For this take ``V\in\mathfrak{so}(N).`` We then have
" * Main.indentation * raw"```math
" * Main.indentation * raw"\frac{d}{dt}\bigg|_{t = 0}\mathrm{Cayley}(tV) = \frac{d}{dt}\bigg|_{t = 0}\left(\mathbb{I} - \frac{1}{2}tV\right)^{-1}\left(\mathbb{I} +\frac{1}{2}tV\right) = \frac{1}{2}V - \frac{1}{2}V^T = V,
" * Main.indentation * raw"```
" * Main.indentation * raw"which satisfies the second condition. We further have
" * Main.indentation * raw"```math
" * Main.indentation * raw"\frac{d}{dt}\bigg|_{t = 0}(\mathrm{Cayley}(tV))^T\mathrm{Cayley}(tV) = (\frac{1}{2}V - \frac{1}{2}V^T)^T + \frac{1}{2}V - \frac{1}{2}V^T = 0.
" * Main.indentation * raw"```
" * Main.indentation * raw"This proves that the Cayley transform maps to ``SO(N)``.")
```

We should mention that the factor ``\frac{1}{2}`` is sometimes left out in the definition of the Cayley transform when used in different contexts. But it is necessary for defining a retraction as without it the second condition is not satisfied.

```@eval
Main.remark(raw"We can also use the Cayley retraction at a different point than the identity ``\mathbb{I}.`` For this consider ``\bar{A}\in{}SO(N)`` and ``\bar{B}\in{}T_{\bar{A}}SO(N) = \{\bar{B}\in\mathbb{R}^{N\times{}N}: \bar{A}^T\bar{B} + \bar{B}^T\bar{A} = \mathbb{O}\}``. We then have ``\bar{A}^T\bar{B}\in\mathfrak{so}(N)`` and 
" * Main.indentation * raw"```math
" * Main.indentation * raw"    \overline{\mathrm{Cayley}}: T_{\bar{A}}SO(N) \to SO(N), \bar{B} \mapsto \bar{A}\mathrm{Cayley}(\bar{A}^T\bar{B}),
" * Main.indentation * raw"```
" * Main.indentation * raw"is a retraction ``\forall{}\bar{A}\in{}SO(N)``.")
```

As a retraction is always an approximation of the geodesic map, we now compare the [`cayley`](@ref) retraction for the example we introduced along [Riemannian manifolds](@ref "Geodesic Sprays and the Exponential Map"):

```@setup s2_retraction
using CairoMakie
```

```@setup s2_retraction
using GeometricOptimizers
import Random # hide
Random.seed!(123) # hide

Y = rand(StiefelManifold, 3, 1)

v = 2 * rand(3, 1)
Δ = v - Y * (v' * Y)

function do_setup(; theme=:light)
    text_color = theme == :dark ? :white : :black # hide
    fig = Figure(; backgroundcolor = :transparent, size = (900, 675)) # hide
    ax = Axis3(fig[1, 1]; # hide
        backgroundcolor = (:tomato, .5), # hide
        aspect = (1., 1., 1.), # hide
        xlabel = L"x_1", # hide
        ylabel = L"x_2", # hide
        zlabel = L"x_3", # hide
        xgridcolor = text_color, # hide
        ygridcolor = text_color, # hide
        zgridcolor = text_color, # hide
        xtickcolor = text_color, # hide
        ytickcolor = text_color, # hide
        ztickcolor = text_color, # hide
        xlabelcolor = text_color, # hide
        ylabelcolor = text_color, # hide
        zlabelcolor = text_color, # hide
        xypanelcolor = :transparent, # hide
        xzpanelcolor = :transparent, # hide
        yzpanelcolor = :transparent, # hide
        limits = ([-1, 1], [-1, 1], [-1, 1]),
        azimuth = π / 7, # hide
        elevation = π / 7, # hide
        # height = 75.,
        ) # hide

    # plot a sphere with radius one and origin 0
    surface!(ax, Main.sphere(1., [0., 0., 0.])...; alpha = .5, transparency = true)

    morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
    point_vec = ([Y[1]], [Y[2]], [Y[3]])
    scatter!(ax, point_vec...; color = morange, marker = :star5, markersize = 30)

    fig, ax, point_vec
end

mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)

nothing
```

```@example s2_retraction
η_increments = 0.2 : 0.2 : 5.4
Δ_increments = [Δ * η for η in η_increments]

Y_increments_geodesic = [geodesic(Y, Δ_increment) for Δ_increment in Δ_increments]
Y_increments_cayley = [cayley(Y, Δ_increment) for Δ_increment in Δ_increments]
nothing # hide
```

```@setup s2_retraction
function make_plot(; theme=:light) # hide

text_color = theme == :light ? :black : :white # hide

fig, ax, point_vec = do_setup(; theme = theme) # hide

Y_zeros = zeros(length(Y_increments_geodesic))
Y_geodesic_reshaped = [copy(Y_zeros), copy(Y_zeros), copy(Y_zeros)]
Y_cayley_reshaped = [copy(Y_zeros), copy(Y_zeros), copy(Y_zeros)]

zip_ob = zip(Y_increments_geodesic, Y_increments_cayley, axes(Y_increments_geodesic, 1))

for (Y_increment_geodesic, Y_increment_cayley, i) in zip_ob
    for d in (1, 2, 3)
        Y_geodesic_reshaped[d][i] = Y_increment_geodesic[d]

        Y_cayley_reshaped[d][i] = Y_increment_cayley[d]
    end
end

scatter!(ax, Y_geodesic_reshaped...; 
        color = mred, label = rich("geodesic retraction"; color = text_color), markersize = 15)

scatter!(ax, Y_cayley_reshaped...; 
        color = mblue, label = rich("Cayley retraction"; color = text_color), markersize = 15)

arrow_vec = ([Δ[1]], [Δ[2]], [Δ[3]]) # hide
arrows!(ax, point_vec..., arrow_vec...; color = mred, linewidth = .02) # hide
backgroundcolor = theme == :light ? :white : :transparent
axislegend(; position = (.82, .75), backgroundcolor = backgroundcolor, color = text_color) # hide

fig, ax, zip_ob, Y_increments_geodesic, Y_increments_cayley # hide
end # hide

fig_light = make_plot(; theme = :light)[1] # hide
fig_dark = make_plot(; theme = :dark)[1] # hide

CairoMakie.save("retraction_comparison_light.png", fig_light; px_per_unit = Main.output_type == :html ? 1.5 : 2) # hide
CairoMakie.save("retraction_comparison_dark.png", fig_dark; px_per_unit = Main.output_type == :html ? 1.5 : 2) # hide

nothing
```

![Comparison between the geodesic and the Cayley retraction.](retraction_comparison_light.png)
![Comparison between the geodesic and the Cayley retraction.](retraction_comparison_dark.png)

We see that for small ``\Delta`` increments the Cayley retraction seems to match the geodesic retraction very well, but for larger values there is a notable discrepancy. We can plot this discrepancy directly: 

```@setup s2_retraction
function plot_discrepancies(discrepancies; theme = :light)
    fig = Figure(; backgroundcolor = :transparent) # hide
    text_color = theme == :dark ? :white : :black # hide
    ax = Axis(fig[1, 1]; # hide
            backgroundcolor = :transparent, # hide
            xlabel = rich("η", font = :italic, color = text_color), # hide
            ylabel = rich("discrepancy", color = text_color), # hide
            ) # hide
    lines!(η_increments, discrepancies; label = rich("Discrepancies between geodesic and Cayley retraction", color = text_color), 
        linewidth = 2, color = mblue)

    axislegend(; position = (.22, .9), backgroundcolor = :transparent, color = text_color) # hide

    fig, ax
end
```

```@example s2_retraction
using LinearAlgebra: norm # hide
zip_ob = zip(Y_increments_geodesic, Y_increments_cayley, axes(Y_increments_geodesic, 1))
_, __, zip_ob, Y_increments_geodesic, Y_increments_cayley = make_plot() # hide
discrepancies = [norm(Y_geo_inc - Y_cay_inc) for (Y_geo_inc, Y_cay_inc, _) in zip_ob]
fig_light = plot_discrepancies(discrepancies; theme = :light)[1] # hide
fig_dark = plot_discrepancies(discrepancies; theme = :dark)[1] # hide
CairoMakie.save("retraction_discrepancy_light.png",        fig_light; px_per_unit = 1.3) # hide
CairoMakie.save("retraction_discrepancy_dark.png",   fig_dark; px_per_unit = 1.3) # hide
nothing
```

![Discrepancy between the geodesic and the Cayley retraction.](retraction_discrepancy_light.png)
![Discrepancy between the geodesic and the Cayley retraction.](retraction_discrepancy_dark.png)

## In `GeometricOptimizers`

The way we use *retractions*[^1] in `GeometricOptimizers` is slightly different from their classical definition:

[^1]: Classical retractions are also defined in `GeometricOptimizers` under the same name, i.e. there is e.g. a method [`cayley(::StiefelLieAlgHorMatrix)`](@ref) and a method [`cayley(::StiefelManifold, ::AbstractMatrix)`](@ref) (the latter being the classical retraction); but the user is *strongly discouraged* from using classical retractions as these are computationally inefficient.

```@eval
Main.definition(raw"Given a section ``\lambda:\mathcal{M}\to{}G,`` where ``\mathcal{M}`` is a homogeneous space, a **retraction** is a map ``\mathrm{Retraction}:\mathfrak{g}^\mathrm{hor}\to{}G`` such that 
" * Main.indentation * raw"```math
" * Main.indentation * raw"\Delta \mapsto \lambda(Y)\mathrm{Retraction}(\lambda(Y)^{-1}\Omega(\Delta)\lambda(Y))E,
" * Main.indentation * raw"```
" * Main.indentation * raw"is a classical retraction.")
```

This map ``\mathrm{Retraction}`` is also what was visualized in the figure on [the general optimization framework](@ref "Generalization to Homogeneous Spaces"). We now discuss how the geodesic retraction (exponential map) and the Cayley retraction are implemented in `GeometricOptimizers`.

## Retractions for Homogeneous Spaces

Here we harness special properties of homogeneous spaces to obtain computationally efficient retractions for the [Stiefel manifold](@ref "The Stiefel Manifold") and the [Grassmann manifold](@ref "The Grassmann Manifold"). This is also discussed in e.g. [bendokat2020grassmann, bendokat2021real](@cite).

The *geodesic retraction* is a retraction whose associated curve is also the unique geodesic. For many matrix Lie groups (including ``SO(N)``) geodesics are obtained by simply evaluating the exponential map [absil2008optimization, o1983semi](@cite):
 
```@eval
Main.theorem(raw"The geodesic on a compact matrix Lie group ``G`` with bi-invariant metric for ``\bar{B}\in{}T_{\bar{A}}G`` is simply
" * Main.indentation * raw"```math
" * Main.indentation * raw"\gamma(t) = \exp(t\cdot{}\bar{B}\bar{A}^{-1})\bar{A} = \bar{A}\exp(t\cdot{}\bar{A}^{-1}\bar{B}),
" * Main.indentation * raw"```
" * Main.indentation * raw"where ``\exp:\mathfrak{g}\to{}G`` is the matrix exponential map.")
```

The last equality in the equation above is a result of:

```math
\begin{aligned}
\exp(\bar{A}^{-1}\hat{B}\bar{A}) = \sum_{k=0}^\infty\frac{1}{k!}(\bar{A}^{-1}\hat{B}\bar{A})^k & = \sum_{k=0}^\infty \frac{1}{k!}\underbrace{(\bar{A}^{-1}\hat{B}\bar{A})\cdots(\bar{A}^{-1}\hat{B}\bar{A})}_{\text{$k$ times}} \\ & = \sum_{k=0}^\infty \frac{1}{k!} \bar{A}^{-1} \hat{B}^k \bar{A} = \bar{A}^{-1}\exp(\hat{B})\bar{A}.
\end{aligned}
```

Because ``SO(N)`` is compact and we furnish it with the canonical metric, i.e. 

```math
    g:T_{\bar{A}}G\times{}T_{\bar{A}}G \to \mathbb{R}, (B_1, B_2) \mapsto \mathrm{Tr}(B_1^TB_2) = \mathrm{Tr}((B_1\bar{A}^{-1})^T(B_2\bar{A}^{-1})),
```

its geodesics are thus equivalent to the exponential maps. We now use this observation to obtain an expression for the geodesics on the [Stiefel manifold](@ref "The Stiefel Manifold"). We use the following theorem from [o1983semi; Proposition 25.7](@cite):

```@eval
Main.theorem(raw"The geodesics for a naturally reductive homogeneous space ``\mathcal{M}`` starting at ``Y`` are given by:
" * Main.indentation * raw"```math
" * Main.indentation * raw"\gamma_{\Delta}(t) = \exp(t\cdot\Omega(\Delta))Y,
" * Main.indentation * raw"```
" * Main.indentation * raw"where the ``\exp`` is the exponential map for the Lie group ``G`` corresponding to ``\mathcal{M}``.")
```

The theorem requires the homogeneous space to be naturally reductive: 

```@eval
Main.definition(raw"A homogeneous space is called **naturally reductive** if the following two conditions hold:
" * Main.indentation * raw"1. ``\bar{A}^{-1}\bar{B}\bar{A}\in\mathfrak{g}^\mathrm{hor}`` for every ``\bar{B}\in\mathfrak{g}^\mathrm{hor}`` and ``\bar{A}\in\exp(\mathfrak{g}^\mathrm{ver}``),
" * Main.indentation * raw"2. ``g([X, Y]^\mathrm{hor}, Z) = g(X, [Y, Z]^\mathrm{hor})`` for all ``X, Y, Z \in \mathfrak{g}^\mathrm{hor}``,
" * Main.indentation * raw"where ``[X, Y]^\mathrm{hor} = \Omega(XYE - YXE)``. If only the first condition holds the homogeneous space is called **reductive** (but not **naturally reductive**).")
```

We state here without proof that the [Stiefel manifold](@ref "The Stiefel Manifold") and the [Grassmann manifold](@ref "The Grassmann Manifold") are naturally reductive. We can however provide empirical evidence here:

```@example naturally_reductive
using GeometricOptimizers # hide
import Random # hide
Random.seed!(123) # hide
B̄ = rand(SkewSymMatrix, 6) # ∈ 𝔤
Ā = exp(B̄ - StiefelLieAlgHorMatrix(B̄, 3)) # ∈ exp(𝔤ᵛᵉʳ)

X = rand(StiefelLieAlgHorMatrix, 6, 3) # ∈ 𝔤ʰᵒʳ
Y = rand(StiefelLieAlgHorMatrix, 6, 3) # ∈ 𝔤ʰᵒʳ
Z = rand(StiefelLieAlgHorMatrix, 6, 3) # ∈ 𝔤ʰᵒʳ

@assert StiefelLieAlgHorMatrix(Ā' * X * Ā, 3) ≈ Ā' * X * Ā # hide
Ā' * X * Ā # this has to be in 𝔤ʰᵒʳ for St(3, 6) to be reductive
```

verifies the first property and

```@example naturally_reductive
using LinearAlgebra: tr # hide
adʰᵒʳ(X, Y) = StiefelLieAlgHorMatrix(X * Y - Y * X, 3)

@assert tr(adʰᵒʳ(X, Y)' * Z) ≈ tr(X' * adʰᵒʳ(Y, Z)) # hide
tr(adʰᵒʳ(X, Y)' * Z) ≈ tr(X' * adʰᵒʳ(Y, Z))
```

verifies the second.

In `GeometricOptimizers` we always work with elements in ``\mathfrak{g}^\mathrm{hor}`` and the Lie group ``G`` is always ``SO(N)``. We hence use:

```math
    \gamma_\Delta(t) = \exp(\lambda(Y)\lambda(Y)^{-1}\Omega(\Delta)\lambda(Y)\lambda(Y)^{-1})Y = \lambda(Y)\exp(\lambda(Y)^{-1}\Omega(\Delta)\lambda(Y))E.
```

Based on this we define the maps: 

```math
\mathtt{geodesic}: \mathfrak{g}^\mathrm{hor} \to G, \bar{B} \mapsto \exp(\bar{B}),
```

and

```math
\mathtt{cayley}: \mathfrak{g}^\mathrm{hor} \to G, \bar{B} \mapsto \mathrm{Cayley}(\bar{B}),
```

where ``\bar{B} = \lambda(Y)^{-1}\Omega(\Delta)\lambda(Y)``. These expressions for [`geodesic`](@ref) and [`cayley`](@ref) are the ones that we typically use in `GeometricOptimizers` for computational reasons. We show how we can utilize the sparse structure of ``\mathfrak{g}^\mathrm{hor}`` for computing the geodesic retraction and the Cayley retraction (i.e. the expressions ``\exp(\bar{B})`` and ``\mathrm{Cayley}(\bar{B})`` for ``\bar{B}\in\mathfrak{g}^\mathrm{hor}``). Similar derivations can be found in [celledoni2000approximating, fraikin2007optimization, bendokat2021real](@cite).

```@eval
Main.remark(raw"Further note that, even though the global section ``\lambda:\mathcal{M} \to G`` is not unique, the final geodesic ``\gamma_\Delta(t) = \lambda(Y)\exp(\lambda(Y)^{-1}\Omega(\Delta)\lambda(Y))E`` does not depend on the particular section we choose.")
```

### The Geodesic Retraction

An element ``\bar{B}`` of ``\mathfrak{g}^\mathrm{hor}`` can be written as:

```math
\bar{B} = \begin{bmatrix}
    A & -B^T \\ 
    B & \mathbb{O}
\end{bmatrix} = \begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} \begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix} =: B'(B'')^T,
```

where we exploit the sparse structure of the array, i.e. it is a multiplication of a ``N\times2n`` with a ``2n\times{}N`` matrix.

We further use the following: 

```math
    \begin{aligned}
    \exp(B'(B'')^T) & = \sum_{n=0}^\infty \frac{1}{n!} (B'(B'')^T)^n = \mathbb{I} + \sum_{n=1}^\infty \frac{1}{n!} B'((B'')^TB')^{n-1}(B'')^T \\
    & = \mathbb{I} + B'\left( \sum_{n=1}^\infty \frac{1}{n!} ((B'')^TB')^{n-1} \right)B'' =: \mathbb{I} + B'\mathfrak{A}(B', B'')B'',
    \end{aligned}
```

where we defined ``\mathfrak{A}(B', B'') := \sum_{n=1}^\infty \frac{1}{n!} ((B'')^TB')^{n-1}.`` Note that evaluating ``\mathfrak{A}`` relies on computing products of *small* matrices of size ``2n\times2n.`` We do this by relying on a simple Taylor expansion, implemented as `GeometricOptimizers.𝔄` (see the [`GeometricOptimizers` documentation](https://juliagni.github.io/GeometricOptimizers.jl/stable/) for its docstring). 

The final expression we obtain is: 

```math
\exp(\bar{B}) = \mathbb{I} + B' \mathfrak{A}(B', B'')  (B'')^T
```

### The Cayley Retraction

For the Cayley retraction we leverage the decomposition of ``\bar{B} = B'(B'')^T\in\mathfrak{g}^\mathrm{hor}`` through the *Sherman-Morrison-Woodbury formula*:

```math
(\mathbb{I} - \frac{1}{2}B'(B'')^T)^{-1} = \mathbb{I} + \frac{1}{2}B'(\mathbb{I} - \frac{1}{2}B'(B'')^T)^{-1}(B'')^T
```

So what we have to compute the inverse of:

```math
\mathbb{I} - \frac{1}{2}\begin{bmatrix}  \mathbb{I} & \mathbb{O} \\ \frac{1}{2}A & -B^T  \end{bmatrix}\begin{bmatrix}  \frac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix} = 
\begin{bmatrix}  \mathbb{I} - \frac{1}{4}A & - \frac{1}{2}\mathbb{I} \\ \frac{1}{2}B^TB - \frac{1}{8}A^2 & \mathbb{I} - \frac{1}{4}A  \end{bmatrix}.
```

By leveraging the sparse structure of the matrices in ``\mathfrak{g}^\mathrm{hor}`` we arrive at the following expression for the Cayley retraction (similar to the case of the geodesic retraction):

```math
\mathrm{Cayley}(\bar{B}) = \mathbb{I} + \frac{1}{2} B' \left(\mathbb{I}_{2n} - \frac{1}{2} (B'')^T B'\right)^{-1} (B'')^T \left(\mathbb{I} + \frac{1}{2} \bar{B}\right),
```

where we have abbreviated ``\mathbb{I} := \mathbb{I}_N.`` We conclude with a remark:

```@eval
Main.remark(raw"As mentioned previously the Lie group ``SO(N)``, i.e. the one corresponding to the Stiefel manifold and the Grassmann manifold, has a bi-invariant Riemannian metric associated with it: ``(B_1,B_2)\mapsto \mathrm{Tr}(B_1^TB_2)``. For other Lie groups (e.g. the symplectic group) the situation is slightly more difficult.")
```

One of such Lie groups is the *group of symplectic matrices* [bendokat2021real](@cite); for this group the expressions presented here are more complicated.

## Where a retraction sits in the algorithm

The optimizer never moves a point of the manifold directly. It keeps a section
``\Lambda^{(t)} \in G`` with ``Y^{(t)} = \Lambda^{(t)}E``, and one step is

```math
\Lambda^{(t+1)} \gets \Lambda^{(t)}\,\mathrm{retraction}(W^{(t)}),
\qquad
Y^{(t+1)} \gets \Lambda^{(t+1)}E,
```

with ``W^{(t)} \in \mathfrak{g}^\mathrm{hor}`` the direction the method produced. So what a
retraction has to return is an element of the *group*, not of the manifold, and the point follows
from it. This is the *extended* retraction of [brantner2023generalizing](@cite); `update_section!`
performs the first line and `apply_section` the second. Because the retracted element is in ``G``, a
retracted point is on the manifold by construction, and [`check`](@ref) — which measures
``\|Y^TY - \mathbb{I}\|`` — returns nothing but accumulated round-off.

An instance is passed to [`Optimizer`](@ref) as `retraction = Cayley()` or `retraction = Geodesic()`.
The default is [`Cayley`](@ref).

## Both retractions factor the lift

A horizontal lift is sparse, and both retractions exploit that in the same way. For the Stiefel
manifold, [`lift_factors`](@ref) writes

```math
\bar{B} = \begin{bmatrix} A & -B^T \\ B & \mathbb{O} \end{bmatrix}
        = \begin{bmatrix} \tfrac{1}{2}A & \mathbb{I} \\ B & \mathbb{O} \end{bmatrix}
          \begin{bmatrix} \mathbb{I} & \mathbb{O} \\ \tfrac{1}{2}A & -B^T \end{bmatrix}
        =: B'(B'')^T,
```

with two ``N\times{}2n`` factors — for a [`GrassmannLieAlgHorMatrix`](@ref) the same expression with
``A \equiv \mathbb{O}``. Whatever matrix function a retraction needs is then evaluated on the
``2n\times{}2n`` product

```math
X := (B'')^TB',
```

which is small even when ``N`` is large. The matrix function is therefore priced by the number of
columns ``n`` and not by the dimension ``N`` of the ambient space; what is left of the retraction is
the assembly around it, which is ``O(N^2n)`` and not the ``O(N^3)`` an ``N\times{}N`` matrix function
would cost. And, as the next sections show, the factorisation is also where the *accuracy* of the
exponential is decided, because ``X`` is a considerably worse-behaved matrix than ``\bar{B}`` is.

## `Cayley` and `Geodesic`

[`Cayley`](@ref) is the Cayley transform,

```math
\mathrm{Cayley}(\bar{B}) = \left(\mathbb{I} - \tfrac{1}{2}\bar{B}\right)^{-1}
                           \left(\mathbb{I} + \tfrac{1}{2}\bar{B}\right),
```

which maps a skew-symmetric matrix into ``SO(N)`` exactly. [`cayley`](@ref) never forms the
``N\times{}N`` inverse: with the factorisation above it inverts a ``2n\times{}2n`` matrix instead. No
matrix *function* is involved anywhere, only a solve, so there is no series to cancel and no step
size at which the transform breaks down the way an unscaled series does. That is not the same as
being insensitive to the size of the lift: `check` still climbs from ``10^{-15}`` to
``3\cdot10^{-13}`` over the sweep [below](@ref "Staying on the manifold"), which is the largest drift
of anything on this page other than [`TaylorSeries`](@ref).

[`Geodesic`](@ref) is the exponential map,

```math
\mathrm{Geodesic}(\bar{B}) = \exp(\bar{B}),
```

i.e. the true geodesic of the manifold [edelman1998geometry](@cite). The difference that matters to
the rest of the package is that ``\alpha \mapsto \exp(\alpha\bar{B})`` is a **one-parameter
subgroup**,

```math
\exp\left((\alpha + \beta)\bar{B}\right) = \exp(\alpha\bar{B})\exp(\beta\bar{B}),
```

and ``\alpha \mapsto \mathrm{Cayley}(\alpha\bar{B})`` is not. Everything in sight is a rational
function of ``\bar{B}``, so it all commutes and the two products can be compared directly: with
``t = \alpha/2`` and ``s = \beta/2``,

```math
\mathrm{Cayley}(\alpha\bar{B})\,\mathrm{Cayley}(\beta\bar{B})
  = \big[\mathbb{I} + (t + s)\bar{B} + ts\bar{B}^2\big]
    \big[\mathbb{I} - (t + s)\bar{B} + ts\bar{B}^2\big]^{-1} ,
```

against ``[\mathbb{I} + (t+s)\bar{B}][\mathbb{I} - (t+s)\bar{B}]^{-1}`` for
``\mathrm{Cayley}((\alpha+\beta)\bar{B})``. The two agree only where ``ts\bar{B}^2 = \mathbb{O}``, and
the gap is not small: on a random ``\mathfrak{g}^\mathrm{hor}`` element of ``\operatorname{St}(6,3)``
with ``\|\bar{B}\| = 2.99`` it is ``1.28`` at ``\alpha = \beta = 1``, where the same difference for
``\exp`` is ``8\times10^{-16}``.

So the generator of the curve's velocity turns with ``\alpha`` instead of staying ``\bar{B}``. That
is what [`retraction_differential`](@ref) supplies, and with it [`trial_slope`](@ref) is the exact
derivative of a line search's merit function under *either* retraction. Before 0.2.0 the slope was
paired against ``\bar{B}`` regardless, which made it first-order under [`Cayley`](@ref) — 8.9% off at
``\alpha = 0.5`` and 36% at ``\alpha = 1`` on the ``\operatorname{St}(6,3)`` problem of
[Linesearches on Manifolds](@ref), which is where that measurement lives.

!!! info "It is the parameterisation and not the approximation"
    The natural reading of "`Cayley` is a retraction and `Geodesic` is the exponential map" is that
    the slope was wrong because the *curve* was approximate. That is not the mechanism, and the
    smallest counterexample separates them. Take ``N = 2`` and ``\bar{B} = J = \left(\begin{smallmatrix}0 & -1\\ 1 & 0\end{smallmatrix}\right)``.
    Then

    ```math
    \mathrm{Cayley}(\alpha{}J) = \exp\big(2\arctan(\tfrac{\alpha}{2})\,J\big)
    ```

    to the last bit — the Cayley curve is the geodesic, *exactly*, with nothing approximate about it.
    It is traversed at a different speed:

    | ``\alpha`` | 0 | 0.5 | 1 | 2 | 4 |
    |---|---|---|---|---|---|
    | angle turned, `Cayley` | 0 | 0.4900 | 0.9273 | 1.5708 | 2.2143 |
    | angle turned, `Geodesic` | 0 | 0.5 | 1 | 2 | 4 |
    | ``d\theta/d\alpha`` for `Cayley` | 1 | 0.9412 | 0.8 | 0.5 | 0.2 |

    and that last row is exactly ``D(\alpha) = \bar{B}(\mathbb{I} - \frac{\alpha^2}{4}\bar{B}^2)^{-1}``
    at ``J^2 = -\mathbb{I}``, i.e. ``J/(1 + \alpha^2/4)``. Pairing the gradient against ``J`` rather
    than against ``D(\alpha)`` therefore overstates ``\varphi'`` by ``1 + \alpha^2/4`` — 6.2% at
    ``\alpha = 0.5``, 25% at ``\alpha = 1``, 100% at ``\alpha = 2``. A curve can be exactly right and
    still give the wrong ``\varphi'``, because ``\varphi'`` is a derivative with respect to the
    *parameter*.

    That also says how to read the percentages above: to leading order the error is
    ``\alpha^2\lambda^2/4`` for an eigenvalue ``\pm{}i\lambda`` of ``\bar{B}``, so it grows with the
    step *and* with the size of the lift, and a figure quoted without its problem means little. The
    same measurement on the ``\operatorname{St}(3,1)`` sphere of `manifold_linesearch_tests.jl` gives
    4.5%, 18%, 72% and 288% at ``\alpha = 0.25, 0.5, 1, 2``.

The retraction used to separate the two polynomial line searches on the SVD problem, where they left
the manifold under [`Cayley`](@ref) for every optimizer method and stayed on it under
[`Geodesic`](@ref). That was issue A1b, and the exact differential closed only one of its four cases:
the cause is the size of the step those searches extrapolate to, not the slope they extrapolate from.
Bounding the step closes it — see [`DEFAULT_STEP_CEILING`](@ref) — and the retraction no longer
separates them. It was the *amplifier* rather than the cause, which is what the `check` table further
down measures.

Cost no longer separates them the way it once did. [`cayley`](@ref) finishes with a product of two
``N\times{}N`` matrices, which is ``O(N^3)``, where [`geodesic`](@ref) only assembles
``\mathbb{I} + B'\mathfrak{A}(X)(B'')^T`` at ``O(N^2n)``, so since 0.2.0 [`Geodesic`](@ref) is the
cheaper of the two for ``N \gtrsim 50``; the table under [What they cost](@ref) has the figures.

## The exponential needs an algorithm

Exponentiating a full ``N\times{}N`` matrix would throw away the sparsity of the lift. The
factorisation avoids it: the exponential of a product taken in this order is

```math
\exp\left(B'(B'')^T\right) = \mathbb{I} + B'\,\mathfrak{A}(X)\,(B'')^T,
\qquad
\mathfrak{A}(X) = \sum_{k=1}^\infty \frac{X^{k-1}}{k!},
```

so the whole computation reduces to one ``2n\times{}2n`` matrix function. ``\mathfrak{A}`` is the
function usually written ``\varphi_1(X) = \left(\exp(X) - \mathbb{I}\right)X^{-1}``, though it is
defined by the series and is perfectly regular at a singular ``X``.

Evaluating ``\mathfrak{A}`` by summing that series is the obvious thing to do and it is what every
version of this package up to 0.2.0 did. It is also wrong for any but a small argument, and the
argument here is not small. ``X``'s lower-left block is ``\tfrac{1}{4}A^2 - B^TB``, so

```math
\|X\| \approx \tfrac{1}{4}\|\bar{B}\|^2
\qquad\text{while}\qquad
\rho(X) \approx \|\bar{B}\|,
```

because the eigenvalues of ``X`` are the nonzero — purely imaginary — eigenvalues of the skew matrix
``\bar{B}``. A norm quadratically larger than the spectral radius is a strongly non-normal matrix,
and on such an argument the terms of the series cancel catastrophically: at
``\|\bar{B}\| \approx 79`` the partial sum reaches ``2.5\cdot10^{18}`` where the result is of order
one. Stopping the summation when a *term* falls below `eps` then leaves a relative error of
``\varepsilon\|\mathfrak{A}(X)\|`` rather than ``\varepsilon``, and the retracted point is not on the
manifold in any sense. That the direct series is not a method for the matrix exponential is a very
old observation [moler2003nineteen](@cite); what is specific here is that the factorisation makes the
argument *worse* than the matrix one started with.

The remedy is a choice, and [`Geodesic`](@ref) makes it one the caller can see:

```julia
Geodesic(ScaledSquaring())   # the default, and `Geodesic()`
Geodesic(NativePade())       # independent and backend-portable
Geodesic(AugmentedPade())
Geodesic(ProjectedSkew())
Geodesic(TaylorSeries())     # the pre-0.2.0 behaviour; not a usable retraction
```

All five are subtypes of [`AbstractExponentialAlgorithm`](@ref) and all five return the exponential
map, so the one-parameter subgroup property above holds for every one of them. What follows is what
each does and what it trades.

## `ScaledSquaring`

[`ScaledSquaring`](@ref) is the default. The series is only inaccurate for a large argument, so halve
the argument until it is small, sum the series there, and undo the halving by squaring — the standard
remedy for a matrix exponential
[higham2005scaling, higham2008functions, almohy2010new](@cite), and what
`Base.exp` itself does.

The one thing that needs care is that squaring must not cost ``O(N^3)``. It does not, because the
low-rank form is closed under squaring:

```math
\left(\mathbb{I} + B'W(B'')^T\right)^2 = \mathbb{I} + B'\left(2W + WXW\right)(B'')^T,
```

so one squaring of the assembled exponential is one application of ``W \mapsto 2W + WXW`` at
``2n\times{}2n``, and no ``N\times{}N`` matrix is ever formed, let alone squared. With ``s`` chosen so
that ``\|X\|_1/2^s \leq \theta``, the algorithm is `s` small-matrix updates on top of a series that
now converges in a handful of terms. That makes it *cheaper* than summing the unscaled series, not
merely more accurate — by 1.7× at ``N = 200``, ``n = 10`` and 4.6× at ``N = 500``, ``n = 50``.

The complete computation is:

1. **Choose the scaling.** Set
   ``s = \max(0, \lceil\log_2(\|X\|_1/\theta)\rceil)`` and ``\alpha = 2^s``.
2. **Evaluate the small series.** Compute
   ``W = \mathfrak{A}(X/\alpha)/\alpha``. The scaled argument has
   ``\|X/\alpha\|_1 \leq \theta``, so the Taylor series converges without catastrophic cancellation.
3. **Undo the scaling.** Repeat ``W \leftarrow 2W + WXW`` exactly ``s`` times. Each update squares
   the represented exponential and restores one factor of two.
4. **Return the result.** After the loop, ``W = \mathfrak{A}(X)``, so
   ``\mathbb{I} + B'W(B'')^T = \exp(B'(B'')^T)``.

The division by ``\alpha`` in the initial ``W`` is essential: it scales the factor ``B'`` implicitly,
allowing every subsequent squaring to remain a ``2n\times{}2n`` update.

The threshold `θ` is the algorithm's one parameter — positional, `ScaledSquaring(0.5)`, and defaulted
to `0.5` — and it barely matters: at ``\|\bar{B}\| \approx 155`` every ``\theta \in [0.125, 4]`` — a
32-fold range — gives a `check` between ``9.9\cdot10^{-15}`` and ``5.0\cdot10^{-14}`` and a forward
error between ``6.4\cdot10^{-15}`` and ``8.2\cdot10^{-15}``. That sweep is [measured at build
time](@ref "The threshold `θ` needs no tuning") below; there is no reason to tune it.

**Advantages.** The cheapest of the five on the ``\mathfrak{A}`` call itself — `0.021 ms` at
``N = 200``, ``n = 10``, against [`NativePade`](@ref)'s `0.037 ms` and [`AugmentedPade`](@ref)'s
`0.053 ms` — and as close to `exp(Matrix(B))` as [`AugmentedPade`](@ref), which is as close as
anything here gets. In the whole-retraction table below it and [`NativePade`](@ref) are level to
within the run-to-run noise at every size except ``n = 50``, where the ``2n\times{}2n`` argument is
finally large enough for the difference to show; the isolated call is where to look. And — because it
uses nothing but matrix products, norms and a kernel-written identity — one of the two usable
algorithms that run unchanged on a `KernelAbstractions` GPU backend. It remains the default because
it is the cheaper of those two.

Keeping that property is why the norm is taken by [`GeometricOptimizers.opnorm₁`](@ref) rather than by
`LinearAlgebra.opnorm(X, 1)` — the latter is a scalar-indexing double loop, and scalar indexing is
exactly what a GPU array cannot serve — and why the identities it needs come from
[`GeometricOptimizers.unit_matrix`](@ref) rather than from `Base.one`, whose diagonal write is the
same hazard one level down.

**Disadvantages.** Its orthogonality is the outcome of an arithmetic cancellation rather than a
structural property, so `check` does drift upwards with the size of the lift — from ``10^{-15}`` to
around ``7\cdot10^{-14}`` over the sweep below, and considerably further in `Float32`. Only
[`ProjectedSkew`](@ref) avoids that drift; [`Cayley`](@ref) has more of it. And it takes about twice
the squarings it needs:

!!! note "The halving count is loose"
    ``s`` is taken from the norm, ``s = \lceil\log_2(\|X\|_1/\theta)\rceil``, and
    ``\|X\| \approx \|\bar{B}\|^2/4`` — so ``s \approx 2\log_2\|\bar{B}\|`` where
    ``\log_2\|\bar{B}\|`` would do, since the spectral radius is only ``\approx\|\bar{B}\|``. Each
    squaring amplifies the error, so this costs both time and accuracy. It is left alone because the
    tighter bound needs the spectral radius, and an eigenvalue computation would forfeit precisely
    the freedom from dense LAPACK that makes this the default algorithm. [`NativePade`](@ref) takes
    ``s`` the same way and inherits the whole of this.

## `NativePade`

[`NativePade`](@ref) evaluates ``\mathfrak{A}`` directly at ``2n\times{}2n`` with the degree-6
diagonal Padé approximant ``q_6(X)^{-1}p_6(X)``. Nothing in it is new; what is assembled is the
pairing. ``q_6`` is the denominator of the ``[7/6]`` Padé approximant of ``\exp``, whose closed form
is standard [higham2005scaling, higham2008functions](@cite), and ``p_6`` is that approximant's
numerator rearranged — with ``\exp(x) \approx N(x)/D(x)``,

```math
\varphi_1(x) = \frac{\exp(x) - 1}{x} \approx \frac{N(x) - D(x)}{x\,D(x)},
```

and ``N - D`` divides by ``x`` exactly, both having constant term one, which makes ``p_6`` degree 6
where ``N`` is degree 7 and inherits the ``O(x^{13})`` order. It scales to ``\|X\|_1 \leq 0.5``
first. There ``q_6`` differs from the identity by at most `0.26` in one-norm, so the Newton--Schulz
iteration [higham2008functions](@cite) can replace the dense solve that a rational approximant
normally needs: starting from the identity, five refinements square the inverse residual five times
over, to ``(\mathbb{I}-q_6)^{32}``. That is the whole point — an LU is no more portable than
`Base.exp` is. The numerator and denominator share ``X^2`` and ``X^4``, and the same low-rank
squaring recursion [`ScaledSquaring`](@ref) uses restores the scale.

!!! note "What justifies `θ = 1/2`, and what does not"
    Not a backward-error table. The ``\theta_m`` of [higham2005scaling, almohy2010new](@cite) are
    derived for ``\exp`` rather than ``\varphi_1``, and they bound a backward error in ``\|X\|``
    — the least informative norm available here, since ``\|X\| \approx \|\bar{B}\|^2/4`` against
    a spectral radius of only ``\approx\|\bar{B}\|``. What justifies the threshold is narrower:
    the Newton--Schulz residual bound, and the measured forward error over the norm sweep and over
    400 random arguments. A backward-error criterion for ``\varphi_1`` on a strongly non-normal
    argument is one of the things [#52](https://github.com/JuliaGNI/GeometricOptimizers.jl/issues/52)
    asked for and this does not settle.

**Advantages.** It never forms a matrix larger than ``2n\times{}2n`` and uses only reductions,
matrix products and a kernel-written identity. That makes it the independent implementation
[`ScaledSquaring`](@ref) can be checked against on a backend that forbids scalar indexing — which is
what `test/retractions/exponential_accuracy.jl` does, on a `JLArray`. At ``\|\bar{B}\| = 361`` its
`check` is `3.9e-14` and its forward error `2.1e-14`, indistinguishable in `Float64` from both
[`ScaledSquaring`](@ref) and [`AugmentedPade`](@ref).

**Disadvantages.** Its fixed rational evaluation does more small matrix products than
[`ScaledSquaring`](@ref): at ``N = 200``, ``n = 10`` the ``\mathfrak{A}`` call costs `0.037 ms`
against `0.021 ms`, though it stays under [`AugmentedPade`](@ref)'s `0.053 ms`. It also **allocates**
the most of the three — `330 KiB` against `201` and `114` — and that is the figure least likely to
stay a constant factor on a GPU backend, where an allocation can cost a synchronisation rather than a
`malloc`. It is a little pointed that [`AugmentedPade`](@ref), which throws three quarters of its work
away, is the lightest allocator of the three: `Base.exp` reuses buffers where both native algorithms
build a fresh ``2n\times{}2n`` temporary per operation.

And the `Float64` indifference above does not carry to `Float32`. At the top of the norm sweep its
`check` is ``1.0\cdot10^{-4}`` against ``4.0\cdot10^{-5}`` for [`ScaledSquaring`](@ref) and
``3.4\cdot10^{-5}`` for [`AugmentedPade`](@ref) — the worst of the three — while its *forward* error
there is ``1.2\cdot10^{-5}`` against ``1.1\cdot10^{-5}`` and ``9.6\cdot10^{-6}``, which is no
outlier at all. So what degrades in `Float32` is the orthogonality of the retracted point rather than
the agreement with the exponential. It is the portable cross-check, not the default.

!!! warning "`θ` is a ceiling here, not a preference"
    `ScaledSquaring(θ)` accepts any positive threshold and stays accurate over a 32-fold range,
    because it sums its series until the terms vanish. `NativePade` does a *fixed* five
    Newton--Schulz steps, so past ``\theta \approx 1`` the inverse it computes stops being one —
    worst relative error over 400 random ``6\times6`` arguments of one-norm exactly ``\theta`` is
    `6e-16` at ``\theta = 1``, `1.2e-10` at ``\theta = 3/2``, `1.1e-5` at ``\theta = 2`` and `169`
    at ``\theta = 3`` — and nothing about the result says so. `NativePade(θ)` therefore refuses
    ``\theta > 1/2``. Lowering it is safe and only adds squarings.

## `AugmentedPade`

[`AugmentedPade`](@ref) evaluates ``\mathfrak{A}`` as a block of a larger *ordinary* exponential. For
the ``4n\times{}4n`` augmented matrix,

```math
\exp\begin{pmatrix} X & \mathbb{I} \\ \mathbb{O} & \mathbb{O} \end{pmatrix}
= \begin{pmatrix} \exp(X) & \mathfrak{A}(X) \\ \mathbb{O} & \mathbb{I} \end{pmatrix},
```

which is the standard device for getting a ``\varphi`` function out of an exponential routine
[sidje1998expokit, higham2008functions](@cite). One call to `Base.exp` therefore returns
``\mathfrak{A}(X)`` in its upper-right block. That hands the numerics to Julia's own exponential — a
degree-13 Padé approximant with its own scaling and squaring [higham2005scaling,
almohy2010new](@cite) — at the cost of exponentiating a matrix four times the size and discarding
three quarters of it.

**Advantages.** It introduces no new numerics at all. Everything delicate is done by the most
heavily exercised matrix-exponential implementation available, which is why it remains the CPU
reference in `test/retractions/exponential_accuracy.jl`. Accuracy is the same order as
[`ScaledSquaring`](@ref)'s.

**Disadvantages.** Three quarters of the work is thrown away, so the ``\mathfrak{A}`` call itself is
about twice as expensive as [`ScaledSquaring`](@ref)'s — though much less than twice once the
``N\times{}N`` assembly around it is counted. In `Float32` it, [`ScaledSquaring`](@ref) and
[`NativePade`](@ref) trade last place across the sweep below, and at the large lifts all three are an
order of magnitude behind [`ProjectedSkew`](@ref). And `Base.exp` on a dense matrix needs LAPACK:

!!! warning "CPU only"
    Neither this nor [`ProjectedSkew`](@ref) runs on a GPU backend. Use [`ScaledSquaring`](@ref),
    or [`NativePade`](@ref) for an independent portable cross-check, there.

## `ProjectedSkew`

[`ProjectedSkew`](@ref) does not go through ``\mathfrak{A}`` at all. It exponentiates the lift in a
basis of the lift's own range, where it is a small skew-symmetric matrix.

``\bar{B}`` is skew-symmetric of rank at most ``2n``, so its range and its row space coincide and
both sit inside the range of ``B'``. A thin QR of ``B'`` gives an ``N\times{}2n`` orthonormal ``Q``
with ``\bar{B} = QMQ^T``, where ``M = Q^T\bar{B}Q`` is skew-symmetric and ``2n\times{}2n``, and

```math
\exp(\bar{B}) = \mathbb{I} + Q\left(\exp(M) - \mathbb{I}\right)Q^T.
```

``\exp(M)`` is then formed from an eigendecomposition rather than from a series: ``iM`` is Hermitian
for real skew ``M``, so ``M = -iV\Lambda{}V^*`` with ``V`` unitary and ``\Lambda`` real, and

```math
\exp(M) = \Re\left(V e^{-i\Lambda} V^*\right),
```

which is orthogonal **by construction** — a product of a unitary matrix, a diagonal of unit-modulus
numbers, and a unitary matrix — rather than by cancellation.

**Advantages.** It is the only algorithm whose `check` does not degrade with the size of the lift.
Over the sweep below it stays between ``2\cdot10^{-15}`` and ``5\cdot10^{-15}`` from
``\|\bar{B}\| \approx 6`` to ``\|\bar{B}\| \approx 770``, where the other three drift from
``10^{-15}`` to around ``7\cdot10^{-14}``. The gap is widest in `Float32`, where the other three are
at the mercy of the format: over the same sweep their `check` climbs into the ``10^{-5}``s — into the
``10^{-4}``s for [`NativePade`](@ref) — while this stays at a few ``10^{-6}`` from one end to the
other. That is the case for choosing it — a long `Float32` run, where
the departure from the manifold accumulates over thousands of steps and staying on the manifold
matters more than agreeing with the exponential to the last bit.

**Disadvantages.** It usually has the largest forward error of the four against `exp(Matrix(B))` —
up to about 4.4× [`ScaledSquaring`](@ref)'s, and largest at all but the top of the sweep measured
below. It needs a `qr` and an `eigen` instead of matrix products, which costs 1.2×–1.5× over the
sizes measured below and rules out a GPU backend. And because it bypasses ``\mathfrak{A}``, it is the
one algorithm that specialises [`geodesic`](@ref) directly rather than supplying a method of
[`GeometricOptimizers.𝔄`](@ref) — worth knowing if you call ``\mathfrak{A}`` yourself, since
`𝔄(X, ProjectedSkew())` does not exist.

## `TaylorSeries`

[`TaylorSeries`](@ref) sums the series for ``\mathfrak{A}`` directly, without scaling, terminating
when a term falls below `eps`. It is the behaviour of every version of this package up to 0.2.0.

!!! danger "This is not a usable retraction"
    It is retained only so that the regression is reproducible from the test suite and so the
    working algorithms have a baseline to be compared against. Its column in the first table
    below is what it does: already at ``10^{-12}`` by ``\|\bar{B}\| \approx 18``, off the manifold
    by any standard at ``37``, meaningless at ``79``, and overflowed to `NaN` by ``767``. Do not
    select it.

Two things about it are worth recording rather than merely deprecating. The failure is *silent*: an
optimizer using it takes a step, gets a matrix back, and nothing anywhere reports that the matrix is
not on the manifold — which is why the defect survived until [`check`](@ref) was made generic over
[`Manifold`](@ref) instead of being defined for [`StiefelManifold`](@ref) alone. And the obvious
first fix does not work: making the termination test relative to the partial sum rather than absolute
was measured to change *none* of the numbers below, at any lift norm. The loss is the cancellation
inside the sum, not the point at which the summation stops. Scaling the argument down is the only
thing that helps, which is [`ScaledSquaring`](@ref) — and that is also 1.7× to 4.6× *faster* here,
because the scaled series converges in a handful of terms where the unscaled one grinds through
hundreds.

## Using them

None of these types is exported, so import the ones you use:

```jldoctest retraction-usage
using GeometricOptimizers
using GeometricOptimizers: Geodesic, Cayley, ScaledSquaring, NativePade, AugmentedPade, ProjectedSkew, check
import Random
Random.seed!(123)

Y = rand(StiefelManifold, 5, 3)
B = GeometricOptimizers.global_rep(GlobalSection(Y), rand(5, 3))

check(Geodesic()(B)) < 1e-14, check(Cayley()(B)) < 1e-14

# output

(true, true)
```

A retraction is passed to [`Optimizer`](@ref) as a keyword argument, and the algorithm travels inside
it:

```julia
optimizer = Optimizer(ps, L; algorithm  = Adam(Float32),
                             retraction = Geodesic(ProjectedSkew()))
```

`Geodesic()` is `Geodesic(ScaledSquaring())`, and `ScaledSquaring(θ)` takes the scaling threshold if
you want to override the default `0.5` — which, per the sweep below, you do not need to:

```jldoctest retraction-usage
Geodesic().algorithm == ScaledSquaring(0.5)

# output

true
```

The algorithm reaches the lower-level entry points too. [`geodesic`](@ref) takes it as an optional
last argument, both for a tangent vector at a point and for a horizontal lift:

```jldoctest retraction-usage
using GeometricOptimizers: geodesic, rgrad

Δ = rgrad(Y, rand(5, 3))

check(geodesic(Y, 300 * Δ, ProjectedSkew())) < 1e-13

# output

true
```

and [`GeometricOptimizers.𝔄`](@ref) can be called on a bare matrix, which is the level at which four
of the five algorithms are implemented — [`ProjectedSkew`](@ref) being the exception, as above:

```jldoctest retraction-usage
using GeometricOptimizers: 𝔄
import Random
Random.seed!(123)

X = randn(6, 6)

isapprox(𝔄(X, ScaledSquaring()), 𝔄(X, NativePade()); rtol = 1e-12) &&
    isapprox(𝔄(X, NativePade()), 𝔄(X, AugmentedPade()); rtol = 1e-12)

# output

true
```

If what you want is the exponential itself rather than ``\mathfrak{A}``,
[`GeometricOptimizers.𝔄exp`](@ref) assembles it — ``\exp(B'(B'')^T) = \mathbb{I} +
B'\mathfrak{A}(B', B'')(B'')^T``, at a cost still set by ``n`` rather than by ``N``, since the only
matrix function evaluated is ``\mathfrak{A}`` on the ``2n\times{}2n`` product. It is what
[`geodesic`](@ref) computes before wrapping the result in a [`Manifold`](@ref), and it defaults to
[`ScaledSquaring`](@ref) for the same reason `geodesic` does:

```jldoctest retraction-usage
using GeometricOptimizers: 𝔄exp, lift_factors
import Random
Random.seed!(1234)

B = 60 * rand(StiefelLieAlgHorMatrix, 20, 3)      # ‖B̄‖ ≈ 393
B̂, B̄ = lift_factors(B)

isapprox(𝔄exp(B̂, B̄), exp(Matrix(B)); rtol = 1e-10)

# output

true
```

Where the algorithms part company is a large step, and that is the whole reason the default changed:

```jldoctest
using GeometricOptimizers
using GeometricOptimizers: Geodesic, TaylorSeries, check
import Random
Random.seed!(1234)

B = 60 * rand(StiefelLieAlgHorMatrix, 20, 3)      # ‖B̄‖ ≈ 393

check(Geodesic()(B)) < 1e-12, check(Geodesic(TaylorSeries())(B)) < 1e-12

# output

(true, false)
```

## What they cost and how accurate they are

Everything in this section other than the timings is recomputed when this page is built, so the
figures are those of the version of the package the documentation was built from rather than a quote
that can go stale. `scripts/retraction_accuracy.jl` produces the same tables — including the
timings — from the command line.

```@setup retractions
using GeometricOptimizers
using GeometricOptimizers: geodesic, cayley, check, ScaledSquaring, NativePade, AugmentedPade, ProjectedSkew, TaylorSeries
using LinearAlgebra: norm
using Markdown
using Printf
import Random

# A `Markdown.MD` rather than a string: Documenter renders it as a table instead of as the code
# block a printed string would give.
function table(header, rows)
    io = IOBuffer()
    println(io, "| ", join(header, " | "), " |")
    println(io, "|", repeat("---|", length(header)))
    for row in rows
        println(io, "| ", join(row, " | "), " |")
    end
    Markdown.parse(String(take!(io)))
end

# `TaylorSeries` overflows at the top of the sweep, so both formatters have to survive a `NaN` and
# an `Inf` — a bare `@sprintf` of one is fine, but the guard makes the intent explicit.
sci(x) = isfinite(x) ? (@sprintf "%.2e" x) : string(x)
fixed(x) = isfinite(x) ? (@sprintf "%.2f" x) : string(x)

"""
A sweep of horizontal lifts of increasing norm, all drawn from the same seed — the same eight
`scripts/retraction_accuracy.jl` sweeps, so that the tables below and the ones it prints agree row
for row.
"""
function sweep(T)
    Random.seed!(1234)
    [T(s) * rand(StiefelLieAlgHorMatrix{T}, 20, 3) for s in (0.1, 1.0, 3.0, 6.0, 12.0, 30.0, 60.0, 120.0)]
end
```

### Staying on the manifold

[`check`](@ref) of the retracted point, ``\|Y^TY - \mathbb{I}\|``, on a random
`StiefelLieAlgHorMatrix(20, 3)` scaled up. This is the quantity a retraction is supposed to keep at
round-off, and it is the one the test suite asserts on. [`Cayley`](@ref) is in the last column as the
reference, since it evaluates no matrix function at all.

```@example retractions
lifts = sweep(Float64)

table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`AugmentedPade`", "`ProjectedSkew`", "`TaylorSeries`", "`Cayley`"],
      [[fixed(norm(Matrix(B))),
        sci(check(geodesic(B, ScaledSquaring()))),
        sci(check(geodesic(B, NativePade()))),
        sci(check(geodesic(B, AugmentedPade()))),
        sci(check(geodesic(B, ProjectedSkew()))),
        sci(check(geodesic(B, TaylorSeries()))),
        sci(check(cayley(B)))] for B in lifts])
```

Every column but `TaylorSeries`'s stays at round-off, and only [`ProjectedSkew`](@ref)'s is *level*.
The other four — [`Cayley`](@ref) included, and it is the one that drifts furthest — grow by two to
three orders of magnitude across the sweep, because their orthogonality is an arithmetic outcome
while [`ProjectedSkew`](@ref)'s is structural. Round-off at ``\|\bar{B}\| \approx 770`` is still
round-off, so this separates the algorithms without condemning any of the four usable ones.

### Agreeing with the exponential

Relative distance to `exp(Matrix(B))`, i.e. to the exponential of the full ``N\times{}N`` lift. This
is a different question from the one above — a retraction that re-orthonormalised its result would
have a perfect `check` and be wrong here — and the test suite asserts both.

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`AugmentedPade`", "`ProjectedSkew`"],
      [begin
           reference = exp(Matrix(B))
           err(algorithm) = norm(Matrix(geodesic(B, algorithm)) - reference) / norm(reference)
           [fixed(norm(Matrix(B))), sci(err(ScaledSquaring())), sci(err(NativePade())),
            sci(err(AugmentedPade())), sci(err(ProjectedSkew()))]
       end for B in lifts])
```

All four grow slowly with the norm of the lift, and the ordering is roughly the reverse of the
previous table: [`ProjectedSkew`](@ref) is the furthest from the exponential at all but the largest
of these norms. [`ScaledSquaring`](@ref), [`NativePade`](@ref) and [`AugmentedPade`](@ref) are within
a factor of `1.6` of each other on every row, so the trade is between that group and
[`ProjectedSkew`](@ref): one is orthogonal by construction, the others agree with `exp` more closely.
The four converge again at the top of the sweep, where the reference `exp(Matrix(B))` is itself no
more accurate than what is being measured against it.

### `Float32`

The same `check`, in the format the MNIST experiment described in
[Optimization on Homogeneous Spaces](@ref) actually runs in. Nothing can do better than about
``10^{-6}`` here, but the four do not degrade alike.

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`AugmentedPade`", "`ProjectedSkew`"],
      [[fixed(norm(Matrix(B))),
        sci(check(geodesic(B, ScaledSquaring()))),
        sci(check(geodesic(B, NativePade()))),
        sci(check(geodesic(B, AugmentedPade()))),
        sci(check(geodesic(B, ProjectedSkew())))] for B in sweep(Float32)])
```

[`ProjectedSkew`](@ref) is flat here too, and by the top of the sweep it is one to two orders of
magnitude below the others. Which of the other three is worst depends on the lift over most of the
range, but not at the top: there [`NativePade`](@ref) is the worst of them by a factor of about
``2.5``, which is the one place the `Float64` indifference of the previous table does not carry over.
In a `Float64` run all of this is academic; in a `Float32` one over thousands of steps it is the
reason to choose [`ProjectedSkew`](@ref).

And the forward error in the same format, which is a different ranking and worth having next to it.
The reference is `exp` of the lift promoted to `Float64`, with the difference taken there as well:
`exp` of a `Float32` matrix is itself only `Float32`-accurate, so comparing against it would measure
the reference as much as the algorithm.

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`AugmentedPade`", "`ProjectedSkew`"],
      [begin
           reference = exp(Matrix{Float64}(Matrix(B)))
           err(algorithm) = norm(Matrix{Float64}(Matrix(geodesic(B, algorithm))) - reference) /
                            norm(reference)
           [fixed(norm(Matrix(B))), sci(err(ScaledSquaring())), sci(err(NativePade())),
            sci(err(AugmentedPade())), sci(err(ProjectedSkew()))]
       end for B in sweep(Float32)])
```

Here [`ProjectedSkew`](@ref) is the *worst* of the four at every norm — by `1.3×` to `2.9×` over most
of the sweep and by `14×` at the smallest lift, where it is the only one not at `Float32` round-off —
and the three ``\mathfrak{A}`` algorithms are within `1.7×` of each other throughout,
[`NativePade`](@ref) included, its `check` outlier above notwithstanding. Taken together the two
tables say what the trade actually is in `Float32`: [`ProjectedSkew`](@ref) buys orthogonality at the
price of agreement, and it is the only one of the four for which that is a structural exchange rather
than an accident of the arithmetic.

### The threshold `θ` needs no tuning

[`ScaledSquaring`](@ref)'s only parameter, swept over a 32-fold range on one lift:

```@example retractions
Random.seed!(99)
B = 30 * rand(StiefelLieAlgHorMatrix{Float64}, 20, 3)
reference = exp(Matrix(B))

table(["θ", "`check`", "error vs `exp`"],
      [begin
           Y = geodesic(B, ScaledSquaring(θ))
           [string(θ), sci(check(Y)), sci(norm(Matrix(Y) - reference) / norm(reference))]
       end for θ in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)])
```

Both columns move by less than a factor of six across the whole range, and not monotonically. The
default of `0.5` sits in that band; nothing in the measurement singles it out, which is the point.

### What they cost

Unlike everything above, these are timings and therefore machine-dependent, so they are quoted rather
than measured at build time. `minimum` of 50 repetitions, a single BLAS thread, on an Apple M-series
laptop; `julia --project=. scripts/retraction_accuracy.jl` reproduces them on yours, and every figure
in this section comes from one run of it.

The ``\mathfrak{A}`` call on its own, at ``N = 200``, ``n = 10``, is what separates the three
algorithms that evaluate it:

| | `ScaledSquaring` | `NativePade` | `AugmentedPade` |
|---|---|---|---|
| runtime | `0.021 ms` | `0.037 ms` | `0.053 ms` |
| allocated | `201 KiB` | `330 KiB` | `114 KiB` |

The allocation row is the one figure here that is *not* machine-dependent — `@allocated` is exact —
and it does not rank the three the way runtime does. [`AugmentedPade`](@ref), which builds a
``4n\times{}4n`` matrix and discards three quarters of the result, allocates the least of the three,
because `Base.exp` works in a few reused buffers where both native algorithms produce a fresh
``2n\times{}2n`` temporary per operation. On a CPU that is a detail. On a backend where an allocation
costs a synchronisation it may not be, which is worth knowing about the algorithm whose whole purpose
is to be portable.

One whole retraction, which adds the ``N\times{}N`` assembly they all share, in milliseconds:

| ``N``, ``n`` | 10, 2 | 20, 3 | 50, 5 | 100, 5 | 200, 10 | 500, 10 | 500, 50 | 1000, 20 |
|---|---|---|---|---|---|---|---|---|
| `Geodesic(ScaledSquaring())` | 0.003 | 0.005 | 0.014 | 0.023 | 0.087 | 0.396 | 3.03 | 2.51 |
| `Geodesic(NativePade())` | 0.004 | 0.005 | 0.013 | 0.023 | 0.091 | 0.410 | 3.33 | 2.57 |
| `Geodesic(AugmentedPade())` | 0.003 | 0.006 | 0.016 | 0.027 | 0.120 | 0.464 | 5.98 | 2.72 |
| `Geodesic(ProjectedSkew())` | 0.004 | 0.008 | 0.023 | 0.028 | 0.130 | 0.464 | 4.04 | 2.89 |
| `Geodesic(TaylorSeries())` | 0.003 | 0.006 | 0.019 | 0.033 | 0.149 | 0.505 | 14.1 | 3.49 |
| `Cayley()` | 0.002 | 0.004 | 0.016 | 0.056 | 0.361 | 4.87 | 6.24 | 38.8 |

and the same in KiB allocated:

| ``N``, ``n`` | 10, 2 | 20, 3 | 50, 5 | 100, 5 | 200, 10 | 500, 10 | 500, 50 | 1000, 20 |
|---|---|---|---|---|---|---|---|---|
| `Geodesic(ScaledSquaring())` | 13.1 | 33.1 | 130 | 338 | 1354 | 6653 | 13813 | 26419 |
| `Geodesic(NativePade())` | 19.8 | 45.9 | 162 | 370 | 1480 | 6782 | 16776 | 26940 |
| `Geodesic(AugmentedPade())` | 12.0 | 29.6 | 117 | 322 | 1278 | 6552 | 11011 | 25982 |
| `Geodesic(ProjectedSkew())` | 15.1 | 33.3 | 122 | 335 | 1315 | 6685 | 10769 | 26417 |
| `Geodesic(TaylorSeries())` | 12.2 | 31.1 | 131 | 345 | 1483 | 6893 | 27586 | 28615 |
| `Cayley()` | 12.9 | 32.9 | 144 | 475 | 1881 | 10546 | 13383 | 41694 |

[`ScaledSquaring`](@ref) and [`NativePade`](@ref) are level on runtime to within the run-to-run noise
at every size but one; the isolated call above is where the extra rational work shows. The exception
is ``n = 50``, the one column where the ``2n\times{}2n`` argument is large enough for the difference
to survive the assembly: `3.33` against `3.03`, with [`AugmentedPade`](@ref)'s `5.98` as it pays for
the ``4n\times{}4n`` embedding. Allocations separate them everywhere and by more, up to `1.5×` at
``n = 50``, since that is the metric the shared assembly dilutes least at small ``N``.
[`ProjectedSkew`](@ref) stays close on both — a QR and an eigendecomposition of a ``2n\times{}2n``
matrix are not expensive things, and it is the *lightest* of the five at ``n = 50``. [`Cayley`](@ref)
is level with the exponential up to ``N \approx 50`` and loses by a factor of 15 by ``N = 1000``,
which is the ``O(N^3)`` against ``O(N^2n)`` of the previous section.

## Choosing one

For the retraction: **[`Geodesic`](@ref) unless you have a reason for [`Cayley`](@ref)**. It is the
exponential map and the cheaper of the two at any size worth worrying about, and it survives an
implausibly large step with a `check` an order of magnitude smaller — which is what issue A1b turned
on. That argument is weaker now than it was: bounding the step ([`DEFAULT_STEP_CEILING`](@ref)) means
an implausibly large step is no longer taken under either retraction, so the tolerance `Geodesic` has
for one is insurance rather than a live difference. A derivative-based line search is exact under
either since 0.2.0, so that is no longer part of the argument. [`Cayley`](@ref) remains the package
default, needs no matrix function at all, and is unconditionally stable.

For the algorithm: **[`ScaledSquaring`](@ref), i.e. the default, unless one of the alternatives has
the property you specifically need.**

| | choose it when | at the price of |
|---|---|---|
| [`ScaledSquaring`](@ref) | almost always; it is the default | `check` drifting up with the size of the lift |
| [`NativePade`](@ref) | you need an independent implementation on a backend that forbids scalar indexing | `1.8×` the isolated ``\mathfrak{A}`` runtime and `1.6×` its allocations, the same accuracy in `Float64`, the worst `check` of the three in `Float32`, and `θ` bounded by `1/2` |
| [`ProjectedSkew`](@ref) | staying on the manifold matters more than the last bit of the exponential — a long `Float32` run, where `check` accumulates over thousands of steps | `1.1×`–`1.6×` the cost, the largest forward error in either format, CPU only |
| [`AugmentedPade`](@ref) | you want a second opinion from an implementation that introduces no numerics of its own | roughly 2× the cost of the ``\mathfrak{A}`` call, no better than [`ScaledSquaring`](@ref) on accuracy, CPU only |
| [`TaylorSeries`](@ref) | never; it exists so the pre-0.2.0 regression stays reproducible | leaving the manifold silently above ``\Vert\bar{B}\Vert \approx 50`` |

On a GPU backend, [`ScaledSquaring`](@ref) remains the production choice and [`NativePade`](@ref)
provides the independent cross-check that was previously missing. Both avoid dense LAPACK and scalar
indexing; [`ScaledSquaring`](@ref) is the default because it does less work.

## Adding one

A new algorithm is a subtype of [`AbstractExponentialAlgorithm`](@ref) that supplies

```julia
GeometricOptimizers.𝔄(X::AbstractMatrix, ::NewAlgorithm)
```

Everything above it — [`geodesic`](@ref) for lifts and for tangent vectors, `retraction`,
`update_section!`, the optimizer — then follows, for both manifolds, with no further methods. An
algorithm that does not evaluate ``\mathfrak{A}`` at all supplies

```julia
GeometricOptimizers.geodesic(B::AbstractLieAlgHorMatrix, ::NewAlgorithm)
```

instead, which is what [`ProjectedSkew`](@ref) does. A new *retraction* is a subtype of
[`AbstractRetraction`](@ref) supplying `retraction(::NewRetraction, x)`; the callable form `R(x)`
comes for free.

Whichever it is, `test/retractions/exponential_accuracy.jl` is where it earns its place: the sweep
there asserts that a retracted point stays on the manifold at every lift norm *and* that it still
agrees with `exp(Matrix(B))`, which together rule out both of the ways an exponential can be wrong
here.


## The retractions on the two manifolds

[`AbstractRetraction`](@ref), [`Geodesic`](@ref), [`Cayley`](@ref), [`geodesic`](@ref),
[`cayley`](@ref) and [`retraction`](@ref). `geodesic` and `cayley` each have a method on an
[`AbstractLieAlgHorMatrix`](@ref) — the efficient form this page derives, for both the Stiefel and
the Grassmann lift — and one on a point together with a tangent vector, which is the classical
retraction of the footnote above. Their docstrings are on the [reference page](@ref GeometricOptimizers), where every docstring in the package is rendered once; the names above link to them.

## Reference

```@bibliography
Pages = ["retractions.md"]
Canonical = false
```
