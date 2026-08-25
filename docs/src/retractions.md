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

This distinction mattered in an earlier line-search regression: large trial steps amplified the
difference between the Cayley curve and the geodesic. The exact differential fixed the derivative,
while [`DEFAULT_STEP_CEILING`](@ref) fixed the excessive extrapolation. With both changes in place,
the regression no longer depends on which retraction is selected.

The cost balance depends on the dimensions. [`cayley`](@ref) finishes with a product of two
``N\times{}N`` matrices, whereas [`geodesic`](@ref) assembles
``\mathbb{I} + B'\mathfrak{A}(X)(B'')^T`` from a ``2n\times{}2n`` matrix function. The benchmark
under [What they cost](@ref) shows one representative comparison.

## The exponential needs an algorithm

The geodesic requires ``\exp(\bar{B})``, but forming and exponentiating the full ``N\times{}N`` lift
would throw away its low-rank structure. Writing ``\bar{B}=B'(B'')^T`` and
``X=(B'')^TB'`` gives, for every ``k\geq 1``,

```math
\left(B'(B'')^T\right)^k = B'X^{k-1}(B'')^T.
```

Substitution into the exponential series therefore gives the exact identity

```math
\exp\left(B'(B'')^T\right) = \mathbb{I} + B'\,\mathfrak{A}(X)\,(B'')^T,
\qquad
\mathfrak{A}(X) = \sum_{k=1}^\infty \frac{X^{k-1}}{k!},
```

so computing ``\mathfrak{A}(X)`` is the central numerical task: once it is available, the
``N\times{}N`` exponential is assembled using only the two thin factors. The argument ``X`` is only
``2n\times{}2n``, which preserves the cost advantage when ``n\ll N``. ``\mathfrak{A}`` is the
function usually written ``\varphi_1(X) = \left(\exp(X) - \mathbb{I}\right)X^{-1}``, though its
series definition remains valid when ``X`` is singular.

There are two separate algorithmic choices:

1. how to approximate the matrix function at a small argument; and
2. whether to scale a large argument down and recover the original value afterwards.

Taylor and Padé are approximation kernels. Scaling and squaring is a framework around such a kernel,
not a competing approximation. Classical dense matrix-exponential routines normally combine a Padé
approximant with scaling and squaring [higham2005scaling, almohy2010new](@cite). Taylor kernels can
also be effective when scaling keeps the argument small [skaflestad2009scaling](@cite).

The algorithms currently available in this package are:

| Package algorithm | Object evaluated | Kernel | Recovery | Backend |
|---|---|---|---|---|
| [`TaylorSeries`](@ref) | ``\mathfrak{A}(X)`` | Taylor series | none | any |
| [`ScaledSquaring`](@ref) | ``\mathfrak{A}(X)`` | Taylor series | low-rank modified squaring | any |
| [`NativePade`](@ref) | ``\mathfrak{A}(X)`` | degree-6 Padé | low-rank modified squaring | matrix products and reductions |
| [`ProjectedSkew`](@ref) | projected lift exponential | eigendecomposition | none | CPU |

These algorithms return the same exponential map, so the one-parameter subgroup property above
holds for every one of them.

## 1. Direct Taylor series

[`TaylorSeries`](@ref) evaluates ``\mathfrak{A}`` directly from its defining series, without scaling,
and stops when a term falls below `eps`. The series converges for every matrix, but convergence alone
does not guarantee an accurate floating-point sum.

The reduced argument here is particularly difficult. ``X``'s lower-left block is
``\tfrac{1}{4}A^2 - B^TB``, so

```math
\|X\| \approx \tfrac{1}{4}\|\bar{B}\|^2
\qquad\text{while}\qquad
\rho(X) \approx \|\bar{B}\|,
```

because the eigenvalues of ``X`` are the nonzero, purely imaginary eigenvalues of the skew matrix
``\bar{B}``. The resulting non-normality can make intermediate Taylor terms much larger than the
final answer. Their cancellation then loses accuracy. This is the familiar reason not to evaluate a
matrix exponential by an unscaled Taylor series [moler2003nineteen](@cite), made more pronounced here
because the norm of the reduced matrix can grow quadratically with the norm of the lift.

!!! danger "This is not a usable retraction"
    `TaylorSeries` is retained as a regression baseline for the pre-0.2.0 implementation. It can
    silently lose orthogonality for large lifts. Do not select it for optimization.

[`Geodesic`](@ref) makes the algorithm choice visible:

```julia
Geodesic(ScaledSquaring())   # the default, and `Geodesic()`
Geodesic(NativePade())
Geodesic(ProjectedSkew())
Geodesic(TaylorSeries())     # the pre-0.2.0 behaviour; not a usable retraction
```

## 2. Padé approximation

The failure of the direct Taylor algorithm suggests scaling the argument before evaluating
``\mathfrak{A}``; that is the subject of the next section. Once the argument has been made small,
however, one still has to choose the approximation evaluated there. A degree-``m`` Taylor polynomial
simply keeps the first ``m+1`` terms,

```math
T_m(z)=\sum_{k=0}^m c_kz^k,
\qquad
f(z)-T_m(z)=O(z^{m+1}).
```

Its coefficients are fixed one by one by the series, so matching more terms requires increasing the
polynomial degree. Padé takes a different route: it chooses a numerator *and* a denominator together,
so that their quotient matches more series coefficients without raising the numerator to the same
degree. It is therefore an alternative small-argument kernel, not an alternative to scaling.

To see how this works, first consider a scalar analytic function ``f``. Its ``[m/n]`` Padé
approximant is the rational function

```math
R_{m,n}(z) = \frac{P_m(z)}{Q_n(z)},
\qquad
\deg P_m\leq m,\quad \deg Q_n\leq n,\quad Q_n(0)=1,
```

and is defined by the matching condition

```math
Q_n(z)f(z)-P_m(z)=O\left(z^{m+n+1}\right).
```

Equating powers of ``z`` determines the coefficients of ``P_m`` and ``Q_n``. The denominator is what
makes this different from truncating a Taylor series: when the quotient is expanded again as a power
series, even low-degree ``P_m`` and ``Q_n`` generate infinitely many powers. For example,

```math
\frac{1+z/2}{1-z/2}=1+z+\frac{z^2}{2}+\frac{z^3}{4}+\cdots
```

matches ``e^z`` through degree two, whereas a polynomial with numerator degree one can match only
``1+z``. More generally, a ``[m/n]`` Padé approximant matches through degree ``m+n`` whenever its
matching condition is solvable — for ``e^z`` it always is, as the next section shows. This
is why a rational kernel can capture much more local information than a Taylor polynomial with a
similar polynomial degree [higham2005scaling, higham2008functions](@cite). The scalar variable ``z``
serves only to fix the coefficients; a matrix takes its place once they are known.

### Deriving the coefficients for ``\mathfrak{A}``

Nothing is fitted to ``\mathfrak{A}``, and nothing is fitted numerically. The coefficients come from
the matching condition above, applied to ``f=\exp`` at ``m=7`` and ``n=6``; one subtraction then
converts that approximant into one for ``\mathfrak{A}``. Write its numerator and denominator as
``P^{\exp}_7`` and ``Q^{\exp}_6``:

```math
e^z = \frac{P^{\exp}_7(z)}{Q^{\exp}_6(z)} + O(z^{14}),
\qquad P^{\exp}_7(0)=Q^{\exp}_6(0)=1.
```

Keep ``m`` and ``n`` general for the derivation, since one formula covers every degree:

```math
P_m(z)=\sum_{k=0}^m a_kz^k,
\qquad
Q_n(z)=\sum_{j=0}^n b_jz^j,
\qquad b_0=1.
```

Because ``e^z=\sum_{r\geq0}z^r/r!``, the ``z^k`` coefficient of ``Q_n(z)e^z`` is the convolution
``\sum_j b_j/(k-j)!``, reading ``1/(k-j)!`` as zero when ``j>k``. The matching condition
``Q_ne^z-P_m=O(z^{m+n+1})`` then says two different things, one on each of the two ranges of ``k``
that it covers:

```math
a_k=\sum_{j=0}^{n}\frac{b_j}{(k-j)!}
\quad (k=0,\ldots,m),
\qquad
\sum_{j=0}^{n}\frac{b_j}{(k-j)!}=0
\quad (k=m+1,\ldots,m+n).
```

Only the right-hand block is a system to be solved: ``n`` equations in the ``n`` unknowns
``b_1,\ldots,b_n``. The left-hand block constrains nothing, since each ``a_k`` occurs in one equation
and nowhere else — with the denominator known it reads the numerator off. The construction is a
statement about the denominator alone. For ``m=7`` and ``n=6`` the ranges are the degrees
``0,\ldots,7`` and ``8,\ldots,13``.

The solution is

```math
a_k=\frac{(m+n-k)!}{(m+n)!}\binom{m}{k},
\qquad
b_k=(-1)^k\frac{(m+n-k)!}{(m+n)!}\binom{n}{k},
```

and one binomial identity settles both blocks at once. Substitute this ``b_j`` into the convolution at
degree ``k`` and clear the constant:

```math
(m+n)!\sum_{j=0}^n\frac{b_j}{(k-j)!}
=\sum_{j=0}^n(-1)^j\binom{n}{j}\frac{(m+n-j)!}{(k-j)!}.
```

Put ``d=m+n-k``. The ratio ``(m+n-j)!/(k-j)!`` is a product of ``d`` consecutive integers, that is
``d!\binom{m+n-j}{d}``, which also reproduces the convention above: for ``j>k`` the binomial has
``m+n-j<d`` and vanishes. So the right-hand side is ``d!`` times

```math
\sum_{j=0}^n(-1)^j\binom{n}{j}\binom{m+n-j}{d}=\binom{m}{d-n},
```

The identity follows directly by extracting the coefficient of ``x^d``:

```math
\begin{aligned}
\sum_{j=0}^n(-1)^j\binom{n}{j}\binom{m+n-j}{d}
&=[x^d](1+x)^m\sum_{j=0}^n\binom{n}{j}(-1)^j(1+x)^{n-j}\\
&=[x^d](1+x)^m((1+x)-1)^n
 =[x^d]x^n(1+x)^m=\binom{m}{d-n}.
\end{aligned}
```

Here ``[x^d]f(x)`` denotes the coefficient of ``x^d`` in ``f``. If ``k\geq m+1``, then
``d-n=m-k<0`` and ``x^n(1+x)^m`` has no ``x^d`` term, so the denominator equations vanish. If
``k\leq m``, that coefficient is ``\binom{m}{m-k}=\binom{m}{k}``; restoring ``d!/(m+n)!`` gives
the claimed ``a_k``.

The two formulas are one expression with ``m`` and ``n`` interchanged and a sign,
``a_k(m,n)=(-1)^kb_k(n,m)``: the reflection that ``e^{-z}=1/e^z`` induces on the table. They are
classical [higham2005scaling, higham2008functions](@cite); the derivation appears here so the numbers
below are traceable rather than quoted. At ``m=7`` and ``n=6``:

```math
\begin{array}{r|ccc}
k & a_k & b_k & a_k-b_k\\\hline
0 & 1 & 1 & 0\\
1 & \tfrac{7}{13} & -\tfrac{6}{13} & 1\\
2 & \tfrac{7}{52} & \tfrac{5}{52} & \tfrac{1}{26}\\
3 & \tfrac{35}{1716} & -\tfrac{5}{429} & \tfrac{5}{156}\\
4 & \tfrac{7}{3432} & \tfrac{1}{1144} & \tfrac{1}{858}\\
5 & \tfrac{7}{51480} & -\tfrac{1}{25740} & \tfrac{1}{5720}\\
6 & \tfrac{7}{1235520} & \tfrac{1}{1235520} & \tfrac{1}{205920}\\
7 & \tfrac{1}{8648640} & 0 & \tfrac{1}{8648640}
\end{array}
```

The last column is the promised subtraction. Since ``\mathfrak{A}(z)=(e^z-1)/z``,

```math
\mathfrak{A}(z)
= \frac{e^z-1}{z}
= \frac{P^{\exp}_7(z)-Q^{\exp}_6(z)}{zQ^{\exp}_6(z)} + O(z^{13})
= \frac{p_6(z)}{q_6(z)} + O(z^{13}),
\qquad
p_6(z)=\frac{P^{\exp}_7(z)-Q^{\exp}_6(z)}{z},
\quad
q_6(z)=Q^{\exp}_6(z).
```

The ``k=0`` entry of that column vanishes, both constant terms being one, so the difference is
divisible by ``z`` and ``p_6`` is a polynomial. Its coefficients are the remaining entries shifted
down one degree, ``(p_6)_k=a_{k+1}-b_{k+1}`` for ``k=0,\ldots,6`` with ``b_7=0``. Numerator and
denominator both have degree six, so this is a ``[6/6]`` approximant of ``\mathfrak{A}``, and it
agrees with

```math
\mathfrak{A}(z)=1+\frac{z}{2!}+\frac{z^2}{3!}+\cdots
```

through the ``z^{12}`` term, where truncating that series after ``z^6`` agrees only through ``z^6``.
The first term missed is small: in exact arithmetic

```math
q_6(z)\mathfrak{A}(z)-p_6(z)=\frac{z^{13}}{149597947699200}+O(z^{14}),
```

so at the scaled argument [`NativePade`](@ref) evaluates, ``|z|\leq1/2``, that term is about
``8\cdot10^{-19}`` — the same order as the inverse-iteration residual derived below, and far under the
`Float64` rounding of a result of size one. The polynomials the implementation uses are therefore

```math
\begin{aligned}
p_6(z)={}&1+\frac{z}{26}+\frac{5z^2}{156}+\frac{z^3}{858}
          +\frac{z^4}{5720}+\frac{z^5}{205920}+\frac{z^6}{8648640},\\
q_6(z)={}&1-\frac{6z}{13}+\frac{5z^2}{52}-\frac{5z^3}{429}
          +\frac{z^4}{1144}-\frac{z^5}{25740}+\frac{z^6}{1235520}.
\end{aligned}
```

### From scalar Padé to matrix Padé

The scalar calculation above has done its only job: it has determined the coefficients. Matrix
functions defined by power series use those same scalar coefficients, with each scalar power ``z^k``
replaced by the matrix power ``Y^k``. Thus ``p_6(z)`` and ``q_6(z)`` become the matrix polynomials
``p_6(Y)`` and ``q_6(Y)`` displayed above with ``z`` replaced by ``Y`` and ``1`` replaced by ``I``.
The matching statement also transfers directly: the expansion of the rational matrix function agrees
with

```math
\mathfrak{A}(Y)=I+\frac{Y}{2!}+\frac{Y^2}{3!}+\cdots
```

through the ``Y^{12}`` term.

The quotient is not taken entry by entry. For scalars, ``r=p/q`` means ``qr=p``. For matrices, the
corresponding operation is therefore to solve

```math
q_6(Y)R=p_6(Y),
```

or, when ``q_6(Y)`` is invertible, to write

```math
R=q_6(Y)^{-1}p_6(Y)\approx\mathfrak{A}(Y).
```

Because ``p_6(Y)`` and ``q_6(Y)`` are polynomials in the same matrix, they commute; the order shown is
the one used by the implementation. Notice that no inverse of ``Y`` occurs. Unlike the expression
``(e^Y-I)Y^{-1}``, the Padé formula remains directly usable when ``Y`` is singular. In
[`_native_pade_polynomials`](@ref), the code first forms ``Y^2`` and ``Y^4`` and groups the displayed
polynomials around those shared powers. This evaluates both degree-6 polynomials with matrix products
instead of constructing the powers one by one.

### Applying the denominator without a matrix solve

A conventional evaluation of this rational matrix function would solve

```math
q_6(Y)W=p_6(Y).
```

!!! info "Why Newton--Schulz rather than LU?"
    For dense CPU matrices, the standard choice is pivoted LU followed by triangular solves; one
    would not normally form an explicit inverse. `NativePade` instead uses Newton--Schulz because
    its updates need only matrix multiplication and addition, keeping the direct algorithm
    independent of backend-specific factorization support. This is a portability choice, not a
    generally faster solver: five matrix products may cost more than LU on a CPU. It is viable here
    because scaling gives the initial-residual bound below, fixing both the iteration count and the
    resulting error.

To derive the iteration [schulz1933iterative](@cite), temporarily write
``A=q_6(Y)`` and seek a matrix ``Z`` satisfying ``Z^{-1}-A=0``. The Fréchet derivative of matrix
inversion is

```math
D(Z^{-1})[H]=-Z^{-1}HZ^{-1}.
```

One Newton correction therefore solves

```math
-Z_j^{-1}H_jZ_j^{-1}=A-Z_j^{-1},
```

which gives ``H_j=Z_j-Z_jAZ_j`` and hence

```math
Z_0=I,
\qquad
Z_{j+1}=Z_j+H_j=Z_j\left(2I-AZ_j\right).
```

This rearrangement is Newton's method using no factorization or linear solve. It is not safe from an
arbitrary starting point: convergence requires the initial inverse residual to be smaller than one
in a submultiplicative norm.

For the present choice ``A=q_6(Y)`` and ``Z_0=I``, define ``E_j=I-AZ_j``. Since
``AZ_j=I-E_j`` and ``2I-AZ_j=I+E_j``, the update gives

```math
AZ_{j+1}=(I-E_j)(I+E_j)=I-E_j^2,
\qquad
E_{j+1}=E_j^2.
```

Consequently ``E_j=E_0^{2^j}`` in exact arithmetic and
``\|E_j\|\leq\|E_0\|^{2^j}``: once ``\|E_0\|<1``, the iteration converges quadratically, squaring
the residual at every step. The code writes the first step explicitly as ``Z_1=2I-q_6(Y)`` and
performs four more, giving
``E_5=(I-q_6(Y))^{32}``. Scaling ensures ``\|Y\|_1\leq 1/2``; from the displayed coefficients,

```math
\|I-q_6(Y)\|_1
\leq \sum_{k=1}^6 |(q_6)_k|\,\|Y\|_1^k
<0.257,
```

so ``\|E_5\|_1<0.257^{32}<1.3\cdot10^{-19}``. This explains both the fixed five Newton--Schulz
steps and the constructor restriction ``0<\theta\leq 1/2``: together they make the solve-free inverse
accurate to approximately `Float64` precision using matrix multiplication alone.

### The complete `NativePade` algorithm

Given the reduced matrix ``X=(B'')^TB'``:

1. Choose ``s=\max(0,\lceil\log_2(\|X\|_1/\theta)\rceil)`` and set ``\alpha=2^s``.
2. Set ``Y=X/\alpha``, so ``\|Y\|_1\leq\theta\leq1/2``, and evaluate ``p_6(Y)`` and ``q_6(Y)``.
3. Compute ``Z_5\approx q_6(Y)^{-1}`` with the five Newton--Schulz steps above.
4. Form ``W_s=Z_5p_6(Y)/\alpha``. The division by ``\alpha`` is required because

   ```math
   \exp\left(B'(B'')^T/\alpha\right)
   =I+B'\left[\mathfrak{A}(X/\alpha)/\alpha\right](B'')^T.
   ```

5. Apply ``W\leftarrow2W+WXW`` exactly ``s`` times. These are the low-rank modified-squaring steps
   derived below; after the last one, ``W\approx\mathfrak{A}(X)``.

Padé is therefore the *small-argument kernel* in `NativePade`, not a replacement for scaling and
squaring. Scaling makes the rational approximation and its solve-free denominator application
reliable; modified squaring transports that small-argument result back to the original ``X``.

!!! note "`AugmentedPade` is a reference implementation"
    [`AugmentedPade`](@ref) obtains ``\mathfrak{A}(X)`` from the upper-right block of
    ``\exp\left(\begin{smallmatrix}X&I\\0&0\end{smallmatrix}\right)`` and delegates that exponential
    to Julia's dense `exp`. It is useful for checking the direct implementations, but normally
    should not be selected: it exponentiates a ``4n\times{}4n`` matrix to recover a
    ``2n\times{}2n`` block, discards the other blocks, and requires dense LAPACK. The identity is a
    standard route from an exponential routine to a ``\varphi``-function
    [sidje1998expokit, higham2008functions](@cite).

## 3. Scaling and modified squaring

Scaling and squaring first evaluates a matrix function at a smaller argument and then reconstructs
the value at the original argument. The classical exponential algorithm repeatedly squares
``\exp(X/2^s)``. For ``\varphi``-functions such as ``\mathfrak{A}=\varphi_1``, the corresponding
recovery formulas are usually called *modified squaring* [skaflestad2009scaling](@cite).

[`ScaledSquaring`](@ref) uses a Taylor kernel, while [`NativePade`](@ref) uses a Padé kernel. Both
scale the argument first and use the same recovery recurrence. `ScaledSquaring`'s advantage over
[`TaylorSeries`](@ref) comes entirely from that scaling: the Taylor series is evaluated only where
``\|X/2^s\|_1\leq\theta``, so its terms remain modest and the catastrophic cancellation of the
unscaled series is avoided. The subsequent modified-squaring steps are algebraic identities, not
additional approximations. In this way `ScaledSquaring` makes the otherwise unreliable Taylor
kernel usable for the reduced matrices encountered here.

The low-rank form is closed under squaring:

```math
\left(\mathbb{I} + B'W(B'')^T\right)^2 = \mathbb{I} + B'\left(2W + WXW\right)(B'')^T,
```

so recovery needs only ``2n\times{}2n`` matrix products. It never squares an ``N\times{}N`` matrix.

Let ``L = B'(B'')^T``, ``X = (B'')^TB'``, and ``\alpha = 2^s``. The scaled exponential is

```math
\exp(L/\alpha)
= \mathbb{I} + B'\left[\frac{\mathfrak{A}(X/\alpha)}{\alpha}\right](B'')^T.
```

Thus the complete computation is:

1. **Choose the scaling.** Set
   ``s = \max(0, \lceil\log_2(\|X\|_1/\theta)\rceil)`` and ``\alpha = 2^s``.
2. **Evaluate the scaled problem.** Compute
   ``W_s = \mathfrak{A}(X/\alpha)/\alpha`` with the Taylor kernel. The argument now satisfies
   ``\|X/\alpha\|_1 \leq \theta``.
3. **Undo the scaling.** If
   ``\exp(L/2^k) = \mathbb{I} + B'W_k(B'')^T``, then squaring gives
   ``W_{k-1} = 2W_k + W_kXW_k``. Apply this update for ``k=s,s-1,\ldots,1``. The recurrence uses the
   original ``X``, not ``X/\alpha``.
4. **Return the result.** After the loop, ``W_0 = \mathfrak{A}(X)``, so
   ``\mathbb{I} + B'W_0(B'')^T = \exp(B'(B'')^T)``.

The initial division by ``\alpha`` follows directly from the scaled-exponential identity; it is not
an additional approximation. The recurrence uses the original ``X`` because it represents repeated
squaring of ``\exp(L/2^s)``. Every recovery step remains a ``2n\times{}2n`` update, so no dense
``N\times{}N`` exponential, square, or solve is formed.

The threshold `θ` defaults to `0.5`. A smaller value performs more scaling steps; a larger value asks
the Taylor kernel to handle a larger argument. The sweep below shows little sensitivity over the
tested range, but it is an empirical check rather than a general error bound.

The implementation uses matrix products and reductions and avoids scalar indexing in package code.
[`GeometricOptimizers.opnorm₁`](@ref) is used instead of `LinearAlgebra.opnorm(X, 1)`. Execution on
an accelerator still depends on the backend's support for those matrix operations.

!!! note "The halving count is loose"
    The implementation chooses `s` from ``\|X\|_1``. For these reduced matrices that norm can be
    much larger than the spectral radius, so the rule may perform more squarings than necessary.
    This is the overscaling phenomenon discussed for the matrix exponential by Al-Mohy and Higham
    [almohy2010new](@cite). Replacing the norm bound by a spectral calculation would undermine the
    backend portability that motivates this implementation.

## 4. `ProjectedSkew`

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

which is orthogonal up to the eigendecomposition and reconstruction errors. Unlike the other
algorithms, `ProjectedSkew` bypasses ``\mathfrak{A}`` and implements [`geodesic`](@ref) directly.
It therefore has no `𝔄(X, ProjectedSkew())` method.

The method gives particularly small orthogonality residuals in the experiments below, including in
`Float32`. Its trade-offs are the QR factorization and eigendecomposition, a somewhat larger forward
error in most rows of the reported sweep, and dependence on dense LAPACK.

## Using them

None of these types is exported, so import the ones you use:

```jldoctest retraction-usage
using GeometricOptimizers
using GeometricOptimizers: Geodesic, Cayley, ScaledSquaring, NativePade, ProjectedSkew, check
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

`Geodesic()` is `Geodesic(ScaledSquaring())`, and `ScaledSquaring(θ)` accepts a scaling threshold;
the default is `0.5`:

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

and [`GeometricOptimizers.𝔄`](@ref) can be called on a bare matrix, which is the level at which the
Taylor- and Padé-kernel algorithms are implemented:

```jldoctest retraction-usage
using GeometricOptimizers: 𝔄
import Random
Random.seed!(123)

X = randn(6, 6)

isapprox(𝔄(X, ScaledSquaring()), 𝔄(X, NativePade()); rtol = 1e-12)

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

The difference between scaled and unscaled Taylor evaluation becomes visible for a large lift:

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

## 5. Numerical comparison

Everything in this section other than the timings is recomputed when this page is built, so the
figures are those of the version of the package the documentation was built from rather than a quote
that can go stale. `scripts/retraction_accuracy.jl` produces the same tables — including the
timings — from the command line.

```@setup retractions
using GeometricOptimizers
using GeometricOptimizers: geodesic, cayley, check, ScaledSquaring, NativePade, ProjectedSkew, TaylorSeries
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

table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`ProjectedSkew`", "`TaylorSeries`", "`Cayley`"],
      [[fixed(norm(Matrix(B))),
        sci(check(geodesic(B, ScaledSquaring()))),
        sci(check(geodesic(B, NativePade()))),
        sci(check(geodesic(B, ProjectedSkew()))),
        sci(check(geodesic(B, TaylorSeries()))),
        sci(check(cayley(B)))] for B in lifts])
```

`TaylorSeries` eventually fails by many orders of magnitude. The other methods remain close to
machine precision throughout this `Float64` sweep. `ProjectedSkew` has the flattest orthogonality
residual; the remaining methods show a modest increase as the lift grows.

### Agreeing with the exponential

Relative distance to `exp(Matrix(B))`, i.e. to the exponential of the full ``N\times{}N`` lift. This
is a different question from the one above — a retraction that re-orthonormalised its result would
have a perfect `check` and be wrong here — and the test suite asserts both.

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`ProjectedSkew`"],
      [begin
           reference = exp(Matrix(B))
           err(algorithm) = norm(Matrix(geodesic(B, algorithm)) - reference) / norm(reference)
           [fixed(norm(Matrix(B))), sci(err(ScaledSquaring())), sci(err(NativePade())),
            sci(err(ProjectedSkew()))]
       end for B in lifts])
```

All three algorithms remain close to the dense reference. In this sweep, the two direct
``\mathfrak{A}`` algorithms usually have the smaller forward error, while `ProjectedSkew` usually
has the smaller orthogonality residual. The table reports an experiment, not an error bound; the
ordering can depend on the matrix and floating-point type.

### `Float32`

The same orthogonality residual in `Float32`, the format used by the MNIST example in
[Optimization on Homogeneous Spaces](@ref):

```@example retractions
table(["‖B̄‖", "`ScaledSquaring`", "`NativePade`", "`ProjectedSkew`"],
      [[fixed(norm(Matrix(B))),
        sci(check(geodesic(B, ScaledSquaring()))),
        sci(check(geodesic(B, NativePade()))),
        sci(check(geodesic(B, ProjectedSkew())))] for B in sweep(Float32)])
```

`ProjectedSkew` again has the flattest residual. The difference is more visible than in `Float64`,
but whether it matters in an optimization run depends on how errors accumulate in that application.

### Sensitivity to the threshold `θ`

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

Both columns vary modestly and non-monotonically over this range. The experiment supports the
default `0.5` as a reasonable value; it does not establish an optimal threshold.

### What they cost

Unlike everything above, these are timings and therefore machine-dependent, so they are quoted rather
than measured at build time. One retraction, `minimum` of 50 repetitions, a single BLAS thread, on an
Apple M-series laptop; `julia --project=. scripts/retraction_accuracy.jl` reproduces them on yours.
Milliseconds:

| ``N``, ``n`` | 10, 2 | 20, 3 | 50, 5 | 100, 5 | 200, 10 | 500, 10 | 500, 50 | 1000, 20 |
|---|---|---|---|---|---|---|---|---|
| `Geodesic(ScaledSquaring())` | 0.003 | 0.005 | 0.014 | 0.023 | 0.087 | 0.396 | 3.03 | 2.51 |
| `Geodesic(NativePade())` | 0.004 | 0.005 | 0.013 | 0.023 | 0.091 | 0.410 | 3.33 | 2.57 |
| `Geodesic(ProjectedSkew())` | 0.004 | 0.008 | 0.023 | 0.028 | 0.130 | 0.464 | 4.04 | 2.89 |
| `Geodesic(TaylorSeries())` | 0.003 | 0.006 | 0.019 | 0.033 | 0.149 | 0.505 | 14.1 | 3.49 |
| `Cayley()` | 0.002 | 0.004 | 0.016 | 0.056 | 0.361 | 4.87 | 6.24 | 38.8 |

On this machine, `ScaledSquaring` and `NativePade` have similar whole-retraction timings at most
measured sizes. `ProjectedSkew` pays for a QR factorization and eigendecomposition. `Cayley` is
competitive for small `N`, but its final dense matrix product becomes dominant as `N` grows. These
timings are illustrative and should be remeasured on the target hardware.

## Choosing one

[`Cayley`](@ref) remains the package default. It requires only a small linear solve and is robust for
large steps. [`Geodesic`](@ref) computes the exponential map and has the one-parameter subgroup
property. In the measurements above it also becomes cheaper once the ambient dimension is large
relative to the manifold dimension. Choose between them according to which map the algorithm needs,
then benchmark representative problem sizes if cost matters.

For the algorithm: **[`ScaledSquaring`](@ref), i.e. the default, unless one of the two special cases
applies.**

| | choose it when | at the price of |
|---|---|---|
| [`ScaledSquaring`](@ref) | the general default | orthogonality residual can grow with the lift |
| [`NativePade`](@ref) | an independent direct Padé calculation is useful | a fixed rational kernel and restricted threshold |
| [`ProjectedSkew`](@ref) | a small orthogonality residual is the priority | QR and eigendecomposition, CPU only |
| [`TaylorSeries`](@ref) | reproducing the historical implementation | unreliable for large lifts |

[`ScaledSquaring`](@ref) and [`NativePade`](@ref) avoid dense LAPACK and scalar indexing in package
code. Whether either runs on a particular accelerator depends on that backend's matrix-multiplication
and reduction support.

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
