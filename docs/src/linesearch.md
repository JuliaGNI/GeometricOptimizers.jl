## Linesearches for Optimizers

In `GeometricOptimizers` we typically build the search direction by multiplying the gradient with a [Hessian](@extref SimpleSolvers Hessians). When starting at ``x_k`` we take:

```math
    p_k = H_{x_k}^{-1}(\nabla_{x_k}f),
```
where ``[H_{x_k}]_{ij} = \partial^2{}f\partial{}x_i\partial{}x_j|_{x_k}`` is the [Hessian](@extref SimpleSolvers Hessians). Note that we often use approximations of this Hessian in practice (such as the [`HessianBFGS`](@ref)).

The linesearch objective is then built as
```math
f^\mathrm{ls}(\alpha) = f(x_k + \alpha{}p_k).
```

For manifolds [absil2008optimization](@cite) defining a Hessian, equivalently to defining a [gradient](@extref SimpleSolvers Gradients), requires a Riemannian metric and the associated Levi-Civita connection ``\nabla``:

```math
\mathrm{Hess}(f) := \nabla\nabla{}f = \nabla{}df \in \Gamma(T^*\mathcal{M}\otimes{}T^*\mathcal{M}).
```

For specific vector fields ``\xi, \eta \in \Gamma(T\mathcal{M})`` we can write this as:

```math
\langle \mathrm{Hess}(f)[\xi], \eta  \rangle = \xi(\eta{}f) - (\nabla_\xi\eta)f.
```

## Example

We look at the following example:

```@example quadratic
f(x::Union{T, Vector{T}}) where {T<:Number} = exp.(x) .* (x .^ 3 .- 5x .+ 2x) .+ 2one(T)
f!(y::AbstractVector{T}, x::AbstractVector{T}) where {T} = y .= f.(x)
F!(y::AbstractVector{T}, x::AbstractVector{T}, params) where {T} = f!(y, x)
```

We hence use [`linesearch_problem`](@ref) not for a [`SimpleSolvers.NewtonSolver`](@extref), but for an [`EuclideanOptimizer`](@ref):

```@example quadratic
using GeometricOptimizers # hide
using GeometricOptimizers: NewtonOptimizerCache, initialize!, gradient, compute_direction, linesearch_problem # hide
x₀ = [0., .1, .2]
x = copy(x₀)
obj = OptimizerProblem(sum∘f, x₀)
grad = GradientAutodiff{Float64}(obj.F, length(x₀))
_cache = NewtonOptimizerCache(x₀)
state = NewtonOptimizerState(x₀)
hess = HessianAutodiff(obj, x₀)
update!(state, grad, x₀)
update!(_cache, state, grad, hess, x₀)
params = (x = state.x, )
ls_obj = linesearch_problem(obj, grad, _cache)

fˡˢ(alpha) = ls_obj.F(alpha, params)
∂fˡˢ∂α(alpha) = ls_obj.D(alpha, params)
nothing # hide
```

```@setup quadratic
using CairoMakie
fig = Figure()
ax = Axis(fig[1, 1])
alpha = -3.:.01:3.
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}_\mathrm{opt}(\alpha)")
axislegend(ax)
save("f_ls_optimizer_light.png", fig)
nothing # hide
```

![](f_ls_optimizer_light.png)

!!! info
    Note the different shape of the line search problem in the case of the optimizer, especially that the line search problem can take negative values in this case!

We now again want to find the minimum with quadratic line search and repeat the procedure above:

```@example quadratic
p₀ = fˡˢ(0.)
```

```@example quadratic
p₁ = ∂fˡˢ∂α(0.)
```

```@example quadratic
using SimpleSolvers: bracket_minimum_with_fixed_point, compute_new_iterate! # hide
params = (x = state.x, )
α₀ = bracket_minimum_with_fixed_point(ls_obj, params, 0.)[1]
@assert !(α₀ == 0. || α₀ == .1) # hide
y = fˡˢ(α₀)
p₂ = (y - p₀ - p₁*α₀) / α₀^2
p(α) = p₀ + p₁ * α + p₂ * α^2
α₁ = -p₁ / (2p₂)
```

```@setup quadratic
using CairoMakie
mred = RGBf(214 / 256, 39 / 256, 40 / 256)
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256)
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)
mblue = RGBf(31 / 256, 119 / 256, 180 / 256)
morange = RGBf(255 / 256, 127 / 256, 14 / 256)

fig = Figure()
ax = Axis(fig[1, 1])
lines!(ax, alpha, fˡˢ.(alpha); label = L"f^\mathrm{ls}_\mathrm{opt}(\alpha)")
lines!(ax, alpha, p.(alpha); label = L"p^{(1)}(\alpha)")
scatter!(ax, α₁, p(α₁); color = mred, label = L"\alpha_1")
axislegend(ax)
save("f_ls_opt1_light.png", fig)
nothing # hide
```

![](f_ls_opt1_light.png)

We now again move the original ``x`` in the Newton direction with step length ``\alpha_1``:

```@example quadratic
sum∘f(x)
```

```@example quadratic
using SimpleSolvers: direction # hide
compute_new_iterate!(x, α₁, direction(_cache))
```

```@example quadratic
sum∘f(x)
```
