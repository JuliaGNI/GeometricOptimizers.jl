```@meta
CurrentModule = GeometricOptimizers
```

# Limited-Memory BFGS

[`BFGS`](@ref) keeps its inverse-Hessian approximation ``Q`` as a dense ``d\times{}d`` matrix, where
``d`` is the number of free entries of the lift, and its cache holds five more. That limits it to
small parameter sets. [`LBFGS`](@ref) keeps the last `memory` secant pairs instead and applies the
``Q`` they define without forming it [nocedal2006numerical; Section 7.2](@cite). Everything else is
the framework of `BFGS`: the gradient [`GeometricOptimizers.gradient_rep!`](@ref), the secant pairs
in the flattened coordinates of the lift, the [`curvature_is_usable`](@ref) test, the line search,
the retraction and the restart, [`restart!(::LBFGSState)`](@ref).

## The two-loop recursion

Let ``(\delta_i, \gamma_i)``, ``i = k-m, \ldots, k-1``, be the stored pairs, ``\rho_i =
1/\delta_i^T\gamma_i``, and ``r = -\nabla{}f`` in the same coordinates. The direction ``Hr`` is
[nocedal2006numerical; Algorithm 7.4](@cite):

```math
\begin{aligned}
&q \gets r, \\
&\text{for } i = k-1, \ldots, k-m: \quad \alpha_i \gets \rho_i\delta_i^Tq, \quad q \gets q - \alpha_i\gamma_i, \\
&z \gets H_0q, \\
&\text{for } i = k-m, \ldots, k-1: \quad \beta \gets \rho_i\gamma_i^Tz, \quad z \gets z + (\alpha_i - \beta)\delta_i, \\
&Hr = z.
\end{aligned}
```

This is the ``Q`` that `BFGS` would reach from ``H_0`` by updating with the same ``m`` pairs, applied
to ``r``. A pair enters only where [`curvature_is_usable`](@ref) holds, and the oldest one is dropped
when `memory` pairs are held.

## The initial matrix

The coordinates of the lift are not orthonormal for the metric everywhere: on a
[`SymplecticStiefelManifold`](@ref) and a [`BiorthogonalManifold`](@ref) the Gram matrix ``M`` of
[`metric`](@ref) in those coordinates depends on the section (see
[The Global Tangent Space for the Symplectic Stiefel Manifold](@ref)). `BFGS` therefore starts ``Q``
at ``M^{-1}``, so that its first direction is that of [`GradientMethod`](@ref) for every section.
`LBFGS` takes ``H_0 = \theta{}M^{-1}`` for the same reason, and solves with ``M`` by the
conjugate-gradient method ([`GeometricOptimizers._metric_solve!`](@ref)), which needs only
[`GeometricOptimizers._metric_apply!`](@ref). So neither ``M`` nor ``M^{-1}`` is formed. On a vector
space, a [`StiefelManifold`](@ref) and a [`GrassmannManifold`](@ref), ``M`` is the identity and the
solve is a copy.

The keyword `initial` chooses ``\theta`` and the section ``M`` is taken at:

- `initial = :scaled`, the default, takes ``\theta = \delta^T\gamma/\gamma^TM^{-1}\gamma`` of the
  newest pair, (7.20) of [nocedal2006numerical](@cite) in the metric, and ``M`` at the current
  section. This rescales ``H_0`` at every step to the curvature last measured.
- `initial = :bfgs` takes ``\theta = 1`` and ``M`` at the section of the start or of the last
  restart. ``H_0`` is then the initial ``Q`` of `BFGS`, and with a `memory` that holds every
  accepted pair `LBFGS` takes the iterates of `BFGS` up to round-off.

`BFGS` corrects its initial matrix with every update, and a short memory cannot. Where ``M`` at the
start is a poor model of the metric later on, `:bfgs` is slow; the docstring of [`LBFGS`](@ref)
gives an example on the biorthogonal manifold.

## Memory and iterations

On ``St(5, 200)`` with ``f(Y) = -\frac{1}{2}\mathrm{tr}(Y^TSYD)``, where ``d = 985``:

```@example lbfgs
using GeometricOptimizers
using LinearAlgebra
import Random

Random.seed!(1)
S = Symmetric(randn(200, 200))
D = Diagonal([5.0, 4.0, 3.0, 2.0, 1.0])
f(Y) = -tr(Y' * S * Y * D) / 2
fstar = -sum(D.diag .* sort(eigvals(S); rev = true)[1:5]) / 2
Y₀ = rand(StiefelManifold{Float64}, 200, 5)

function compare(algorithm)
    Random.seed!(2)  # `global_section` draws the completion
    Y = copy(Y₀)
    opt = Optimizer(Y, f; algorithm, max_iterations = 1000)
    state = OptimizerState(algorithm, Y)
    solve!(Y, state, opt)
    (method = string(algorithm), iterations = state.iterations, gap = f(Y) - fstar,
        MiB = round((Base.summarysize(state) + Base.summarysize(opt)) / 2^20; digits = 1))
end

[compare(algorithm) for algorithm in (BFGS(), LBFGS(; memory = 5), LBFGS(),
    LBFGS(; initial = :bfgs))]
```

The memory column counts the state and the optimizer together, with the iterates, the gradients and
the sections that every method holds. For `BFGS` it grows as ``d^2``, for `LBFGS` as
``\mathtt{memory}\cdot{}d``.

`LBFGS` runs wherever `BFGS` runs: on vectors, on the manifolds of this package, and on a
`NetworkParameters` mixing them.
