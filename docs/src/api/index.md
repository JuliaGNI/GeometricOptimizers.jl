# API Reference

Every docstring in the package, on one page per part of the manual. The chapters explain what these
are *for* and link into these pages; a docstring may only be rendered once, which is why they link
rather than repeat.

```@autodocs
Modules = [GeometricOptimizers]
Pages = ["src/GeometricOptimizers.jl"]
```

```@index
Pages = ["manifolds.md", "special_matrices.md", "retractions.md", "optimizers.md",
         "optimizer_methods.md", "linesearch.md"]
```
