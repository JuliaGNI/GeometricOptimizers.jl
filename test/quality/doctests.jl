# The docstring and manual doctests, as the Doctests job of `Documenter.yml` runs them.
#
# Documenter evaluates a page's `@meta` block in `Main`, and this file runs in a module of its own,
# so the package is imported into `Main` first.

using Documenter
using GeometricOptimizers

@eval Main import GeometricOptimizers

DocMeta.setdocmeta!(GeometricOptimizers, :DocTestSetup, :(using GeometricOptimizers);
    recursive = true)

doctest(GeometricOptimizers)
