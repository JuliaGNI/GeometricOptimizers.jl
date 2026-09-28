using ExplicitImports: test_explicit_imports
using GeometricOptimizers: GeometricOptimizers

test_explicit_imports(GeometricOptimizers;
    # the package brings whole modules in with `using`; making each name explicit is not this check
    no_implicit_imports = false,
    # the explicit imports include non-public names of dependencies such as LinearAlgebra
    all_explicit_imports_are_public = false,
    # the qualified accesses include non-public names of dependencies such as Base
    all_qualified_accesses_are_public = false)
