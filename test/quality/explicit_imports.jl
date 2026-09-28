using ExplicitImports: test_explicit_imports
using GeometricOptimizers: GeometricOptimizers

test_explicit_imports(GeometricOptimizers;
    # off by the plan's decision; the package has no implicit import today
    no_implicit_imports = false,
    # the explicit imports include non-public names: `Base.Callable` and SimpleSolvers internals such as `alloc_h`
    all_explicit_imports_are_public = false,
    # the qualified accesses include non-public names of dependencies such as Base
    all_qualified_accesses_are_public = false)
