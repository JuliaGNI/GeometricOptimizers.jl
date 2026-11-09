using ExplicitImports: test_explicit_imports
using GeometricOptimizers: GeometricOptimizers

test_explicit_imports(GeometricOptimizers;
    # off by a maintainer decision; the package relies on no implicit import
    no_implicit_imports = false,
    # the explicit imports include non-public names: `Base.Callable` and SimpleSolvers internals such as `alloc_h`
    all_explicit_imports_are_public = false,
    # the qualified accesses include non-public names of dependencies such as Base
    all_qualified_accesses_are_public = false,
    # `KernelAbstractions.zeros` is the documented spelling, and KernelInterface, which owns the
    # function, is no dependency of the package
    all_qualified_accesses_via_owners = (; ignore = (:zeros,)))
