# Convenience targets for building the documentation locally. Run from this directory:
#
#   make            build into build/ (instantiating the docs environment first)
#   make figures    compile the TikZ figures, which are build products and not committed
#   make documenter build without touching the environment, once it is instantiated
#   make clean      remove build/
#
# `figures` is the same step .github/workflows/Documenter.yml runs before `make.jl`; without it the
# build is six images short. It needs `xelatex` and `pdftocairo` — see src/tikz/Makefile — so it is
# also the one target a machine without a TeX installation cannot run.
#
# `instantiate` is the same pair of calls .github/workflows/Documenter.yml makes, so a local build
# resolves exactly as CI's does: `develop` points the environment at the working tree rather than at
# the registered version. Note that on Julia ≥ 1.11 `Pkg.develop` records this as a `[sources]`
# entry in `docs/Project.toml`; that edit is not meant to be committed.

.PHONY: all instantiate figures documenter clean

all: instantiate figures documenter

instantiate:
	julia --color=yes --project=. -e 'using Pkg; Pkg.develop(PackageSpec(path = "..")); Pkg.instantiate()'

figures:
	$(MAKE) all -C src/tikz

documenter:
	julia --color=yes --project=. make.jl

clean:
	rm -Rf build
