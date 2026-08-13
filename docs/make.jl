using GeometricOptimizers
using GeometricOptimizers: HessianBFGS, linesearch_problem
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using SimpleSolvers
import Bibliography

links = InterLinks(
    "SimpleSolvers" => (
        "https://juliagni.github.io/SimpleSolvers.jl/stable",
        "https://juliagni.github.io/SimpleSolvers.jl/stable/objects.inv",
        joinpath(@__DIR__, "inventories", "SimpleSolvers.toml")
    ),
)

bib = CitationBibliography(joinpath(@__DIR__, "src", "GeometricOptimizers.bib"))
Bibliography.sort_bibliography!(bib.entries, :nyt)  # name-year-title

DocMeta.setdocmeta!(GeometricOptimizers, :DocTestSetup, :(using GeometricOptimizers); recursive=true)

makedocs(;
    plugins=[bib, links],
    modules=[GeometricOptimizers],
    authors="Michael Kraus",
    repo="https://github.com/JuliaGNI/GeometricOptimizers.jl/blob/{commit}{path}#{line}",
    sitename="GeometricOptimizers.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://JuliaGNI.github.io/GeometricOptimizers.jl",
        edit_link="main",
        assets=String[],
        # `index.md` is a single `@autodocs` block holding every docstring in the package, so it grows
        # with the package and has already passed Documenter's 200 KiB default. Raising the hard limit
        # keeps the build green; `size_threshold_warn` is deliberately left at its default so the
        # warning keeps nagging until the reference is split across pages.
        size_threshold=400 * 2^10,
    ),
    pages=[
        "Home" => "index.md",
        "Optimization on Homogeneous Spaces" => "manifold_optimizers.md",
        "Linesearch" => "linesearch.md",
        "Linesearches on Manifolds" => "linesearch_on_manifolds.md",
        "Weight Decay on Manifolds" => "weight_decay.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/GeometricOptimizers.jl",
    devurl="latest",
    devbranch="main",
)
