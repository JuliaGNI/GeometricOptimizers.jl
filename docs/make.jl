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
    ),
    pages=[
        "Home" => "index.md",
        "References" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/GeometricOptimizers.jl",
    devurl="latest",
    devbranch="main",
)
