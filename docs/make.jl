using GeometricOptimizers
using Documenter
using DocumenterCitations
import Bibliography

bib = CitationBibliography(joinpath(@__DIR__, "src", "GeometricOptimizers.bib"))
Bibliography.sort_bibliography!(bib.entries, :nyt)  # name-year-title

DocMeta.setdocmeta!(GeometricOptimizers, :DocTestSetup, :(using GeometricOptimizers); recursive=true)

makedocs(;
    plugins = [bib],
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
    repo   = "github.com/JuliaGNI/GeometricOptimizers.jl",
    devurl = "latest",
    devbranch = "main",
)
