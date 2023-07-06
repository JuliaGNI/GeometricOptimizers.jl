using GeometricOptimizers
using Documenter

DocMeta.setdocmeta!(GeometricOptimizers, :DocTestSetup, :(using GeometricOptimizers); recursive=true)

makedocs(;
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
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/GeometricOptimizers.jl",
    devbranch="main",
)
