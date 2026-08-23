using GeometricOptimizers
using GeometricOptimizers: HessianBFGS, linesearch_problem
using Documenter
using DocumenterCitations
using DocumenterInterLinks
using Markdown
using SimpleSolvers
import Bibliography

links = InterLinks(
    "SimpleSolvers" => (
        "https://juliagni.github.io/SimpleSolvers.jl/stable",
        "https://juliagni.github.io/SimpleSolvers.jl/stable/objects.inv",
        joinpath(@__DIR__, "inventories", "SimpleSolvers.toml")
    ),
    # The manifold and optimizer chapters moved here from `GeometricMachineLearning` in
    # GeometricMachineLearning#239, and a handful of their cross-references point at chapters that
    # stayed there — the pullback machinery, the SympNet and transformer architectures. Those are
    # neural-network material and do not belong here, so the references are kept as references.
    # This is a documentation-only edge: nothing under `src/` or `test/` may depend on GML, and
    # nothing does. `DocumenterInterLinks` reads a published `objects.inv`, not a local build, so
    # linking both ways creates no build-order cycle.
    "GeometricMachineLearning" => (
        "https://juliagni.github.io/GeometricMachineLearning.jl/latest",
        "https://juliagni.github.io/GeometricMachineLearning.jl/latest/objects.inv",
        joinpath(@__DIR__, "inventories", "GeometricMachineLearning.toml")
    ),
)

# The chapters that moved here from `GeometricMachineLearning` state their definitions, theorems and
# proofs through these, so that one source renders as an admonition in HTML and as a LaTeX
# environment in the PDF build. They are `Main.`-qualified from the pages, which is why they are
# defined at top level here rather than in a module.
const output_type = isempty(ARGS) ? :html : ARGS[1] == "html_output" ? :html : :latex

function theorem(statement::String, name::Nothing; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! info "Theorem"
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{th:" * label * raw"}"
        Markdown.parse(raw"\begin{thrm}" * statement * theorem_label * raw"\end{thrm}")
    end
end

function theorem(statement::String, name::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! info "Theorem ($(name))"
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{th:" * label * raw"}"
        Markdown.parse(raw"\begin{thrm}[" * name * "]" * statement * theorem_label * raw"\end{thrm}")
    end
end

function theorem(statement::String; name::Union{Nothing, String} = nothing, label::Union{Nothing, String} = nothing)
    theorem(statement, name; label = label)
end

function definition(statement::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! info "Definition"
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{def:" * label * raw"}"
        Markdown.parse(raw"\begin{dfntn}" * statement * theorem_label * raw"\end{dfntn}")
    end
end

function example(statement::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! info "Example"
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{xmpl:" * label * raw"}"
        Markdown.parse(raw"\begin{xmpl}" * statement * theorem_label * raw"\end{xmpl}")
    end
end

function remark(statement::String; label::Union{Nothing, String} = nothing)
    if Main.output_type == :html
        Markdown.parse("""!!! tip "Remark"
            \t $(statement)""")
    else
        theorem_label = isnothing(label) ? "" : raw"\label{rmrk:" * label * raw"}"
        Markdown.parse(raw"\begin{rmrk}" * statement * theorem_label * raw"\end{rmrk}")
    end
end

function proof(statement::String)
    if Main.output_type == :html
        Markdown.parse("""!!! details "Proof"
            \t $(statement)""")
    else
        Markdown.parse(raw"\begin{proof}" * statement * raw"\end{proof}")
    end
end

function sphere(r, C)   # r: radius; C: center [cx,cy,cz]
    n = 100
    u = range(-π, π; length = n)
    v = range(0, π; length = n)
    x = C[1] .+ r * cos.(u) * sin.(v)'
    y = C[2] .+ r * sin.(u) * sin.(v)'
    z = C[3] .+ r * ones(n) * cos.(v)'
    x, y, z
end

# this is needed if we have multiline definitions or proofs

# the indentation the multi-line `Main.definition`/`Main.theorem` bodies are written with
const indentation = output_type == :html ? "\t" : ""

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
        # The moved chapters include each of their figures twice, once per theme; without this
        # stylesheet both variants render, stacked. See `docs/src/assets/extra_styles.css`.
        assets=["assets/extra_styles.css"],
        # `api.md` is one catch-all `@autodocs` over the whole package, so it is large by design.
        # `size_threshold_warn` is left at its default: the warning it prints for that one page is
        # the reminder that the docstrings are still not distributed over the chapters that explain
        # them.
        size_threshold=400 * 2^10,
    ),
    pages=[
        "Home" => "index.md",
        # The `Manifolds`, `Special Matrices` and `Optimizers` chapters came over from
        # `GeometricMachineLearning` with the types they describe; see GeometricMachineLearning#239.
        "Manifolds" => [
            "Concepts from General Topology" => "manifolds/basic_topology.md",
            "Metric and Vector Spaces" => "manifolds/metric_and_vector_spaces.md",
            "Foundations of Differential Manifolds" => "manifolds/inverse_function_theorem.md",
            "General Theory on Manifolds" => "manifolds/manifolds.md",
            "Differential Equations and the EAU theorem" => "manifolds/existence_and_uniqueness_theorem.md",
            "Riemannian Manifolds" => "manifolds/riemannian_manifolds.md",
            "Homogeneous Spaces" => "manifolds/homogeneous_spaces.md",
        ],
        "Special Matrices" => [
            "Symmetric, Skew-Symmetric and Triangular Matrices" => "special_matrices.md",
            "Global Tangent Spaces" => "global_tangent_spaces.md",
        ],
        "Optimizers" => [
            "Optimization on Homogeneous Spaces" => "manifold_optimizers.md",
            "Retractions" => "retractions.md",
            "Parallel Transport" => "parallel_transport.md",
            "Optimizer Methods" => "optimizer_methods.md",
        ],
        "Linesearch" => [
            "Linesearch" => "linesearch.md",
            "Linesearches on Manifolds" => "linesearch_on_manifolds.md",
            "Weight Decay on Manifolds" => "weight_decay.md",
        ],
        "References" => "references.md",
        "API" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/JuliaGNI/GeometricOptimizers.jl",
    devurl="latest",
    devbranch="main",
)
