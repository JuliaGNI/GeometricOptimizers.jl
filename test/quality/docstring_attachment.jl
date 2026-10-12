# A comment or a blank line between a docstring and its definition detaches the docstring: the
# binding has no documentation, and the docs build reports it only where an `@ref` names it.

using Test

# The lines of `file` that follow a top-level docstring and are blank or a comment.
function detached_docstrings(file)
    lines = readlines(file)
    hits = Int[]
    i = 1
    while i ≤ length(lines)
        if occursin(r"^(@doc\s+(raw)?)?\"\"\"", lines[i])
            # A one-line docstring closes on its opening line.
            j = count("\"\"\"", lines[i]) ≥ 2 ? i :
                findnext(l -> occursin("\"\"\"", l), lines, i + 1)
            j === nothing && break
            if j < length(lines)
                next = strip(lines[j + 1])
                (isempty(next) || startswith(next, "#")) && push!(hits, j + 1)
            end
            i = j + 1
        else
            i += 1
        end
    end
    hits
end

@testset "no comment or blank line between a docstring and its definition" begin
    root = normpath(joinpath(@__DIR__, "..", ".."))
    sources = [joinpath(path, file)
               for dir in ("src", "ext")
               for (path, _, files) in walkdir(joinpath(root, dir))
               for file in files if endswith(file, ".jl")]
    @test length(sources) > 50
    for f in sources
        @test (relpath(f, root), detached_docstrings(f)) == (relpath(f, root), Int[])
    end
end
