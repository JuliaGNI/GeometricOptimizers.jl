# How many ambiguous method pairs the row-vector tie-breakers remove, and how many they create.
#
# Run with the repository as the active project, in a **cold process**:
#
#     julia --startup-file=no --project=. scripts/row_vector_ambiguity_count.jl
#
# This is the check behind the figures in the CHANGELOG entry *A row vector times one of this
# package's matrix types is an ordinary product again*. The account of what those methods are for is
# in `src/ambiguities.jl`, under *A row vector meets an owned matrix*.
#
# ## What is compared
#
# `Test.detect_ambiguities(GeometricOptimizers; recursive = false)` over the whole loaded set,
# twice: once as the package stands, and once with the tie-breakers deleted from the running
# session. The second count is what the package had before them, measured in the same process and
# against the same loaded dependency set — which is what makes the two comparable. Two counts taken
# in two sessions are not: `detect_ambiguities` sees every method of `*` that is loaded, so a
# different dependency set gives a different number for reasons that have nothing to do with this
# package.
#
# Deleting rather than checking out the previous revision is deliberate for the same reason, and it
# is why this script leaves its session unusable afterwards. It is a measurement, not a test.
#
# ## Which methods are deleted
#
# A method of `*` that this package owns and whose **left** operand is an `Adjoint` or a `Transpose`
# of a vector. The whole class is listed, because the list is what says the account in
# `src/ambiguities.jl` is complete. Only the subset the row-vector change added is deleted:
# `StiefelProjection`'s pair and the two that take the *adjoint* of a symplectic point on the right
# were already there, so deleting them would measure two changes at once.

using GeometricOptimizers
using LinearAlgebra: Adjoint, Transpose
using Test

const ROWVEC = Union{Adjoint, Transpose}

"Whether `T` is an `Adjoint` or a `Transpose` wrapping an `AbstractVector`."
function is_row_vector_type(@nospecialize(T))
    S = Base.unwrap_unionall(T)
    S isa DataType || return false
    S <: ROWVEC || return false
    parent = Base.unwrap_unionall(S.parameters[2])
    parent isa DataType ? parent <: AbstractVector : parent isa TypeVar
end

"Every row-vector tie-breaker this package owns."
function is_tie_breaker(m::Method)
    parentmodule(m) === GeometricOptimizers || return false
    sig = Base.unwrap_unionall(m.sig)
    length(sig.parameters) == 3 || return false
    is_row_vector_type(sig.parameters[2])
end

"The ones the row-vector change added, i.e. every one whose right operand is a bare owned matrix."
function is_new_tie_breaker(m::Method)
    right = Base.unwrap_unionall(Base.unwrap_unionall(m.sig).parameters[3])
    !(right <: StiefelProjection) && !(right <: ROWVEC)
end

function ambiguous_pairs()
    Set(Test.detect_ambiguities(GeometricOptimizers; recursive = false))
end

"The module of whichever method of the pair is not this package's."
function opposing_module(pair)
    a, b = pair
    parentmodule(a) === GeometricOptimizers ? parentmodule(b) : parentmodule(a)
end

function tally_by_module(pairs)
    counts = Dict{Module, Int}()
    for pair in pairs
        counts[opposing_module(pair)] = get(counts, opposing_module(pair), 0) + 1
    end
    sort(collect(counts); by = last, rev = true)
end

tie_breakers = sort(filter(is_tie_breaker, collect(methods(*)));
    by = m -> (string(m.file), m.line))

println("row-vector tie-breakers in the package: ", length(tie_breakers))
for m in tie_breakers
    println("  ", is_new_tie_breaker(m) ? "new " : "was ",
        basename(string(m.file)), ":", m.line, "  ", m.sig)
end

added = filter(is_new_tie_breaker, tie_breakers)

with = ambiguous_pairs()
println("\ndetect_ambiguities with the ", length(added), " this change adds:    ", length(with))

foreach(Base.delete_method, added)

without = ambiguous_pairs()
println("detect_ambiguities without them: ", length(without))
println("difference: ", length(with) - length(without))

# The net is not the whole story, and the CHANGELOG says so: the tie-breakers remove a pair against
# `LinearAlgebra` and one each against `FillArrays` and `ArrayLayouts`, then stand in the removed
# methods' place against the latter two. Printing both directions is what makes that checkable
# rather than inferred from one subtraction.
println("\nremoved by the ", length(added), " methods: ", length(setdiff(without, with)))
for (m, n) in tally_by_module(setdiff(without, with))
    println("  ", n, "\t", m)
end
println("created by them: ", length(setdiff(with, without)))
for (m, n) in tally_by_module(setdiff(with, without))
    println("  ", n, "\t", m)
end
