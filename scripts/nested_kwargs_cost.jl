# What nested `kwargs...` splatting costs to compile when its result feeds a large call tree in the
# same inferred body. Open issue **D1**.
#
#     julia --project=. scripts/nested_kwargs_cost.jl              # the four controls
#     julia --project=. scripts/nested_kwargs_cost.jl --levels 5   # deeper splatting
#     julia --project=. scripts/nested_kwargs_cost.jl --width 400  # a bigger call tree
#     julia --project=. scripts/nested_kwargs_cost.jl --with-package  # the real Optimizer + solve!
#
# **Run this on 1.12 and 1.13 at least.** The whole point of D1 is that the two disagree by two orders
# of magnitude, so a figure without its neighbour says nothing.
#
# ## What D1 recorded, and why this file exists
#
# Constructing an `Optimizer` through three nested levels of `kwargs...` splatting and calling `solve!`
# on it *in the same inferred function body* cost **940.86 s** of compile time on Julia 1.12.6 against
# **4.35 s** on 1.13.0-rc2. Neither half is slow alone: 0.99 s for the constructor, 2.35 s for `solve!`.
# Flattening the splatting to one level took it to 6.53 s. A `@noinline` barrier around the construction
# did not help (925.27 s), and neither did `@nospecialize` on the enclosing function (965 s). Before it
# was worked around by PR #35 the CI suite took 31–42 minutes on 1.12 against 3–5 on every other
# version, with one test file accounting for the whole difference.
#
# That is four controls, and between them they rule out the two obvious explanations — so they are worth
# reporting together or not at all. **The reproducer they were measured with lived in `/tmp` and is
# gone.** D1's own text says that is the case for writing it down somewhere durable this time, and this
# is that. It is also the precondition for reporting the thing upstream: a bug report has to reduce to
# something that does not depend on this package, which is what everything below is written not to.
#
# ## Nothing here imports GeometricOptimizers
#
# Deliberately, and it is the file's main design constraint. `Base` only, so that a figure from it can
# go into a JuliaLang issue as it stands.
#
# ## As written, it does not reproduce D1 — and that is the current state of the issue
#
# Measured, four controls, one process each, `--levels 3`:
#
#   |               | 1.12.7, width 200 | 1.12.7, width 800 | 1.13.0-rc3, width 200 | 1.13.0-rc3, width 800 |
#   |---------------|-------------------|-------------------|-----------------------|-----------------------|
#   | plain         |       0.09        |       0.21        |         0.05          |         0.19          |
#   | `@noinline`   |       0.10        |       0.21        |         0.06          |         0.21          |
#   | `@nospecialize`|      0.11        |       0.27        |         0.09          |         0.24          |
#   | one level     |       0.09        |       0.21        |         0.08          |         0.19          |
#
# **No version gap and no plain-versus-one-level gap**, at any width tried, and `--levels 6` moves
# nothing either. D1's signature is a 200× spread between the first and last rows and a 200× spread
# between the two versions; this has neither. A second variant was tried and is also negative: the work
# chain specialised on the `Widget`'s type parameters rather than taking it abstractly, with a
# function-typed field carried through the `kwargs` — 0.14 s for both controls on both versions at
# width 400.
#
# ## Neither does the real thing, which is the more surprising half
#
# `--with-package` is the second mode, and it is not a reduction at all: it builds a real `Optimizer`
# through three nested `kwargs...` levels and calls `solve!` in the same body, on an `St(20, 3)` SVD
# problem. That is PR #35's before-shape, reconstructed. It imports this package, so it is a diagnostic
# rather than something to attach to a JuliaLang issue — but it is the control that says whether D1 is
# still live at all.
#
#   | | 1.12.6 | 1.12.7 | 1.13.0-rc3 |
#   |---|---|---|---|
#   | flat (one level)              | 3.44 | 3.13 | 3.36 |
#   | nested (three levels)         | 3.44 | 3.12 | 3.39 |
#   | nested, three algorithm types | —    | 5.05 | —    |
#   | flat, three algorithm types   | —    | 5.05 | —    |
#
# **Not a trace of it**, on either 1.12 patch, and the nested and flat columns are equal to the
# hundredth of a second. D1 recorded 4.35 s against **940.86 s** for these two on 1.12.6. The 1.13
# figures agree with D1's 4.35 s to within a cold measurement's spread, so the harness is measuring the
# right thing; it is the 940 that will not come back.
#
# ## So four reconstructions are negative and D1 is not reportable
#
# Two synthetic, two with this package's own types, on both 1.12 patch releases:
#
#   1. `Base`-only, `Widget` taken abstractly through a chain of 800 — no gap.
#   2. `Base`-only, specialising on the type parameters, function-typed field — no gap.
#   3. real `Optimizer` + `solve!`, one algorithm — no gap.
#   4. real `Optimizer` + `solve!`, looping over three algorithm types and two retractions in one body,
#      which is what `test/optimizer_convergence/svd_optim.jl:223-234` does and what its comment says
#      the loop exists for — no gap.
#
# A bug report needs a reproducer and four independent negatives are not one. What these *do* establish
# is that D1's own description of itself — "a constructor reached through N nested `kwargs...` levels
# whose result is passed to a second function with a large call tree, both in one inferred body" — is
# **not sufficient**, four ways. That description is what anyone would work from, so knowing it is
# incomplete is the useful part.
#
# Candidates for what is still missing, none tested, cheapest first:
#
#   * The `Options(T; options_kwargs...)` call. Keyword *defaults* computed from other keywords make
#     inference resolve a dependency order, and it is the one part of the real constructor that none of
#     these four vary.
#   * The objective. D1's was the SVD problem's closure over a captured `A`; #3 and #4 use the same
#     shape but a smaller `N`, and inference cost may follow the objective's own tree rather than the
#     construction's.
#   * Something in this package changed between PR #35 and now that removed the sensitivity, in which
#     case D1 is *closeable* and the interesting artefact is whichever commit did it. This is the
#     hypothesis the table above most supports and the hardest to test: it means bisecting #35..HEAD
#     with the nested shape reinstated at each step.
#
# What is *ruled out* is the patch-release explanation, which was the cheap hypothesis and the one worth
# eliminating first: 1.12.6 and 1.12.7 agree to within 0.3 s on every control.
#
# ## How it measures
#
# One process per cell (`fan_out`), `Base.invokelatest` at the timed call, and `time()` rather than
# `@elapsed` — all three for the reasons `scripts/walk_compile_cost.jl` gives at length above its own
# `first_call`. The short version: compiling the harness infers through the call in its body, so a
# direct call spends the inference *before* the clock starts and prints the leftover.
#
# And read the load average before believing a figure. If a cell drives the box into swap the cells
# after it are nonsense; that has happened to the sibling harness and is recorded there.

const LEVELS = let i = findfirst(==("--levels"), ARGS)
    i === nothing ? 3 : parse(Int, ARGS[i + 1])
end

# The size of the "second function with a large call tree". 200 distinct one-line functions, each
# calling the next, is a stand-in for `solve!` — what matters to D1's shape is that the tree is large
# and that its argument type comes from the construction, not what it computes.
const WIDTH = let i = findfirst(==("--width"), ARGS)
    i === nothing ? 200 : parse(Int, ARGS[i + 1])
end

# Four type parameters, as `AdamCache` has, so the constructed type is not a trivial one to infer
# through. `D` is left unconstrained on purpose: it is what the deepest `kwargs...` level supplies.
struct Widget{A, B, C, D}
    a::A
    b::B
    c::C
    d::D
end

# `LEVELS` functions, each splatting `kwargs...` into the next and adding one positional argument of
# its own. This is D1's first half: a constructor reached through N nested `kwargs...` levels.
#
# Generated rather than written out so that `--levels` means something. `_ctor_1` is the entry point and
# `_ctor_$LEVELS` is the one that actually builds the `Widget`.
let
    @eval $(Symbol(:_ctor_, LEVELS))(x; kwargs...) =
        Widget(x, get(kwargs, :b, 0), get(kwargs, :c, 0.0), get(kwargs, :d, Int8(0)))
    for l in (LEVELS - 1):-1:1
        @eval $(Symbol(:_ctor_, l))(x; kwargs...) = $(Symbol(:_ctor_, l + 1))(x + $l; kwargs...)
    end
end

# The innermost constructor under a stable name, for the one-level control below.
const _ctor_innermost = getfield(@__MODULE__, Symbol(:_ctor_, LEVELS))

# The second half: a large call tree whose entry argument is whatever the construction returned.
# `_work_1` calls `_work_2` calls … `_work_$WIDTH`, each doing something typed with the `Widget`'s
# fields so that none of it can be folded away.
let
    @eval $(Symbol(:_work_, WIDTH))(w::Widget) = w.a + w.b + w.c + w.d
    for k in (WIDTH - 1):-1:1
        @eval $(Symbol(:_work_, k))(w::Widget) = $(Symbol(:_work_, k + 1))(w) + $k * w.a
    end
end

const _work_entry = getfield(@__MODULE__, Symbol(:_work_, 1))

# ## The four controls
#
# Each is "construct, then work, in one inferred body" — which is the shape, and splitting the two is
# the thing that fixes it.

# 1. plain: the shape D1 measured at 940.86 s on 1.12.6.
plain() = _work_entry(_ctor_1(1.0; b = 2, c = 3.0, d = Int8(4)))

# 2. a `@noinline` barrier around the construction. D1: 925.27 s, i.e. nothing.
@noinline _ctor_barrier() = _ctor_1(1.0; b = 2, c = 3.0, d = Int8(4))
noinline_barrier() = _work_entry(_ctor_barrier())

# 3. `@nospecialize` on the enclosing function. D1: 965 s, i.e. nothing, and slightly worse.
function nospecialized(@nospecialize(x))
    _work_entry(_ctor_1(x; b = 2, c = 3.0, d = Int8(4)))
end

# 4. the splatting flattened to one level, which is what PR #35 did in this package. D1: 6.53 s, i.e.
#    the fix. Calls the innermost constructor directly, so no level splats into another.
one_level() = _work_entry(_ctor_innermost(1.0 + LEVELS - 1; b = 2, c = 3.0, d = Int8(4)))

# ## The package-typed controls
#
# Loaded only for `--with-package`, so the default run stays `Base`-only and attachable to a JuliaLang
# issue. This is PR #35's before-shape reconstructed: a real `Optimizer` reached through three nested
# `kwargs...` levels, with `solve!` in the same inferred body. It lives here rather than in `/tmp`
# because the last copy lived in `/tmp` and that is why D1 had to be re-measured from nothing.
const WITH_PACKAGE = "--with-package" in ARGS

if WITH_PACKAGE
    @eval begin
        using GeometricOptimizers
        using GeometricOptimizers: Optimizer, OptimizerState, solve!, GradientMethod, MomentumMethod,
                                   Adam, StiefelManifold, Geodesic, Cayley
        using SimpleSolvers: Static
        import Random

        const _N, _n = 20, 3
        const _A = randn(Random.Xoshiro(1234), _N, _N)
        _objective(ps) = sum(abs2, _A - ps.Y * (ps.Y' * _A))
        _start() = (Y = rand(Random.Xoshiro(3), StiefelManifold{Float64}, _N, _n),)

        # three levels, as the constructor had before #35 flattened it
        _pkg_lvl3(x, F; kwargs...) = Optimizer(x, F; kwargs...)
        _pkg_lvl2(x, F; kwargs...) = _pkg_lvl3(x, F; kwargs...)
        _pkg_lvl1(x, F; kwargs...) = _pkg_lvl2(x, F; kwargs...)

        _opts(alg, retr) = (retraction = retr, algorithm = alg, linesearch = Static(0.01),
                            max_iterations = 2, warn_iterations = 0)

        # one algorithm: construction and solve in one body, nested against flat
        function pkg_nested()
            ps = _start(); alg = GradientMethod()
            opt = _pkg_lvl1(ps, _objective; _opts(alg, Cayley())...)
            solve!(ps, OptimizerState(alg, ps), opt)
        end

        function pkg_flat()
            ps = _start(); alg = GradientMethod()
            opt = Optimizer(ps, _objective; _opts(alg, Cayley())...)
            solve!(ps, OptimizerState(alg, ps), opt)
        end

        # and the same with three algorithm types and two retractions in the one frame, which is the
        # breadth `svd_optim.jl`'s loop has and the single-algorithm controls above do not
        function pkg_nested_multi()
            for retr in (Geodesic(), Cayley()), alg in (GradientMethod(), MomentumMethod(), Adam())
                ps = _start()
                opt = _pkg_lvl1(ps, _objective; _opts(alg, retr)...)
                solve!(ps, OptimizerState(alg, ps), opt)
            end
        end

        function pkg_flat_multi()
            for retr in (Geodesic(), Cayley()), alg in (GradientMethod(), MomentumMethod(), Adam())
                ps = _start()
                opt = Optimizer(ps, _objective; _opts(alg, retr)...)
                solve!(ps, OptimizerState(alg, ps), opt)
            end
        end
    end
end

# `time()` and not `@elapsed`, `invokelatest` and not a direct call: see the header.
function first_call(f, args...)
    t = time()
    Base.invokelatest(f, args...)
    round(time() - t; digits = 2)
end

function row(label, f, args...)
    print("  ", rpad(label, 22), ": ")
    try
        println(first_call(f, args...), " s")
    catch e
        println(nameof(typeof(e)))
    end
end

# One control per process, because each is a *first* call and two in one process share whatever the
# first of them compiled. That is the same reason `walk_compile_cost.jl` fans out per cell, and it is
# the mistake that produced two of the wrong figures recorded in this repository's changelog.
const CONTROLS = ("plain", "noinline", "nospecialize", "one-level")
const PKG_CONTROLS = ("pkg-flat", "pkg-nested", "pkg-flat-multi", "pkg-nested-multi")

function run_one(which)
    println("LEVELS = ", LEVELS, ", WIDTH = ", WIDTH, "   [", which, "]")
    which == "plain"            ? row("plain", plain) :
    which == "noinline"         ? row("@noinline barrier", noinline_barrier) :
    which == "nospecialize"     ? row("@nospecialize", nospecialized, 1.0) :
    which == "one-level"        ? row("one level", one_level) :
    which == "pkg-flat"         ? row("pkg, one level", pkg_flat) :
    which == "pkg-nested"       ? row("pkg, three levels", pkg_nested) :
    which == "pkg-flat-multi"   ? row("pkg, flat, 3 algs", pkg_flat_multi) :
                                  row("pkg, nested, 3 algs", pkg_nested_multi)
end

function fan_out()
    extra = WITH_PACKAGE ? ["--with-package"] : String[]
    for which in (WITH_PACKAGE ? PKG_CONTROLS : CONTROLS)
        run(`$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project())
             $(@__FILE__) --in-process $which --levels $LEVELS --width $WIDTH $extra`)
    end
end

function main(args)
    if "--in-process" in args
        i = findfirst(==("--in-process"), args)
        return run_one(args[i + 1])
    end
    fan_out()
end

main(ARGS)
