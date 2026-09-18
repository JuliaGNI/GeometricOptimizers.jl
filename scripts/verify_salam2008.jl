# Checks `symplectic_householder!` against the paper it implements, Salam, Al-Aidarous and
# El Farouk, "Optimal symplectic Householder transformations for SR decomposition", Linear Algebra
# and its Applications 429 (2008) 1334-1353, doi:10.1016/j.laa.2008.02.029.
#
# Run with the repository as the active project:
#
#     julia --project=. scripts/verify_salam2008.jl
#
# The correspondence is not obvious from reading the code beside the paper, which is why it is
# checked rather than asserted. Three things differ in presentation without differing in
# mathematics, and each one looks like a discrepancy until it is expanded:
#
#   * The reflector vectors are stored negated. The paper has `v₁ = ρe₁ - a` and this stores
#     `a - ρe₁`; likewise for `v₂`. `T = I + cvvᴶ` is quadratic in `v`, so the sign cancels.
#   * `c₂` looks as though it has the wrong sign. The paper's Theorem 4.5 writes
#     `c̄₂ = -1/(±ξ u_{n+1})`, and the code has `c₂ = +s/(ξν)`. Expanding the general form in its
#     Theorem 3.2 settles it: `uᴶ(μe₁ + νe_{n+1}) = ν(u₁ - μ) = -νsξ`, so
#     `c₂ = -1/(-νsξ) = s/(ξν)`, which is the code. `c2_from_theorem_3_2` below evaluates that
#     general form directly and compares.
#   * `J` is never formed in the package, so `aᴶb` is `symplectic_form(a, b)`. Here it is written
#     out against an explicit `J` so that the two are independent.
#
# What is checked, per draw:
#
#   * (4.2)  `T₁a = ρe₁`
#   * (4.3)  `T₂e₁ = e₁` and `T₂T₁b = μe₁ + νe_{n+1}`
#   * (3.4)  `ρν = aᴶb`, the condition the free parameters must satisfy
#   * Lemma 2.4: `T₁` and `T₂` are symplectic
#   * Lemma 4.3: `ρ = sign(a₁)‖a‖₂` minimizes `κ₂(T₁)` over `ρ`
#   * Theorem 3.2: both coefficients equal the general expressions
#
# Lemma 4.3 is the one worth checking by search rather than by formula: it is the claim that makes
# the algorithm "optimal", and a plausible-looking alternative sign for `ρ` satisfies every other
# equation here while giving a worse-conditioned reflector.

using LinearAlgebra
using Printf
using Random

using GeometricOptimizers
using GeometricOptimizers: symplectic_form, symplectic_householder

poisson(n) = [zeros(n, n) I(n); -I(n) zeros(n, n)]

transvection(c, v, J) = I + c * v * (v' * J)

c1_from_theorem_3_2(a, ρ, e₁, J) = -1 / (a' * J * (ρ * e₁))
c2_from_theorem_3_2(u, μ, ν, e₁, eₙ₁, J) = -1 / (u' * J * (μ * e₁ + ν * eₙ₁))

"The 2-norm condition number of `I + cvvᴶ`, which Lemma 4.1 gives in closed form."
condition_number(c, v, J) = cond(transvection(c, v, J))

"""
Lemma 4.3 says `κ₂(T₁)` is minimized at `ρ = sign(a₁)‖a‖₂`. Scan `ρ` over both signs and a range of
magnitudes and return the best one found, to compare against what the code picks.
"""
function best_rho(a, J, e₁)
    nrm = norm(a)
    best, best_κ = NaN, Inf
    for sgn in (-1.0, 1.0), scale in range(0.25, 4.0; length = 601)

        ρ = sgn * scale * nrm
        v = a - ρ * e₁
        norm(v) < 1e-12 && continue
        c = c1_from_theorem_3_2(a, ρ, e₁, J)
        isfinite(c) || continue
        κ = condition_number(c, v, J)
        if κ < best_κ
            best, best_κ = ρ, κ
        end
    end
    best, best_κ
end

function verify(; n = 4, draws = 200, seed = 20260918)
    Random.seed!(seed)
    n2 = 2n
    J = poisson(n)
    e₁ = zeros(n2)
    e₁[1] = 1
    eₙ₁ = zeros(n2)
    eₙ₁[n + 1] = 1

    worst = Dict(k => 0.0
    for k in ("4.2", "4.3a", "4.3b", "3.4", "sympl T1", "sympl T2", "3.2 c1", "3.2 c2"))
    rho_optimal = 0

    for _ in 1:draws
        a, b = randn(n2), randn(n2)
        c₁, c₂, ρ, ν, μ, v₁, v₂ = symplectic_householder(a, b)
        T₁ = transvection(c₁, v₁, J)
        T₂ = transvection(c₂, v₂, J)
        u = T₁ * b

        worst["4.2"] = max(worst["4.2"], norm(T₁ * a - ρ * e₁))
        worst["4.3a"] = max(worst["4.3a"], norm(T₂ * e₁ - e₁))
        worst["4.3b"] = max(worst["4.3b"], norm(T₂ * u - (μ * e₁ + ν * eₙ₁)))
        worst["3.4"] = max(worst["3.4"], abs(ρ * ν - symplectic_form(a, b)))
        worst["sympl T1"] = max(worst["sympl T1"], norm(T₁' * J * T₁ - J))
        worst["sympl T2"] = max(worst["sympl T2"], norm(T₂' * J * T₂ - J))

        # Relative, because the coefficients themselves span orders of magnitude.
        ĉ₁ = c1_from_theorem_3_2(a, ρ, e₁, J)
        worst["3.2 c1"] = max(worst["3.2 c1"], abs(c₁ - ĉ₁) / abs(ĉ₁))

        # At `n == 1` there is no second reflector to build: `ξ` sums over the components outside
        # positions 1 and `n+1`, and there are none, so `T₁b` already has the required shape and
        # the code returns `c₂ = 0` with `T₂ = I`. Theorem 3.2's general expression divides by a
        # quantity that is not zero there, so comparing against it would report a difference where
        # the degenerate case is simply outside its scope.
        if n > 1
            ĉ₂ = c2_from_theorem_3_2(u, μ, ν, e₁, eₙ₁, J)
            worst["3.2 c2"] = max(worst["3.2 c2"], abs(c₂ - ĉ₂) / abs(ĉ₂))
        end

        # Lemma 4.3: the code's `ρ` should be the minimizer the scan finds.
        ρ_scan, _ = best_rho(a, J, e₁)
        rho_optimal += isapprox(ρ, ρ_scan; rtol = 1e-2)
    end

    println("Salam, Al-Aidarous and El Farouk (2008), 2n = $n2, $draws draws, seed $seed")
    println()
    for k in ("4.2", "4.3a", "4.3b", "3.4", "sympl T1", "sympl T2", "3.2 c1")
        @printf("  %-10s worst %.3e\n", k, worst[k])
    end
    if n > 1
        @printf("  %-10s worst %.3e\n", "3.2 c2", worst["3.2 c2"])
    else
        println("  3.2 c2     not applicable at n = 1: T₂ = I and c₂ = 0 by construction")
    end
    @printf("\n  Lemma 4.3  the code's ρ is the scanned minimizer in %d/%d draws\n",
        rho_optimal, draws)
    worst, rho_optimal
end

if abspath(PROGRAM_FILE) == @__FILE__
    for n in (1, 2, 4, 8)
        verify(; n = n)
        println()
    end
end
