# How far off the manifold an iterate may be: `check(Y)`, the distance of `YᵀY` from the identity.
# Every retraction maps onto the manifold exactly in exact arithmetic, so this bounds round-off
# accumulated over a solve and nothing else. It is a multiple of `eps(T)`: in `Float64` it is
# 9.1e-13, under the `1e-12` this suite used, where the observed values are of the order of 1e-14.
#
# The one copy of this number: `test/verification/svd_optim.jl`,
# `test/integration/manifold_linesearch_tests.jl` and `test/retractions/retractions.jl` include it,
# and `scripts/retraction_accuracy.jl` keeps a copy that points here.
manifold_tolerance(::Type{T}) where {T <: AbstractFloat} = 4096 * eps(T)
