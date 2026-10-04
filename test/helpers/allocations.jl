# Two `@allocated` readings compared as a difference under a tolerance, and not as an equality.
#
# A byte count is not bit-reproducible on every platform. On Windows the same `Cayley` call at two
# sizes came back 3 671 and 3 719 bytes, and `update_section!` 3 831 and 3 815 -- 48 and 16 apart,
# in *both* directions, so it is quantisation inside `inv`'s own allocation and not a term that
# grows with `N`. `exp` of the same `4n × 4n` matrix reads up to 96 bytes apart between two calls
# there. Linux and macOS give the two readings byte for byte. An exact equality is therefore a
# platform lottery.
#
# The tolerance does not weaken what is asserted as long as the fixture is large enough that the
# smallest term the assertion has to catch -- one reintroduced `N × N` or `N × 2n` temporary -- is
# well above it. Each caller states that size beside its assertion.
const N_INDEPENDENCE_TOLERANCE = 1024

function n_independent(a, b; tolerance = N_INDEPENDENCE_TOLERANCE)
    abs(a - b) < tolerance
end
