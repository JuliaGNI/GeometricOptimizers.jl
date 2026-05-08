using GeometricOptimizers
using GeometricOptimizers: _copyto!
using Zygote
using Test

const a = [1.0, 2.0, 3.0]
const b = (a=a, b=[4.0, 5.0, 6.0])

loss(a::Vector) = sum(a .^ 2)
loss(b::NamedTuple{(:a, :b),Tuple{VT,VT}}) where {VT<:Vector} = loss(b.a - b.b)

g₁ = GradientFunction(loss, (_b, _a) -> (_copyto!(_b, Zygote.gradient(loss, _a)[1])), a)
g₂ = GradientFunction(loss, (_b, _a) -> (_copyto!(_b, Zygote.gradient(loss, _a)[1])), b)
g₁ᶜ = GradientAutodiff(loss, a)

@test g₁(a) == g₁ᶜ(a)
@test typeof(g₂(b)) <: NamedTuple{(:a, :b),Tuple{VT,VT}} where {VT<:Vector}
