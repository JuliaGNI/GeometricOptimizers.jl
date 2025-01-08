# Kernel that is needed for functions relating to `SymmetricMatrix` and `SkewSymMatrix` 
@kernel function write_ones_kernel!(unit_matrix::AbstractMatrix{T}) where T
    i = @index(Global)
    unit_matrix[i, i] = one(T)
end

function apply_toNT(fun, ps::NamedTuple...)
    for p in ps
        @assert keys(ps[1]) == keys(p)
    end
    NamedTuple{keys(ps[1])}(fun(p...) for p in zip(ps...))
end

function add!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::AbstractVecOrMat)
    @assert size(A) == size(B) == size(C)
    C .= A + B
end

function add!(dx₁::NamedTuple, dx₂::NamedTuple, dx₃::NamedTuple)
    apply_toNT(add!, dx₁, dx₂, dx₃)
end