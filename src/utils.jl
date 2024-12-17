# Kernel that is needed for functions relating to `SymmetricMatrix` and `SkewSymMatrix` 
@kernel function write_ones_kernel!(unit_matrix::AbstractMatrix{T}) where T
    i = @index(Global)
    unit_matrix[i, i] = one(T)
end