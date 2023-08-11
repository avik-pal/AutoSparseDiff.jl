@concrete struct MatrixColoringResult
    colorvec
    jacobian_sparsity
    idx_vec
    nz_rows
    nz_cols
end

struct NoMatrixColoring end

# Using Non-Sparse AD / NoSparsityDetection results in NoMatrixColoring
(::NoSparsityDetection)(::AbstractADType, args...; kwargs...) = NoMatrixColoring()

## If no specialization is available, we don't perform sparsity detection
(::AbstractMaybeSparsityDetection)(::AbstractADType, args...; kws...) = NoMatrixColoring()

# Prespecified Jacobian Structure
function (alg::JacPrototypeSparsityDetection)(ad::AbstractSparseADType, args...; kwargs...)
    J = alg.jac_prototype
    colorvec = __colorvec(ad, J, alg.alg)
    (nz_rows, nz_cols) = collect.(ArrayInterface.findstructralnz(J))
    return MatrixColoringResult(colorvec, J, collect(1:size(J, 2)), nz_rows, nz_cols)
end

function __colorvec(ad, J, alg)
    if alg === nothing
        return matrix_colors(ad isa AbstractSparseReverseModeAD ? J' : J)
    else
        return matrix_colors(J, alg; partition_by_rows=(ad isa AbstractSparseReverseModeAD))
    end
end

# TODO: Heuristics to decide whether to use Sparse Differentiation or not
