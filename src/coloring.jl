@concrete struct MatrixColoringResult
    colorvec
    jacobian_sparsity
    idx_vec
    nz_rows
    nz_cols
end

struct NoMatrixColoring end

# Using Non-Sparse AD / NoSparsityDetection results in NoMatrixColoring
(::NoSparsityDetection)(::AbstractADType, _, _) = NoMatrixColoring()

## If no specialization is available, we don't perform sparsity detection
(alg::AbstractMaybeSparsityDetection)(::AbstractADType, _, _) = NoMatrixColoring()

# Prespecified Jacobian Structure
function (alg::JacPrototypeSparsityDetection)(::Union{AbstractSparseFiniteDifferencesMode,
        AbstractSparseForwardModeAD}, _, _)
    J = alg.jac_prototype
    colorvec = alg.alg === nothing ? matrix_colors(J) : matrix_colors(J, alg.alg)
    (nz_rows, nz_cols) = collect.(ArrayInterface.findstructralnz(J))
    return MatrixColoringResult(colorvec, J, collect(1:size(J, 2)), nz_rows, nz_cols)
end

function (alg::JacPrototypeSparsityDetection)(::AbstractSparseReverseModeAD, _, _)
    J = alg.jac_prototype
    colorvec = alg.alg === nothing ? matrix_colors(J') :
               matrix_colors(J, alg.alg; partition_by_rows=true)
    (nz_rows, nz_cols) = collect.(ArrayInterface.findstructralnz(J))
    return MatrixColoringResult(colorvec, J, collect(1:size(J, 1)), nz_rows, nz_cols)
end

# TODO: Using Symbolics to Generate Sparsity Pattern

# TODO: Heuristics to decide whether to use Sparse Differentiation or not
