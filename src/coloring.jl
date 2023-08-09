@concrete struct MatrixColoringResult
    colorvec
    jacobian_sparsity
    idx_vec
    nz_rows
    nz_cols
end

# Prespecified Jacobian Structure
function (alg::JacPrototypeSparsityDetection)(::Union{AbstractFiniteDifferencesMode,
        AbstractForwardModeAD}, _, _)
    J = alg.jac_prototype
    colorvec = alg.alg === nothing ? matrix_colors(J) : matrix_colors(J, alg.alg)
    (nz_rows, nz_cols) = collect.(ArrayInterface.findstructralnz(J))
    return MatrixColoringResult(colorvec, J, collect(1:size(J, 2)), nz_rows, nz_cols)
end

function (alg::JacPrototypeSparsityDetection)(::AbstractReverseModeAD, _, _)
    J = alg.jac_prototype
    colorvec = alg.alg === nothing ? matrix_colors(J') :
               matrix_colors(J, alg.alg; partition_by_rows=true)
    (nz_rows, nz_cols) = collect.(ArrayInterface.findstructralnz(J))
    return MatrixColoringResult(colorvec, J, collect(1:size(J, 1)), nz_rows, nz_cols)
end

# Using Symbolics to Generate Sparsity Pattern
