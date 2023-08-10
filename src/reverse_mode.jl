function sparse_jacobian_setup(ad::AbstractReverseModeAD, sd::AbstractSparsityDetection, f,
    x; dx=nothing)
    return sd(ad, f, x)
end

function sparse_jacobian!(J::AbstractMatrix, f, x, ad, cache::MatrixColoringResult)
    for c in 1:maximum(cache.colorvec)
        @. cache.idx_vec = cache.colorvec == c
        gs = __gradient(ad, f, x, cache.idx_vec)
        pick_idxs = [i for i in 1:length(cache.nz_rows)
                         if cache.colorvec[cache.nz_rows[i]] == c]
        row_idxs = cache.nz_rows[pick_idxs]
        col_idxs = cache.nz_cols[pick_idxs]
        len_cols = length(col_idxs)
        unused_cols = setdiff(1:size(J, 2), col_idxs)
        perm_cols = sortperm(vcat(col_idxs, unused_cols))
        row_idxs = vcat(row_idxs, zeros(Int, size(J, 2) - len_cols))[perm_cols]
        # FIXME: Assumes fast scalar indexing currently. Very easy to write a kernel to do
        #        this in parallel using KA.jl.
        for i in axes(J, 1), j in axes(J, 2)
            i == row_idxs[j] && (J[i, j] = gs[j])
        end
    end
    return J
end

function sparse_jacobian!(J::AbstractMatrix, f, x, ad, cache::NoMatrixColoring)
    return __jacobian!(J, ad, f, x)
end

function sparse_jacobian(ad::AbstractReverseModeAD, sd::AbstractSparsityDetection, f, x;
    dx=nothing)
    cache = sparse_jacobian_setup(ad, sd, f, x; dx)
    if cache isa NoMatrixColoring
        dx = dx === nothing ? similar(f(x)) : dx
        J = similar(dx, length(dx), length(x))
    else
        if ad isa AbstractSparseReverseModeAD
            J = similar(cache.jacobian_sparsity, eltype(x))
        else
            J = similar(x, size(cache.jacobian_sparsity))
        end
    end
    return sparse_jacobian!(J, f, x, ad, cache)
end
