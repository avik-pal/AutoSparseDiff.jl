@concrete struct ReverseModeJacobianCache <: AbstractMaybeSparseJacobianCache
    coloring
    cache
    jac_prototype
    fx
    x
end

function sparse_jacobian_setup(ad::AbstractReverseModeAD, sd::AbstractSparsityDetection, f,
    x; fx=nothing)
    fx = fx === nothing ? similar(f(x)) : fx
    coloring_result = sd(ad, f, x)
    jac_prototype = __getfield(coloring_result, Val(:jacobian_sparsity))
    return ReverseModeJacobianCache(coloring_result, nothing, jac_prototype, fx, x)
end

function sparse_jacobian_setup(ad::AbstractReverseModeAD, sd::AbstractSparsityDetection, f!,
    fx, x)
    coloring_result = sd(ad, f!, fx, x)
    jac_prototype = __getfield(coloring_result, Val(:jacobian_sparsity))
    return ReverseModeJacobianCache(coloring_result, nothing, jac_prototype, fx, x)
end

function sparse_jacobian!(J::AbstractMatrix, f, x, ad, cache::ReverseModeJacobianCache)
    if cache.coloring isa NoMatrixColoring
        return __jacobian!(J, ad, f, x)
    else
        return __sparse_jacobian_reverse_impl!(J, f, x, ad, cache.coloring)
    end
end

function sparse_jacobian!(J::AbstractMatrix, f!, fx, x, ad, cache::ReverseModeJacobianCache)
    if cache.coloring isa NoMatrixColoring
        return __jacobian!(J, ad, f!, fx, x)
    else
        return __sparse_jacobian_reverse_impl!(J, f!, fx, x, ad, cache.coloring)
    end
end

function __sparse_jacobian_reverse_impl!(J::AbstractMatrix, f, x, ad,
    cache::MatrixColoringResult)
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

function __sparse_jacobian_reverse_impl!(J::AbstractMatrix, f!, fx, x, ad,
    cache::MatrixColoringResult)
    for c in 1:maximum(cache.colorvec)
        @. cache.idx_vec = cache.colorvec == c
        gs = __gradient!(ad, f!, fx, x, cache.idx_vec)
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
