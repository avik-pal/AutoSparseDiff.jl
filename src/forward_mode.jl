function sparse_jacobian_cache(ad::AutoSparseForwardDiff, sd::AbstractSparsityDetection, f,
    x; dx=nothing)
    dx = dx === nothing ? similar(f(x)) : dx
    coloring_result = sd(ad, f, x)
    cache = ForwardColorJacCache(f, x, __chunksize(ad); coloring_result.colorvec, dx,
        sparsity=coloring_result.jacobian_sparsity)
    return cache
end

function sparse_jacobian!(J::AbstractMatrix, f, x, cache::ForwardColorJacCache)
    return forwarddiff_color_jacobian(J, f, x, cache)
end

function sparse_jacobian(ad::AutoSparseForwardDiff, sd::AbstractSparsityDetection, f,
    x)
    cache = sparse_jacobian_cache(ad, sd, f, x)
    J = similar(cache.dx, length(cache.dx), length(x))
    return sparse_jacobian!(J, f, x, cache)
end
