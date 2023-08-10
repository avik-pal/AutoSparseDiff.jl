function sparse_jacobian_setup(ad::Union{AutoSparseForwardDiff, AutoForwardDiff},
    sd::AbstractSparsityDetection, f, x; dx=nothing)
    coloring_result = sd(ad, f, x)
    if coloring_result isa NoMatrixColoring
        cache = ForwardDiff.JacobianConfig(f, x)
    else
        dx = dx === nothing ? similar(f(x)) : dx
        cache = ForwardColorJacCache(f, x, __chunksize(ad); coloring_result.colorvec, dx,
            sparsity=coloring_result.jacobian_sparsity)
    end
    return cache
end

function sparse_jacobian!(J::AbstractMatrix, f, x, _, cache::ForwardColorJacCache)
    return forwarddiff_color_jacobian(J, f, x, cache)
end

function sparse_jacobian!(J::AbstractMatrix, f, x, _, cache::ForwardDiff.JacobianConfig)
    return ForwardDiff.jacobian!(J, f, x, cache)
end

function sparse_jacobian(ad::Union{AutoSparseForwardDiff, AutoForwardDiff},
    sd::AbstractSparsityDetection, f, x; dx=nothing)
    cache = sparse_jacobian_setup(ad, sd, f, x; dx)
    if cache isa ForwardColorJacCache
        J = similar(cache.dx, length(cache.dx), length(x))
    else
        dx = dx === nothing ? similar(f(x)) : dx
        J = similar(dx, length(dx), length(x))
    end
    return sparse_jacobian!(J, f, x, ad, cache)
end
