@concrete struct ForwardDiffJacobianCache <: AbstractMaybeSparseJacobianCache
    coloring
    cache
    jac_prototype
    fx
    x
end

function sparse_jacobian_setup(ad::Union{AutoSparseForwardDiff, AutoForwardDiff},
    sd::AbstractSparsityDetection, f, x; fx=nothing)
    coloring_result = sd(ad, f, x)
    fx = fx === nothing ? similar(f(x)) : fx
    if coloring_result isa NoMatrixColoring
        cache = ForwardDiff.JacobianConfig(f, x)
        jac_prototype = nothing
    else
        cache = ForwardColorJacCache(f, x, __chunksize(ad); coloring_result.colorvec, dx=fx,
            sparsity=coloring_result.jacobian_sparsity)
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return ForwardDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian_setup(ad::Union{AutoSparseForwardDiff, AutoForwardDiff},
    sd::AbstractSparsityDetection, f!, fx, x)
    coloring_result = sd(ad, f!, fx, x)
    if coloring_result isa NoMatrixColoring
        cache = ForwardDiff.JacobianConfig(f!, fx, x)
        jac_prototype = nothing
    else
        cache = ForwardColorJacCache(f!, x, __chunksize(ad); coloring_result.colorvec,
            dx=fx, sparsity=coloring_result.jacobian_sparsity)
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return ForwardDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian!(J::AbstractMatrix, f, x, _, cache::ForwardDiffJacobianCache)
    if cache.cache isa ForwardColorJacCache
        forwarddiff_color_jacobian(J, f, x, cache.cache) # Use Sparse ForwardDiff
    else
        ForwardDiff.jacobian!(J, f, x, cache.cache) # Don't try to exploit sparsity
    end
    return J
end

function sparse_jacobian!(J::AbstractMatrix, f!, fx, x, _, cache::ForwardDiffJacobianCache)
    if cache.cache isa ForwardColorJacCache
        forwarddiff_color_jacobian!(J, f!, x, cache.cache) # Use Sparse ForwardDiff
    else
        ForwardDiff.jacobian!(J, f!, fx, x, cache.cache) # Don't try to exploit sparsity
    end
    return J
end
