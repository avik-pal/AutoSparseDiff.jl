@concrete struct FiniteDiffJacobianCache <: AbstractMaybeSparseJacobianCache
    coloring
    cache
    jac_prototype
    fx
    x
end

function sparse_jacobian_setup(fd::Union{AutoSparseFiniteDiff, AutoFiniteDiff},
    sd::AbstractSparsityDetection, f, x; fx=nothing)
    coloring_result = sd(fd, f, x)
    fx = fx === nothing ? similar(f(x)) : fx
    if coloring_result isa NoMatrixColoring
        cache = FiniteDiff.JacobianCache(x, fx)
        jac_prototype = nothing
    else
        cache = FiniteDiff.JacobianCache(x, fx; coloring_result.colorvec,
            sparsity=coloring_result.jacobian_sparsity)
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return FiniteDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian!(J::AbstractMatrix, f, x, _, cache::FiniteDiffJacobianCache)
    f!(y, x) = (y .= f(x))
    FiniteDiff.finite_difference_jacobian!(J, f!, x, cache.cache)
    return J
end

function sparse_jacobian(fd::Union{AutoSparseFiniteDiff, AutoFiniteDiff},
    sd::AbstractSparsityDetection, f, x; fx=nothing)
    cache = sparse_jacobian_setup(fd, sd, f, x; fx)
    J = __init_𝒥(cache)
    return sparse_jacobian!(J, f, x, fd, cache)
end
