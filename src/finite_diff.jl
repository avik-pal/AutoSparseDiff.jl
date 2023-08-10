function sparse_jacobian_setup(fd::Union{AutoSparseFiniteDiff, AutoFiniteDiff},
    sd::AbstractSparsityDetection, f, x; dx=nothing)
    coloring_result = sd(fd, f, x)
    dx = dx === nothing ? similar(f(x)) : dx
    if coloring_result isa NoMatrixColoring
        cache = FiniteDiff.JacobianCache(x, dx)
    else
        cache = FiniteDiff.JacobianCache(x, dx; coloring_result.colorvec,
            sparsity=coloring_result.jacobian_sparsity)
    end
    return cache
end

function sparse_jacobian!(J::AbstractMatrix, f, x, _, cache::FiniteDiff.JacobianCache)
    f!(y, x) = (y .= f(x))
    FiniteDiff.finite_difference_jacobian!(J, f!, x, cache)
    return J
end

function sparse_jacobian(fd::Union{AutoSparseFiniteDiff, AutoFiniteDiff},
    sd::AbstractSparsityDetection, f, x; dx=nothing)
    cache = sparse_jacobian_setup(fd, sd, f, x; dx)
    J = similar(cache.fx, length(cache.fx), length(x))
    sparse_jacobian!(J, f, x, fd, cache)
    return J
end
