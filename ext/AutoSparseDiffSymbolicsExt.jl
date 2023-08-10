module AutoSparseDiffSymbolicsExt

using AutoSparseDiff, Symbolics
import AutoSparseDiff: AbstractSparseADType

function (alg::SymbolicsSparsityDetection)(ad::AbstractSparseADType, f, x, args...;
    dx=nothing, kwargs...)
    dx = dx === nothing ? similar(f(x)) : dx
    f!(y, x) = (y .= f(x))
    J = Symbolics.jacobian_sparsity(f!, dx, x)
    _alg = JacPrototypeSparsityDetection(J, alg.alg)
    return _alg(ad, f, x, args...; dx, kwargs...)
end

end
