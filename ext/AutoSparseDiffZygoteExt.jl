module AutoSparseDiffZygoteExt

using AutoSparseDiff, Zygote
import AutoSparseDiff: __f̂, __jacobian!, __gradient, AutoSparseZygote, AutoZygote

function __gradient(::Union{AutoSparseZygote, AutoZygote}, f, x, cols)
    _, ∂x, _ = Zygote.gradient(__f̂, f, x, cols)
    return vec(∂x)
end

# Zygote doesn't provide a way to accumulate directly into `J`. So we modify the code from
# https://github.com/FluxML/Zygote.jl/blob/82c7a000bae7fb0999275e62cc53ddb61aed94c7/src/lib/grad.jl#L140-L157C4
import Zygote: _jvec, _eyelike, _gradcopy!

@views function __jacobian!(J::AbstractMatrix, ::Union{AutoSparseZygote, AutoZygote}, f, x)
    y, back = Zygote.pullback(_jvec ∘ f, x)
    δ = _eyelike(y)
    for k in LinearIndices(y)
        grad = only(back(δ[:, k]))
        _gradcopy!(J[k, :], grad)
    end
    return J
end

end
