module AutoSparseDiffZygoteExt

using AutoSparseDiff, Zygote
import AutoSparseDiff: __f̂, __jacobian!, __gradient, AutoSparseZygote, AutoZygote

function __gradient(::Union{AutoSparseZygote, AutoZygote}, f, x, cols)
    _, ∂x, _ = Zygote.gradient(__f̂, f, x, cols)
    return vec(∂x)
end

function __jacobian!(J::AbstractMatrix, ::Union{AutoSparseZygote, AutoZygote}, f, x)
    Jz = only(Zygote.jacobian(f, x))
    J .= Jz
    return J
end

end
