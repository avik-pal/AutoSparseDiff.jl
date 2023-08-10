module AutoSparseDiff

using ArrayInterface, Reexport, SparseDiffTools

import ADTypes: AbstractADType
import ArrayInterface: matrix_colors
import ConcreteStructs: @concrete
import SparseDiffTools: ForwardColorJacCache
@reexport using ADTypes

import PackageExtensionCompat
function __init__()
    PackageExtensionCompat.@require_extensions
end

include("common.jl")
include("coloring.jl")
include("forward_mode.jl")
include("reverse_mode.jl")
include("finite_diff.jl")

export NoSparsityDetection,
    SymbolicsSparsityDetection, JacPrototypeSparsityDetection, AutoSparsityDetection
export sparse_jacobian, sparse_jacobian_cache, sparse_jacobian!

end
