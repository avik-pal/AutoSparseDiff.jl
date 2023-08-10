# See upstream issue to fix this https://github.com/SciML/ADTypes.jl/issues/14
## Reverse Mode
struct AutoSparseZygote <: AbstractADType end

const AbstractSparseReverseModeAD = Union{AutoSparseZygote}
const AbstractReverseModeAD = Union{AutoEnzyme, AutoZygote, AutoTracker, AutoReverseDiff,
    AbstractSparseReverseModeAD}

## Forward Mode
const AbstractSparseForwardModeAD = Union{AutoSparseForwardDiff}
const AbstractForwardModeAD = Union{AutoForwardDiff, AbstractSparseForwardModeAD}

## Finite Differences
const AbstractSparseFiniteDifferencesMode = Union{AutoSparseFiniteDiff}
const AbstractFiniteDifferencesMode = Union{AutoFiniteDiff, AutoFiniteDifferences,
    AbstractSparseFiniteDifferencesMode}

const AbstractSparseADType = Union{AbstractSparseReverseModeAD, AbstractSparseForwardModeAD,
    AbstractSparseFiniteDifferencesMode}

# Sparsity Detection
abstract type AbstractMaybeSparsityDetection end
abstract type AbstractSparsityDetection <: AbstractMaybeSparsityDetection end

struct NoSparsityDetection <: AbstractMaybeSparsityDetection end
@concrete struct SymbolicsSparsityDetection <: AbstractSparsityDetection
    alg
end
@concrete struct JacPrototypeSparsityDetection <: AbstractSparsityDetection
    jac_prototype
    alg
end
@concrete struct AutoSparsityDetection <: AbstractMaybeSparsityDetection
    alg
end

# Function Specifications
function sparse_jacobian end
function sparse_jacobian! end
function sparse_jacobian_setup end

function __gradient end
function __jacobian! end

# Misc Functions
__chunksize(::AutoSparseForwardDiff{C}) where {C} = C

__f̂(f, x, cols) = dot(vec(f(x)), cols)
