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
abstract type AbstractMaybeSparseJacobianCache end

"""
    sparse_jacobian!(J::AbstractMatrix, f, x, ad, cache::AbstractMaybeSparseJacobianCache)
    sparse_jacobian!(J::AbstractMatrix, f!, fx, x, ad,
        cache::AbstractMaybeSparseJacobianCache)
    sparse_jacobian!(J::AbstractMatrix, ad::AbstractADType, sd::AbstractSparsityDetection,
        f, x; fx=nothing)
    sparse_jacobian!(J::AbstractMatrix, ad::AbstractADType, sd::AbstractSparsityDetection,
        f!, fx, x)

Inplace update the matrix `J` with the Jacobian of `f` at `x` using the AD backend `ad`.

`cache` is the cache object returned by `sparse_jacobian_setup`.
"""
function sparse_jacobian! end

"""
    sparse_jacobian_setup(ad::AbstractADType, sd::AbstractSparsityDetection, f, x;
        fx=nothing)
    sparse_jacobian(ad::AbstractADType, sd::AbstractSparsityDetection, f!, fx, x)

Takes the underlying AD backend `ad`, sparsity detection algorithm `sd`, function `f`,
and input `x` and returns a cache object that can be used to compute the Jacobian.

If `fx` is not specified, it will be computed by calling `f(x)`.

## Returns

A cache for computing the Jacobian of type `AbstractMaybeSparseJacobianCache`.
"""
function sparse_jacobian_setup end

"""
    sparse_jacobian(ad::AbstractADType, sd::AbstractSparsityDetection, f, x;
        fx=nothing)
    sparse_jacobian(ad::AbstractADType, sd::AbstractSparsityDetection, f!, fx, x)

Sequentially calls `sparse_jacobian_setup` and `sparse_jacobian!` to compute the Jacobian of
`f` at `x`. Use this if the jacobian for `f` is computed exactly once. In all other
cases, use `sparse_jacobian_setup` once to generate the cache and use `sparse_jacobian!`
with the same cache to compute the jacobian.
"""
function sparse_jacobian(ad::AbstractADType, sd::AbstractSparsityDetection, f, x;
    fx=nothing)
    cache = sparse_jacobian_setup(ad, sd, f, x; fx)
    J = __init_𝒥(cache)
    return sparse_jacobian!(J, f, x, ad, cache)
end

function sparse_jacobian(ad::AbstractADType, sd::AbstractSparsityDetection, f!, fx, x)
    cache = sparse_jacobian_setup(ad, sd, f!, fx, x)
    J = __init_𝒥(cache)
    return sparse_jacobian!(J, f!, fx, x, ad, cache)
end

function sparse_jacobian!(J::AbstractMatrix, ad::AbstractADType,
    sd::AbstractSparsityDetection, f, x; fx=nothing)
    cache = sparse_jacobian_setup(ad, sd, f, x; fx)
    return sparse_jacobian!(J, f, x, ad, cache)
end

function sparse_jacobian!(J::AbstractMatrix, ad::AbstractADType,
    sd::AbstractSparsityDetection, f!, fx, x)
    cache = sparse_jacobian_setup(ad, sd, f!, fx, x)
    return sparse_jacobian!(J, f!, fx, x, ad, cache)
end

## Internal
function __gradient end
function __gradient! end
function __jacobian! end

function __init_𝒥 end

# Misc Functions
__chunksize(::AutoSparseForwardDiff{C}) where {C} = C

__f̂(f, x, cols) = dot(vec(f(x)), cols)

@generated function __getfield(c::T, ::Val{S}) where {T, S}
    hasfield(T, S) && return :(c.$(S))
    return :(nothing)
end

function __init_𝒥(c::AbstractMaybeSparseJacobianCache)
    T = promote_type(eltype(c.fx), eltype(c.x))
    return __init_𝒥(__getfield(c, Val(:jac_prototype)), T, c.fx, c.x)
end
__init_𝒥(::Nothing, ::Type{T}, fx, x) where {T} = similar(fx, T, length(fx), length(x))
__init_𝒥(J, ::Type{T}, _, _) where {T} = similar(J, T, size(J, 1), size(J, 2))
