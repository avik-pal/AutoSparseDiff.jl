# AutoSparseDiff

**DO NOT USE THIS PACKAGE. THIS IS MOSTLY A TESTBED BEFORE MOVING THINGS TO SparseDiffTools.jl**

Fast Jacobian Computation using Matrix Coloring via:

* Finite Differences
  * FiniteDiff.jl *(native)*
* Forward Mode AD
  * ForwardDiff.jl *(native)*
* Reverse Mode AD
  * Zygote.jl *(via extensions)*

All using a single unified API.

Packages supported via extensions need to be installed separately, and loaded before that
functionality is available.

## Example

```julia
using AutoSparseDiff
using Symbolics # For Sparsity Detection

@views function f(y, x) # in-place
    y[(begin + 1):(end - 1)] .= x[begin:(end - 2)] .- 2 .* x[(begin + 1):(end - 1)] .+
                                x[(begin + 2):end]
    y[begin] = -2 * x[begin] + x[begin + 1]
    y[end] = x[end - 1] - 2 * x[end]
    return nothing
end

@views function g(x) # out-of-place
    y₂ = x[begin:(end - 2)] .- 2 .* x[(begin + 1):(end - 1)] .+ x[(begin + 2):end]
    y₁ = -2x[1] + x[2]
    y₃ = x[end - 1] - 2x[end]
    return vcat(y₁, y₂, y₃)
end

x = randn(Float32, 100);

difftype = AutoSparseFiniteDiff()

# Using setup + jacobian API
sd = SymbolicsSparsityDetection(nothing)

cache = sparse_jacobian_setup(difftype, sd, g, x)
J = similar(x, length(x), length(x))

sparse_jacobian!(J, g, x, difftype, cache)

# Single high-level API
J = sparse_jacobian(AutoSparseFiniteDiff(), sd, g, x)
```

## (Simple) Benchmark

```julia
using BenchmarkTools, AutoSparseDiff, SparseDiffTools
using Zygote, ForwardDiff, FiniteDiff
using Symbolics, Test

@views function f(y, x) # in-place
    y[(begin + 1):(end - 1)] .= x[begin:(end - 2)] .- 2 .* x[(begin + 1):(end - 1)] .+
                                x[(begin + 2):end]
    y[begin] = -2 * x[begin] + x[begin + 1]
    y[end] = x[end - 1] - 2 * x[end]
    return nothing
end

@views function g(x) # out-of-place
    y₂ = x[begin:(end - 2)] .- 2 .* x[(begin + 1):(end - 1)] .+ x[(begin + 2):end]
    y₁ = -2x[1] + x[2]
    y₃ = x[end - 1] - 2x[end]
    return vcat(y₁, y₂, y₃)
end

x = randn(Float32, 100);

t₁ = @belapsed ForwardDiff.jacobian($g, $x)

t₂ = @belapsed Zygote.jacobian($g, $x)

@info "`ForwardDiff.jacobian` time: $(t₁)s"
@info "`Zygote.jacobian` time: $(t₂)s"

# AutoSparseDiff API
J_sparsity = Symbolics.jacobian_sparsity(f, similar(x), x);

for difftype in (AutoSparseZygote(), AutoZygote(), AutoSparseForwardDiff(),
    AutoForwardDiff(), AutoSparseFiniteDiff(), AutoFiniteDiff())
    sd = JacPrototypeSparsityDetection(J_sparsity, nothing)

    cache = sparse_jacobian_setup(difftype, sd, g, x)
    J = AutoSparseDiff.__init_𝒥(cache)

    sparse_jacobian!(J, g, x, difftype, cache)

    @test J ≈ J_true

    t₁ = @belapsed sparse_jacobian!($J, $g, $x, $difftype, $cache)
    @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` time: $(t₁)s"

    J = sparse_jacobian(difftype, sd, g, x)

    @test J ≈ J_true

    t₂ = @belapsed sparse_jacobian($difftype, $sd, $g, $x)
    @info "$(nameof(typeof(difftype)))() `sparse_jacobian` time: $(t₂)s"
end

for difftype in (AutoSparseForwardDiff(), AutoForwardDiff(), AutoSparseFiniteDiff(),
    AutoFiniteDiff())
    sd = JacPrototypeSparsityDetection(J_sparsity, nothing)
    y = similar(x)

    cache = sparse_jacobian_setup(difftype, sd, f, y, x)
    J = AutoSparseDiff.__init_𝒥(cache)

    sparse_jacobian!(J, f, y, x, difftype, cache)

    @test J ≈ J_true

    t₁ = @belapsed sparse_jacobian!($J, $f, $y, $x, $difftype, $cache)
    @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` time: $(t₁)s"

    J = sparse_jacobian(difftype, sd, f, y, x)

    @test J ≈ J_true

    t₂ = @belapsed sparse_jacobian($difftype, $sd, $f, $y, $x)
    @info "$(nameof(typeof(difftype)))() `sparse_jacobian` time: $(t₂)s"
end
```

### Result

```julia
[ Info: `ForwardDiff.jacobian` time: 3.0039e-5s
[ Info: `Zygote.jacobian` time: 0.001191653s

## Out of Place Version

[ Info: AutoSparseZygote() `sparse_jacobian!` time: 7.0368e-5s
[ Info: AutoSparseZygote() `sparse_jacobian` time: 0.000143805s
[ Info: AutoZygote() `sparse_jacobian!` time: 0.001079399s
[ Info: AutoZygote() `sparse_jacobian` time: 0.001092648s
[ Info: AutoSparseForwardDiff() `sparse_jacobian!` time: 6.9178e-5s
[ Info: AutoSparseForwardDiff() `sparse_jacobian` time: 0.000126426s
[ Info: AutoForwardDiff() `sparse_jacobian!` time: 2.6629e-5s
[ Info: AutoForwardDiff() `sparse_jacobian` time: 2.9779e-5s
[ Info: AutoSparseFiniteDiff() `sparse_jacobian!` time: 2.6021111111111115e-6s
[ Info: AutoSparseFiniteDiff() `sparse_jacobian` time: 6.6648e-5s
[ Info: AutoFiniteDiff() `sparse_jacobian!` time: 3.4919e-5s
[ Info: AutoFiniteDiff() `sparse_jacobian` time: 3.6929e-5s

## Inplace Version

[ Info: AutoSparseForwardDiff() `sparse_jacobian!` time: 9.55375e-7s
[ Info: AutoSparseForwardDiff() `sparse_jacobian` time: 7.1688e-5s
[ Info: AutoForwardDiff() `sparse_jacobian!` time: 1.5859e-5s
[ Info: AutoForwardDiff() `sparse_jacobian` time: 1.945e-5s
[ Info: AutoSparseFiniteDiff() `sparse_jacobian!` time: 1.3099000000000002e-6s
[ Info: AutoSparseFiniteDiff() `sparse_jacobian` time: 6.3978e-5s
[ Info: AutoFiniteDiff() `sparse_jacobian!` time: 5.024833333333333e-6s
[ Info: AutoFiniteDiff() `sparse_jacobian` time: 6.2898e-6s
```
