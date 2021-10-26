
| Docs | Build | Tests |
|------|--------|-------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://joshday.github.io/OnlineStats.jl/stable) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://joshday.github.io/OnlineStats.jl/latest)  | [![Build status](https://github.com/joshday/OnlineStatsBase.jl/workflows/CI/badge.svg)](https://github.com/joshday/OnlineStatsBase.jl/actions?query=workflow%3ACI+branch%3Amaster) | [![codecov](https://codecov.io/gh/joshday/OnlineStatsBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/OnlineStatsBase.jl) |

# OnlineStatsBase

This package defines the basic types and interface for [OnlineStats](https://github.com/joshday/OnlineStats.jl).

# Interface

### Required
- **`_fit!(stat, y)`**: Update the "sufficient statistics" of the estimator from a single observation `y`.

### Optional
- **`_merge!(stat1, stat2)`** Merge `stat2` into `stat1`.  By default, a warning will occur.

### Defaults
- **`value(stat, args...; kw...)`**:  Calculate the value of the estimator from the "sufficient statistics".  By default, this returns the first field of the OnlineStat.
- **`nobs(stat)`**: Return the number of observations.  By default, this returns `stat.n`.



# Example

- Make a subtype of OnlineStat and give it a `_fit!(::OnlineStat{T}, y::T)` method.
- `T` is the type of a single observation.  Make sure it's adequately wide.

```julia
using OnlineStatsBase

mutable struct MyMean <: OnlineStat{Number}
    value::Float64
    n::Int
    MyMean() = new(0.0, 0)
end
function OnlineStatsBase._fit!(o::MyMean, y)
    o.n += 1
    o.value += (1 / o.n) * (y - o.value)
end
```

## That's all there is to it!

```julia
y = randn(1000)

o = fit!(MyMean(), y)
```
