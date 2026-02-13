[![CI](https://github.com/joshday/OnlineStatsBase.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/joshday/OnlineStatsBase.jl/actions/workflows/CI.yml)
[![Docs Build](https://github.com/joshday/OnlineStatsBase.jl/actions/workflows/Docs.yml/badge.svg)](https://github.com/joshday/OnlineStatsBase.jl/actions/workflows/Docs.yml)
[![Stable Docs](https://img.shields.io/badge/docs-stable-blue)](https://joshday.github.io/OnlineStatsBase.jl/stable/)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue)](https://joshday.github.io/OnlineStatsBase.jl/dev/)

<br>

<h1 align="center">OnlineStatsBase</h1>

This package defines the basic types and interface for [OnlineStats](https://github.com/joshday/OnlineStats.jl).

<br><br>

# Interface

### Required

- **`_fit!(stat, y)`**: Update the "sufficient statistics" of the estimator from a single observation `y`.

#### Required (with Defaults)

- **`value(stat, args...; kw...) = <first field of struct>`**:  Calculate the value of the estimator from the "sufficient statistics".
- **`nobs(stat) = stat.n`**: Return the number of observations.

### Optional

- **`_merge!(stat1, stat2)`**: Merge `stat2` into `stat1` (an error by default in OnlineStatsBase versions >= 1.5).
- **`Base.empty!(stat)`**: Return the stat to its initial state (an error by default).

<br><br>

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

<br><br>

## That's all there is to it!

```julia
y = randn(1000)

o = fit!(MyMean(), y)
# MyMean: n=1_000 | value=0.0530535
```
