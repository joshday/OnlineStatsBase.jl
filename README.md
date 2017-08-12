# OnlineStatsBase

[![Build Status](https://travis-ci.org/joshday/OnlineStatsBase.jl.svg?branch=master)](https://travis-ci.org/joshday/OnlineStatsBase.jl)
[![codecov](https://codecov.io/gh/joshday/OnlineStatsBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/OnlineStatsBase.jl)

OnlineStats: [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://joshday.github.io/OnlineStats.jl/stable) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://joshday.github.io/OnlineStats.jl/latest)

This package defines the basic types and interface used by [OnlineStats](https://github.com/joshday/OnlineStats.jl).  Extending functionality of OnlineStats should be accomplished through OnlineStatsBase and can thus avoid the extra dependencies.

# Example of creating a mean


### A new OnlineStat needs the following methods:
- `StatsBase.fit!(stat, observation, weight::Float64)`
- `Base.merge!(stat1, stat2, weight::Float64)` (optional)


```julia
using OnlineStatsBase, StatsBase

mutable struct MyMean <: OnlineStat{0, 0, EqualWeight}
    value::Float64
    MyMean() = new(0.0)
end
StatsBase.fit!(o::MyMean, y::Real, w::Float64) = (o.value += w * (y - o.value))
```
### That's all there is to it
```
y = randn(1000)

s = Series(MyMean(), Variance())

for yi in y
    fit!(s, yi)
end

value(s)
mean(y), var(y)
```

# Details

### OnlineStat Parameters
`OnlineStat{InDim, OutDim, Weight}` is parameterized by
- `InDim`: size of a single observation
    - `0` = `Union{Real, AbstractString, Symbol}`
    - `1` = `Union{Vector, NTuple}`,
    - `(0, 1)` = `Real`, `Union{Vector, NTuple}` pair
- `OutDim`: size of output (the value of the stat)
- `Weight`: default weight

### `fit!` and `value`
Many parameters are based on [sufficient statistics](https://en.wikipedia.org/wiki/Sufficient_statistic).  The `fit!` method should update the sufficient statistics and not necessarily the parameter directly.  If `fit!` does not update the parameter directly, an additional method is required:

```julia
OnlineStatsBase._value(stat)
```

The reason for this is to avoid unnecessary calculations while updating an OnlineStat with a batch of observations.
