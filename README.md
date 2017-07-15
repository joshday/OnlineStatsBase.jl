# OnlineStatsBase

[![Build Status](https://travis-ci.org/joshday/OnlineStatsBase.jl.svg?branch=master)](https://travis-ci.org/joshday/OnlineStatsBase.jl)
[![codecov](https://codecov.io/gh/joshday/OnlineStatsBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/OnlineStatsBase.jl)


This package defines the base types used by [OnlineStats](https://github.com/joshday/OnlineStats.jl).

Extending functionality of OnlineStats should be accomplished through OnlineStatsBase and can thus avoid the extra dependencies.

# Example of creating a mean

A new OnlineStat needs
- a constructor
- a `fit!(stat, observation, weight::Float64)` method.

Note that `OnlineStat{InDim, OutDim, Weight}` is parameterized by
- `InDim`: size of a single observation (0=scalar, 1=vector, (0,1)=scalar-vector pair)
- `OutDim`: size of output (same convention as `InDim`)
- `Weight`: default weight

```julia
using OnlineStatsBase, StatsBase

mutable struct MyMean <: OnlineStat{0, 0, EqualWeight}
    value::Float64
    MyMean() = new(0.0)
end
StatsBase.fit!(o::MyMean, y::Real, w::Float64) = (o.value += w * (y - o.value))

# Just like that, it works
using OnlineStats
y = randn(1000)

s = Series(MyMean(), Variance())
for yi in y
    fit!(s, yi)
end

value(s)
mean(y), var(y)
```
