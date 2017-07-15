# OnlineStatsBase

[![Build Status](https://travis-ci.org/joshday/OnlineStatsBase.jl.svg?branch=master)](https://travis-ci.org/joshday/OnlineStatsBase.jl)
[![codecov](https://codecov.io/gh/joshday/OnlineStatsBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/OnlineStatsBase.jl)


This package defines the base types used by [OnlineStats](https://github.com/joshday/OnlineStats.jl).

Extending functionality of OnlineStats should be accomplished through OnlineStatsBase and can thus avoid the extra dependencies.

# Example of creating a mean

`OnlineStat{InDim, OutDim, Weight}` is parameterized by
- `InDim`: size of a single observation (0=scalar, 1=vector, 2=matrix, (0,1)=scalar-vector pair)
- `OutDim`: size of output (same convention as `InDim`)
- `Weight`: default weight

```julia
using OnlineStatsBase, StatsBase

mutable struct MyMean <: OnlineStat{0, 0, EqualWeight}
    value::Float64
    MyMean() = new(0.0)
end

function StatsBase.fit!(o::MyMean, y::Real, w::Float64)
    o.value = (1 - w) * o.value + w * y
end

using OnlineStats
y = randn(1000)
s = Series(y, MyMean())
@show value(s)
@show mean(y)
```
