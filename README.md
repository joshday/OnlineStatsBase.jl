# OnlineStatsBase

| OnlineStats Docs | Release | Master Build | Test Coverage |
|------------------|---------|--------------|---------------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://joshday.github.io/OnlineStats.jl/stable) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://joshday.github.io/OnlineStats.jl/latest) | [![OnlineStatsBase](https://pkg.julialang.org/badges/OnlineStatsBase_0.6.svg)](https://pkg.julialang.org/detail/OnlineStatsBase) [![OnlineStatsBase](https://pkg.julialang.org/badges/OnlineStatsBase_0.7.svg)](https://pkg.julialang.org/detail/OnlineStatsBase)| [![Build Status](https://travis-ci.org/joshday/OnlineStatsBase.jl.svg?branch=master)](https://travis-ci.org/joshday/OnlineStatsBase.jl) [![Build status](https://ci.appveyor.com/api/projects/status/99i0vq2crpwgqonp/branch/master?svg=true)](https://ci.appveyor.com/project/joshday/onlinestatsbase-jl/branch/master) | [![codecov](https://codecov.io/gh/joshday/OnlineStatsBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/OnlineStatsBase.jl) |


This package defines the basic types and interface for [OnlineStats](https://github.com/joshday/OnlineStats.jl).  

## Interface

### Required Methods
- **`_fit!(stat, y)`**: Update the "sufficient statistics" of the estimator from a single observation `y`.
- **`merge!(stat1, stat2)`** (optional, no default): Merge OnlineStat `stat2` into `stat1`.

### Default Methods
- **`value(stat, args...)`**:  Calculate the value of the estimator from the "sufficient statistics".  By default, this returns the first field of the OnlineStat.
- **`nobs(stat)`**: Return the number of observations.  By default, this returns `stat.n`.



## Basic Example

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

### That's all there is to it

```julia
y = randn(1000)

o = fit!(MyMean(), y)
```
