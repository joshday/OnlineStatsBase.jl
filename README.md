# OnlineStatsBase

| OnlineStats Documentation | Master Build | Test Coverage |
|-------|---------------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://joshday.github.io/OnlineStats.jl/stable) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://joshday.github.io/OnlineStats.jl/latest) | [![Build Status](https://travis-ci.org/joshday/OnlineStatsBase.jl.svg?branch=master)](https://travis-ci.org/joshday/OnlineStatsBase.jl) [![Build status](https://ci.appveyor.com/api/projects/status/99i0vq2crpwgqonp/branch/master?svg=true)](https://ci.appveyor.com/project/joshday/onlinestatsbase-jl/branch/master) | [![codecov](https://codecov.io/gh/joshday/OnlineStatsBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/OnlineStatsBase.jl) |


This package defines the basic types and interface used by [OnlineStats](https://github.com/joshday/OnlineStats.jl).  Extending functionality of OnlineStats should be accomplished through OnlineStatsBase and can thus avoid the extra dependencies.



# Creating a new OnlineStat
```julia
using OnlineStatsBase

mutable struct MyMean <: OnlineStat{0, EqualWeight}
    value::Float64
    MyMean() = new(0.0)
end

OnlineStatsBase.fit!(o::MyMean, y::Real, w::Float64) = (o.value += w * (y - o.value))

# optional
Base.merge!(o::MyMean, o2::MyMean, w::Float64) = (o.value += w * (o2.value - o.value))
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

- An OnlineStat is parameterized by the size of a single observation.
  - 0: a `Number`, `Symbol`, or `String`
  - 1: an `AbstractVector` or `Tuple`
  - (1, 0): one of each
- OnlineStat Interface
  - `fit!(o, new_observation, weight)`
    - Update the "sufficient statistics", not necessarily the value
  - `value(o)`
    - Create the value from the "sufficient statistics".  By default, this will return the first field of an OnlineStat
  - `merge!(o1, o2, weight)`
