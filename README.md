# OnlineStatsBase

| OnlineStats Docs | Master Build | Test Coverage |
|------------------|--------------|---------------|
| [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://joshday.github.io/OnlineStats.jl/stable) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://joshday.github.io/OnlineStats.jl/latest) | [![Build Status](https://travis-ci.org/joshday/OnlineStatsBase.jl.svg?branch=master)](https://travis-ci.org/joshday/OnlineStatsBase.jl) [![Build status](https://ci.appveyor.com/api/projects/status/99i0vq2crpwgqonp/branch/master?svg=true)](https://ci.appveyor.com/project/joshday/onlinestatsbase-jl/branch/master) | [![codecov](https://codecov.io/gh/joshday/OnlineStatsBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/OnlineStatsBase.jl) |


## Interface

- This package defines the basic types and interface for [OnlineStats](https://github.com/joshday/OnlineStats.jl).  
- Extending functionality of OnlineStats should be accomplished through OnlineStatsBase.
- If a new OnlineStat `o` implements the following interface, it will automatically work within [OnlineStats](https://github.com/joshday/OnlineStats.jl) and [JuliaDB](https://github.com/JuliaComputing/JuliaDB.jl):

---

- `fit!(o, y, w)`: Update the "sufficient statistics" of the estimator from a single observation `y` and arbitrary weight (in (0, 1]) `w`.
- `value(o, args...)` (optional):  Calculate the value of the estimator from the "sufficient statistics".  By default, this returns the first field of the OnlineStat.
- `merge!(o1, o2, w)` (optional, no default): Merge OnlineStat `o2` into `o1` where `w` (in (0, 1]) is the amount of influence `o2` has over `o1`.
- `default_weight(o)` (optional): The default weighting mechanism of the OnlineStat.
  - For `<: ExactStat{N}`, something that can reproduce the same estimate as its offline counterpart, this is `EqualWeight()`.
  - For `<: StochasticStat{N}`, something that uses stochastic approximation, this is `LearningRate(.6)`



## Basic Example

### Make a subtype of OnlineStat and give it a `fit!` method.

```julia
import OnlineStatsBase: ExactStat, fit!

mutable struct MyMean <: ExactStat{0}
    value::Float64
    MyMean() = new(0.0)
end
fit!(o::MyMean, y, w) = (o.value += w * (y - o.value))
```

### That's all there is to it

```julia
using OnlineStats

y = randn(1000)

s = Series(MyMean(), Counter(), Variance())

for yi in y
    fit!(s, yi)
end

value(s)
mean(y), nobs(s), var(y)
```

## Other Notes

- An OnlineStat is parameterized by the "size" of a single observation (e.g. `Mean <: ExactStat{0}`).
  - 0 (scalar): Anything that's not a `VectorOb`
  - 1 (vector): an `VectorOb = Union{AbstractVector, Tuple, NamedTuple}`
  - (1, 0) (vector/scalar pair): `Tuple{VectorOb, Any}`
