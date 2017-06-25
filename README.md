# OnlineStatsBase

[![Build Status](https://travis-ci.org/joshday/OnlineStatsBase.jl.svg?branch=master)](https://travis-ci.org/joshday/OnlineStatsBase.jl)
<!-- [![codecov.io](http://codecov.io/github/joshday/OnlineStatsBase.jl/coverage.svg?branch=master)](http://codecov.io/github/joshday/OnlineStatsBase.jl?branch=master) -->


This package contains the abstract types used by OnlineStats.

```julia
abstract type Weight end
abstract type OnlineStat{INDIM, OUTDIM} end
abstract type StochasticStat{I, O} <: OnlineStat{I, O} end
abstract type AbstractSeries end
```
