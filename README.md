[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://joshday.github.io/OnlineStats.jl/stable) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://joshday.github.io/OnlineStats.jl/latest) [![Build status](https://github.com/joshday/OnlineStatsBase.jl/workflows/CI/badge.svg)](https://github.com/joshday/OnlineStatsBase.jl/actions?query=workflow%3ACI+branch%3Amaster) [![codecov](https://codecov.io/gh/joshday/OnlineStatsBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/OnlineStatsBase.jl)

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

## A little more complex example...

- Make a subtype of OnlineStat
- `OHLC` is the type of a single observation
- `S` is the type of the "sufficient statistic"
- Give it a `_fit!(::OnlineStat{T}, y::T)` method.

```julia
using OnlineStatsBase

struct OHLC{Tprice}
    open::Tprice
    high::Tprice
    low::Tprice
    close::Tprice
end

mutable struct TypicalPrice{S} <: OnlineStat{OHLC}
    value::S
    n::Int
    TypicalPrice{S}() where {S} = new{S}(0.0, 0)
end
function OnlineStatsBase._fit!(o::TypicalPrice, candle)
    o.n += 1
    o.value = (candle.high + candle.low + candle.close) / 3
end
```
### Usage
```julia
o = TypicalPrice{Float64}()
fit!(o, OHLC{Float64}(10.0, 11.0, 9.0, 10.5))
println(o)
fit!(o, OHLC{Float64}(10.0, 11.0, 9.0, 10.5))
println(o)
```
