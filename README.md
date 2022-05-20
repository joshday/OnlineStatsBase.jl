[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://joshday.github.io/OnlineStats.jl/stable) [![](https://img.shields.io/badge/docs-latest-blue.svg)](https://joshday.github.io/OnlineStats.jl/latest) [![Build status](https://github.com/joshday/OnlineStatsBase.jl/workflows/CI/badge.svg)](https://github.com/joshday/OnlineStatsBase.jl/actions?query=workflow%3ACI+branch%3Amaster) [![codecov](https://codecov.io/gh/joshday/OnlineStatsBase.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/joshday/OnlineStatsBase.jl)

<br>

<h1 align="center">OnlineStatsBase</h1>

This package defines the basic types and interface for [OnlineStats](https://github.com/joshday/OnlineStats.jl).

<br><br>

## Interface

- Every statistic is an `OnlineStat{T}` where `T` is the type of a single observation.
- Statistics should be either `<: ExactStat{T}` or `<: ApproxStat{T}`, depending on whether the online calculation can be equivalent to the offline calculation.
- Required methods:

```julia
# update the "sufficient statistics" from a single observation with weight `w`.
_fit!(o::OnlineStat{T}, x::T, w) where {T}

# Or if weighting isn't possible/defined (e.g. Maximum)
_fit!(o::OnlineStat{T}, x::T) where {T}

# Calculate the value from the "sufficient statistics".
value(o)  # (Returns the first field of the type by default)
```

- Optional methods:

```julia
# Update `a` with the
_merge!(a, b)

# Return statistic to original state
Base.empty!(o)

# Create a copy
Base.copy(o)

# Additional info that should be printed
OnlineStatsBase.keyvalues(o) = (; key=value)
```

<br><br>

## Example

```julia
using OnlineStatsBase

mutable struct Mean <: ExactStat{Real}
    value::Float64
    n::Int
    Mean() = new(0.0, 0)
end
function OnlineStatsBase._fit!(o::Mean, y, w)
    o.n += 1
    o.value += w * (y - o.value)
end
function OnlineStatsBase._merge!(a::Mean, b::Mean)
    a.n += b.n
    a.value += (b.n / a.n) * (b.value - a.value)
end
OnlineStatsBase.keyvalues(o::Mean) = (; nobs=o.n)
```

<br><br>

## That's all there is to it!

```julia
y = randn(10^6)
y2 = randn(10^6)

a = fit!(Mean(), y)
b = fit!(Mean(), y2)

merge!(a, b)
# Mean: value=-0.0021030658536789247 | nobs=2_000_000

mean(vcat(y,y2)) ≈ value(a)
# true
```
