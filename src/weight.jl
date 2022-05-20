module Weight

using ..OnlineStatsBase: name

abstract type AbstractWeight end

#-----------------------------------------------------------------------# Equal
"""
    Equal()

Equally weighted observations.

``γ(n) = 1 / n``
"""
struct Equal <: AbstractWeight end
(::Equal)(n) = inv(n)

#-----------------------------------------------------------------------# Exponential
"""
    Exponential(λ::Float64)
    Exponential(lookback::Int)

Exponentially weighted observations.  Each weight is `λ = 2 / (lookback + 1)`.

`Exponential` does not satisfy the usual assumption that `γ(1) == 1`.  Therefore, some
statistics have an implicit starting value.

```
# E.g. Mean has an implicit starting value of 0.
o = Mean(weight=ExponentialWeight(.1))
fit!(o, 10)
value(o) == 1
```

``γ(n) = λ``
"""
struct Exponential <: AbstractWeight
    λ::Float64
    Exponential(λ::Real = .1) = new(λ)
    Exponential(lookback::Integer) = new(2 / (lookback + 1))
end
(w::Exponential)(n) = w.λ
Base.show(io::IO, w::Exponential) = print(io, name(w) * "(λ = $(w.λ))")

#-----------------------------------------------------------------------# LearningRate
"""
    LearningRate(r = .6)

Slowly decreasing weight.  Satisfies the standard stochastic approximation assumption
``∑ γ(t) = ∞, ∑ γ(t)^2 < ∞`` if ``r ∈ (.5, 1]``.

``γ(n) = inv(n ^ r)``
"""
struct LearningRate <: AbstractWeight
    r::Float64
    LearningRate(r = .6) = new(r)
end
(w::LearningRate)(n) = inv(n ^ w.r)
Base.show(io::IO, w::LearningRate) = print(io, name(w) * "(r = $(w.r))")

#-----------------------------------------------------------------------# LearningRate2
"""
    LearningRate2(c = .5)

Slowly decreasing weight.

``γ(n) = inv(1 + c * (n - 1))``
"""
struct LearningRate2 <: AbstractWeight
    c::Float64
    LearningRate2(c = .5) = new(c)
end
(w::LearningRate2)(n) = 1 / (1 + w.c * (n - 1))
Base.show(io::IO, w::LearningRate2) = print(io, name(w) * "(c = $(w.c))")

#-----------------------------------------------------------------------# Harmonic
"""
    Harmonic(a = 10.0)

Weight determined by harmonic series.

``γ(n) = a / (a + n - 1)``
"""
struct Harmonic <: AbstractWeight
    a::Float64
    Harmonic(a = 10.0) = new(a)
end
(w::Harmonic)(n) = w.a / (w.a + n - 1)
Base.show(io::IO, w::Harmonic) = print(io, name(w) * "(a = $(w.a))")

#-----------------------------------------------------------------------# Mcclain
"""
    Mcclain(α = .1)

Weight which decreases into a constant.

``γ(n) = γ(n-1) / (1 + γ(n-1) - α)``
"""
mutable struct Mcclain <: AbstractWeight
    α::Float64
    last::Float64
    Mcclain(α = .1) = new(α, 1.0)
end
(w::Mcclain)(n) = n == 1 ? 1.0 : (w.last = w.last / (1 + w.last - w.α))
Base.show(io::IO, w::Mcclain) = print(io, name(w) * "(α = $(w.α))")

end #module
