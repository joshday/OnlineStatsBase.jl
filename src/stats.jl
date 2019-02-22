#-----------------------------------------------------------------------# Mean
"""
    Mean(T = Float64; weight=EqualWeight())

Track a univariate mean, stored as type `T`.

# Example

    @time fit!(Mean(), randn(10^6))
"""
mutable struct Mean{T,W} <: OnlineStat{Number}
    μ::T
    weight::W
    n::Int
end
Mean(T::Type{<:Number} = Float64; weight = EqualWeight()) = Mean(zero(T), weight, 0)
_fit!(o::Mean{T}, x) where {T} = (o.μ = smooth(o.μ, x, T(o.weight(o.n += 1))))
function _merge!(o::Mean, o2::Mean)
    o.n += o2.n
    o.μ = smooth(o.μ, o2.μ, o2.n / o.n)
end
Statistics.mean(o::Mean) = o.μ
Base.copy(o::Mean) = Mean(o.μ, o.weight, o.n)

#-----------------------------------------------------------------------# Variance
"""
    Variance(T = Float64; weight=EqualWeight())

Univariate variance, tracked as type `T`.

# Example

    o = fit!(Variance(), randn(10^6))
    mean(o)
    var(o)
    std(o)
"""
mutable struct Variance{T, W} <: OnlineStat{Number}
    σ2::T
    μ::T
    weight::W
    n::Int
end
function Variance(T::Type{<:Number} = Float64; weight = EqualWeight()) 
    Variance(zero(T), zero(T), weight, 0)
end
Base.copy(o::Variance) = Variance(o.σ2, o.μ, o.weight, o.n)
function _fit!(o::Variance{T}, x) where {T}
    μ = o.μ
    γ = T(o.weight(o.n += 1))
    o.μ = smooth(o.μ, T(x), γ)
    o.σ2 = smooth(o.σ2, (T(x) - o.μ) * (T(x) - μ), γ)
end
function _merge!(o::Variance, o2::Variance)
    γ = o2.n / (o.n += o2.n)
    δ = o2.μ - o.μ
    o.σ2 = smooth(o.σ2, o2.σ2, γ) + δ ^ 2 * γ * (1.0 - γ)
    o.μ = smooth(o.μ, o2.μ, γ)
    o
end
value(o::Variance) = o.n > 1 ? o.σ2 * bessel(o) : 1.0
Statistics.var(o::Variance) = value(o)
Statistics.mean(o::Variance) = o.μ