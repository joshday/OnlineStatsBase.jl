
#-----------------------------------------------------------------------------# Maximum
mutable struct Maximum{T} <: ExactStat{T}
    value::T
end
Maximum(T::Type=Float64) = Maximum(typemin(T))
_fit!(o::Maximum, y) = (o.value = max(o.value, y))
_merge!(a::Maximum, b::Maximum) = (a.value = max(a.value, b.value))

#-----------------------------------------------------------------------------# Mean
"""
    Mean(T=Float64)

"""
mutable struct Mean{T, W} <: ExactStat{Number}
    value::T
    weight::W
    n::Int
end
Mean(T::Type=Float64; weight=Weight.Equal()) = Mean(zero(T) / one(T), weight, 0)
_fit!(o::Mean, y) = (o.value = smooth(o.value, y, o.weight(o.n += 1)))
_merge!(a::Mean, b::Mean) = (a.value += (b.n / (a.n += b.n)) * (b.value - a.value))
showparams(o::Mean) = false
keyvalues(o::Mean) = (; nobs=o.n)
Statistics.mean(o::Mean) = value(o)

#-----------------------------------------------------------------------------# Minimum
mutable struct Minimum{T} <: ExactStat{T}
    value::T
end
Minimum(T::Type=Float64) = Minimum(typemax(T))
_fit!(o::Minimum, y) = (o.value = min(o.value, y))
_merge!(a::Minimum, b::Minimum) = (a.value = min(a.value, b.value))


#-----------------------------------------------------------------------------# Nobs
"""
    Nobs(T)

Track the number of observations of type `T`.
"""
mutable struct Nobs{T} <: ExactStat{T}
    n::Int
    Nobs{T}() where {T} = new{T}(0)
    Nobs(T::Type) = new{T}(0)
    Nobs(data) = new{eltype(data)}(length(data))
end
_fit!(o::Nobs{T}, ::T) where {T} = (o.n += 1)
_merge!(a::Nobs{T}, b::Nobs{T}) where {T} = (a.n += b.n)

#-----------------------------------------------------------------------------# Variance
mutable struct Variance{T, S} <: ExactStat{Number}
    σ2::S  # store the biased variance
    μ::T
    n::Int
end
Variance(T::Type{<:Number} = Float64) = Variance(zero(T) ^ 2 / one(T), zero(T) / one(T), 0)
Base.copy(o::Variance) = Variance(o.σ2, o.μ, o.weight, o.n)
function _fit!(o::Variance{T}, x, w::Number) where {T}
    μ = o.μ
    o.n += 1
    o.μ = smooth(o.μ, T(x), w)
    o.σ2 = smooth(o.σ2, (T(x) - o.μ) * (T(x) - μ), w)
end
function _merge!(o::Variance, o2::Variance)
    γ = o2.n / (o.n += o2.n)
    δ = o2.μ - o.μ
    o.σ2 = smooth(o.σ2, o2.σ2, γ) + δ ^ 2 * γ * (1.0 - γ)
    o.μ = smooth(o.μ, o2.μ, γ)
end
value(o::Variance{T}) where {T} = o.n > 1 ? o.σ2 * T(bessel(o.n)) : Inf
Statistics.var(o::Variance) = value(o)
Statistics.mean(o::Variance) = o.μ
