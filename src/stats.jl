#--------------------------------------------------------------------# Mean
"""
    Mean()
Univariate mean.
    s = Series(randn(100), Mean())
    value(s)
"""
mutable struct Mean <: OnlineStat{0, EqualWeight}
    μ::Float64
    Mean() = new(0.0)
end
fit!(o::Mean, y::Real, γ::Float64) = (o.μ = smooth(o.μ, y, γ))
Base.merge!(o::Mean, o2::Mean, γ::Float64) = (fit!(o, value(o2), γ); o)
Base.mean(o::Mean) = value(o)

#--------------------------------------------------------------------# Variance
"""
    Variance()
Univariate variance.
    s = Series(randn(100), Variance())
    value(s)
"""
mutable struct Variance <: OnlineStat{0, EqualWeight}
    σ2::Float64     # biased variance
    μ::Float64
    nobs::Int
    Variance() = new(0.0, 0.0, 0)
end
function fit!(o::Variance, y::Real, γ::Float64)
    μ = o.μ
    o.nobs += 1
    o.μ = smooth(o.μ, y, γ)
    o.σ2 = smooth(o.σ2, (y - o.μ) * (y - μ), γ)
end
function Base.merge!(o::Variance, o2::Variance, γ::Float64)
    o.nobs += o2.nobs
    δ = o2.μ - o.μ
    o.σ2 = smooth(o.σ2, o2.σ2, γ) + δ ^ 2 * γ * (1.0 - γ)
    o.μ = smooth(o.μ, o2.μ, γ)
    o
end
value(o::Variance) = o.σ2 * unbias(o)
Base.var(o::Variance) = value(o)
Base.std(o::Variance) = sqrt(var(o))
Base.mean(o::Variance) = o.μ
nobs(o::Variance) = o.nobs


#-------------------------------------------------------------------------# CovMatrix
"""
    CovMatrix(d)
Covariance Matrix of `d` variables.
    y = randn(100, 5)
    Series(y, CovMatrix(5))
"""
mutable struct CovMatrix <: OnlineStat{1, EqualWeight}
    value::Matrix{Float64}
    cormat::Matrix{Float64}
    A::Matrix{Float64}  # X'X / n
    b::Vector{Float64}  # X * 1' / n (column means)
    nobs::Int
    CovMatrix(p::Integer) = new(zeros(p, p), zeros(p, p), zeros(p, p), zeros(p), 0)
end
function fit!(o::CovMatrix, x::VectorOb, γ::Float64)
    smooth!(o.b, x, γ)
    smooth_syr!(o.A, x, γ)
    o.nobs += 1
    o
end
function value(o::CovMatrix)
    o.value[:] = full(Symmetric((o.A - o.b * o.b')))
    scale!(o.value, unbias(o))
end
# Base.length(o::CovMatrix) = length(o.b)  # What is this here for?
Base.mean(o::CovMatrix) = o.b
Base.cov(o::CovMatrix) = value(o)
Base.var(o::CovMatrix) = diag(value(o))
Base.std(o::CovMatrix) = sqrt.(var(o))
function Base.cor(o::CovMatrix)
    copy!(o.cormat, value(o))
    v = 1.0 ./ sqrt.(diag(o.cormat))
    scale!(o.cormat, v)
    scale!(v, o.cormat)
    o.cormat
end
function Base.merge!(o::CovMatrix, o2::CovMatrix, γ::Float64)
    smooth!(o.A, o2.A, γ)
    smooth!(o.b, o2.b, γ)
    o.nobs += o2.nobs
    o
end

#--------------------------------------------------------------------# Extrema
"""
    Extrema()
Maximum and minimum.
    s = Series(randn(100), Extrema())
    value(s)
"""
mutable struct Extrema <: OnlineStat{0, EqualWeight}
    min::Float64
    max::Float64
    Extrema() = new(Inf, -Inf)
end
function fit!(o::Extrema, y::Real, γ::Float64)
    o.min = min(o.min, y)
    o.max = max(o.max, y)
    o
end
function Base.merge!(o::Extrema, o2::Extrema, γ::Float64)
    o.min = min(o.min, o2.min)
    o.max = max(o.max, o2.max)
    o
end
value(o::Extrema) = (o.min, o.max)
Base.extrema(o::Extrema) = value(o)

#--------------------------------------------------------------------# OrderStats
"""
    OrderStats(b)
Average order statistics with batches of size `b`.
    s = Series(randn(1000), OrderStats(10))
    value(s)
"""
mutable struct OrderStats <: OnlineStat{0, EqualWeight}
    value::Vector{Float64}
    buffer::Vector{Float64}
    i::Int
    nreps::Int
    OrderStats(p::Integer) = new(zeros(p), zeros(p), 0, 0)
end
function fit!(o::OrderStats, y::Real, γ::Float64)
    p = length(o.value)
    buffer = o.buffer
    o.i += 1
    buffer[o.i] = y
    if o.i == p
        sort!(buffer)
        o.nreps += 1
        o.i = 0
        smooth!(o.value, buffer, 1 / o.nreps)
    end
    o
end

#--------------------------------------------------------------------# Moments
"""
    Moments()
First four non-central moments.
    s = Series(randn(1000), Moments(10))
    value(s)
"""
mutable struct Moments <: OnlineStat{0, EqualWeight}
    m::Vector{Float64}
    nobs::Int
    Moments() = new(zeros(4), 0)
end
function fit!(o::Moments, y::Real, γ::Float64)
    o.nobs += 1
    @inbounds o.m[1] = smooth(o.m[1], y, γ)
    @inbounds o.m[2] = smooth(o.m[2], y * y, γ)
    @inbounds o.m[3] = smooth(o.m[3], y * y * y, γ)
    @inbounds o.m[4] = smooth(o.m[4], y * y * y * y, γ)
end
Base.mean(o::Moments) = o.m[1]
Base.var(o::Moments) = (o.m[2] - o.m[1] ^ 2) * unbias(o)
Base.std(o::Moments) = sqrt.(var(o))
function skewness(o::Moments)
    v = value(o)
    (v[3] - 3.0 * v[1] * var(o) - v[1] ^ 3) / var(o) ^ 1.5
end
function kurtosis(o::Moments)
    v = value(o)
    (v[4] - 4.0 * v[1] * v[3] + 6.0 * v[1] ^ 2 * v[2] - 3.0 * v[1] ^ 4) / var(o) ^ 2 - 3.0
end
function Base.merge!(o1::Moments, o2::Moments, γ::Float64)
    smooth!(o1.m, o2.m, γ)
    o1.nobs += o2.nobs
    o1
end


# #-----------------------------------------------------------------------# Quantiles
# """
#     Quantiles(q = [.25, .5, .75])  # default algorithm is :MSPI
#     Quantiles{:SGD}(q = [.25, .5, .75])
#     Quantiles{:MSPI}(q = [.25, .5, .75])
# Approximate quantiles via the specified algorithm (`:SGD` or `:MSPI`).
#     s = Series(randn(10_000), Quantiles(.1:.1:.9)
#     value(s)
# """
# struct Quantiles{T} <: OnlineStat{0, LearningRate}
#     value::Vector{Float64}
#     τvec::Vector{Float64}
#     function Quantiles{T}(value, τvec) where {T}
#         for τ in τvec
#             0 < τ < 1 || throw(ArgumentError("provided quantiles must be in (0, 1)"))
#         end
#         new(value, τvec)
#     end
# end
# Quantiles{T}(τvec = [.25, .5, .75]) where {T} = Quantiles{T}(zeros(τvec), collect(τvec))
# Quantiles(τvec= [.25, .5, .75]) = Quantiles{:MSPI}(collect(τvec))
#
# function fit!(o::Quantiles{:SGD}, y::Float64, γ::Float64)
#     γ == 1.0 && fill!(o.value, y)
#     for i in eachindex(o.τvec)
#         @inbounds o.value[i] -= γ * deriv(QuantileLoss(o.τvec[i]), y, o.value[i])
#     end
# end
# function fit!(o::Quantiles{:MSPI}, y::Real, γ::Float64)
#     γ == 1.0 && fill!(o.value, y)
#     for i in eachindex(o.τvec)
#         w = abs(y - o.value[i]) + ϵ
#         b = o.τvec[i] - .5 * (1 - y / w)
#         o.value[i] = (o.value[i] + γ * b) / (1 + γ / 2w)
#     end
# end
# TODO
# function fit!(o::Quantiles{:OMAP}, y::Real, γ::Float64)
#     for i in eachindex(o.τvec)
#         u = y - o.value[i]
#         l = QuantileLoss(o.τvec[i])
#         c = (value(l, -u) - value(l, u) - 2deriv(l, u) * u) / (2 * u ^ 2)
#         o.value[i] -= γ * deriv(l, u) / c
#     end
# end
#
# function Base.merge!(o::Quantiles, o2::Quantiles, γ::Float64)
#     o.τvec == o2.τvec || throw(ArgumentError("objects track different quantiles"))
#     smooth!(o.value, o2.value, γ)
# end

#--------------------------------------------------------------------# QuantileMM
"""
    QuantileMM(q = 0.5)
Approximate quantiles via an online MM algorithm.
    s = Series(randn(1000), QuantileMM())
    value(s)
"""
mutable struct QuantileMM <: OnlineStat{0, LearningRate}
    value::Vector{Float64}
    τ::Vector{Float64}
    s::Vector{Float64}
    t::Vector{Float64}
    QuantileMM(τ = [.25, .5, .75]) = new(zeros(τ), τ, zeros(τ), zeros(τ))
end
function fit!(o::QuantileMM, y::Real, γ::Float64)
    γ == 1.0 && fill!(o.value, y)
    @inbounds for j in 1:length(o.τ)
        w = 1.0 / (abs(y - o.value[j]) + ϵ)
        o.s[j] = smooth(o.s[j], w * y, γ)
        o.t[j] = smooth(o.t[j], w, γ)
        o.value[j] = (o.s[j] + (2.0 * o.τ[j] - 1.0)) / o.t[j]
    end
end



#--------------------------------------------------------------------# Diff
"""
    Diff()
Track the difference and the last value.
    s = Series(randn(1000), Diff())
    value(s)
"""
mutable struct Diff{T <: Real} <: OnlineStat{0, EqualWeight}
    diff::T
    lastval::T
end
Diff() = Diff(0.0, 0.0)
Diff{T<:Real}(::Type{T}) = Diff(zero(T), zero(T))
Base.last(o::Diff) = o.lastval
Base.diff(o::Diff) = o.diff
function fit!{T<:AbstractFloat}(o::Diff{T}, x::Real, γ::Float64)
    v = convert(T, x)
    o.diff = v - last(o)
    o.lastval = v
end
function fit!{T<:Integer}(o::Diff{T}, x::Real, γ::Float64)
    v = round(T, x)
    o.diff = v - last(o)
    o.lastval = v
end

#--------------------------------------------------------------------# Sum
"""
    Sum()
Track the overall sum.
    s = Series(randn(1000), Sum())
    value(s)
"""
mutable struct Sum{T <: Real} <: OnlineStat{0, EqualWeight}
    sum::T
end
Sum() = Sum(0.0)
Sum{T<:Real}(::Type{T}) = Sum(zero(T))
Base.sum(o::Sum) = o.sum
fit!{T<:AbstractFloat}(o::Sum{T}, x::Real, γ::Float64) = (v = convert(T, x); o.sum += v)
fit!{T<:Integer}(o::Sum{T}, x::Real, γ::Float64) =       (v = round(T, x);   o.sum += v)
Base.merge!{T <: Sum}(o::T, o2::T, γ::Float64) = (o.sum += o2.sum)

#-----------------------------------------------------------------------# Hist
"""
    OHistogram(range)
Make a histogram with bins given by `range`.  Uses left-closed bins.
    y = randn(100)
    s = Series(y, OHistogram(-4:.1:4))
    value(s)
"""
struct OHistogram{H <: Histogram} <: OnlineStat{0, EqualWeight}
    h::H
end
OHistogram(r::Range) = OHistogram(Histogram(r, :left))
function fit!(o::OHistogram, y::ScalarOb, γ::Float64)
    H = o.h
    x = H.edges[1]
    a = first(x)
    δ = step(x)
    k = floor(Int, (y - a) / δ) + 1
    if 1 <= k <= length(x)
        @inbounds H.weights[k] += 1
    end
end
